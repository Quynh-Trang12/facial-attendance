"""Reference collection for session-only and persistent local profiles."""

from dataclasses import dataclass
from time import monotonic
from uuid import uuid4

from ..recognition.evaluation import (
    EvaluationResult,
    RecognitionEvaluation,
    SandboxInference,
    frame_fingerprint,
)
from ..recognition.gallery_matching import empty_gallery
from ..recognition.inference import cosine
from .emotion_presentation import geometry
from .live_preview import SingleWorker
from .liveness_preview import (
    CHECKPOINT_SHA256,
    EPOCH,
    MODEL_ID,
    MODEL_VERSION,
    THRESHOLD,
    valid_spoof_score,
)
from .registration_pose import FacePose
from .webcam_prototype import (
    CONTRACT_VERSION,
    Completion,
    Context,
    Liveness,
    OperationError,
)

REGISTRATION_PREPROCESSING = "aenet-full-yunet-bgr-rgb-bilinear224-div255-fp32-v1"
REGISTRATION_POLICY = "aenet-full-per-capture-threshold-v1"
REGISTRATION_RELEASE = {
    "model_id": MODEL_ID,
    "model_version": MODEL_VERSION,
    "epoch": EPOCH,
    "checkpoint_sha256": CHECKPOINT_SHA256,
    "threshold": THRESHOLD,
    "precision": "cpu_float32",
}


@dataclass(frozen=True)
class Pose:
    instruction: str
    direction: tuple[int, int] | None = None


@dataclass(frozen=True)
class RegistrationGuidance:
    pose: Pose | None
    captured: int
    total: int
    feedback: str


@dataclass(frozen=True)
class ReferenceResult:
    evaluation: EvaluationResult
    pose: FacePose | None


class TrialProfiles:
    # Direction is expressed in the user's mirrored preview coordinates.
    POSES = (
        Pose("Look straight ahead"),
        Pose("Turn slightly left", (-1, 0)),
        Pose("Turn slightly right", (1, 0)),
        Pose("Look slightly up", (0, -1)),
        Pose("Look slightly down", (0, 1)),
    )

    def __init__(
        self,
        parent,
        adapters,
        on_gallery_change,
        *,
        store=None,
        loaded_gallery=None,
        liveness_provenance=None,
        liveness_preview=None,
        pose_estimator=None,
    ):
        import tkinter as tk
        from tkinter import ttk

        self.session = RecognitionEvaluation()
        self.store = store
        self._liveness_provenance = liveness_provenance
        self._liveness_preview = liveness_preview
        self.liveness_threshold = (
            liveness_provenance["threshold"] if self._has_liveness_release() else None
        )
        self.active = False
        self._guidance_feedback = "guidance"
        self._control_states = None
        if loaded_gallery is not None:
            self.session.matcher._gallery = loaded_gallery
        inference = SandboxInference(adapters.face_inference, self.session)
        adapters.matcher = self.session.matcher

        def recognize(request):
            if (
                self.session.reference_hashes
                and frame_fingerprint(request.frame) in self.session.reference_hashes
            ):
                return Completion(
                    request.context,
                    OperationError(
                        "recognition",
                        "invalid_input",
                        "Wait for a frame distinct from the reference captures.",
                        True,
                    ),
                )
            return inference.adapter(request)

        adapters.recognition_adapter = recognize
        self.on_gallery_change = on_gallery_change

        def reference(job):
            evaluation = inference(job)
            pose = pose_estimator(job.request) if pose_estimator is not None else None
            return ReferenceResult(evaluation, pose)

        self.worker = SingleWorker(
            reference if store is not None else inference, "trial-reference"
        )
        self._neutral_pose = None
        self.source = None
        self._attempt_started_at = None
        self._last_face = None
        self._attempt_id = None
        self._frame_liveness = self._pending_liveness = None
        self._accepted_liveness = []
        self._capture_request = None
        self._capture_used = False
        self.save_notice = None
        self.form = ttk.LabelFrame(
            parent,
            text="Local profile" if store else "Test profile · this session only",
            padding=8,
        )
        self.form.grid(row=4, column=0, sticky="new", pady=8)
        self.form.grid_remove()
        self.name = tk.StringVar(master=parent)
        self.consent = tk.BooleanVar(master=parent)
        guidance = (
            "Enter a name and confirm consent to begin."
            if store
            else "Confirm consent, then capture 3–10 photos of one person."
        )
        self.status = tk.StringVar(master=parent, value=guidance)
        ttk.Label(
            self.form,
            text="Display name (up to 24 characters)" if store else "Display name",
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(self.form, textvariable=self.name, width=22).grid(
            row=0, column=1, sticky="w", padx=8
        )
        consent_text = (
            "I am the subject or have consent to store a face template on this device."
            if store
            else "I am the subject or have informed consent."
        )
        ttk.Checkbutton(
            self.form,
            variable=self.consent,
            command=self.change_consent,
            text=consent_text,
        ).grid(row=1, column=0, columnspan=4, sticky="w")
        ttk.Label(self.form, textvariable=self.status, wraplength=600).grid(
            row=2, column=0, columnspan=4, sticky="w"
        )
        self.capture_button = ttk.Button(
            self.form, text="Capture reference", command=self.capture
        )
        self.capture_button.grid(row=3, column=0, sticky="w")
        self.save_button = ttk.Button(
            self.form,
            text="Save profile" if store else "Save test profile",
            command=self.save,
            state="disabled",
        )
        self.save_button.grid(row=3, column=1, sticky="w")
        ttk.Button(
            self.form, text="Retry" if store else "Discard photos", command=self.discard
        ).grid(row=3, column=2, sticky="w")
        ttk.Button(
            self.form,
            text="Cancel" if store else "Clear test profiles",
            command=self.cancel if store else self.clear,
        ).grid(row=3, column=3, sticky="w")

    def show(self):
        if self.store is not None:
            if self.active or self.liveness_threshold is None:
                return
            self.active = True
            self._restart("Enter a name and confirm consent to begin.")
        self.form.grid()

    def observe(self, token, request, captured_at):
        if self.store is not None:
            if not self.active or self._attempt_id is None:
                return
            if request is None:
                self.invalidate_source()
                return
            if captured_at < self._attempt_started_at:
                return
            previous = self._last_face
            if previous is not None:
                last_token, last_at, last_box, last_shape = previous
                if (
                    token[0] != last_token[0]
                    or not 0 < captured_at - last_at <= 1.0
                    or request.frame.shape != last_shape
                    or geometry(last_box, request.bbox)[0] < 0.20
                ):
                    self.invalidate_source()
                    return
            self._last_face = (token, captured_at, request.bbox, request.frame.shape)
            self._frame_liveness = None
            self._capture_used = False
        self.source = (
            (token, request, captured_at)
            if request is not None and self.session.consent
            else None
        )

    def _has_liveness_release(self):
        metadata = self._liveness_provenance
        return isinstance(metadata, dict) and all(
            metadata.get(key) == value for key, value in REGISTRATION_RELEASE.items()
        )

    def observe_liveness(self, token, request, prediction):
        # The webcam supplies the original inference request after checking its worker token.
        if (
            not self.active
            or self._attempt_id is None
            or self.source is None
            or self.source[0] != token
            or self.source[1] is not request
            or self._frame_liveness is not None
        ):
            return
        if (
            not self._has_liveness_release()
            or not isinstance(prediction, dict)
            or prediction.get("source", "model") != "model"
            or any(
                prediction.get(key) != REGISTRATION_RELEASE[key]
                for key in ("model_id", "model_version", "threshold", "precision")
            )
        ):
            self._abort("Liveness verification failed. Select Retry.")
            return

        status = prediction.get("status")
        allowed = prediction.get("allow_attendance")
        score = prediction.get("spoof_score")
        if status == "uncertain" and score is None and allowed is False:
            reason = "uncertain_sample"
        elif (
            status in ("live", "spoof")
            and type(score) in (int, float)
            and valid_spoof_score(prediction, self.liveness_threshold) is not None
            and (status == "live") == (score < self.liveness_threshold)
            and allowed is (status == "live")
        ):
            reason = "accepted_live" if status == "live" else "detected_spoof"
        else:
            self._abort("Invalid liveness result. Select Retry.")
            return

        camera_context = request.context
        context = Context(
            camera_context.contract_version,
            self._attempt_id,
            "registration",
            camera_context.capture_id,
            camera_context.face_id,
        )
        self._frame_liveness = Completion(
            context,
            Liveness(
                status,
                reason,
                score,
                MODEL_VERSION,
                REGISTRATION_PREPROCESSING,
                REGISTRATION_POLICY,
                "model",
            ),
        )
        if self._capture_request is self.source:
            self._finish_capture()
        self._update_controls()

    def invalidate_source(self):
        if self.store is not None:
            if (
                self.active
                and self._attempt_id is not None
                and self._last_face is not None
            ):
                self._restart(
                    "Face lost or changed. Show one face and retake all five photos."
                )
            return
        self.source = None
        if self.session.pending is not None:
            self.session.cancel()
            self.status.set("Capture cancelled. Wait for one face and capture again.")
            self._update_controls()

    def change_consent(self):
        self.session.set_consent(self.consent.get())
        if self.store is not None:
            if not self.active or self._attempt_id is not None:
                self.discard()
            return
        if not self.session.consent:
            self.discard()
            self.source = None
        self._update_controls()

    def capture(self):
        if self.store is not None:
            if not self._can_capture(monotonic()):
                return
            self._capture_request = self.source
            self._capture_used = True
            if self._frame_liveness is None:
                self._show_progress("Checking liveness for the selected photo…")
            else:
                self._finish_capture()
            self._update_controls()
            return

        if not self.session.consent:
            self.status.set("Confirm consent before capturing photos.")
            return
        if self.source is None or self.worker.busy:
            self.status.set("Wait for one detected face and a free capture slot.")
            return
        self._submit_reference(self.source)

    def _finish_capture(self):
        selected = self._capture_request
        self._capture_request = None
        if not self._has_fresh_face(monotonic()):
            self._show_progress(
                "Selected photo expired. Capture a fresh frame.", feedback="retry"
            )
            return

        evidence = self._frame_liveness
        if evidence.result.status == "spoof":
            self._abort(
                "Selected photo was flagged as spoof. Registration aborted. Select Retry.",
                feedback="spoof",
            )
        elif evidence.result.status == "uncertain":
            self._show_progress(
                "Uncertain — retry capture with a fresh frame.", feedback="retry"
            )
        elif self._is_live_for_attempt(evidence):
            self._submit_reference(selected, evidence)
        else:
            self._abort("Reference verification failed. Select Retry.")

    def _submit_reference(self, source, evidence=None):
        token, request, captured_at = source
        try:
            context = evidence.context if evidence is not None else None
            job = self.session.job(
                "reference",
                token,
                request.frame,
                [request.bbox],
                captured_at,
                now=monotonic(),
                registration_context=context,
            )
        except (ValueError, PermissionError) as error:
            if self.store is not None:
                self._show_progress(str(error), feedback="retry")
            else:
                self.status.set(str(error))
            return
        if not self.worker.submit(token, job):
            self.session.cancel()
            return
        self._pending_liveness = evidence
        count = len(self.session.references)
        self._guidance_feedback = "guidance"
        self.status.set(
            f"{count}/{len(self.POSES)} · Processing reference…"
            if self.store
            else "Processing reference…"
        )
        self._update_controls()

    def poll(self):
        if (
            self._capture_request is not None
            and monotonic() - self._capture_request[2] > self.session.MAX_SAMPLE_AGE
        ):
            self._capture_request = None
            self._show_progress(
                "Selected photo expired. Capture a fresh frame.", feedback="retry"
            )
        result = self.worker.take()
        if result is None or self.session.pending is None:
            if self.store is not None:
                self._update_controls()
            return
        current = self.source[0] if self.source is not None else None
        evaluation = result.value
        try:
            if self.store is not None:
                job = self.session.pending
                if result.token != job.source:
                    return
                if result.error is not None:
                    self._abort("Reference inference failed. Select Retry.")
                    return
                if (
                    not isinstance(result.value, ReferenceResult)
                    or result.value.evaluation.job is not job
                ):
                    return
                evaluation = result.value.evaluation
                if (
                    not self._is_live_for_attempt(self._pending_liveness)
                    or self._pending_liveness.context != job.request.context
                ):
                    self._abort("Reference verification failed. Select Retry.")
                    return
                if self.session.references and not self.session.policy.accepts_cosine(
                    cosine(self.session.references[0], evaluation.vector)
                ):
                    self._restart(
                        "Face changed. Retake all five photos of the same person."
                    )
                    return
                expected = self.POSES[len(self.session.references)]
                pose = result.value.pose
                if pose is None:
                    raise ValueError(
                        f"Could not check face direction. {expected.instruction}, then capture again."
                    )
                if not pose.matches(expected.direction, self._neutral_pose):
                    raise ValueError(
                        f"Face direction does not match. {expected.instruction}, then capture again."
                    )
                # Newer frames may arrive while embedding this already-verified capture.
                current = job.source
            elif result.error is not None:
                self.session.cancel()
                raise ValueError("Reference capture failed. Try another photo.")
            self.session.accept(evaluation, current_source=current, now=monotonic())
        except (ValueError, PermissionError) as error:
            if self.store is not None:
                self.session.cancel()
                self._show_progress(str(error), feedback="retry")
            else:
                self.status.set(str(error))
        else:
            count = len(self.session.references)
            if self.store is None:
                self.status.set(
                    f"{count} reference photos captured. Capture more or save once you have at least three."
                )
            else:
                if count == 1:
                    self._neutral_pose = result.value.pose
                self._accepted_liveness.append(
                    (job.request.context, self._pending_liveness)
                )
                message = (
                    f"Sample accepted. {self.POSES[count].instruction}, then select Capture reference."
                    if count < len(self.POSES)
                    else "Registration ready. Select Save profile."
                )
                self._show_progress(message, feedback="accepted")
        self._pending_liveness = None
        self._update_controls()

    def save(self):
        if self.store is not None and not self._can_save(monotonic()):
            return
        name = self.name.get().strip()
        try:
            test_id = self.session.commit_reference(
                display_name=name or None,
                save_gallery=self.store.save if self.store else None,
            )
        except (ValueError, PermissionError) as error:
            if self.store is not None:
                self._show_progress(f"Save failed. {error}", feedback="retry")
            else:
                self.status.set(str(error))
            return
        if self.store is not None:
            self.cancel()
            self.save_notice = f"Registration saved for {name}."
        else:
            self.name.set("")
            self.consent.set(False)
            self.source = None
            self._update_controls()
        self.on_gallery_change()
        scope = "on this device" if self.store else "for this session"
        self.status.set(
            f"Profile {name or test_id} saved {scope}. Look at the camera to check the match."
        )

    def discard(self):
        if self.store is not None:
            self._restart()
            return
        self.session.discard_draft()
        self.status.set(
            "Reference photos discarded. Confirm consent and capture fresh photos."
        )
        self._update_controls()

    def cancel(self):
        self.active = False
        self.name.set("")
        self.consent.set(False)
        self.session.set_consent(False)
        self._restart()
        self.form.grid_remove()

    def _restart(self, message=None):
        self._clear_attempt()
        self.save_notice = None
        self._attempt_id = uuid4().hex
        self._attempt_started_at = monotonic()
        self._show_progress(message)
        self._update_controls()

    def _abort(self, message, *, feedback="guidance"):
        self._clear_attempt()
        self._show_progress(message, feedback=feedback)
        self._update_controls()

    def _clear_attempt(self):
        self.session.discard_draft()
        self._neutral_pose = None
        self._attempt_id = None
        self.source = self._last_face = None
        self._frame_liveness = self._pending_liveness = None
        self._accepted_liveness.clear()
        self._capture_request = None
        self._capture_used = False

    def guidance(self):
        if self.store is None or not self.active:
            return None
        if self._attempt_id is None:
            if self._guidance_feedback == "spoof":
                return RegistrationGuidance(None, 0, len(self.POSES), "spoof")
            return None
        count = len(self.session.references)
        pose = self.POSES[count] if count < len(self.POSES) else None
        return RegistrationGuidance(
            pose, count, len(self.POSES), self._guidance_feedback
        )

    def _show_progress(self, message=None, *, feedback="guidance"):
        self._guidance_feedback = feedback
        count = len(self.session.references)
        if message is None:
            if not self.session.consent:
                message = "Confirm consent before capturing photos."
            elif count < len(self.POSES):
                message = f"{self.POSES[count].instruction}. Wait for a live 3-second scan, then select Capture reference."
            else:
                message = "Registration ready. Select Save profile."
        self.status.set(f"{count}/{len(self.POSES)} · {message}")

    def clear(self):
        if self.store:
            from tkinter import messagebox

            if not messagebox.askyesno(
                "Clear local profiles",
                "Delete all saved local face templates from this device?",
                parent=self.form,
            ):
                return
            try:
                self.store.save(empty_gallery())
            except ValueError as error:
                self.status.set(str(error))
                return
        self.session.reset()
        if self.store is not None:
            self.cancel()
        else:
            self.invalidate_source()
            self.consent.set(False)
            self.name.set("")
            self._update_controls()
        self.on_gallery_change()
        self.status.set("Profiles and reference photos cleared.")

    def close(self):
        self.invalidate_source()
        self.session.reset()
        self.worker.close()

    def _has_fresh_face(self, now):
        return (
            self.source is not None
            and 0 <= now - self.source[2] <= self.session.MAX_SAMPLE_AGE
        )

    def _is_live_for_attempt(self, completion):
        if (
            completion is None
            or self._attempt_id is None
            or not self._has_liveness_release()
        ):
            return False
        context, result = completion.context, completion.result
        return (
            context.contract_version == CONTRACT_VERSION
            and context.attempt_id == self._attempt_id
            and context.action == "registration"
            and bool(context.capture_id)
            and bool(context.face_id)
            and result.status == "live"
            and result.reason == "accepted_live"
            and result.source == "model"
            and result.model_version == MODEL_VERSION
            and result.preprocessing_version == REGISTRATION_PREPROCESSING
            and result.decision_policy_version == REGISTRATION_POLICY
            and type(result.p_spoof) in (int, float)
            and 0 <= result.p_spoof < self.liveness_threshold
        )

    def _has_live_scan(self, now):
        preview = self._liveness_preview
        if (
            preview is None
            or preview.last_check is None
            or preview.last_check_window is None
        ):
            return False
        started_at, completed_at = preview.last_check_window
        return (
            preview.last_check.decision == "live"
            and started_at >= self._attempt_started_at
            and 0
            <= now - completed_at
            <= preview.SCAN_SECONDS + self.session.MAX_SAMPLE_AGE
        )

    def _can_capture(self, now):
        return (
            self.active
            and self._attempt_id is not None
            and self.session.consent
            and not self.worker.busy
            and not self._capture_used
            and self._capture_request is None
            and self._has_fresh_face(now)
            and self._has_live_scan(now)
            and len(self.session.references) < len(self.POSES)
        )

    def _can_save(self, now):
        return (
            self.active
            and self.session.consent
            and self.session.pending is None
            and self._capture_request is None
            and len(self.session.references)
            == len(self._accepted_liveness)
            == len(self.POSES)
            and self._has_fresh_face(now)
            and bool(self.name.get().strip())
            and all(
                self._is_live_for_attempt(sample) and sample.context == context
                for context, sample in self._accepted_liveness
            )
            and len({context.capture_id for context, _ in self._accepted_liveness})
            == len(self.POSES)
        )

    def _update_controls(self):
        if self.store is None:
            ready = (
                self.session.consent
                and self.session.pending is None
                and len(self.session.references) >= self.session.MIN_REFERENCES
            )
            self.save_button.configure(state="normal" if ready else "disabled")
            return
        now = monotonic()
        states = self._can_capture(now), self._can_save(now)
        if states != self._control_states:
            self.capture_button.configure(state="normal" if states[0] else "disabled")
            self.save_button.configure(state="normal" if states[1] else "disabled")
            self._control_states = states
