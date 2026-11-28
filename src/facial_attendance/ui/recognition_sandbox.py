"""Standalone Recognition Evaluation / SANDBOX. No production registration API."""

import argparse
from pathlib import Path
from time import monotonic

from ..recognition.evaluation import (
    CONDITIONS,
    RecognitionEvaluation,
    SandboxInference,
    SandboxReports,
)
from ..recognition.inference import FaceInference
from .face_detection import ScaledFaceDetector, YuNetFaceDetector
from .live_preview import CameraFeed, SingleWorker, prepare_preview_image
from .sandbox_status import SandboxStatusPanel, next_step, set_enabled


class RecognitionSandboxApp:
    INTERVAL = 0.4  # Minimum spacing between analysis samples, not preview FPS.

    def __init__(
        self,
        root,
        inference,
        open_camera,
        detector,
        *,
        source_kind="webcam",
        preflight=None,
        repository=None,
    ):
        import tkinter as tk
        from tkinter import ttk

        self.root = root
        self.open_camera = open_camera
        self.preflight = preflight
        self.session = RecognitionEvaluation(source_kind=source_kind)
        self.service = SandboxInference(inference, self.session)
        self._detector_fn = detector
        self._cleanup_pending = False
        self.reports = SandboxReports(repository or Path(__file__).resolve().parents[3])
        self.detector = SingleWorker(detector, "sandbox-detector")
        self.worker = SingleWorker(self.service, "sandbox-face-inference")
        self.feed = None
        self._retiring_feed = None
        self._camera_serial = 0
        self._frame = None
        self._detection_frame = None
        self._detected = None
        self._current_source = None
        self._next_detection = 0.0
        self._drawn = None
        self._decision = None
        self.photo = None
        self._image = None
        self._closed = False
        self._last_text = None
        root.title("Recognition Evaluation / SANDBOX — no attendance")
        root.geometry("1120x740")
        root.minsize(1100, 690)
        root.protocol("WM_DELETE_WINDOW", self.close)
        body = ttk.Frame(root, padding=14)
        body.pack(fill="both", expand=True)
        ttk.Label(
            body, text="Face recognition test", font=("TkDefaultFont", 19, "bold")
        ).grid(row=0, column=0, sticky="w")
        self.details_button = ttk.Button(
            body, text="Settings and model details…", command=self._show_details
        )
        self.details_button.grid(row=0, column=1, sticky="e", padx=12)
        ttk.Label(
            body,
            text="SANDBOX • Identity unverified • No liveness, real registration or attendance",
            wraplength=1050,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=5)
        self.consent = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            body,
            text="I am testing myself or everyone being tested has given informed consent.",
            variable=self.consent,
            command=self._consent_changed,
        ).grid(row=2, column=0, sticky="w")
        self.profile_name = tk.StringVar(master=root)
        self.profile_labels = {}
        profile_field = ttk.Frame(body)
        profile_field.grid(row=2, column=1, sticky="w", padx=12)
        ttk.Label(profile_field, text="Test profile name (optional):").pack(side="left")
        self.profile_name_entry = ttk.Entry(
            profile_field, textvariable=self.profile_name, width=19
        )
        self.profile_name_entry.pack(side="left", padx=(6, 0))
        self.canvas = tk.Canvas(
            body, width=640, height=360, background="#162332", highlightthickness=0
        )
        self.canvas.grid(row=3, column=0, sticky="nw", pady=8)
        self.details = tk.Toplevel(root)
        self.details.withdraw()
        self.details.title("Face recognition test — settings and details")
        self.details.protocol("WM_DELETE_WINDOW", self.details.withdraw)
        detail_body = ttk.Frame(self.details, padding=14)
        detail_body.pack(fill="both", expand=True)
        self.status_panel = SandboxStatusPanel(body, detail_body, self.session.policy)
        # Retain the current textual summary for diagnostic consumers. Visible
        # fields occupy independent fixed cells, rather than a resizing label.
        self.live_text = tk.StringVar(
            value="Camera stopped. Confirm consent, then start."
        )
        self.progress = tk.StringVar()
        self.start_button = ttk.Button(
            self.status_panel.actions[0],
            text="Start / restart camera",
            command=self.start_camera,
        )
        self.reference_button = ttk.Button(
            self.status_panel.actions[1],
            text="Take reference photo",
            command=lambda: self.capture("reference"),
        )
        self.commit_button = ttk.Button(
            self.status_panel.actions[2],
            text="Save test profile",
            command=self.commit_reference,
        )
        self.evaluation_button = ttk.Button(
            self.status_panel.actions[3],
            text="Record a test result",
            command=lambda: self.capture("evaluation"),
        )
        for button in (
            self.start_button,
            self.reference_button,
            self.commit_button,
            self.evaluation_button,
        ):
            button.pack(anchor="w", pady=4)
        conditions = ttk.LabelFrame(
            detail_body, text="Optional test conditions", padding=8
        )
        conditions.pack(fill="x", anchor="w", pady=8)
        self.conditions = {}
        for column, (key, values) in enumerate(CONDITIONS.items()):
            row, column = divmod(column, 3)
            ttk.Label(conditions, text=key.replace("_", " ").title()).grid(
                row=row * 2, column=column, sticky="w"
            )
            var = tk.StringVar(value=values[0])
            self.conditions[key] = var
            ttk.Combobox(
                conditions, textvariable=var, values=values, state="readonly", width=18
            ).grid(row=row * 2 + 1, column=column, padx=(0, 8), pady=(0, 5))
        actions = ttk.Frame(detail_body)
        actions.pack(anchor="w", pady=6)
        ttk.Button(
            actions, text="Discard unsaved photos", command=self.discard_draft
        ).grid(row=0, column=0, sticky="w")
        self.reload_button = ttk.Button(
            actions,
            text="Check gallery reload (memory only)",
            command=self.reload_gallery,
        )
        self.reload_button.grid(row=0, column=1, sticky="w", padx=6)
        self.export_button = ttk.Button(
            actions, text="Export scores only", command=self.export_scores
        )
        self.export_button.grid(row=1, column=0, sticky="w", pady=6)
        self.reset_button = ttk.Button(
            actions, text="Delete all sandbox data and exports", command=self.reset
        )
        self.reset_button.grid(row=1, column=1, sticky="w", padx=6, pady=6)
        self.status = tk.StringVar(
            value="Test profiles and display names stay in memory for this session."
        )
        notice = ttk.Frame(body, width=1060, height=44)
        notice.grid(row=4, column=0, columnspan=2, sticky="ew")
        notice.pack_propagate(False)
        ttk.Label(notice, textvariable=self.status, anchor="nw", wraplength=1040).pack(
            fill="both", expand=True
        )
        trials = ttk.Frame(body)
        trials.grid(row=5, column=0, columnspan=2, sticky="w", pady=4)
        ttk.Label(trials, text="Expected person for recorded tests (optional):").pack(
            side="left"
        )
        self.truth = tk.StringVar(value="unspecified")
        self.truth_picker = ttk.Combobox(trials, state="readonly", width=30)
        self.truth_picker.pack(side="left", padx=6)
        self._refresh_truth_choices()
        self.truth_picker.bind(
            "<<ComboboxSelected>>", lambda _: self._subject_changed()
        )
        ttk.Label(trials, text="Choose a profile if you know who is in view.").pack(
            side="left"
        )
        self.table = ttk.Treeview(
            body,
            columns=(
                "trial",
                "truth",
                "predicted",
                "score",
                "decision",
                "outcome",
                "presentation",
            ),
            show="headings",
            height=6,
        )
        for key, label, width in (
            ("trial", "Test", 45),
            ("truth", "Expected person", 115),
            ("predicted", "Model result", 115),
            ("score", "Score", 100),
            ("decision", "Match", 85),
            ("outcome", "Outcome", 140),
            ("presentation", "Input", 110),
        ):
            self.table.heading(key, text=label)
            self.table.column(key, width=width, stretch=True)
        self.table.grid(row=6, column=0, columnspan=2, sticky="ew", pady=6)
        self.counts = tk.StringVar()
        ttk.Label(detail_body, textvariable=self.counts, wraplength=620).pack(
            anchor="w", pady=4
        )
        self._refresh_controls()
        self._timer = root.after(0, self._tick)

    def _show_details(self):
        self.details.deiconify()
        self.details.lift()

    def _profile_caption(self, test_id):
        name = self.profile_labels.get(test_id)
        return f"{name} ({test_id})" if name else test_id

    def _refresh_truth_choices(self):
        self._truth_ids = ("unspecified", "unknown", *self.session.test_ids)
        self.truth_picker.configure(
            values=tuple(self._profile_caption(test_id) for test_id in self._truth_ids)
        )
        self.truth_picker.current(self._truth_ids.index(self.truth.get()))

    def _subject_changed(self):
        index = self.truth_picker.current()
        if index < 0:
            return
        self.truth.set(self._truth_ids[index])
        self.consent.set(False)
        self._consent_changed()
        self.status.set(
            "Reconfirm informed consent for the ground-truth subject, then start the camera."
        )

    def _consent_changed(self):
        self.session.set_consent(self.consent.get())
        if not self.consent.get():
            self._stop_camera()
        self._refresh_controls()

    def start_camera(self):
        if not self.consent.get():
            self.status.set("Informed consent is required before camera sampling.")
            return False
        self._stop_camera()
        self.session.set_consent(True)
        if not self._ensure_workers():
            self.status.set(
                "Previous sample is still finishing cleanup; wait and retry."
            )
            return False
        if self._retiring_feed is not None and self._retiring_feed.thread.is_alive():
            self.status.set("Previous camera is still closing; wait and retry.")
            return False
        try:
            if self.preflight is not None and not self.preflight():
                raise OSError("Camera access unavailable")
            self.feed = CameraFeed(self.open_camera)
            self.feed.restart()
            self._camera_serial += 1
        except Exception:
            self.status.set("Camera unavailable. Check OS consent/device and restart.")
            return False
        self.status.set("Camera started. Keep one face in view.")
        return True

    def _stop_camera(self):
        if self.feed is not None:
            self.feed.close()
            self._retiring_feed = self.feed if self.feed.thread.is_alive() else None
        self.feed = None
        self._frame = self._detection_frame = self._detected = None
        self._current_source = None
        self._drawn = None
        self._decision = None
        self._next_detection = 0.0
        self.session.cancel()
        self.canvas.delete("all")
        self.photo = None
        self._image = None

    def _conditions(self):
        return {k: v.get() for k, v in self.conditions.items()}

    def capture(self, kind):
        if self.worker.busy or self._detected is None:
            self.status.set("Wait for one fresh detected face and a free model worker.")
            return False
        source, frame, boxes, timestamp = self._detected
        try:
            job = self.session.job(
                kind,
                source,
                frame,
                boxes,
                timestamp,
                now=monotonic(),
                truth=self.truth.get(),
                conditions=self._conditions(),
            )
            if not self.worker.submit((job.epoch, job.serial), job):
                self.session.cancel()
                return False
        except (ValueError, PermissionError) as error:
            self.status.set(str(error))
            return False
        self.status.set(
            "Processing explicit reference capture..."
            if kind == "reference"
            else "Processing held-out evaluation capture..."
        )
        self._refresh_controls()
        return True

    def commit_reference(self):
        name = self.profile_name.get().strip()
        if len(name) > 24 or (name and not name.isprintable()):
            self.status.set(
                "Use a single-line test profile name of up to 24 characters."
            )
            return False
        try:
            test_id = self.session.commit_reference()
        except (ValueError, PermissionError) as error:
            self.status.set(str(error))
            return False
        if name:
            self.profile_labels[test_id] = name
        self.profile_name.set("")
        self.consent.set(False)
        self._stop_camera()
        self._refresh_truth_choices()
        self.status.set(
            f"Test profile {self._profile_caption(test_id)} saved in memory. Confirm consent and restart the camera to test it."
        )
        self._refresh_controls()
        return True

    def discard_draft(self):
        self.profile_name.set("")
        self.session.discard_draft()
        self.status.set(
            "Reference draft discarded. Its exact frames remain excluded from evaluation in this run."
        )
        self._refresh_controls()

    def reload_gallery(self):
        try:
            self.session.reload_in_memory()
        except Exception:
            self.status.set(
                "Gallery reload check unavailable; wait for idle and retry."
            )
            return False
        self._decision = None
        self._current_source = None
        self.status.set(
            "Gallery schema/model round-trip passed in memory. No biometric file was written."
        )
        return True

    def export_scores(self):
        try:
            self.reports.export(self.session)
        except (ValueError, OSError):
            self.status.set("Sandbox export failed; destination was not used.")
            return False
        self.status.set(
            "Scores/conditions exported under artifacts/recognition-evaluation for this run. No images, embeddings or real names."
        )
        return True

    def reset(self):
        self._stop_camera()
        self.session.reset()
        self.consent.set(False)
        # SingleWorker locals can retain the last input/result while idle.
        # Close those actors, then replace only after both have actually exited.
        self.detector.close()
        self.worker.close()
        self._cleanup_pending = True
        self._ensure_workers()
        self.profile_name.set("")
        self.profile_labels.clear()
        self.truth.set("unspecified")
        self._refresh_truth_choices()
        self.table.delete(*self.table.get_children())
        try:
            self.reports.delete_all()
            message = (
                "All memory references/trials and exported sandbox score logs deleted."
            )
        except (ValueError, OSError):
            message = "Memory cleared. Export cleanup refused an unexpected path/file; inspect the local sandbox directory."
        self.status.set(message)
        self._refresh_controls()

    def _ensure_workers(self):
        if not self._cleanup_pending:
            return True
        if self.detector.thread.is_alive() or self.worker.thread.is_alive():
            return False
        self.detector = SingleWorker(self._detector_fn, "sandbox-detector")
        self.worker = SingleWorker(self.service, "sandbox-face-inference")
        self._cleanup_pending = False
        return True

    def _refresh_controls(self):
        ready = (
            not self._cleanup_pending
            and self.session.consent
            and self.feed is not None
            and self._detected is not None
            and len(self._detected[2]) == 1
            and monotonic() - self._detected[3] <= self.session.MAX_SAMPLE_AGE
            and not self.worker.busy
        )
        set_enabled(self.reference_button, ready)
        set_enabled(self.evaluation_button, ready)
        set_enabled(
            self.commit_button,
            self.session.consent
            and len(self.session.references) >= 3
            and not self.worker.busy,
        )
        progress = (
            f"Next alias: {self.session.next_test_id}\nReferences: {len(self.session.references)}/5 target (3–10)\n"
            + f"Isolated subjects: {len(self.session.test_ids)} • recorded trials: {len(self.session.records)}"
        )
        if self.progress.get() != progress:
            self.progress.set(progress)
        self.status_panel.update(
            alias=f"Next alias: {self.session.next_test_id}",
            references=f"References: {len(self.session.references)}/5 target (3–10)",
            counts=f"Isolated subjects: {len(self.session.test_ids)} • recorded trials: {len(self.session.records)}",
        )
        references = len(self.session.references)
        profiles = len(self.session.test_ids)
        if profiles and not references:
            sample_progress = "Test profile saved in memory"
        else:
            sample_progress = f"Reference photos: {references} / {self.session.TARGET_REFERENCES} recommended"
        self.status_panel.update(
            guide=next_step(
                consent=self.session.consent,
                running=self.feed is not None,
                references=references,
                profiles=profiles,
                target=self.session.TARGET_REFERENCES,
                minimum=self.session.MIN_REFERENCES,
            ),
            sample_progress=sample_progress,
            session_summary=f"Test profiles: {profiles}  •  Recorded tests: {len(self.session.records)}",
        )
        decision = self._decision
        display_name = None
        if decision is not None and decision.test_id in self.profile_labels:
            display_name = self._profile_caption(decision.test_id)
        self.status_panel.render(
            decision,
            running=self.feed is not None,
            face_count=None if self._detected is None else len(self._detected[2]),
            display_name=display_name,
        )
        if getattr(self, "_count_key", None) != (
            self.session.run_id,
            len(self.session.records),
        ):
            self._count_key = self.session.run_id, len(self.session.records)
            report = self.session.report()
            self.counts.set(
                "Counts (no small-sample accuracy claim): "
                + ", ".join(f"{k}={v}" for k, v in report["counts"].items())
                + f" • photograph accepts={report['photo_accepts']} (NOT liveness success)"
            )

    def _draw(self, packet):
        from PIL import ImageTk

        image = prepare_preview_image(packet.frame)
        self._preview_size = image.size
        self.photo = ImageTk.PhotoImage(image, master=self.root)
        if self._image is None:
            self._image = self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        else:
            self.canvas.itemconfigure(self._image, image=self.photo)

    def _tick(self):
        if self._closed:
            return
        if not self._ensure_workers():
            self._refresh_controls()
            self._timer = self.root.after(33, self._tick)
            return
        now = monotonic()
        packet = self.feed.latest() if self.feed is not None else None
        usable = (
            packet is not None
            and packet.error is None
            and now - packet.captured_at <= 1.0
        )
        if usable and packet.sequence != self._drawn:
            self._draw(packet)
            self._drawn = packet.sequence
        elif packet is not None and not usable:
            self._stop_camera()
            self.status.set(
                "Camera failure: current sandbox result cleared. Restart after checking access."
            )
        finished = self.worker.take()
        if finished is not None:
            if finished.error is not None:
                self.session.cancel()
                self._decision = None
                self.status.set("Model sample failed. No reference/trial was recorded.")
            else:
                try:
                    result = finished.value
                    self._decision = self.session.accept(
                        result, current_source=self._current_source, now=monotonic()
                    )
                    if result.job.kind == "reference":
                        self.status.set(
                            "Reference photo added. Move slightly before taking the next one."
                        )
                    if result.job.kind == "evaluation":
                        record = self.session.records[-1]
                        self.table.insert(
                            "",
                            "end",
                            values=(
                                record["trial"],
                                record["ground_truth"],
                                record["predicted_test_id"] or "—",
                                "N/A"
                                if record["cosine"] is None
                                else f"{record['cosine']:.6f}",
                                record["decision"],
                                record["outcome"],
                                record["presentation"],
                            ),
                        )
                        self.status.set("Test result recorded in the table below.")
                except (ValueError, PermissionError) as error:
                    self._decision = None
                    self.status.set(str(error))
        detection = self.detector.take()
        if detection is not None:
            source = self._detection_frame
            if (
                usable
                and source is not None
                and detection.token == self._current_source
            ):
                boxes = detection.value if detection.error is None else None
                try:
                    from .webcam_prototype import box_in_frame, frame_size

                    width, height = frame_size(source.frame)
                    if not isinstance(boxes, list) or not all(
                        box_in_frame(b, width, height) for b in boxes
                    ):
                        raise ValueError("invalid detection")
                    self._detected = (
                        detection.token,
                        source.frame,
                        boxes,
                        source.captured_at,
                    )
                    self.canvas.delete("region")
                    for x1, y1, x2, y2 in boxes:
                        vw, vh = self._preview_size
                        self.canvas.create_rectangle(
                            x1 * vw / width,
                            y1 * vh / height,
                            x2 * vw / width,
                            y2 * vh / height,
                            outline="#ffce65",
                            width=2,
                            tags="region",
                        )
                    if len(boxes) != 1:
                        self.session.cancel()
                        self._decision = None
                    # Live recognition renders the current face count. The action
                    # message below the preview must not retain a transient miss.
                    elif self.session.consent and not self.worker.busy:
                        job = self.session.job(
                            "preview",
                            detection.token,
                            source.frame,
                            boxes,
                            source.captured_at,
                            now=monotonic(),
                        )
                        self.worker.submit((job.epoch, job.serial), job)
                except (ValueError, PermissionError):
                    self.session.cancel()
                    self._detected = None
                    self._decision = None
                    self.status.set(
                        "Detector/sample invalid or expired. Wait for a fresh single face."
                    )
            self._detection_frame = None
        if (
            usable
            and self.session.consent
            and not self.detector.busy
            and now >= self._next_detection
        ):
            token = (self._camera_serial, packet.sequence)
            if token != self._current_source and self.detector.submit(
                token, packet.frame
            ):
                self._current_source = token
                self._detection_frame = packet
                self._detected = None
                self._decision = None
                self._next_detection = now + self.INTERVAL
        decision = self._decision
        text = (
            (
                "Camera stopped."
                if self.feed is None
                else "Checking current single-face sample..."
            )
            if decision is None
            else (
                f"{'Known: ' + decision.test_name if decision.accepted else 'Unknown'}\n"
                f"Cosine: {'N/A (empty gallery)' if decision.score is None else format(decision.score, '.6f')}\n"
                f"Threshold: {decision.threshold}\nUNVERIFIED recognition preview only"
            )
        )
        if text != self._last_text:
            self.live_text.set(text)
            self._last_text = text
        self._refresh_controls()
        self._timer = self.root.after(33, self._tick)

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.root.after_cancel(self._timer)
        self._stop_camera()
        self.detector.close()
        self.worker.close()
        self.session.reset()
        self.root.destroy()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Isolated Recognition Evaluation / SANDBOX — no liveness or production registration."
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--face-checkpoint", required=True)
    parser.add_argument("--yunet-model", required=True)
    args = parser.parse_args(argv)
    try:
        inference = FaceInference(args.face_checkpoint).load()
        import tkinter as tk

        import cv2

        cv2.setNumThreads(1)
        detector = ScaledFaceDetector(YuNetFaceDetector(args.yunet_model))
        root = tk.Tk()
    except Exception as error:
        parser.exit(1, f"Sandbox initialization failed: {error}\n")

    def camera():
        capture = cv2.VideoCapture(args.camera)
        capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return capture

    def preflight():
        capture = camera()
        opened = capture.isOpened()
        capture.release()
        return opened

    RecognitionSandboxApp(root, inference, camera, detector, preflight=preflight)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
