# SYS-05 — Owner verification and independent-review handover

Vỹ's Windows manual run on **8 November 2026** verified physical webcam 0
through frame freeze, manual crop and the displayed stub payload. The local
branch `feature/sys-05-webcam-gui-stubs` is ready to commit and share with
Trang for independent review. The working tree is based on `ddf3fc6`.
Assign the review commit after committing the four SYS-05 files listed below.

## WBS acceptance audit

Trang is the independent reviewer for these ten subtasks. The status below
reflects the evidence available for this handover. [Requirements rows 40–50](https://docs.google.com/spreadsheets/d/1ItpF11JBlpcTnNiADI-Qm-cDmexBuLRzuHBW5uDChLw/edit#gid=2016101501&range=A40:P50)

| Task | Acceptance criterion | Owner evidence and remaining action |
| --- | --- | --- |
| SYS-05.1 | Define webcam/crop flow and three stub interfaces. Agree the sample outputs. | Flow, fields and four scenarios documented below. Trang's agreement on the sample payloads is pending. |
| SYS-05.2 | Connect webcam, stubs and GUI with inspectable, clearly labelled outputs. | Owner verified webcam 0, manual crop and the displayed scenarios on Windows. Ready for review. |
| SYS-05.3 | Reviewer runs the prototype against SYS-02 and records reproducible interface or connection findings. | Pending Trang's independent run. Use the checklist below and record expected versus actual behavior. |
| SYS-05.4 | Correct reviewer findings and have the reviewer check affected flows. If there are no findings, record a passing check. | Awaiting SYS-05.3 findings. Apply and verify corrections only where the review identifies an issue. |
| SYS-05.5 | Reviewer reruns the corrected flow and prior failures and checks that the stubs are replaceable. | Owner no-face and camera-error checks passed. Independent retest remains pending. Extra camera and multi-region checks are supplementary. |
| SYS-05.6 | Smoke test on the intended demo runtime and check outstanding fixes or limitations. | Windows owner smoke evidence is available. Confirm the intended demo machine and rerun on that runtime if it differs. Recheck any later fixes there. |
| SYS-05.7 | Identify and freeze the reviewed source and configuration for later adapter replacement. | Replacement points documented below. Freeze a reviewed commit and record its runtime and launch configuration after review and retest. |
| SYS-05.8 | Reviewer launches the identified release candidate and confirms required files and reproducibility. | Pending a frozen candidate and Trang's fresh launch from it. |
| SYS-05.9 | Demonstrate capture, crop, adapter calls and output. Identify the simulated components and answer questions. | Owner has run the flow. The walkthrough with Trang and any follow-up questions remain pending. |
| SYS-05.10 | Hand the runnable version, instructions and execution evidence to the integrating members. | Handover package prepared here. Share the identified version and record receipt by the UI-03/UI-04 integrating members. |

## Read-only compatibility audit

The local remote-tracking commits below matched the GitHub branch tips when
checked on 8 November 2026. Files were read directly from those commits.
The comparison uses [SYS-02 Final v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d).

| Boundary | Verified source and contract | SYS-05 treatment |
| --- | --- | --- |
| Capture and face region | SYS-02 §3. BGR `uint8[H,W,3]`, one face, original-frame integer half-open `[x1,y1,x2,y2]`. Zero/multiple regions or invalid geometry block evaluation. `origin/main` (`ddf3fc6`), `docs/architecture.md`, has a UI package boundary and no camera/detector implementation. | Use OpenCV capture. In this prototype the operator marks the face region on a frozen frame. Crop the source pixels without applying a model transform. Automatic detection and final boundary policy belong to later integration. |
| Recognition | `origin/docs/sys-04-experiment-settings` (`71f3b56`), `docs/embedding-schema.md`, `docs/experiment-settings.md`. Square crop with proposed margin 1.2, clipped to frame, 64×64 bilinear RGB, /255 then ImageNet mean/std, float32 NCHW. Output L2-normalized 512D embeddings. Compare by cosine against compatible templates using the selected checkpoint threshold. SYS-02 §4 adds canonical `person_id`, `name`, `score`, `accepted`, `model_id`, `model_version` and null semantics. | Return controlled match/Unknown payloads. Keep checkpoint fields null for stubs. No model input resizing, scoring, threshold or padding policy is selected here. |
| Older face schema and loaders | `origin/fac-01/face-data-protocol-audit` (`82e34b0`), `docs/embedding-schema.md`, has pending S/D, `identity_id`, singular `embedding` and checkpoint-file hash in `model_id`. SYS-04 and SYS-02 supersede those fields with 64/512, `person_id`, one template in `embeddings`, run ID and weights version. FAC-01 `docs/face-data-protocol.md`, `src/facial_attendance/recognition/audit.py`, `tests/test_face_audit.py` cover dataset roles, scoring direction and integrity. `origin/main` `src/facial_attendance/recognition/data_loader.py`, `scripts/check_face_data.py`, `tests/test_face_data.py` cover RGB decoding and pair fixtures. CLI size 112 is for inspection. | Keep loaders/audit in their training and evaluation roles. They provide no live recognition adapter. Use SYS-02 field names and defer the trained checkpoint. |
| Liveness and reset | `origin/docs/liv-02-liveness-gating` (`48279ac`), `docs/liveness-gating.md`, and SYS-02 §§5, 7–8. RGB binary labels 0 live, 1 spoof. Finite `p_spoof` in [0,1], null when scoring fails. Adapter owns `live/spoof/uncertain`, `reason`, model/preprocessing/policy versions and `source`. Equality at the selected threshold gives spoof. Below threshold requires a valid quality decision. Spoof aborts, uncertain needs a fresh capture, and stale completions are discarded. Stub results cannot authorize either operation. | Stub decisions are chosen by scenario. Operational errors are separate. Clear active results on capture invalidation/retry/mode change, block further callbacks after spoof, keep `can_save=false`. The branch has design review cases and no runtime adapter. |
| Emotion | `origin/feature/emotion-eda` (`c3db30c`), `configs/emotion.toml`, `src/facial_attendance/emotion/aggregate.py`, `src/facial_attendance/emotion/tensor_data.py`, `tests/test_emotion_aggregation.py`, `tests/test_emotion_loader.py`. Explicit 0 Negative (Angry/Disgust/Fear/Sad), 1 Neutral, 2 Positive (Happy). Surprise excluded. Pillow grayscale and bilinear 48×48, float32 /255, batch shape B×1×48×48, no additional normalization or augmentation. `docs/emotion-display-and-webcam-tests.md` and SYS-02 §6 specify `class_id`, `label`, optional finite softmax `confidence` in [0,1], and clearing unavailable results. | Use the verified label mapping in demo payloads. Reuse the branch's grayscale tensor transform when trained inference is integrated. Dataset preparation and loading are outside the live path. |
| Context and errors | SYS-02 §§3, 7–8 plus LIV-02 `docs/liveness-gating.md`. Correlate `contract_version`, `attempt_id`, `action`, `capture_id`, `face_id`. Request carries frame and bbox. Controller envelope contains context, recognition, liveness, emotion, ui_state, can_save and errors. Errors have `module/code/message/blocks_persistence`. Emotion-only errors are display-only. | Use a common context wrapper for all three stubs and compare it to the active context. JSON view shows the envelope, with pixels kept in memory. Process commands serially on the GUI thread in this phase. |
| Gallery | `origin/feature/ui-01-gallery-base` (`82a74f8`), `src/facial_attendance/ui/gallery.py`, `tests/test_gallery.py`, `docs/ui-01-4-gallery-revisions.md`. Current v0.2 satisfies the record shape SYS-02 §4.3 describes. `load(path, model_id=…, model_version=…, embedding_dim=…)` verifies compatibility. `register(id, name, embedding, n_images=…)` takes one normalized averaged vector, 3–10 accepted samples. `update_template` replaces explicitly, `add_embedding` rejects. Casefold IDs, atomic save, rollback on failure, explicit rejection of v0.1. | SYS-02's v0.1 warning describes an earlier snapshot. UI-03 can reuse v0.2 after it is available on the integration branch. The controller owns liveness, sample selection/averaging, identity checks and save authorization. SYS-05 has no gallery dependency. |

The model owners still need to supply the deployed detector/crop policy, actual
checkpoint values, liveness input dimensions and normalization, the LIV-04
threshold/quality policy, and the trained emotion checkpoint. The LIV-02
proposal allows parallel provisional recognition, while the face design places
recognition after liveness. This serial demo evaluates liveness first. Real
model scheduling, attendance session/duplicate-event rules and persistence
ordering remain integration work.

## Phase 1 flow and sample payloads

OpenCV captures and previews actual video in webcam mode. The operator freezes
one frame and manually draws a face region. The app maps that rectangle to the
source frame, validates and crops it, and passes the same request context to
the stub adapters. The GUI displays text summaries and the result JSON.
Automatic face detection has yet to be demonstrated. The selected region and
the scenario dropdown drive separate parts of the demo. Recognition, liveness
and emotion are simulated by the selected scenario in both camera modes.

Use **Retry / resume preview** for a fresh attempt. Changing operation also
invalidates the active capture. Zero regions produce `no_face`. Two regions
produce `multiple_faces`. These checks count manually marked regions.

| Scenario | Liveness | Recognition | Emotion |
| --- | --- | --- | --- |
| Live match | `live`, demo score 0.08 | `accepted=true`, `demo-001`, Demo Person, score 0.91 | 2 Positive |
| Live unknown | `live`, demo score 0.08 | `accepted=false`, null identity and score | 1 Neutral |
| Spoof | `spoof`, demo score 0.96 | Skipped | Cleared |
| Uncertain | `uncertain`, demo score 0.45 | Skipped | 0 Negative |

The scores are fixtures. The GUI labels results as DEMO and the liveness
payload carries `source="stub"`. The frozen `Envelope` fixes `can_save=false`
with an `init=False` field. Both action modes use that envelope, and the
prototype has no persistence control or storage call for enrollment or
attendance. Webcam frames and crops remain in memory.

## Reproduce on Windows

Use the project Python 3.13 virtual environment with OpenCV, Pillow, NumPy
and Tk. From the repository root in PowerShell, run one GUI command at a time.
Close the current window before launching the next command.

```powershell
$env:PYTHONPATH = 'src'

# Generated frames
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --demo-camera

# Physical webcam on the owner's machine
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0

# Unavailable camera case on the owner's machine
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 99

# Reopen the physical webcam after the failure case
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0

# Contract tests and complete suite
.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_webcam_prototype.py' -v
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Choose a scenario, select **Freeze frame**, draw one rectangle and select
**Run stubs**. Inspect the summary and scroll through the result JSON.
Camera indices depend on the machine. If the review host uses another webcam
index or has a device at index 99, record the indices actually tested.

## Manual verification — owner, Windows, 8 November 2026

These observations were supplied by the owner from the physical Windows run.

| Check | Observed result |
| --- | --- |
| Open webcam 0 | **Pass.** Live video displayed. |
| Freeze, draw one manual region, run stubs | **Pass.** The crop-to-result flow completed. |
| Live match | **Pass.** Demo Person, `liveness.status=live`, Positive and `can_save=false`. |
| Live unknown | **Pass.** `recognition.accepted=false`, `person_id=null`, `liveness.status=live` and `can_save=false`. |
| Spoof | **Pass.** `liveness.status=spoof`, `recognition=null`, `emotion=null`, `ui_state=spoof` and `can_save=false`. |
| Uncertain | **Pass.** `liveness.status=uncertain`, `recognition=null`, `ui_state=uncertain` and `can_save=false`. A demo emotion may still be displayed. |
| Retry / resume preview | **Pass.** Previously displayed identity and emotion cleared. |
| Clear regions, then Run stubs | **Pass for behavior.** `ui_state=no_face`, null recognition/liveness/emotion, `can_save=false` and a blocking `no_face` error. The owner did not separately check whether the rectangle disappeared visually. |
| Close and reopen webcam 0 | **Pass.** Live video was available after reopening. |
| Launch camera 99, then reopen webcam 0 | **Pass.** The unavailable-camera handling appeared on 99 and webcam 0 reopened successfully. |

## Automated verification — 8 November 2026

Rerun on Windows with Python 3.13.17 and `PYTHONPATH=src`, using the commands
above. The full suite includes the ten contract tests.

| Check | Rerun result |
| --- | --- |
| SYS-05 contract tests | **10 passed**, 0 failed, 0 skipped. Covers original-frame cropping, context rejection, fail-closed demo results and reset/error behavior. |
| Full unit suite | **56 passed**, 0 failed, 0 skipped. |
| Synthetic Tk smoke | **Passed.** Exercised generated preview, selection-to-crop coordinates, all four scenarios, JSON and display states, operation change, retry, zero/two regions, simulated read failure, recovery and camera release. Asserted the canvas rectangle item was removed by Clear regions. |

The earlier Codex physical-camera probe reached the unavailable-camera state.
The physical-camera success evidence above comes from the separate owner run.
The current automated rerun used generated frames for the GUI check.

## Trang's independent-review checklist

1. Record the review commit, date, OS, Python version and camera indices. Check
   the four files below and run both test commands.
2. Start with `--demo-camera`, then open the physical webcam. Freeze a frame,
   mark one region and run each scenario with a fresh capture. Compare the
   summary and JSON with the manual evidence table. Confirm the demo labels.
3. Compare the capture, bbox and result flow with SYS-02. Check
   `contract_version=sys02-v1`, `attempt_id`, `action`, `capture_id`, `face_id`,
   all three module fields, `ui_state`, `can_save` and typed errors. Confirm
   whether the four sample payloads cover the expected review cases.
4. Switch between attendance and registration. Check that current results
   clear and that even a live match leaves `can_save=false` in both modes.
   After spoof, use **Retry / resume preview** before evaluating a fresh frame.
5. Draw a rectangle, select **Clear regions** and explicitly check that the
   overlay disappears. Then run stubs and confirm the blocking `no_face`
   error and cleared module fields. Check the unavailable-camera launch and
   close/reopen behavior on the physical device. Two marked regions are an
   optional check and should produce `multiple_faces`.
6. For each check, record the scenario, steps, expected result, actual result
   and pass/fail or not-run. Use text and the demo JSON fields for evidence.
   Leave webcam screenshots and personal face images out of Git. Retest any
   reviewer findings before freezing a candidate and running it afresh.

## Integration and files to share

The current controller dispatches adapters serially on the Tk thread.
Use the three `StubAdapters` methods and the `run_stubs` call site as the
UI-03/UI-04 integration boundary. Keep the request context and original-frame
coordinates. The demo controller accepts only `source="stub"` for liveness.
Trained results need a reviewed controller path with verified preprocessing
and result validation. Genuine persistence also requires explicit authorization
based on liveness, gallery/checkpoint compatibility and the applicable
enrollment or attendance rules. The automatic detector, deployed crop policy
and trained-model behavior remain integration work.

The four intended files are currently untracked:

- `docs/sys-05-phase-1.md` — compatibility audit, verification and handover.
- `src/facial_attendance/ui/webcam_app.py` — camera lifecycle, crop selection and Tk display.
- `src/facial_attendance/ui/webcam_prototype.py` — request context, controller and stub adapter boundary.
- `tests/test_webcam_prototype.py` — ten contract tests.

`reports/environment/vy-independent-retest.json` is a pre-existing, unrelated
untracked report. Preserve it locally and exclude it from the SYS-05 commit.
