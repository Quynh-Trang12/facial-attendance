# SYS-05 — Owner verification and independent-review handover

The committed baseline is `578e6e9` on `feature/sys-05-webcam-gui-stubs`.
Vỹ verified the revised build on a physical Windows webcam on
**8 November 2026**. The automatic detector, filtering, mode switching,
reset changes and associated tests are in the **unstaged working tree**
above that baseline. The owner evidence and final automated rerun support
using these six files as one review candidate. Assign a commit ID when the
owner commits the candidate, then give that ID and the runtime instructions
to Trang for an independent run.

## WBS acceptance audit

Trang is the independent reviewer for these ten subtasks. The status below
reflects the evidence available for this handover. [Requirements rows 40–50](https://docs.google.com/spreadsheets/d/1ItpF11JBlpcTnNiADI-Qm-cDmexBuLRzuHBW5uDChLw/edit#gid=2016101501&range=A40:P50)

| Task | Acceptance criterion | Owner evidence and remaining action |
| --- | --- | --- |
| SYS-05.1 | Define webcam/crop flow and three stub interfaces. Agree the sample outputs. | Owner implementation and documentation cover both modes, the shared fields and four scenarios. Trang's agreement on the sample payloads is pending. |
| SYS-05.2 | Connect webcam, stubs and GUI with inspectable, clearly labelled outputs. | Owner verified the revised automatic and manual paths on Windows, including a 205 × 205 automatic crop and the registration Live match stub. |
| SYS-05.3 | Reviewer runs the prototype against SYS-02 and records reproducible interface or connection findings. | Owner evidence and checklist are ready. Trang's independent run and findings are pending. |
| SYS-05.4 | Correct reviewer findings and have the reviewer check affected flows. If there are no findings, record a passing check. | The owner addressed the nested-box false positive and verified the revised single-face and two-region behavior. Any independent findings and their disposition follow SYS-05.3. |
| SYS-05.5 | Reviewer reruns the corrected flow and prior failures and checks that the stubs are replaceable. | Owner no-face, camera-error and manual fallback checks passed. Trang's independent retest and assessment of adapter replacement are pending. |
| SYS-05.6 | Smoke test on the intended demo runtime and check outstanding fixes or limitations. | Owner smoke passed on Windows webcam 0. Haar's angled-face limitation and the phone-based two-region test are recorded below. Repeat on the intended demo machine if it differs from the owner's test machine. |
| SYS-05.7 | Identify and freeze the reviewed source and configuration for later adapter replacement. | Candidate files, baseline, wheel and launch configuration recorded. Assign the review commit ID after this audit. Freeze the reviewed version after any findings are resolved. |
| SYS-05.8 | Reviewer launches the identified release candidate and confirms required files and reproducibility. | Trang's fresh setup and launch from the identified candidate are pending. |
| SYS-05.9 | Demonstrate capture, crop, adapter calls and output. Identify the simulated components and answer questions. | Walkthrough prepared below. The demonstration with Trang and any follow-up are pending. |
| SYS-05.10 | Hand the runnable version, instructions and execution evidence to the integrating members. | Owner evidence and integration handover are ready. Sharing the identified version and recording receipt remain pending. |

## Read-only compatibility audit

The local remote-tracking commits below matched the GitHub branch tips when
checked on 8 November 2026. Files were read directly from those commits.
The comparison uses [SYS-02 Final v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d).

| Boundary | Verified source and contract | SYS-05 treatment |
| --- | --- | --- |
| Capture and face region | SYS-02 §3. BGR `uint8[H,W,3]`, one face, original-frame integer half-open `[x1,y1,x2,y2]`. Zero/multiple regions or invalid geometry block evaluation. `origin/main` (`ddf3fc6`), `docs/architecture.md`, has a UI package boundary and no camera/detector implementation. | Capture with OpenCV. Find regions on a frozen source frame with the bundled Haar detector or mark them manually on the preview. Convert either path to validated source coordinates. The final recognition crop policy belongs to UI-03. |
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

1. **Camera → detector.** OpenCV previews actual video in webcam mode.
   **Freeze frame** copies the latest BGR source frame. **Automatic detection**
   converts that frame to grayscale and runs OpenCV's bundled frontal-face Haar
   cascade with `scaleFactor=1.1` and `minNeighbors=5`. It draws amber overlays
   for the validated, filtered regions. The detector is separate from the three stubs.
2. **Detector → crop.** Haar's `(x,y,w,h)` boxes become integer half-open
   source rectangles. An invalid rectangle rejects the batch. **Manual
   selection** draws a green rectangle and maps it from the scaled preview
   to the same source coordinates, with the start rounded down and end rounded
   up. Switching to manual on a frozen frame is the fallback for an empty or
   failed detection. **Run stubs** gives `no_face` for zero regions or
   `multiple_faces` for more than one, with no crop evaluated.
3. **Crop → stubs → controller → GUI.** With one region, the controller creates
   a `FaceRequest` for that source and box. Liveness runs first, recognition
   runs only for live, and emotion is display-only. The controller checks
   the capture context and renders the summary and SYS-02 JSON. The three
   results are determined by the scenario dropdown in both selection modes.

Retry, operation changes and scenario changes discard the frozen frame and
current results. Clearing or switching region modes invalidates the previous
attempt. Spoof and uncertain require a fresh frame. Camera failure clears the
capture, selections and results, and retry opens the camera again.

Haar is a frontal-face locator and may miss faces or return extra boxes,
especially when pose or lighting changes. Results describe the frozen frame.
Returning to live preview clears them. The generated demo placeholder is for
manual selection and GUI testing. It provides no evidence of real-face
detection or model accuracy.

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

## Nested Haar detection regression

On the physical Windows webcam, the owner observed one person with a large
face box and a smaller eye-region box fully inside it. Counting both boxes
produced `ui_state=multiple_faces` with `can_save=false`. This was a safe
block, but the eye box was a false second detection.

The Haar adapter validates every box before filtering and considers larger
boxes first. It drops a candidate when it is fully contained in a retained
box and has at most half its area, or when intersection over union with a
retained box is at least 0.80. The first rule targets small internal
features. The second requires nearly coincident regions and implicitly
requires similar areas. These are conservative prototype heuristics.
Separate boxes and ambiguous overlaps survive. The surviving rectangles
keep their original half-open source coordinates.

The GUI draws the same filtered list that it passes to the controller.
The regression fixture produces one overlay and one crop from a face plus
an eye box. Adding a separate face yields two overlays and
`multiple_faces` with no crop or stub evaluation. Empty detections and
invalid boxes still take their existing guard paths. Manual selection
bypasses the Haar filter. In the revised physical run the owner observed
one frontal face produce one 205 × 205 crop, and a person plus a face on
a phone produce two regions and a block.

## Reproduce on Windows

Use the existing project `.venv` with Python 3.13, Pillow, NumPy and Tk.
The setup was verified locally on Windows AMD64 with Python 3.13.17,
NumPy 2.2.6, Pillow 11.2.1 and Tk 8.6. See [project setup](setup.md)
if the base environment is missing.

The pinned OpenCV 5.0.0.93 main wheel lacks the Haar classifier and cascade
XML. The same-version contrib wheel supplied the classifier but had no bundled
XML in the Windows wheel tested here. The OpenCV 4.13.0.92 Windows wheel includes
both. The inspected isolated wheel is `cp37-abi3-win_amd64` and uses the
base environment's NumPy 2.x. Load it for SYS-05 from the ignored
`artifacts/sys05-haar` directory. `--target` isolates the install and
`--no-deps` avoids reinstalling NumPy or replacing the base packages.
[OpenCV migration guide](https://github.com/opencv/opencv/wiki/OpenCV-4-to-5-migration)

From the repository root in PowerShell, install once on a fresh review
checkout. Reuse the target directory if it already passes the check below.

```powershell
.venv\Scripts\python.exe -m pip install --no-deps --only-binary=:all: --target artifacts/sys05-haar opencv-python==4.13.0.92
```

Set the import path for the GUI and verify the wheel and bundled classifier.
The path printed by the check should be under `artifacts/sys05-haar`.

```powershell
$env:PYTHONPATH = 'artifacts/sys05-haar;src'
.venv\Scripts\python.exe -c "from pathlib import Path; import importlib.metadata as md, cv2; p=Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'; assert md.version('opencv-python') == '4.13.0.92'; assert Path(cv2.__file__).resolve().is_relative_to(Path('artifacts/sys05-haar').resolve()); assert p.is_file() and not cv2.CascadeClassifier(str(p)).empty(); print(cv2.__version__, cv2.__file__, p)"
```

Launch one GUI at a time. Close the window before opening another.

```powershell
# Generated frames for manual selection
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --demo-camera

# Physical webcam. Select Automatic detection or Manual selection in the GUI.
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0

# Unavailable camera case on the owner's machine
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 99

# Reopen the physical webcam after the failure case
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0
```

For tests, return to the ordinary project import path. Set the Haar path
again before launching automatic detection.

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m unittest discover -s tests -p 'test_webcam*.py' -v
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The GUI starts in **Manual selection**, which also works with
`PYTHONPATH=src` and the project's original OpenCV wheel. Select
**Automatic detection** before freezing to draw Haar's boxes. If the wheel
lacks the cascade, the GUI reports a detector error and allows manual fallback.
Camera indices depend on the machine. Record the indices actually tested.
The import and classifier check passed on this Windows worktree. Trang will
run it on her Windows machine as part of independent review.

## Manual verification — owner, Windows, 8 November 2026

The owner supplied the following results from the **revised** working tree
on physical webcam 0. The earlier manual-only run on committed baseline
`578e6e9` also passed preview, manual crop, all four stub scenarios,
reset and camera recovery.

| Check | Observed result |
| --- | --- |
| Automatic detection, one frontal face | **Pass.** One valid source crop, 205 × 205 pixels. |
| Automatic detection, person plus a face displayed on a phone | **Pass.** Two distinct regions, `multiple_faces` and `can_save=false`. This was a two-region test with one physically present person. |
| No face | **Pass.** `no_face`, null recognition/liveness/emotion and `can_save=false`. |
| Manual selection with Haar available | **Pass.** Manual selection and crop still worked. |
| Angled face | **Pass for fallback.** Frontal Haar missed the angled face. Manual selection worked. |
| Manual-to-automatic mode change | **Pass.** Old recognition cleared and the owner returned to preview. |
| Registration Live match | **Pass.** `context.action=registration`, expected demo identity/liveness/emotion and `can_save=false`. |
| Reset | **Pass.** Prior identity and emotion cleared, and `ui_state` returned to `idle`. |
| Camera 99 error, then reopen camera 0 | **Pass.** The expected unavailable-camera error appeared. Webcam 0 reopened successfully. |
| Close and reopen the physical webcam | **Pass.** The webcam remained usable after reopening. |

The phone case verifies the region-count guard. Two physically present
people and spoof-model behavior were outside that check. Frontal Haar can
miss angled or profile faces. The detector locates regions, while identity,
liveness and emotion remain deterministic stubs and all persistence stays
disabled.

## Automated verification — 8 November 2026

The unit suites and both Tk smoke checks were rerun for this candidate
on Windows with Python 3.13.17. The full suite includes all 31 SYS-05
tests. Tk smoke checks used generated frames.

| Check | Rerun result |
| --- | --- |
| SYS-05 tests | **31 passed**, 0 failed, 0 skipped. Includes five regressions for nested and duplicate detections, ambiguous overlaps, separate faces, and agreement between the overlay and controller. Empty/invalid detections, mode changes, resets and stub safety also passed. |
| Full unit suite | **77 passed**, 0 failed, 0 skipped. |
| Tk with pinned OpenCV 5 | **Passed.** Typed missing-Haar error, manual fallback, all four scenarios, registration, reset, Clear regions, simulated camera failure, recovery and release. |
| Tk with isolated OpenCV 4.13 | **Passed.** Actual Haar returned `no_face` for a blank frame. Controlled nested and separate boxes exercised the filter, overlays, crop and multiple-face blocking. The manual, scenario, registration and camera lifecycle checks also passed. |

The unit checks use controlled detections. The physical results above come
from the separate owner run.

## Windows reproduction checklist for review

1. Launch `--camera 0` with the isolated wheel. Confirm live preview.
   Select **Automatic detection** and freeze a well-lit frontal face.
   Reproduce the framing that previously yielded the eye box. Check that
   the overlay and crop retain the enclosing face without a second eye box,
   then inspect the result JSON and `can_save=false`.
2. Freeze with no face, then with a person plus a face on a phone to repeat
   the owner's two-region case. Check `no_face` and `multiple_faces`.
   Record the actual count and scene if Haar misses a face. When two
   boxes appear, verify that neither one is selected for evaluation.
   Testing two physically present people is a separate supplementary case.
3. Switch to **Manual selection** on a frozen frame after an empty detection.
   Draw a green rectangle and evaluate it. Clear regions and verify both
   overlay removal and the blocking `no_face` result. Try an angled face
   and confirm the manual fallback when Haar misses it.
4. Use fresh captures for all four scenarios. Check `can_save=false` in
   attendance and registration. Confirm retry, operation changes, mode
   changes and scenario changes clear the previous result. After spoof
   and uncertain, confirm a fresh capture is required.
5. Close and reopen webcam 0. Run camera 99 or another unused index and
   confirm the typed unavailable-camera state, then reopen the working webcam.
   Record results as text, including the machine, camera index, resolution
   and runtime. Keep webcam images out of Git.

## Trang's independent-review checklist

1. Record the review commit, date, OS, Python version, OpenCV wheel and
   camera indices. Verify the isolated wheel and bundled cascade using
   the setup check above. Run both test commands and confirm the files below.
2. Start with `--demo-camera` in manual mode, then open the physical webcam
   and work through the reproduction checklist independently. Follow the walkthrough
   above and compare the four scenario payloads with the documented values.
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

The controller dispatches serially on the Tk thread. For a different
detector, replace `WebcamApp.detect_faces` with a callable that accepts
the frozen BGR frame and returns source boxes. For recognition, liveness and
emotion, use the corresponding `StubAdapters` methods and the `run_stubs`
call in `WebcamApp.evaluate` as the UI-03/UI-04 boundary. Preserve
`FaceRequest` source and context fields and return context-bound
`Completion` results. Trained results need verified preprocessing and a
reviewed controller path. The current liveness guard accepts only
`source="stub"`. Genuine persistence requires liveness, compatible gallery
and checkpoint metadata, and explicit enrollment or attendance authorization.

Only the following six paths belong in the SYS-05 commit:

- `docs/sys-05-phase-1.md` — updated audit, verification and handover.
- `src/facial_attendance/ui/webcam_app.py` — updated camera lifecycle, region modes and Tk display.
- `src/facial_attendance/ui/webcam_prototype.py` — updated shared box validation.
- `src/facial_attendance/ui/face_detection.py` — new Haar adapter.
- `tests/test_webcam_app.py` — new deterministic GUI flow tests.
- `tests/test_webcam_detection.py` — new deterministic Haar and filtering tests.

The existing `tests/test_webcam_prototype.py` remains part of the baseline
and requires no staging. Keep the unrelated
`reports/environment/vy-independent-retest.json` local. Exclude the
`artifacts/sys05-haar/` wheel and cascade files, `.venv/`, caches,
webcam images and recordings from the commit.
