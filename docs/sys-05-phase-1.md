# SYS-05 — Automatic live webcam prototype

The dropdown/freeze workflow has been superseded by an automatic attendance
preview. This local redesign is based on `39ac3df` on
`feature/sys-05-webcam-gui-stubs`; it has **no new review-candidate commit**.
The owner physically retested YuNet on macOS on **16 November 2026** and reported
all eight checks PASS, including glasses, responsiveness, clearing and recovery.
Slight box hesitation remains documented. Independent review, final freeze and
actual handover delivery remain pending. Do not treat this document as Owner Evidence or
mark tasks complete.
Do not commit or push this redesign before the owner validates the GUI.

## Current owner verification — 16 November 2026

Candidate **SYS05-LOCAL-20261116-c00c004fc23a** identifies the current UI runtime;
HEAD is still `39ac3df` and no new commit/final release freeze is claimed.
The [consolidated owner self-review](sys-05-owner-self-review.md) contains the
11-task acceptance matrix, actual results, attributed physical evidence and final
Owner Evidence for completed owner-side portions. [SYS-02 traceability](sys-05-contract-traceability.md)
maps requirements to functions and tests. [Handover and demonstration](sys-05-handover.md)
provides current clean-environment commands, integration boundaries and an unsent
message for Trang/UI-03/UI-04.

- **AUTOMATED TEST PASS:** 131 tests, 0 failed, 0 skipped, including all previous
  120. The same suite passed from an isolated source snapshot in a fresh minimal
  venv. Fourteen native Tk controlled cases passed in both runtimes.
- **PHYSICAL TEST PASS:** owner-reported macOS/YuNet checks for glasses/no glasses,
  normal indoor light, movement, face-loss/return, Retry and camera recovery.
  No physical FPS, backlight or Windows result was invented.
- **READY FOR INDEPENDENT REVIEW:** identified local source/package and instructions.
  **REVIEWER CONFIRMATION PENDING:** Trang's run, findings, walkthrough and final
  reviewed freeze. Actual handover delivery/receipt have not occurred here.

The earlier dated 11/12 November sections below retain their original historical
measurements and test counts. Their not-tested/open statements describe that
stage; the 16 November evidence above is the current status.

## Sources and acceptance scope

Read on 11 November 2026, before implementation:

- [SYS-02 Final v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d):
  source/crop coordinates, context correlation, liveness gating, clearing and stub safety.
- [UI-02 scheduling and states](https://app.notion.com/p/412863094ebe80c09ffce13f894f16ae):
  bounded background inference, main-thread UI changes and stale-result rejection.
- [Requirements rows 39–49](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit#gid=2016101501&range=A39:S49):
  SYS-05 parent and SYS-05.1–SYS-05.10. The Sheet was read only, never edited.

The criteria require a runnable, clearly identified stub prototype with replaceable
interfaces; trained model completion is separate. Reviewer runs, sample-output
agreement, retesting, release checks, demonstration and handover remain pending.
No reviewer approval or revised task completion is claimed here. Any future Owner
Evidence must be a final self-review, at most 60 English words per task, grounded
in the owner's actual revised-runtime checks.

## Normal workflow

1. Launch the module. Camera 0 opens automatically and the live preview starts.
   There are no operation, scenario or detection-mode dropdowns.
2. Keep one frontal face in view. CPU YuNet detects source-frame face regions in the
   background; amber boxes show the sampled detection. A dashed grey box is a previous,
   unconfirmed detection and can persist only up to 450 ms from its source sample.
   Zero faces and multiple faces block adapter evaluation and clear previous model results.
3. Exactly one valid region automatically creates a fresh attendance capture and
   runs liveness, recognition (only after simulated live) and emotion adapters.
   Human-readable identity, liveness, emotion and status update automatically.
   No **Freeze frame** or **Run stubs** action is required.
4. **DEMO / SIMULATED OUTPUT** remains visible. **Show / hide SYS-02 debug JSON**
   exposes the secondary envelope. **Retry / restart camera** clears the current
   attempt and reopens the camera on its existing worker.

Face detection actually examines image pixels. The default downstream scenario
is the deterministic **Live match** fixture: every eligible face gets the same
fabricated Demo Person / live / Positive values. Those values are **not inferred
from image content** and are not genuine recognition, spoof or emotion predictions.
`liveness.source="stub"`, null model versions and `can_save=false` distinguish the
payload. There are no attendance, enrollment, gallery or storage writes.
Emotion never authorizes persistence. Images and crops stay in memory.

## Scheduling and SYS-02 safety

`live_preview.py` supplies three persistent daemon workers:

- One camera thread owns open/read/release and publishes only its latest owned,
  read-only BGR `uint8[H,W,3]` snapshot. It budgets a 30-Hz period and subtracts
  blocking read/copy time before sleeping; this is a target, not measured
  physical-camera FPS. Retry increments its generation; an old open/read cannot
  publish into the restarted camera session.
- One detector thread has one job/result slot. The GUI samples the latest camera
  frame at most every 300 ms, with no detector backlog. Both automatic detectors
  receive aspect-preserving images with maximum edge 640, then map validated
  boxes outward to the original source using the actual integer resize ratios.
- One adapter thread has one job/result slot and a worker-local controller for
  serial liveness gating. It returns the original context-bound completions.
  Busy slots reject extra submissions; no pending frame queue or new per-frame
  threads are created. If old adapter work is still busy, the current sample
  stays processing and a later eligible sample tries again.

The Tk timer only consumes slots, converts/draws the preview, submits work and
updates widgets. Workers never touch Tk or the active GUI controller. Camera
I/O, detection and adapters execute outside the rendering loop.

Each analysis sample supersedes the previous sample and clears its output before
new work is accepted. The detector token is `(camera_generation, frame_sequence)`.
The selected analysis snapshot is the active inference frame; raw preview frames
continue updating independently and are not all inference requests. Its validated
half-open `[x1,y1,x2,y2]` box and the same original BGR source go to every adapter.
`contract_version="sys02-v1"`, `attempt_id`, `action`, `capture_id` and `face_id`
remain correlated through the original request and all completions.

Only matching analysis tokens and matching SYS-02 contexts can update output.
Retry, supersession, camera failure, detector failure and analysis age above one
second invalidate context and clear all boxes/results. A no-face sample
immediately clears model fields; only the explicitly unconfirmed outline may
remain briefly. Multiple faces clear single-box smoothing history and block
adapters immediately. New samples clear all prior
identity/liveness/emotion values, including after a face change; no identity
tracking is inferred from overlapping boxes. No-face and multiple-face samples
clear the model fields. Camera frames older than one second are unavailable;
opening without a frame shows a camera problem after five seconds. A later fresh
frame or explicit retry can recover. Detector errors remain visible until retry.

A spoof aborts its sample, blocks recognition/emotion callbacks and never saves.
Uncertain skips recognition; its emotion is informational. Every subsequent
sample is a fresh automatic demo attempt; explicit Retry also starts a new
attempt. Neither a positive stub nor emotion can authorize genuine records.

On close, timers stop, contexts invalidate and worker slots close. The camera
is released by its owning thread. Joins are bounded to 100 ms per worker so Tk
cannot hang on a blocked native call. A hung native camera/model call cannot be
forcibly interrupted safely: its daemon may finish later, but cannot publish or
update the closed UI. Retry reuses the same camera worker rather than creating
replacement threads. A permanently hung driver may require restarting the app.

## Isolated OpenCV setup and compatibility

Local read-only checks on this Mac confirmed the shared OpenCV 5.0.0 package
(from the 5.0.0.93 wheel) lacks `CascadeClassifier`. The existing isolated
4.13.0.92 wheel loads `haarcascade_frontalface_default.xml`. Historical Windows
checks found the same 4.13 wheel usable. This is evidence about those tested
wheels, not a guarantee about all OpenCV distributions.

Keep `requirements.txt`, the shared environment and its OpenCV pin unchanged.
`requirements-sys05-haar.txt` retains its compatibility filename and contains
only the isolated OpenCV pin, now used for both Haar and YuNet.
`--no-deps --target artifacts/sys05-haar` uses the project's existing NumPy,
Pillow and Tk without replacing shared packages. Installation is explicit;
the application never downloads classifiers or models. Reuse the target if the
check passes; do not mix multiple OpenCV wheels in it.

From the repository root, use an existing Python 3.13 `.venv` with NumPy 2.2.6,
Pillow 11.2.1 and Tk. For a new GUI-only checkout, create `.venv` with Python 3.13
and install those two packages there first; the GUI does not require PyTorch.
The full repository suite requires the ordinary project test dependencies.
If Tk is missing, install a Python 3.13 distribution with working Tk support.

### macOS (zsh)

```sh
# Once, only if the isolated wheel is absent:
.venv/bin/python -m pip install --no-deps --only-binary=:all: --target artifacts/sys05-haar -r requirements-sys05-haar.txt

# Verify that imports come from the isolated target and the classifier loads:
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -c "from pathlib import Path; import importlib.metadata as md, cv2; p=Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'; assert md.version('opencv-python') == '4.13.0.92'; assert Path(cv2.__file__).resolve().is_relative_to(Path('artifacts/sys05-haar').resolve()); assert p.is_file() and not cv2.CascadeClassifier(str(p)).empty(); print(cv2.__version__, cv2.__file__, p)"

# Explicit model setup (network only with --download; already present on this Mac):
.venv/bin/python scripts/setup_sys05_yunet.py --download

# Default automatic physical webcam preview:
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --diagnostics

# Synthetic source: real YuNet should report no face, no simulated results:
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --demo-camera

# Camera-unavailable case (choose an unused index on this machine):
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 99

# Detector-unavailable case with the tested shared OpenCV 5 wheel:
PYTHONPATH=src .venv/bin/python -m facial_attendance.ui.webcam_app --demo-camera
```

Allow camera access for the launching terminal/Python application in macOS
Privacy & Security if needed. Close each window before the next launch. Return
to camera 0 after the unavailable-camera test and check that it opens again.

### Windows (PowerShell)

```powershell
# Once, only if the isolated wheel is absent:
.venv\Scripts\python.exe -m pip install --no-deps --only-binary=:all: --target artifacts/sys05-haar -r requirements-sys05-haar.txt

$env:PYTHONPATH = 'artifacts/sys05-haar;src'
.venv\Scripts\python.exe -c "from pathlib import Path; import importlib.metadata as md, cv2; p=Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'; assert md.version('opencv-python') == '4.13.0.92'; assert Path(cv2.__file__).resolve().is_relative_to(Path('artifacts/sys05-haar').resolve()); assert p.is_file() and not cv2.CascadeClassifier(str(p)).empty(); print(cv2.__version__, cv2.__file__, p)"

# Explicit model setup (network only with --download):
.venv\Scripts\python.exe scripts/setup_sys05_yunet.py --download

# Default automatic physical webcam preview:
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --diagnostics

# Synthetic no-face source:
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --demo-camera

# Camera unavailable, then close and relaunch camera 0:
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 99
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0

# Detector unavailable with the tested shared wheel:
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --demo-camera
# Restore the isolated path before further webcam checks:
$env:PYTHONPATH = 'artifacts/sys05-haar;src'
```

Camera indices are device-dependent. Check Windows camera privacy permissions.
The revised GUI and setup commands have not been executed on Windows in this
redesign session; historical Windows results do not validate this implementation.

## Explicit developer harness

No fake-outcome selector appears in the normal UI. `--dev-scenario` injects one
fixture at startup for developer testing, with the simulation banner unchanged:

| CLI scenario | Simulated liveness | Simulated recognition | Simulated emotion |
| --- | --- | --- | --- |
| `Live match` (default) | live, 0.08 | Demo Person, accepted, 0.91 | 2 Positive |
| `Live unknown` | live, 0.08 | Unknown, null identity/score | 1 Neutral |
| `Spoof` | spoof, 0.96 | Skipped | Cleared |
| `Uncertain` | uncertain, 0.45 | Skipped | 0 Negative |

macOS examples (use the same arguments after the Windows module command):

```sh
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 0 --dev-scenario Spoof
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 0 --dev-scenario Uncertain
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 0 --dev-scenario 'Live unknown'

# Separate manual fallback: freeze, draw a source region, then Run stubs.
# This synthetic source contains no actual face; a marked rectangle is manual input.
PYTHONPATH=src .venv/bin/python -m facial_attendance.ui.webcam_app --demo-camera --dev-manual --dev-scenario Spoof

# Separate non-persistent registration harness (never normal attendance UI):
PYTHONPATH=src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 0 --dev-manual --dev-action registration
```

In PowerShell, for example:

```powershell
$env:PYTHONPATH = 'artifacts/sys05-haar;src'
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0 --dev-scenario Spoof
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0 --dev-scenario Uncertain
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --demo-camera --dev-manual --dev-scenario Spoof
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0 --dev-manual --dev-action registration
```

`--dev-action registration` requires `--dev-manual`. This is an independent,
non-persistent developer operation. The retained `WebcamApp` selection methods
also support the old deterministic regression harness; normal launch uses
`LiveWebcamApp`. Manual developer evaluation is synchronous and is intended only
for the cheap deterministic stubs. Trained models belong on the bounded live path.

## Verification — 11 November 2026 (before these findings)

Before changes: **77 passed, 0 failed, 0 skipped**. All 77 original tests and
historical evidence are preserved. After changes: **99 passed, 0 failed,
0 skipped** (22 new hardware-free tests). Command:

```sh
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

PowerShell equivalent:

```powershell
$env:PYTHONPATH = 'src'
.venv\Scripts\python.exe -m unittest discover -s tests -v
```

New tests mock camera, detector, widgets and adapters for automatic scheduling,
no controls required, one/no/multiple face states, bbox scaling, clear-on-loss,
missing detector, stalled camera, retry, expiry, mismatched contexts and late
completions. Thread tests exercise actual bounded workers, off-main-thread camera
ownership, immutable frame publication, exception reporting and safe shutdown.
Existing scenario, crop, nested-Haar filtering and contract regressions remain.

Native checks: the shared wheel lacks the binding; the isolated classifier loads;
actual Haar returns `[]` on `DemoCamera`'s generated image. CLI help also works.
These are not physical-webcam or trained-model checks.

**NOT TESTED in the 11 November agent run:** revised native Tk window rendering, physical webcam behavior,
visual responsiveness, macOS camera permissions/recovery and all revised Windows
GUI/webcam checks. Automated widget mocks are not native GUI acceptance. The owner
must run the checklist below before selecting a review candidate or finalizing
Owner Evidence. No trained model accuracy or genuine spoof detection is claimed.

## Owner manual acceptance checklist

Record date, OS, Python/OpenCV versions, camera index, capture resolution, observed
steps and pass/fail/NOT TESTED. Keep face images and recordings out of Git.

- [ ] Launch camera 0: opening state leads directly to moving preview, with no
  dropdown, freeze or manual-evaluation requirement. Read the simulation banner.
- [ ] Present one frontal face: see an aligned sampled box and automatically
  refreshed identity, liveness, emotion and status. Confirm that changing facial
  expression does not turn the deterministic fixture into a real prediction.
- [ ] Open/close debug JSON; verify SYS-02 fields, `source=stub`, fresh identifiers,
  `action=attendance` and `can_save=false`. No record-saving control exists.
- [ ] Remove the face: previous outputs clear on the next analysis sample. Present
  two faces: when the selected detector finds two distinct boxes, both appear and evaluation
  blocks. Record misses/false positives honestly rather than claiming success.
- [ ] Re-enter after no face; use Retry after a good result. Previous outputs
  clear, a fresh camera generation/attempt starts, and new results recover.
- [ ] Run developer Spoof and Uncertain commands with a detected face. Verify the
  explicitly simulated warning, hidden identity and blocked persistence. Retry
  and verify no old success briefly reappears. Repeat Live unknown if needed.
- [ ] Run the synthetic camera with either detector: no face and no downstream model results.
  Check the separate manual fallback and registration harness if needed.
- [ ] Launch the shared-wheel detector-error case: moving synthetic preview,
  detector-unavailable text, no fabricated overlay and no old results. Close and
  relaunch with the isolated wheel to recover.
- [ ] Launch an unused camera index; check unavailable status and cleared output.
  Reopen camera 0, then test disconnect/reconnect and Retry if hardware permits.
- [ ] Check that preview, buttons and debug panel stay responsive during analysis.
  Close while processing, then reopen. Check camera release/recovery. Expired and
  superseded callbacks are covered deterministically in the new automated suite.
- [ ] Only after owner validation, identify the next candidate commit for Trang's
  independent run, findings/retest, demo and handover. Do not claim those steps now.

## Adapter replacement and remaining limitations

`HaarFaceDetector` and `YuNetFaceDetector` are replaceable callables returning
validated boxes; `ScaledFaceDetector` maps bounded inputs back to source boxes.
`LiveWebcamApp(detector=..., adapters=...)` supports
hardware-free injection and future integration. `FaceRequest` contains the
original source and half-open bbox, `face_crop`, and shared context.
`StubAdapters.liveness/recognition/emotion` return `Completion` objects.
`evaluate_adapters` runs those on its worker-local controller; `run_stubs` now
also returns the completions it applied. The active controller revalidates each
completion on the main thread, retaining all existing envelope/error semantics.

UI-03/UI-04 must supply trained adapters with verified model-owned preprocessing,
checkpoint metadata and policy. The prototype's liveness validator intentionally
accepts only `source=stub`; replacing it with a model requires a reviewed controller
integration, not just changing a label. No trained checkpoint, threshold, crop
padding or normalization is guessed here. No persistence path is introduced.

The optional Haar fallback is a frontal-face heuristic: pose, lighting, occlusion, phone images and
nested features can cause misses/false positives. Existing containment/IoU
filtering is retained and tested; it does not make Haar robust or anti-spoofing.
Model results expire after one second; confirmed sampled boxes can lag the
moving preview, and dashed historical outlines expire after 450 ms. Detection and
adapter latency are not measured physical-camera performance claims. Slow work
may be superseded and never displayed; native calls cannot be forcibly canceled.
Registration is only the developer manual harness; no normal registration UX or
final attendance/session rules are implemented under SYS-05.

## Performance and glasses findings — 12 November 2026

At the 12 November stage, the two owner findings remained **OPEN pending physical
comparison**. On 16 November the owner subsequently reported eight YuNet/macOS
physical checks PASS; see the current self-review. The earlier statements here
are retained as dated history, not current unresolved owner failures. No final Owner Evidence is created.

### Findings: measured software costs versus suspected physical causes

The previous camera loop slept another 30 ms after a blocking read and immutable
copy. The Tk timer likewise added 33 ms after rendering. Rendering converted the
full source to RGB/Pillow before shrinking it. Haar saw full-resolution input;
canvas regions were deleted/recreated every tick; the hidden JSON widget was
still serialized/updated. These are confirmed code paths. Their contribution to
the owner's physical lag is not yet measured. One owned frame copy is retained
for safe source association even if a camera backend reuses its read buffer.

A reproducible **synthetic CPU microbenchmark**, not a webcam/GUI accuracy test,
ran on this Mac with Python 3.13.17, isolated OpenCV 4.13, one native thread,
1920×1080 seeded noise, two warmups and eight timed runs (median below):

| Software operation | Measured result |
| --- | --- |
| Previous full-frame RGB/Pillow preparation | 14.419 ms |
| Resize BGR first, then convert only the preview | 2.319 ms |
| Full-HD Haar | 790.131 ms |
| Haar at maximum edge 640, including coordinate mapping | 75.877 ms |
| CPU YuNet at maximum edge 640, including coordinate mapping | 13.088 ms |
| Paced fake 30-Hz source, reproduced old fixed-wait loop | 13.85 published frames/s |
| Same paced source through current CameraFeed | 25.95 published frames/s |

Noise can be a very different Haar workload from a face scene. Capture numbers
include copying and software scheduling. They are **not physical webcam FPS**,
Tk presentation FPS, end-to-end camera latency or evidence of glasses accuracy.
Reproduce without opening a webcam or Tk:

```sh
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python scripts/benchmark_sys05.py --runs 8
```

PowerShell: set `$env:PYTHONPATH = 'artifacts/sys05-haar;src'`, then run
`.venv\Scripts\python.exe scripts/benchmark_sys05.py --runs 8`.

Camera driver buffering, actual exposure/read timing, macOS Tk image upload and
CPU contention remain suspected contributors requiring the owner's diagnostics.
The previous OpenCV default was eight native threads on this Mac; the CLI now
uses one to limit competition between capture/rendering/detection, configurable
with `--opencv-threads`. A best-effort one-frame camera buffer hint may be ignored
by the backend. Neither setting proves physical latency improvement.

[OpenCV's Haar explanation](https://docs.opencv.org/4.13.0/db/d28/tutorial_cascade_classifier.html)
describes eye/nose contrast features. Glasses frames, glare, occlusion and pose
can alter those features; this is a plausible explanation for the reported misses,
not a verified diagnosis of the owner's scene. No glasses images were collected
or evaluated during this change.

### Selected detector and explicit model provenance

**YuNet is the provisional default** because it provides a replaceable trained
face detector with low measured CPU cost at the bounded input size. Its real
face localization does not make the downstream identity, liveness or emotion
stubs real predictions. Glasses reliability still needs a paired physical test.
`--detector haar` selects the bounded Haar comparison/fallback, without manual
selection. There is no silent fallback when YuNet is missing or fails.

The chosen model is OpenCV Zoo's `face_detection_yunet_2023mar.onnx`, 232,589 bytes,
SHA-256 `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
The explicit setup script downloads model and licence into ignored
`artifacts/sys05-yunet/`, validates both size and hash, and refuses mismatches.
The URLs use upstream `main`; the bytes are pinned by hash, so upstream changes
fail validation rather than silently changing the selected model. Without
`--download`, the setup script only checks existing local files. Application
startup has no download code. No model weights are added to Git.

Sources checked on 12 November 2026:

- [OpenCV Zoo model and provenance](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet)
- [Published model LFS SHA-256](https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx)
- [MIT licence, Shiqi Yu, 2020](https://github.com/opencv/opencv_zoo/blob/main/models/face_detection_yunet/LICENSE),
  retained beside the downloaded model; preserve its notice when redistributing.
- [OpenCV 4.13 FaceDetectorYN API](https://docs.opencv.org/4.13.0/df/d20/classcv_1_1FaceDetectorYN.html):
  CPU backend/target, input sizing and 15-value output rows.

The actual 4.13.0.92 macOS wheel loaded and executed the model on CPU. YuNet
uses score threshold 0.9 and NMS 0.3 from the OpenCV API defaults. These locate
faces only and are unrelated to liveness/recognition authorization thresholds.
Finite positive DNN rectangles crossing an edge are clipped to their intersection;
empty, fully outside, nonfinite or malformed regions reject the batch. Starts are
floored, ends ceiled, then mapped to source half-open coordinates. Multiple valid
faces remain multiple; no single-person selection is fabricated.

The chosen 2023 model is restricted to the tested OpenCV 4.x path. OpenCV Zoo now
also has a 2026 dynamic-shape export for OpenCV 5; it is not substituted here.
Windows uses the same isolated wheel version and standard-library setup script,
but **revised Windows model execution is NOT TESTED** in this environment.

### Diagnostics and reduced GUI work

Append `--diagnostics` to either detector command. A `SYS05_PERF` JSON line prints
to stderr every five seconds and once on shutdown. Counters and timing aggregates
have bounded memory and contain no images or identity data. No per-frame logging.

- `capture_fps` counts successfully read/copied/published frames; `preview_fps`
  counts distinct frames rendered by Tk. The first window includes startup.
- `camera_read`, `camera_copy`, `render`, `gui_tick`, `detector`, `adapters` and
  visible `debug_json` report mean/max duration and sample count in milliseconds.
- `preview_age` measures read-return to completion of GUI rendering. It excludes
  sensor exposure and backend buffering and is not photon-to-screen latency.
- `preview_dropped` counts camera sequence gaps between displayed frames.
  `analysis_unsampled` counts frames intentionally skipped by analysis throttling;
  `analysis_superseded` counts replaced active samples. These are distinct from
  `detection_discarded`, `adapter_discarded`, `adapter_busy_dropped` and
  `analysis_expired`; `capture_discarded` counts publications rejected after reset/close. Intentional sampling is not a detection failure rate.

The camera now subtracts time already spent reading/copying from its frame-period
wait. The Tk timer subtracts its own work from its 30-Hz target. Rendering resizes
BGR before RGB/Pillow conversion and reuses the canvas image item. Region redraws
occur only when their appearance changes. Summary text is cached separately from
request IDs. Hidden JSON is not serialized; opening it refreshes current state.
Expensive detection and adapters remain off the Tk thread, with one slot each;
when a slot frees, analysis samples the latest camera frame, not a queued backlog.

### Bounded display stability and safety

Only one previous display box, source size and timestamp are retained. Nearby
single-face detections (IoU above 0.5, within 450 ms) blend 70% current geometry
and 30% previous geometry for the outline. Adapter requests always use the current
unsmoothed box and original source frame. Large moves or size changes do not blend.

While a sample is processing or after a brief miss, a dashed grey outline may
show that **unconfirmed previous detection**, at most 450 ms from its source
sample. No old identity, liveness or emotion is retained for it. A no-face sample
immediately clears every model field; multiple faces, failures and Retry clear
the single-box history. This is display smoothing, not identity tracking or
substitute detection evidence. Old contexts/tokens still cannot overwrite current
state, and `can_save=false` remains enforced for all scenarios.

### Verification and owner comparison still required

The full suite now reports **120 passed, 0 failed, 0 skipped**: all 99 previously
passing tests plus 21 new regressions. No earlier tests or evidence were removed.
New tests cover scaling and half-open rounding, detector throttling/latest-frame
selection, preview drops, frame-period budgeting, hidden JSON, region caching,
rapid face loss, display-only smoothing, multiple faces, stale completions,
missing/corrupt models, CPU API selection and diagnostics. Existing worker
shutdown and camera ownership tests still pass. Both real detectors returned
`[]` on the generated camera image; that is not a real-face accuracy test.

**NOT TESTED in the 12 November agent run:** physical webcam FPS/latency, native Tk visual
responsiveness, glasses improvement, backlight robustness, physical camera
recovery and Windows execution. Owner feedback was unresolved pending these
checks at that stage. The owner report above supersedes the pending physical
checks for the eight confirmed YuNet/macOS cases. Run each detector in separate
launches for any further A/B comparison, same camera, distance and scene;
wait at least ten seconds after startup, then record several diagnostic windows:

```sh
# macOS: default candidate and bounded Haar comparison
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --diagnostics
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector haar --diagnostics
```

```powershell
$env:PYTHONPATH = 'artifacts/sys05-haar;src'
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --diagnostics
.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0 --detector haar --diagnostics
```

| Owner comparison | Record / expected safety |
| --- | --- |
| Without glasses, normal indoor light | Acquisition, missed boxes, preview FPS and local age for both detectors |
| With glasses, same light/distance/pose | Compare misses/jitter; do not infer improvement from the model choice alone |
| Bright backlight, with and without glasses | Misses/false positives and preview responsiveness; record failures honestly |
| Entering/leaving frame repeatedly | Model fields clear on no-face; any dashed outline expires; fresh context on return |
| Two people / additional face image | Multiple detected faces block adapters immediately; no prior identity remains |
| Retry, camera failure, close/reopen | Clear results/history, reject late output and release/recover camera |

Also repeat the developer Spoof/Uncertain scenarios and inspect `source=stub`,
context fields and `can_save=false`. Keep webcam images out of Git. Record
pass/fail/NOT TESTED as a comparison report, not final Owner Evidence.

## Earlier verified evidence

[The verbatim document snapshot from 39ac3df](sys-05-legacy-39ac3df.md) preserves
the prior Windows physical-webcam and automated/Tk observations, compatibility
audit and original commands. Its old dropdown instructions and references are
historical, not current setup guidance. Earlier macOS evidence remains in the
owner's original records; `reports/environment/vy-mac-m1-review.json` is untouched
and must not be modified, staged or committed. None of that evidence is relabelled
as acceptance of this redesigned GUI.
