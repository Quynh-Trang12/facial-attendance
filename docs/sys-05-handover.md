# SYS-05 runnable handover and demonstration

Owner: Vỹ. Independent reviewer: Trang. Prepared 16 November 2026.

Candidate: **SYS05-LOCAL-20261116-c00c004fc23a**, uncommitted source on
`feature/sys-05-webcam-gui-stubs`, baseline `39ac3df`. The precise files and hashes
are in the candidate's `SYS05-CANDIDATE.json`. This is READY FOR INDEPENDENT REVIEW,
not an independently reviewed/frozen release. No message or artifact has been
sent to another member by this work session. Receipt/acceptance are unconfirmed.

## Reproduce in a separate minimal environment

Use a Python 3.13 distribution with Tk support. The GUI needs only NumPy, Pillow
and OpenCV; installing the shared Torch environment is not required for this
prototype. Do not install these pins over the existing shared environment.

From the checkout root on macOS:

```sh
python3.13 -m venv artifacts/sys05-runtime
artifacts/sys05-runtime/bin/python -m pip install -r requirements-sys05-gui.txt
# Explicit model download, with hash/size checks and retained MIT licence:
artifacts/sys05-runtime/bin/python scripts/setup_sys05_yunet.py --download
PYTHONPATH=src artifacts/sys05-runtime/bin/python -m unittest discover -s tests -v
PYTHONPATH=src artifacts/sys05-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --diagnostics
```

From the checkout root in Windows PowerShell:

```powershell
py -3.13 -m venv artifacts/sys05-runtime
.\artifacts\sys05-runtime\Scripts\python.exe -m pip install -r requirements-sys05-gui.txt
.\artifacts\sys05-runtime\Scripts\python.exe scripts/setup_sys05_yunet.py --download
$env:PYTHONPATH = 'src'
.\artifacts\sys05-runtime\Scripts\python.exe -m unittest discover -s tests -v
.\artifacts\sys05-runtime\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --diagnostics
```

The fresh macOS environment and isolated source verification are recorded in the
owner report. **Windows execution remains OWNER VALIDATION REQUIRED**; historical
Windows Haar evidence is not new YuNet verification. No fixed FPS or accuracy
threshold is introduced as an acceptance criterion.

For the owner's already configured project environment on macOS:

```sh
PYTHONPATH=artifacts/sys05-haar:src .venv/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --diagnostics
```

The existing `requirements-sys05-haar.txt` / `artifacts/sys05-haar` path is retained
for compatibility. In a fresh minimal environment, use `PYTHONPATH=src` so it uses
that environment's own pinned OpenCV. Both paths leave team requirements untouched.
When the model is already supplied and verified, omit `--download` to check it
offline. The local review snapshot includes the explicitly copied model and
licence, but the source commit must not include model weights or environments.

## Useful verification commands

macOS commands below assume the minimal environment. Windows uses its
`Scripts\python.exe` executable and `$env:PYTHONPATH = 'src'` with identical arguments.

```sh
# Headless interface walkthrough; generated BGR image and supplied test box:
PYTHONPATH=src artifacts/sys05-runtime/bin/python scripts/demo_sys05.py

# Actual native Tk widgets with generated frames and controlled detections:
PYTHONPATH=src artifacts/sys05-runtime/bin/python scripts/smoke_sys05_gui.py

# Real detector, synthetic source: should be no-face, not a fabricated match:
PYTHONPATH=src artifacts/sys05-runtime/bin/python -m facial_attendance.ui.webcam_app --demo-camera

# Explicit scenario testing on eligible camera detections:
PYTHONPATH=src artifacts/sys05-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --dev-scenario Spoof
PYTHONPATH=src artifacts/sys05-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --dev-scenario Uncertain
PYTHONPATH=src artifacts/sys05-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --dev-scenario 'Live unknown'

# Bounded Haar A/B comparison; no manual selection required:
PYTHONPATH=src artifacts/sys05-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector haar --diagnostics
```

Close one camera window before opening another. Camera index 99 is only an example
of an unavailable device; use an actually unused index. Retry clears the current
generation and reopens the device. Check OS camera permissions for the launching
Python/terminal application. Missing/corrupt models and unavailable APIs fail with
a detector error; startup never downloads or silently chooses another detector.
If a native driver hangs, exit/restart the application; daemon shutdown is bounded
and late work cannot update closed UI. A backend may ignore the one-frame buffer hint.

## Adapter replacement for UI-03 / UI-04

| Boundary | Current entry point | Required replacement behavior |
| --- | --- | --- |
| Detection | `ScaledFaceDetector(YuNetFaceDetector())`, or injected callable | Accept immutable original BGR uint8 source; return validated source-pixel half-open boxes. Scaling/rounding stays at the boundary. No box invented on missing detection. |
| Request | `FaceRequest(context, bbox, frame)` / `.face_crop` | Use the exact source and unsmoothed box. Each model owns crop policy, colour conversion, resizing and normalization; never mutate the shared frame. |
| Recognition | `StubAdapters.recognition(request)` | Return `Completion(request.context, Recognition(...))`. `person_id/name/score/accepted/model_id/model_version` follow SYS-02. Convert raw model vectors to normalized matching inputs and verify checkpoint/gallery compatibility. |
| Liveness | `StubAdapters.liveness(request)` | Return context-bound `Liveness(status, reason, p_spoof, version fields, source)`. Trained preprocessing, threshold and quality policy come from the selected model, never guessed by the GUI. |
| Emotion | `StubAdapters.emotion(request)` | Return context-bound `Emotion(class_id, label, confidence)`, explicit 0 Negative / 1 Neutral / 2 Positive. Failed/invalid emotion clears display only. |
| Errors | `Completion(original_context, OperationError(...))` | Use `module/code/message/blocks_persistence`; the controller owns severity. Camera/recognition/liveness errors block the operation. Invalid responses become typed errors. Stale contexts are discarded. |
| Scheduling | `evaluate_adapters` / `run_stubs`; `SingleWorker` | One detector and one adapter job/result slot, serial liveness first. Worker-local validation returns original completions; only Tk applies active state. No unbounded jobs or worker widget calls. |
| Application | `LiveWebcamApp(..., detector=..., adapters=...)` | Construct/load adapters once; preserve source association, cancellation by supersession, expiry, Retry and shutdown. |
| Genuine authorization | No implementation in SYS-05 | Current liveness validator accepts only `source=stub`, `Envelope.can_save` is always false, and no attendance/gallery calls exist. A trained adapter alone does not enable saving. UI-04 must add and verify model/policy versions, gallery checks, sessions/duplicates, final serialized authorization and persistence. |

New model integration must use a separately reviewed trained path. Do not weaken
the stub guard to make a model appear integrated. Current remote model/gallery
interfaces and exact branch heads are reconciled in
[the contract audit](sys-05-contract-traceability.md#current-team-interfaces-read-without-merging).
LIV-03 dataset `_BB.txt` cropping is not the webcam coordinate contract.

## Concise English demonstration script

1. “This is an automatic attendance **preview**. Camera capture and face detection
   are real; identity, liveness and emotion are simulated. No records are saved.”
2. Launch the camera command. Show continuous preview and one source-frame box.
   Explain bounded 640-edge detector input and outward mapping back to the original
   BGR frame. A grey dashed outline is an unconfirmed historical display box.
3. Open debug JSON. Show `sys02-v1`, `attempt_id`, `action`, `capture_id`, `face_id`
   and the three result groups. Explain that face IDs are not person IDs.
4. Run `scripts/demo_sys05.py`. Show the generated source shape, half-open box,
   exact shared crop and distinct BGR fixture values. Do not describe this supplied
   test rectangle as real face detection.
5. Explain the adapter order: liveness first, recognition only for live, emotion
   as information. Compare Live match, Live unknown, Spoof and Uncertain fixtures.
6. Point to `source=stub`, null model metadata and `can_save=false`. A positive
   stub cannot establish a real identity or permit attendance/enrollment.
7. Leave the physical frame or use the controlled native face-loss case. Show
   cleared results; multiple detected faces block the operation.
8. Retry and show fresh identifiers. The headless/native stale tests demonstrate
   rejection of an old callback, without relabelling it for the new frame.
9. Show diagnostics: capture/preview rate, duration, local frame age and intentional
   sampling/drops. Quote measured values only with their actual input/runtime scope.
10. Hand over the adapter table and source snapshot manifest. Record Trang's actual
    questions/findings, then retest corrections. Do not declare reviewer approval.

This walkthrough was exercised locally through four controlled interface scenarios
and 14 native Tk cases. **It has not been presented to Trang in this session.**

## Likely integration questions

- **Why does every eligible face show Demo Person?** The default scenario is a
  deterministic fixture; image content does not determine the simulated outputs.
- **Does YuNet detect spoofing or recognize a person?** No. It locates face regions.
- **Where do trained input sizes and thresholds come from?** From each selected
  checkpoint and its versioned configuration/policy, reconciled with SYS-02.
- **Can a late worker completion change the new person's result?** Only a matching
  token and original full context can be applied; reset/supersession reject old work.
- **Does a held outline allow an old identity to remain current?** No. Outline
  smoothing is display-only and bounded; model fields clear on every new sample/loss.
- **Is this final training or accuracy evidence?** No. Synthetic benchmark timings
  are not real webcam FPS or recognition accuracy. Physical passes are the owner's
  observations on YuNet/macOS, with slight box hesitation still documented.
- **Can gallery v0.2 be used later?** Yes, subject to UI-03/UI-04 integration and
  checkpoint compatibility. The gallery does not enforce liveness on behalf of callers.

## Copy-ready handover message — not sent

> Hi Trang and UI-03/UI-04 team,
>
> SYS-05 is prepared as local candidate SYS05-LOCAL-20261116-c00c004fc23a on
> feature/sys-05-webcam-gui-stubs, based on 39ac3df. The automatic YuNet preview
> keeps identity, liveness and emotion simulated and disables genuine saving.
> All 131 automated tests and 14 native Tk cases passed. I confirmed the eight
> macOS physical checks, including glasses, responsiveness, clearing and recovery;
> slight box hesitation during movement remains documented.
>
> Please use docs/sys-05-handover.md for setup and adapter boundaries, and
> docs/sys-05-owner-self-review.md for the acceptance matrix and evidence. Confirm
> the exact snapshot/commit you receive, then record your independent findings and
> rerun the relevant cases. Windows execution and reviewer confirmation remain
> pending. This is an uncommitted local candidate, not a frozen release.
>
> Please acknowledge receipt when the package has actually been shared. Thanks, Vỹ.

Do not send this text claiming an attached/published package until the owner actually
supplies the snapshot or authorizes a source commit/push. No delivery or receipt is
claimed by this preparation document.
