# UI-03/UI-04 webcam presentation regression — 18 November 2026

> Follow-up: the owner physically failed the subsequent presentation change too. See [Emotion continuity investigation](ui-03-emotion-continuity.md) for measured camera failures, the independent Emotion display, current 284-test/20-native-check evidence and pending physical retest. Measurements below remain the historical geometry-fix record.

**Status: implementation and automated/native verification complete; owner physical retest pending.** The owner reported stuttering and clarified that the result panel jumps or resizes. That physical acceptance finding is not marked resolved by these generated-input measurements.

Work stays on `feature/ui03-ui04-emotion-integration` in `/Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration`. The starting implementation had 217 passing automated tests and 15 passing controlled native checks. Source was frozen under ignored `artifacts/ui03-performance-before` before changing product code; verified SYS-05 `798f2e7` was separately read into `artifacts/ui03-performance-sys05`. No original branch, environment, model file, protected JSON or Google Sheet was changed. Nothing is staged, committed, pushed or merged.

## Diagnosis: measured versus suspected

**Reproduced:** under mapped native Tk, generated 1280×720 input, actual Face/Emotion weights, real CPU YuNet work and controlled jittering single-face boxes, the integrated GUI resized **56 and 58 times in two 10-second windows**. The verified SYS-05 implementation had **zero** geometry changes with the same generated source and detector workload. The integrated result panel grew beyond the preview's height: alternating Processing/results and short/long metadata text changed requested geometry. This directly matches the owner's clarified observation.

Model metadata was written 88–91 times per 10 seconds, although the selected checkpoint never changed. Identity/Emotion fields clear on each sampled context; both detection-pending and adapter-pending phases also rewrote the summary. Registration stayed packed throughout the old implementation: its enabled state and prompt changed, not its visibility. Model metadata, prompt wrapping and overall panel requested size caused measured geometry changes; there is no evidence of a registration show/hide loop.

**Already correct and retained:** capture, detection and model work use persistent bounded workers; preview rendering does not call inference. Models load once, use eval/inference mode, and preserve their preprocessing. BGR preview resizing occurs before RGB conversion, and the canvas image item is reused. No unbounded inference queue was found. Camera capture pacing and source-copy behavior match SYS-05. The old default-hidden debug JSON was not serialized during the normal closed-diagnostics runs, so JSON serialization does not explain the normal panel jump.

**Measured costs:** final generated-input run had median Face adapter input-to-vector work 20.919 ms, Emotion 2.409 ms, mean detector work 13.039 ms, preview rendering 4.182 ms and GUI tick 3.906 ms. The detector executes real YuNet on generated pixels, then a controlled region is supplied to exercise the models; these boxes are not claimed as genuine detections. Complete raw distributions, model-forward timings and runtime versions are in the reports.

**Not established:** the physical camera driver's latency, real face-box movement, lighting/glasses effects, background CPU/window-server load and the owner's exact frame timings. Large outliers varied between pre-change runs; they cannot all be attributed to inference or CPU contention. These runs do not establish physical webcam FPS, detection accuracy or resolution of the owner finding.

## Equivalent-condition before/after measurements

Each timed native window used generated 1280×720 immutable BGR pixels, the same controlled box movement, CPU YuNet at its existing detector size, OpenCV one thread, actual supplied models, a 1.5-second GUI warm-up and 10 seconds of measurement. Models were warmed before the GUI measurement. Runs were sequential on this Mac, not simultaneous. Native window presentation intentionally differs because its layout is the correction. OS/background timing remains variable.

| Metric | Before | Before repeat | Final after, one Torch thread |
| --- | ---: | ---: | ---: |
| Capture FPS, generated source | 25.90 | 26.20 | 26.70 |
| Preview FPS, generated source | 24.40 | 24.50 | 26.70 |
| Dropped/superseded preview frames | 15 | 18 | 0 |
| Window geometry transitions | 56 | 58 | **0** |
| Preview interval p95, ms | 67.419 | 67.920 | 67.173 |
| 10 ms heartbeat excess delay p95, ms | 30.533 | 32.389 | 15.027 |
| Preview image render mean, ms | 4.185 | 4.463 | 4.182 |
| Face preprocessing + vector median, ms | 23.593 | 30.120 | 20.919 |
| Emotion preprocessing + prediction median, ms | 2.378 | 2.583 | 2.409 |
| Summary writes | 88 | 91 | 66 |
| Selected Emotion metadata rewrites during timed window | 88 | 91 | **0** |

The preview interval tail remains around 67 ms: this is not a claim of perfect 30 FPS or removal of every timing irregularity. The strongest repeated result is removal of layout transitions. FPS and heartbeat observations are synthetic comparisons, not physical-camera guarantees. Before model-latency outliers varied considerably; trained inference code and inputs were not changed to obtain shorter numbers.

With diagnostics open, before/after preview FPS was 24.80/25.38, geometry transitions 58/0, and heartbeat excess-delay p95 29.128/16.471 ms. JSON serialization itself averaged 0.139/0.178 ms. Debug JSON remains an optional cost; the change does not claim reduced total JSON serialization in this workload. Correct context resets and first results intentionally bypass throttling.

### CPU thread experiment

The actual models were benchmarked at 1, 2 and 4 Torch intra-op threads, eight warm-up pairs and 50 measured pairs each. Median Face times were 17.434, 16.022 and 14.791 ms; Emotion 2.213, 2.211 and 2.142 ms. In native GUI comparisons, one/four threads gave preview 26.39/26.59 FPS and heartbeat excess-delay p95 16.031/16.397 ms. Four threads shortened model work but did not establish a consistent GUI-tail advantage. The existing **one-thread default is retained**; OpenCV also remains one thread. No inference interval, capture resolution, trained weights, threshold, crop preprocessing or label mapping is changed.

### Slow-model stress check

An injected 450 ms wait before Face inference produced an approximately 477 ms adapter duration while generated preview remained 26.67 FPS, with zero layout changes. The single inference slot rejected 17 busy submissions and discarded 16 obsolete completions; there was no backlog and all workers stopped. The injected wait is reported separately from actual model computation.

With inference slower than the existing 300 ms analysis cadence, strict context supersession can discard all current model outputs. The GUI may continue showing Checking/Unavailable in that case. This is a documented safety/performance limit; no old result is relabelled to a newer face, no unsafe stale authorization is introduced, and no interval is increased to hide the condition.

## Changes and presentation

| File | Change |
| --- | --- |
| `src/facial_attendance/ui/webcam_app.py` | Fixed initial window, result-panel dimensions and pixel-sized label containers; cached unchanged text, model metadata and button state; equivalent pending statuses share one summary. Detailed checkpoint information and SYS-02 JSON move to one collapsible diagnostics section, hidden by default. |
| `src/facial_attendance/ui/diagnostics.py` | Adds bounded recent p95 samples (at most 256 per timing) while preserving existing mean/max/count fields and aggregate log compatibility. No per-frame printing. |
| `src/facial_attendance/ui/live_preview.py` | Adds capture intervals and individually timed liveness/recognition/Emotion boundaries through the existing liveness-first dispatcher. No scheduling or authorization relaxation. |
| `scripts/benchmark_ui03_preview.py` | Reproducible frozen-source comparisons, actual-model CPU thread trials, mapped Tk frame/heartbeat/geometry measurements, diagnostics-open and slow-model experiments. No physical camera access. |
| `tests/test_ui03_preview_regression.py` | Ten focused regressions for caching, bounded instrumentation, module timing/gating, slow-work preview independence, stale/negative-result presentation and JSON behavior. |
| `reports/ui03-performance/` | Before-source hashes, SYS-05/before/after/debug/thread/stress measurements, final 15-check native report and verification summary. No camera frames or biometric templates. |
| This document and the notice in `docs/ui-03-integration-delivery.md` | Current findings, measured comparison, reproduction, owner retest and evidence limitations. Earlier integration reports remain historical. |

The primary panel keeps preview, identity, Emotion, simulated Liveness and registration visible. Checkpoint IDs, full weight/file hashes and current SYS-02 JSON are available under **Show / hide diagnostics**. Selected model metadata is labelled as configuration, not current-face evidence, and is cached across fresh samples and Retry. The registration button remains visible; its enabled state still follows the current Unknown result and the existing safety guards.

Routine model context changes still immediately clear current model fields. A short Checking transition remains; two processing phases now share the same text, and text changes cannot resize the panel. The existing 450 ms box hold/smoothing remains display-only; unsmoothed source boxes still feed inference, and no identity tracking is introduced. No historical identity or liveness value is held as a current result.

Hidden diagnostics perform no JSON serialization. Identical envelopes are not rewritten. Repeated changes within the same context are limited to 250 ms refresh spacing; first current results, context clearing and spoof/uncertain/errors bypass the cap so diagnostic JSON cannot retain an old live decision. No GUI label or debug output grants save permission.

## Verification and evidence

- **227 automated tests passed, zero failures/errors/skips:** all prior 217 plus ten new tests. Actual Face and renamed Emotion checkpoints were enabled. The previous tests were retained without edits.
- **All 15 existing controlled native checks passed**, including generated registration/subsequent recognition, stale Retry rejection, face loss, multiple faces, model/detector/camera failures and recovery, simulated spoof/uncertain handling and blocked genuine enrollment.
- Four extra mapped-native layout checks passed: long text does not change geometry; opening diagnostics preserves preview position; metadata is outside primary results; controls fit inside the fixed panel. The diagnostics button's bottom was 471 pixels inside a 540-pixel panel.
- No callback errors; worker shutdown passed. The current native report's source hashes match the tested source, config, scripts and tests.
- Both supplied checkpoints retain their recorded hashes: Face `bc4502836f306c886323c444f11d6ddd7fe02515ba40a467d22f486fbfdcbc4f`; Emotion `323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd`.
- Simulated liveness remains explicit and `can_save=false`; no attendance/gallery writer or liveness bypass is added. The protected original review JSON retains SHA-256 `6a8f55dbd0ae32f88528d164ed17fc61dc3dc416eaf0417c4aac48545fbe43cc`.

Report files are `before-source.json`, `gui-before.json`, `gui-before-repeat.json`, `gui-sys05.json`, `cpu-threads-before.json`, `gui-after-threads1.json`, `gui-after-threads4.json`, `gui-after-final.json`, `gui-slow-model.json`, `gui-before-debug.json`, `gui-after-debug.json`, `native-compatibility.json` and `verification.json`, all under `reports/ui03-performance/`.

Reproduce tests from this worktree:

```sh
FA_FACE_CHECKPOINT='/Users/elvyx/Downloads/best (Que An).pt' FA_EMOTION_CHECKPOINT='/Users/elvyx/Downloads/best(Dung).pt' PYTHONPATH=src artifacts/ui04-runtime/bin/python -m unittest discover -s tests -q
PYTHONPATH=src artifacts/ui04-runtime/bin/python scripts/check_ui03_integration.py --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' --native-gui --output artifacts/ui03-native-retest.json
artifacts/ui04-runtime/bin/python scripts/benchmark_ui03_preview.py --duration 10 --output artifacts/ui03-performance-retest.json
```

Native commands require desktop access; outputs refuse to overwrite existing reports. Add `--debug`, `--slow-ms 450` or `--thread-bench` for the corresponding generated-input experiment. The ignored frozen-source snapshots are local comparison assets, not commit candidates. Before/after reports retain the actual source hashes used in each run.

## Exact macOS launch and owner physical retest

Close the old GUI first, then launch this worktree:

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' --opencv-threads 1 --diagnostics
```

1. Leave diagnostics collapsed. Watch a stationary face for at least 15 seconds, then move/turn slowly. Check that the result panel and video position do not jump during Checking/Unknown/Emotion changes. Record preview responsiveness and box movement separately.
2. Repeat without glasses and with glasses, in normal indoor light and bright backlight, using the same camera/lighting for comparison. No new glasses or recognition accuracy claim is made by this change.
3. Expand/collapse diagnostics once. Confirm model details and JSON remain accessible, the primary video position stays stable, and the selected metadata is clearly separate from current face evidence. Return to collapsed diagnostics for normal use.
4. Leave/re-enter the frame and introduce a second face. Current identity/Emotion/Liveness fields must clear appropriately; previous results must never reappear as current. Check Retry, camera interruption/relaunch and clean close.
5. Confirm Liveness is labelled SIMULATED, identities UNVERIFIED, and saving disabled throughout. Unknown may open the registration form; genuine enrollment must remain blocked. Use the separate controlled generated-input demo for test enrollment only.
6. Record PASS/FAIL for panel stability, preview, box behavior, diagnostics, clearing, Retry/recovery and disabled saving. If possible retain two or three aggregate `SYS05_PERF` lines after warm-up. These logs contain timing/count data, not face pixels or identity values.

Physical resolution remains **PENDING OWNER RETEST**. Full UI-03.2/UI-04.2 acceptance and genuine enrollment/attendance remain blocked by trained liveness and the previously documented authorization work. No independent reviewer approval or Final Owner Evidence is claimed.
