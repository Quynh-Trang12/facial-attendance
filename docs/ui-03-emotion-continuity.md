# Emotion continuity investigation — 18 November 2026

> Presentation extension: [Standard/Developer modes](ui-03-dual-mode.md) moves technical ages, policies and raw data into a reusable diagnostics window. Standard Mode uses concise labels and safety warnings; the continuity/freshness behavior and historical evidence below are retained. Current physical verification remains pending.

**Updated implementation: physical acceptance pending.** The owner failed the previous presentation change: Emotion became Unavailable during movement and alternated Neutral/Unavailable while stationary. The preceding 254 automated tests and 20 native checks did not represent camera noise, exposure changes, pose/scale variation or intermittent missed detections adequately. They remain regression tests, not physical acceptance evidence.

Work stays in `artifacts/ui03-ui04-emotion-integration`, branch `feature/ui03-ui04-emotion-integration`. The fixed window geometry, trained models, preprocessing, recognition threshold, label mapping, 300 ms analysis interval and camera settings are preserved. No tracker, original SYS-05 branch, protected review JSON or supplied checkpoint was changed. Nothing was staged, committed, pushed or merged. No independent reviewer approval is claimed.

## Current entry point

This record retains the behavior, commands and environment of its original UI revision. Use the [launch guide](attendance-demo.md) and [setup](setup.md) for the current bundled-model application.

## Measured physical failure

The owner authorized two 15-second camera probes, including a repeat with instructions to remain still for five seconds, move gently for five, then remain still. The reports do not establish compliance with every movement instruction; those phases are timing bins. No images, crops or recordings were saved.

The clean repeat, `reports/ui03-emotion/physical-before-repeat.json`, used the frozen **previous** implementation in ignored `artifacts/ui03-emotion-before`:

| Measurement | Observed |
| --- | ---: |
| Raw detector batches / single-face batches | 48 / 48 |
| Visible Emotion text changes / transitions to Unavailable | 80 / 40 |
| Approximate unavailable time | 6.05 of 15.00 seconds |
| Strict preview appearance rejections | 21 of 331 comparisons |
| Strict display completion acceptance / rejection | 42 / 7 |
| Capture / preview FPS | 28.0 / 27.2 |
| Recognition / Emotion / combined adapter mean | 20.654 / 1.551 / 22.318 ms |
| Detector mean / recent p95 | 20.658 / 22.384 ms |
| GUI frame rendering mean / recent p95 | 4.306 / 4.877 ms |
| Event-loop delay mean / recent p95 | 1.208 / 1.179 ms |
| Preview age mean / recent p95 | 22.321 / 37.109 ms |
| Dropped preview frames | 12 |
| Geometry transitions / callback errors / can_save | 0 / 0 / false |

Thus flickering occurred even with continuously successful sampled detection and fast Emotion inference. These are measurements of the failed implementation, not a measured physical improvement.

An earlier valid probe (`physical-before-retry.json`) found no face in 19 of 48 raw detector batches; the owner confirmed their face remained visible. It also recorded 29 strict pixel rejections in 43 comparisons and 22 display completion rejections. That probe overlapped a generated workload, so its throughput/model timings are excluded from performance comparisons. Detector misses are a second observed failure mode, not a necessary explanation for the clean repeat. Lighting/pose causes of individual misses cannot be established without images, which were intentionally not retained. The first permission-failed probe (`physical-before.json`, zero captured frames) is invalid as a webcam measurement.

## Confirmed mechanisms and test gap

`StablePresentation` shared one history and invalidation token across Identity, Liveness and Emotion. A 16×16 point-sampled BGR guard required mean pixel difference ≤6 and pixel p95 ≤18 on previews. Normal noise, movement and exposure could reset its token before a correctly context-bound model completion was displayed. Detection continuity also required IoU ≥0.90, area ratio 0.90–1.10 and center motion ≤4% of the smaller face side.

After a reset, a fresh result could appear, then the next routine `begin_sample` hid it until two compatible detections had accumulated again. Every missed detection and uncertain/error state also cleared the shared history. These code paths explain repeated clearing independently of inference speed. The baseline instrumentation counts pixel/completion rejection but cannot allocate each of the 40 physical Unavailable transitions to one unique cause.

The earlier controlled presentation comparison used a noiseless, unchanged generated image with only ±4 pixel box jitter. It exercised processing and safety transitions but not realistic sensor noise or exposure/scale variation. A new stationary-noise workload reproduces rejected completions even with perfect, identical boxes. The frozen baseline and current source are exercised by the same diagnostic harness.

## Separate informational display

`ui/emotion_presentation.py` owns an independent Emotion cache, original source context/capture time, a continuity epoch and one pending completion association. It never writes to the authoritative controller or authorizes any action.

- A new routine analysis context immediately clears authoritative identity/liveness/Emotion as before. It does **not** clear informational Emotion text.
- Freshness remains **800 ms from the original capture**. Rendering, previews and routine contexts do not renew it. Expiring old history does not cancel a newer, still-fresh pending prediction.
- Display continuity uses an area-averaged 24×24 grayscale descriptor of the central face region, normalized for brightness, at detection frequency. Raw preview pixel differences no longer erase Emotion. This transformation is display-only; neither trained model's preprocessing changes.
- Display guards require one validated box, unchanged dimensions/generation, detection evidence no older than 750 ms, IoU ≥0.40, area ratio 0.60–1.70, center movement ≤35% of the smaller box side, and descriptor correlation ≥0.55 with sufficient texture. These are informational display heuristics, not recognition thresholds or identity tracking.
- A missed detection may retain clearly marked informational Emotion for at most **350 ms from the first missed sample**, still subject to the original 800 ms capture deadline. Raw authority clears immediately. Sustained loss, multiple faces, ambiguous replacement/box jumps, camera failure, Retry, action changes and spoof cancel display history and pending old completions.
- Correct original-context Emotion can be displayed after the active analysis context advances only if its registered source, epoch, frame, region and age remain valid. It is never relabelled as a prediction for the new context. The raw controller continues to reject superseded identity/liveness/completion data.
- A recognition adapter failure still blocks the authoritative operation. In the trained combined bundle only, Emotion can run once independently after that failure. Its own error clears its display; no historical identity or liveness is supplied to inference or authorization. Spoof still prevents Emotion dispatch.
- Latest valid Emotion labels update directly; no majority smoothing delays a new class. Confidence remains raw in diagnostics and is omitted from the primary cell, preventing percentage shimmer. Unchanged strings are not written again. The cell is labelled **display only ≤800 ms**.

Identity/liveness retain their preceding strict guards. A visually similar face replacement could evade a coarse display guard, so this is never proof of identity or liveness. Every displayed identity remains unverified, Liveness remains SIMULATED, genuine enrollment/attendance remain disabled and `can_save=false` remains fixed. Registration navigation and submission restrictions are unchanged.

## Aggregate reasons and accurate diagnostics

With `--diagnostics`, aggregate `emotion_display_*` counters appear in the existing five-second performance summaries. Counts distinguish detector misses, ambiguous appearance/geometry, stale source/context, expiry, camera failure, own model failure, accepted completions and accepted original superseded contexts. `unavailable_<reason>` counts actual visible-to-unavailable transitions. Histograms retain at most 256 geometry/appearance measurements; no pixels or identity are logged.

The collapsible diagnostics show the Emotion display reason and original source context separately from the SYS-02 JSON. The JSON remains the raw current authoritative envelope, including null fields and current errors. Display audit never overwrites JSON fields or confers saving eligibility.

The finite diagnostic script separately measures pixel/box guard rejection, face counts, actual label writes, approximate available/unavailable time (20 ms polling), capture/preview intervals, dropped frames, per-model latency, rendering, event-loop delay and frame age. `--camera` explicitly opens hardware; generated modes do not. Reports refuse overwriting. Aggregate display audit includes immutable Emotion prediction metadata, never a person identity or image.

## Equivalent controlled comparisons

Each row used the same generated 1280×720 workload, actual supplied models, one OpenCV/Torch thread, bounded workers, controlled detections and a ten-second timed mapped-Tk window after warm-up. Paired final runs were serial, with no simultaneous camera benchmark. Generated detector regions are not evidence of real face detection or emotion accuracy.

| Generated workload | Unavailable transitions before → after | Approximate unavailable seconds before → after | Preview FPS before → after |
| --- | ---: | ---: | ---: |
| Stationary Gaussian sensor noise, σ7 | 4 → 0 | 9.85 → 0 | 27.39 → 27.39 |
| Moving/scaling box and moving pixels | 21 → 0 | 3.04 → 0 | 26.70 → 26.79 |
| Changing illumination, ±18 intensity | 19 → 0 | 2.23 → 0 | 26.67 → 26.78 |
| Injected variable recognition delay, 150/380 ms | 20 → 9 | 4.00 → 0.50 | 26.89 → 26.76 |
| Deliberate one-second detector loss | 2 → 1 | 1.61 → 0.96 | 26.70 → 26.66 |

The after-run writes zero Emotion strings during the timed noise/motion/illumination windows because the model label stays unchanged. All nine remaining slow-workload transitions are explicitly `freshness_expired`; ten superseded original-source Emotion completions were displayed while their raw controller completions remained rejected. That workload still exposes short freshness gaps when service time and dropped busy slots exceed the capture-age bound. The bound was not increased to hide those gaps. Deliberate loss produces one appropriate `sustained_face_loss` transition followed by a fresh result on return.

There is no meaningful FPS improvement claimed. Reports are `noise-{before,after}-final.json`, `{motion,illumination,latency,loss}-{before,after}.json` under `reports/ui03-emotion/`. Earlier intermediate noise runs are retained; the first overlapped run is not a timing comparison. Final native/layout and controlled after-run product source hashes match the delivered source.

## Verification and files

**284 automated tests passed, zero failures/errors/skips:** all 254 previous tests plus 30 new tests for motion, scale/exposure, noise, bounded miss grace, sustained loss, same-box replacement, multiple faces, source/region/context mismatch, expiry, variable latency, superseded completion display without authorization, Retry/camera failure, accurate raw JSON and unchanged rendering. Both supplied actual checkpoint tests were enabled. Full output and source/test fingerprints: `reports/ui03-emotion/automated-tests-final.json`.

All **20 prior native checks passed again**: 13 controlled integration/recovery checks, two genuine-enrollment-blocked form checks (`native-final.json`), and five mapped layout checks (`layout-final.json`). Long text, diagnostics expansion and separate cells preserve fixed positions. Zero geometry transitions/callback errors, all workers stopped, saving remained false. These use generated images, not owner physical acceptance.

Changes in this follow-up: `ui/emotion_presentation.py`, `ui/webcam_app.py`, `ui/live_preview.py`, `tests/test_emotion_presentation.py`, `scripts/diagnose_emotion_preview.py`, `reports/ui03-emotion/`, this document and historical-document notices. Earlier integration/performance work remains uncommitted; it is not newly attributed to this fix.

Face checkpoint SHA-256 remains `bc4502836f306c886323c444f11d6ddd7fe02515ba40a467d22f486fbfdcbc4f`; Emotion remains `323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd`. Original SYS-05 HEAD remains `798f2e7b9484a682e50c70b0b4689cd2fde316dc`. Protected review JSON remains SHA-256 `6a8f55dbd0ae32f88528d164ed17fc61dc3dc416eaf0417c4aac48545fbe43cc`. Nothing is staged.

## macOS launch and required physical retest

Close any old GUI, then run:

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' --opencv-threads 1 --diagnostics
```

1. Keep diagnostics collapsed. Stay still five seconds, move/turn gently five seconds, then stay still five seconds. Repeat with/without glasses and normal indoor/backlit conditions. Report whether Emotion still alternates with Unavailable; real changes between valid Emotion classes are distinct from blanking.
2. Leave the frame for at least one second and return. Confirm sustained loss clears informational Emotion, raw identity/liveness clear immediately, and only a fresh result returns. Check multiple faces and a different person entering too.
3. Retry and camera recovery must clear old display; registration form must still block genuine submission. Confirm no window/panel movement and visible SIMULATED liveness/disabled saving.
4. If Emotion blanks while continuously visible, open diagnostics and report its reason plus the aggregate `emotion_display_*` counters. Distinguish `detector_miss`, `ambiguous_replacement`, `ambiguous_box_jump`, `model_error_*` and `freshness_expired`.

Optional authorized 15-second aggregate-only camera audit (choose a new output name for each run; normal macOS camera consent may appear):

```sh
artifacts/ui04-runtime/bin/python scripts/diagnose_emotion_preview.py --camera 0 --request-camera-access --duration 15 --output artifacts/emotion-physical-retest.json
```

No automatic new physical run or resolution claim is substituted for owner retesting. Sustained detector misses, ambiguous appearances, overload beyond freshness, large motion and camera errors can still intentionally clear Emotion. Windows execution, real emotion accuracy and recognition/liveness security are not established by these Mac/generated-input checks.
