# UI-03/UI-04 stable result presentation — 18 November 2026

**Historical implementation: OWNER PHYSICAL RETEST FAILED.** Emotion alternated between Neutral and Unavailable, including while stationary. The 254-test/20-native-check results below did not establish physical acceptance. See [Emotion continuity investigation](ui-03-emotion-continuity.md) for measured camera failures, the subsequent independent Emotion display, current verification and pending physical retest. The following describes the preceding implementation.

Work remains in `/Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration` on `feature/ui03-ui04-emotion-integration`. The preceding fixed window/panel geometry is preserved. No weights, preprocessing, similarity threshold, label mapping, capture resolution or model-inference interval changed. Nothing is staged, committed, pushed or merged; no Google Sheet or original SYS-05 branch change was made.

## Reproduced cause and measured result

The old visible summary directly reflected the authoritative envelope. Every 300 ms sample correctly cleared that envelope, causing Identity, Emotion and Liveness to switch to Processing/Unavailable and then repopulate after inference. Per-capture confidence percentages changed too. The registration navigation button alternated disabled/normal with those same phases. A fixed outer window did not prevent these text changes or the line offsets inside one combined multiline label.

Before changing product code, its source was frozen in ignored `artifacts/ui03-presentation-before`, with hashes in `reports/ui03-presentation/before-source.json`. The shared mapped-Tk harness counts changes to actual visible fields, or parses the old combined summary when separate fields do not exist. It also measures actual registration-button state changes.

Each comparison used the same generated 1280×720 BGR pattern, controlled ±4-pixel box jitter, real CPU YuNet work, actual Face/Emotion checkpoints, one Torch/OpenCV thread, model warm-up, 1.5 seconds of GUI warm-up and a 10-second timed window. No physical camera or real-person accuracy was tested.

| Timed changes in 10 seconds | Before | Delivery run |
| --- | ---: | ---: |
| Identity text | 66 | **0** |
| Emotion text | 66 | **0** |
| Liveness text | 66 | **0** |
| Registration button state | 66 | **0** |
| Combined summary writes | 66 | **0** |
| Window geometry transitions | 0 | **0** |
| Generated-input preview FPS | 26.70 | 26.78 |

Earlier after-runs recorded 0–4 changes per field. The run with four changes also had camera/GUI/model timing interruptions, including a roughly 326 ms preview interval and a roughly 706 ms Face call. Display history intentionally expires or clears when continuity/freshness is lost. These results support reduced routine-cycle flicker, not elimination of every legitimate result transition. FPS is effectively unchanged in this experiment; no physical performance improvement is claimed.

Final comparison: `reports/ui03-presentation/labels-before.json` and `labels-delivery.json`. Intermediate after-runs retain their own source fingerprints and are not substitutes for final-source verification.

## Separate display state

`ui/presentation.py` owns only immutable display strings, their original source context, a bounded coarse appearance sample and up to three emotion votes. It has no persistence operation, no `can_save` field, and no route back into a model completion or controller envelope. The controller and diagnostic JSON remain authoritative.

The primary panel now has four separately positioned, fixed-size cells: status, Identity, Emotion and Liveness. Each StringVar is written only when that cell's visible text changes. Newly completed predictions and qualified retained predictions are consistently labelled **Previous sample • UNVERIFIED • display only** because they describe a captured frame, not live authorization. The separate historical flag/source context retains whether the data is from an older active context.

The whole display sample expires **800 ms after the original capture**, not 800 ms after each redraw or preview check. Fresh previews and repeated renders cannot renew that deadline. The bound is checked on GUI ticks; a blocked OS event loop can delay the next visible repaint. Source context is never relabelled to the new capture.

## Conservative continuity guard

Previous text may bridge routine processing only after two distinct, compatible detections and continued compatible previews. Guards include:

- Exactly one validated original-frame box and unchanged source dimensions/camera generation.
- Consecutive detection timestamps strictly increase, with a maximum 450 ms gap.
- Box IoU at least 0.90, area ratio 0.90–1.10, and center movement at most 4% of the shorter box side.
- A copied 16×16 BGR sample has spatial texture; uniform/low-information regions do not establish continuity.
- Sampled-pixel mean absolute difference at most 6 and 95th-percentile difference at most 18 on the uint8 scale.
- Preview gaps at most 150 ms, plus the independent 800 ms original-sample expiry.

These are **display guard defaults**, not trained recognition/liveness decision thresholds or validated identity tracking. They do not prove that two frames depict the same person. A visually similar replacement can evade appearance checks, so retained text remains explicitly previous/unverified and never authorizes anything. Physical camera noise, exposure changes and movement may conservatively clear history and must be checked by the owner.

New context creation still immediately clears authoritative Identity, Emotion and Liveness. History clears on face loss, multiple faces, ambiguous appearance/geometry, changed dimensions/generation/action, Retry, spoof, uncertain liveness, camera/model failure or expiry. The latest preview can clear a contrasting same-box replacement before the next sampled detector result. Old model completions cannot repopulate the display after that barrier. Completion acceptance also checks the original context, token, region, source sample and timestamp.

## Emotion and confidence

Primary confidence percentages are omitted to avoid numerical shimmer. Full raw confidence remains in current diagnostic JSON, unchanged. No confidence is rounded inside inference or matching.

The display emotion uses a majority only when at least two of the last three compatible, distinct samples agree. Votes are bounded to 800 ms by their original capture times and are cleared with continuity/identity changes or invalidation. A late model result cannot revive expired votes. The primary Emotion cell is marked **display-smoothed**. A genuine class transition may therefore take an additional compatible sample to appear. Raw `class_id`, label, probability and decision policies are untouched and remain reviewable in JSON.

## Navigation and authorization

**Registration options…** is stable navigation that opens a form; it is not an enrollment authorization. Opening the form clears presentation history and begins a new registration action. The normal webcam form still cannot submit genuine enrollment. Controlled generated-input enrollment still requires a fresh authoritative Unknown result with a matching live simulation/registration context and the existing isolated-test guards. A previous Unknown/live display cannot satisfy those checks.

Simulated Liveness warnings remain visible. `Envelope.can_save` remains fixed false; `GalleryMatcher.register_live` still rejects genuine enrollment. Historical identity, emotion votes and liveness text are never supplied to these decisions. Attendance persistence remains absent.

The collapsible diagnostics section explicitly distinguishes raw current JSON from the previous-sample primary display. JSON may correctly show checking/null results while previous text remains visible. It never substitutes a smoothed label, old identity or old live result for current model data. Essential invalidation updates still bypass the earlier JSON refresh throttle.

## Changed files

| File | Purpose |
| --- | --- |
| `src/facial_attendance/ui/presentation.py` | Pure bounded display state, source/continuity checks, expiry and display-only emotion votes |
| `src/facial_attendance/ui/webcam_app.py` | Observes preview/detection/completion events; renders four cached cells; preserves raw authorization/JSON; stable form navigation |
| `scripts/benchmark_ui03_preview.py` | Measures actual per-field/button changes and verifies independent cell positions, retaining the prior four layout checks |
| `tests/test_stable_presentation.py` | 27 new expiry, replacement, flicker, stale-source, smoothing, raw-JSON and authorization regressions |
| `reports/ui03-presentation/` | Frozen-source audit, before/after measurements, actual test results and final native verification |
| This document and the notice in the preceding performance report | Current behavior, measured evidence, limitations and physical retest |

Model inference, controller authorization, gallery storage and previous tests are preserved. Coarse pixel samples exist only in memory; benchmark reports contain no webcam frames, personal gallery templates or face crops.

## Verification

- **254 automated tests passed, zero failures/errors/skips:** all previous 227 plus 27 new tests, with both supplied actual checkpoints enabled. Results and tested source hashes: `reports/ui03-presentation/automated-tests-final.json`.
- All **19 previous native checks** passed again: 15 integration checks in `native-delivery.json`, plus the original four mapped layout checks in `labels-delivery.json`.
- One additional mapped-native check passed: independent status/Identity/Emotion/Liveness cell positions remain fixed after long text and diagnostics expansion. **20 native checks total**.
- No callback errors, all workers stopped, zero geometry changes. Final native/source fingerprints match the delivered implementation.
- Regression tests prove that raw fields clear while historical text is visible, source/token mismatches are rejected, same-box contrasting replacements clear history, votes/captures expire, and historical Unknown/live text cannot submit enrollment.
- Face and Emotion files retain SHA-256 `bc4502836f306c886323c444f11d6ddd7fe02515ba40a467d22f486fbfdcbc4f` and `323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd`. Original SYS-05 and protected files remain unchanged.

Reproduce from the integration worktree with new output filenames:

```sh
FA_FACE_CHECKPOINT='/Users/elvyx/Downloads/best (Que An).pt' FA_EMOTION_CHECKPOINT='/Users/elvyx/Downloads/best(Dung).pt' PYTHONPATH=src artifacts/ui04-runtime/bin/python -m unittest discover -s tests -q
artifacts/ui04-runtime/bin/python scripts/benchmark_ui03_preview.py --duration 10 --output artifacts/ui03-label-retest.json
PYTHONPATH=src artifacts/ui04-runtime/bin/python scripts/check_ui03_integration.py --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' --native-gui --output artifacts/ui03-native-retest.json
```

## Exact macOS launch and physical retest

Close the old GUI before launching the changed source:

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' --opencv-threads 1 --diagnostics
```

1. Keep diagnostics collapsed. Hold a face still for 15 seconds after initial warm-up. Check that Identity, Emotion and Liveness no longer alternate with Checking/Unavailable every analysis cycle, and that each row stays in the same position. The previous-sample/unverified warning must remain clear.
2. Move slowly, blink and change expression. Check the distinction between routine processing, genuine class changes and conservative history clearing. Repeat with/without glasses and under indoor/backlit conditions; do not interpret the display smoothing as improved model accuracy.
3. Leave/re-enter; replace the person, including near the same box location; introduce a second face. History must clear when continuity is lost or ambiguous. Nothing should authorize enrollment/attendance from a previous label.
4. Press Retry during inference and test camera interruption/relaunch. Check immediate clearing and fresh contexts. If inference stalls, previous text must expire rather than remain indefinitely.
5. Open diagnostics: compare the raw current JSON with the explicitly previous/smoothed primary text. During fresh processing the JSON should have current/null fields, not copied historical results. Verify SIMULATED Liveness and `can_save=false` throughout.
6. Open Registration options while processing or after a previous Unknown. Opening the form should stay visually stable; actual submission still requires fresh current evidence and remains disabled for genuine enrollment.

Record PASS/FAIL for routine label stability, fixed rows, person replacement/loss, expiry, Retry/recovery, diagnostics separation and disabled saving. The physical finding remains unresolved until the owner reports the outcome. No independent reviewer approval or Final Owner Evidence is generated.
