# UI-03 / UI-04 dual-mode GUI and diagnostics repair

> Historical UI-03 verification for the earlier simulated-liveness build. See the [current demo guide](attendance-demo.md) for the trained webcam and review interface.

Prepared 18 November 2026 in the existing `feature/ui03-ui04-emotion-integration` worktree. **Owner physical verification is pending.** This implements presentation and diagnostics only. No independent reviewer approval, trained liveness, genuine enrollment or recorded attendance is claimed.

> Diagnostics usability extension: [pause, copy, privacy-safe export and completed snapshots](ui-03-diagnostic-controls.md) documents the current controls. The historical implementation/results below are retained; current diagnostics can freeze independently of the running pipeline and explicitly labels frozen/completed data.

> Visibility follow-up: the diagnostics button was unconditionally packed in Standard Mode. It now starts hidden and is packed only in Developer Mode. Returning to Standard Mode hides the button and withdraws the inspector; returning to Developer Mode reopens the same inspector. The mode switch remains visible. All preceding 299 automated tests and 47 native checks pass again, plus three automated and nine actual-widget visibility checks: **302 automated tests and 56 native checks passed**, zero failures/skips/callback errors. New local evidence is ignored under `artifacts/ui03-visibility-evidence/`. The benchmark history below belongs to the preceding dual-mode delivery; no new physical performance claim is made.

## Diagnosed cause

The Diagnostics button callback toggled successfully; its binding was not broken. The panel was packed into the bottom of the fixed main window, after the preview/results and optional registration rows. It did not have its own usable viewport for all technical content.

A mapped, generated-input probe with actual loaded model results measured:

| Before repair | Measured value |
| --- | ---: |
| Main requested height after diagnostics populated | 826 px |
| Fixed default height / supported minimum height | 800 / 700 px |
| Diagnostics panel top relative to main content | 651 px |
| Populated diagnostics panel height | 153 px |
| Portion fitting inside the 700 px main window | 49 px |

The callback opened and closed the panel, but content could be clipped below the main viewport. The previous native geometry checks verified that opening diagnostics did not move the preview; they did not verify usable diagnostic content visibility or scrolling. This reproduces a concrete layout defect rather than proving the precise size/state of the owner's earlier window.

Probe records are in ignored `artifacts/ui03-modes-evidence/diagnostics-before.json` and `diagnostics-live-before.json`. No camera images or recordings were used. Source was frozen under ignored `artifacts/ui03-modes-before` before product changes. The actual pre-change full suite passed **284 tests, zero failures/errors/skips**; all **20 prior native checks** passed before edits too.

## Standard Mode

Standard Mode is the default, even with `--diagnostics`. Its fixed result cells show Identity, Emotion and Liveness, with human-readable checking, no-face, multiple-face, uncertain, spoof and camera/model failure states. Emotion labels remain Negative, Neutral and Positive. Confidence, hashes, context IDs, timing bounds and scheduling explanations are absent from standard result labels.

Concise visible warnings retain the important boundaries: **Liveness SIMULATED**, **Identity UNVERIFIED**, recent preview results only, genuine registration and attendance saving disabled. An unverified preview is never presented as a verified attendance event. Registration options and the existing form remain available, while genuine submission remains blocked. The generated-input controlled test form still has its distinct memory-only mechanics.

The main window remains 1120×800 with a 1120×700 minimum; the results panel stays fixed at 420×540. Each result label retains its own fixed area and cached text updates. Retry and guidance share a compact row, and the existing registration form uses the space beneath the preview so its controls fit at the supported minimum height. Mode changes do not clear typed ID/name values or hide the form unnecessarily.

## Developer Mode and diagnostics

Select **Developer Mode** beside Live results. It shows the existing technical primary descriptions and opens a separate, resizable, non-modal diagnostics window. **Open / close diagnostics** is available exclusively in Developer Mode. The first explicit open creates one window; later opens reuse it. Closing the inspector leaves Developer Mode selected; select **Standard Mode** to hide the diagnostics button, inspector and technical presentation. Returning to Developer Mode reopens the existing inspector. The mode-switching control remains available in both modes.

The inspector tabs scroll vertically and horizontally:

- **Runtime and presentation:** provided camera index/input/detector, camera delivery state, face count, detector/inference activity, processing device, selected checkpoint provenance, embedding dimensions, recognition threshold and score/decision when available, gallery count/status, raw and displayed Emotion, display age/reason/counters, existing retention policies, simulated Liveness fields, authorization restrictions and measured timing data.
- **Raw current SYS-02:** the authoritative envelope, original context, current raw predictions/errors and fixed false `can_save`. It never substitutes historical or smoothed values for current fields.
- **Completed inference (historical):** an explicitly captured, timestamped diagnostic copy of the latest accepted completed batch, separate from the current checking state. It never authorizes anything.

Checkpoint metadata comes from the already-loaded adapters. No model is run or loaded by inspecting diagnostics. The raw on-screen envelope may contain the current matched person's name/ID. Copy preserves that displayed JSON exactly; privacy-safe file export instead constructs a new allowlisted scalar summary and omits identity, contexts, gallery records, arbitrary messages and paths. Images and embeddings are absent from both diagnostic snapshots and exports. See [diagnostic controls](ui-03-diagnostic-controls.md).

Liveness is explicitly `stub / SIMULATED`. Trained-liveness threshold and model metadata remain null because no trained adapter supplies them. Recognition acceptance is a model match decision, not verified identity or saving authorization. Genuine registration and attendance availability remain false.

Live runtime diagnostics refresh at most four times per second while visible. Pause freezes both current diagnostic tabs and their timestamps; the main preview and authoritative state continue updating. Scroll position and unchanged or uniquely relocated selection are retained where practical. Hidden/paused diagnostics do not build the runtime report. Current raw JSON retains its existing invalidation/first-result updates and other-change throttle when live. `Diagnostics.peek()` does not consume/reset periodic console measurements. `--diagnostics` enables actual timing collection; without it, timing information remains unavailable rather than invented.

## Shared architecture and unchanged policies

There is still one application, camera feed, detector worker, inference worker, adapter/model bundle, controller, gallery and registration workflow. The optional controlled enrollment worker is unchanged. A mode switch changes labels and window visibility only. It never calls camera restart, checkpoint loading, gallery reset, inference submission or authorization methods.

The existing flow is unchanged: continuous latest-frame camera preview → sampled CPU YuNet detection with original-frame box mapping → immutable context-bound request → simulated liveness, trained Face embedding/matching and trained Emotion on the bounded worker → authoritative controller acceptance plus separate display views. Model instances are loaded once before Tk. Existing recognition-error/independent Emotion handling is preserved.

The display retains its existing policies, with their numbers moved out of Standard Mode:

- Original-capture presentation freshness remains **800 ms**; updates/peeks/mode switching cannot renew it.
- Previous bounding-box retention remains **450 ms**, with existing display-only outline smoothing. Adapter requests retain the original unsmoothed box.
- Trained informational Emotion retains the latest valid label, independent of the strict identity/liveness display guard. No new class smoothing or changed prediction policy was introduced. Existing legacy bounded voting remains where it was previously used.
- Routine pixel changes do not reintroduce Emotion blanking. Existing continuity checks, bounded detector-miss grace, ambiguous replacement, multiple faces, sustained loss, spoof, Retry, camera failure and expiry behavior remain unchanged.
- A fresh context immediately invalidates actionable raw results. Historical identity/liveness never authorizes registration or attendance. Original-context stale completion rejection and `can_save=false` remain enforced.
- Capture resolution, analysis cadence, bounded scheduling and OpenCV/Torch thread configuration are unchanged. Architecture, weights, preprocessing, embeddings, thresholds, label mappings and selection policies are unchanged.

The separate display state explains why a previous Emotion label can remain visible while raw JSON correctly reports a newer checking/null state. The label describes a bounded recent preview; it is not current authorization or identity tracking.

## Verification and performance

The final full suite passes **299 tests**, comprising all 284 previous tests and 15 new tests, with the actual supplied checkpoint tests enabled and zero failures/errors/skips. New tests cover concise supported statuses, both modes, shared object/state preservation, form input, one reused inspector, read-only metrics, raw/history separation, bounded refresh and blocked authorization.

All **20 previous native checks pass again**: 13 controlled integration/recovery checks, two genuine-enrollment-blocked checks and five layout checks. **27 additional mapped native checks pass** for default Standard Mode with metrics enabled, visible screen-bounded diagnostics, runtime metadata, scrollable read-only content, actual window-manager close/reopen, one Toplevel, uninterrupted capture, preserved models/gallery/input, minimum-height registration visibility and bounded refresh. **47 native checks total.** Zero callback errors, all workers stopped, saving stayed false. These checks use generated input and are not physical acceptance.

Equivalent ten-second generated-input comparisons used actual supplied models, one CPU thread, the existing analysis cadence and the same benchmark harness/source snapshot:

| Workload | Preview FPS before → after | Emotion Unavailable transitions before → after |
| --- | ---: | ---: |
| Fixed pattern, real CPU YuNet cost, diagnostics closed | 26.67 → 26.49 | 0 → 0 |
| Fixed pattern, diagnostics open | 24.09 → 26.32 | 0 → 0 |
| Stationary Gaussian sensor noise | 26.67 → 26.70 | 0 → 0 |
| Moving pixels and face box/scale | 26.59 → 26.46 | 0 → 0 |
| Changing illumination | 26.79 → 26.77 | 0 → 0 |
| Injected variable recognition latency | 26.40 → 26.46 | 10 → 11 |

All slow-workload Unavailable transitions are the unchanged **freshness_expired** policy. The 800 ms bound was not raised to conceal them. Mean result-render time with diagnostics open increased from 0.115 ms to 0.253 ms for the richer inspector; its after-run p95 was 1.108 ms. Mean total GUI tick was 4.295 → 4.289 ms in those runs. Closed diagnostics result-render mean was 0.0156 → 0.0184 ms. There were zero geometry transitions in these runs. The closed after-run dropped three preview frames and included a 54 ms maximum event-loop delay; the before-run dropped none. Short single runs do not establish universal regression-free performance or physical FPS improvement.

New generated evidence/logs remain ignored under `artifacts/ui03-modes-evidence/`, including full test output/fingerprints, baseline/after native results, mode checks, benchmarks, asset verification and complete Git status. They are not staged or added to Git. Existing historical reports remain intact.

## Changed-file inventory for this follow-up

| File | Change |
| --- | --- |
| `src/facial_attendance/ui/webcam_app.py` | Shared Standard/Developer controls, concise standard labels, lazy inspector lifecycle, compact registration layout, runtime descriptors |
| `src/facial_attendance/ui/developer_view.py` | Read-only runtime/presentation data, standard formatting, reusable screen-bounded scrollable Toplevel |
| `src/facial_attendance/ui/diagnostics.py` | Non-consuming measured-counter peek; original default snapshot/reset behavior preserved |
| `tests/test_view_modes.py` | 15 new automated mode, diagnostics, presentation and authorization regressions |
| `scripts/check_ui03_modes.py` | 27 mapped native checks using generated input and actual models |
| `README.md` | Current branch scope, launch, modes, diagnostics, restrictions and links |
| `docs/ui-03-dual-mode.md` | This delivery, root cause, shared architecture, measurements and retest |
| `docs/ui-03-integration-delivery.md` | Current presentation follow-up pointer |
| `docs/ui-03-emotion-continuity.md` | Standard/Developer presentation extension pointer; historical evidence retained |

This inventory describes this request only; earlier uncommitted integration work is preserved. Local snapshots and generated logs are excluded from a future commit. No checkpoint, environment, dataset, private gallery or protected review JSON belongs in a commit.

## Integrity and restrictions

Before/after verification retains these SHA-256 values:

- Face: `bc4502836f306c886323c444f11d6ddd7fe02515ba40a467d22f486fbfdcbc4f`.
- Emotion: `323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd`.
- YuNet: `8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4`.
- Protected original review JSON: `6a8f55dbd0ae32f88528d164ed17fc61dc3dc416eaf0417c4aac48545fbe43cc`.

Original SYS-05 remains on `feature/sys-05-webcam-gui-stubs` at `798f2e7b9484a682e50c70b0b4689cd2fde316dc`. Existing ML, inference, controller, presentation-policy and scheduling source files match the pre-change manifest. Nothing is staged, committed, pushed, merged or submitted as a PR. Google Sheets, Notion evidence and reviewer fields were not modified.

## Launch, troubleshooting and owner retest

Use the unchanged launch configuration; Standard Mode remains the default:

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app \
  --camera 0 --detector yunet \
  --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx \
  --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' \
  --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' \
  --opencv-threads 1 --diagnostics
```

1. Confirm Standard Mode, the webcam/results and concise SIMULATED/UNVERIFIED/saving-disabled warnings.
2. Stay still five seconds, move naturally five seconds, then change expression. Check for inappropriate Emotion blanking and any geometry movement.
3. Select Developer Mode. Confirm technical runtime/model information appears in a separate visible window.
4. Open Raw current SYS-02; scroll. Close diagnostics with the window close control, then reopen with Open / close diagnostics. Confirm the camera continues.
5. Switch modes repeatedly. Open registration options, type test text without submitting, switch again and check the input remains.
6. Leave/re-enter the frame, try multiple faces when practical, and use Retry. Check that invalid state clears and only fresh results return.
7. Confirm genuine enrollment and attendance saving remain blocked. Report PASS/FAIL/NOT TESTED for these checks; none is pre-marked passed.

For unavailable Emotion, inspect its display reason/age and measured counters in Developer Mode. Sustained missed detection, ambiguous replacement or inference delay beyond freshness can still clear it deliberately. If timing is null, relaunch with `--diagnostics` to collect measurements. Missing/corrupt checkpoints require a correct verified local file; there is no bypass or automatic download. A transient camera failure uses Retry and normal OS camera consent.

Physical Emotion stability after the preceding correction remains unconfirmed. This request adds no physical accuracy/FPS evidence. Windows execution, trained-liveness integration, genuine persistence, physical owner acceptance and independent review remain pending. See [Emotion continuity](ui-03-emotion-continuity.md), [Face contract](ui-03-face-inference-registration-contract.md) and [trained adapter contract](ui-04-trained-adapter-contract.md) for the detailed policies and model requirements.
