# UI-03 identity label above the webcam outline

Prepared 21 November 2026 on `feature/ui03-ui04-emotion-integration`, in the existing isolated worktree. This is a small display addition; owner physical acceptance and independent review are pending.

## Existing path and implementation

The existing renderer converts/resizes BGR frames for one Tk Canvas image and draws the existing YuNet-derived source boxes as canvas rectangles. Identity, Emotion and Liveness remain in the side panel. `LiveController.visual_regions()` already supplies the displayed box, source dimensions and whether it is an unconfirmed retained outline; `StablePresentation.view()` supplies the existing strict visual-validity result.

`ui/identity_overlay.py` adds two reusable Canvas items: a dark, opaque background with a yellow border and light 12-point bold text. It does not burn text into camera images, copy image arrays for the label, run recognition, change a gallery, or add a worker/camera session. Font measurements and unchanged drawing content are cached. Hidden labels retain the same canvas items; Retry/camera clearing resets their handles so deleted items are not reused.

The overlay is updated from the same GUI render pass that computes the existing presentation view, before the side-panel unchanged-content early return. Thus an unchanged box can receive a newly completed label, while an invalidated view can clear the name immediately. It does not depend on a previous render's cached side-panel text or frozen diagnostic snapshot.

## Known, Unknown and Checking

| Condition | Overlay |
| --- | --- |
| Exactly one current confirmed face, trained accepted recognition, same full raw/view context, current visually valid view agreeing with that recognition | Registered name followed by `· UNVERIFIED` |
| Same conditions with the existing rejected gallery match | `Unknown` |
| Exactly one visible/temporarily retained box while detection or recognition is genuinely processing | `Checking...` |
| Invalid visual guard, expired/unusable result, spoof, uncertain, camera/model error, no face or multiple faces | No identity label |
| Trained recognition not active | No invented identity from the stub adapter |

Names come directly from the existing trained Recognition result. The overlay requires the existing liveness gate and compares the full current `Context` with the visually valid view's source context. Historical view data, retained-box continuity alone and earlier diagnostic completions cannot supply a name. If the validated view and current recognition disagree, the label is omitted.

**The overlay deliberately does not hold a previous name across new capture contexts.** A new sample can briefly show Checking... while a fresh result is processed. The existing side-panel historical presentation policy is unchanged. Ordinary motion can also neutralize/hide a name when the existing strict visual guard rejects continuity; this change does not relax that guard to hide transitions. The canvas objects remain stable, and these legitimate text transitions do not rebuild widgets or resize the interface.

## Alignment and boundaries

The outline and label share the same source-to-preview, half-open coordinate transform. The label anchors above the displayed smoothed outline, follows its horizontal/vertical movement, and clamps its full background inside the actual displayed image dimensions. At a top edge it prefers space below the box; when neither exterior position fits, it uses a clamped inside-edge fallback. Excessively long names are ellipsized while preserving UNVERIFIED; an impossibly small preview omits the label.

The image renderer, outline and label share the same optional internal mirror flag. Mirroring is a display transform after preview conversion, never a model-input transformation. Tests exercise a mirrored image and a smaller aspect-preserving preview. No mirror button, extra UI mode, camera setting or preview-resize feature was added: normal launch preserves the existing unmirrored preview, dimensions/aspect ratio and camera resolution.

## Loss, replacement, multiple faces and authorization

- New contexts clear current raw results under the existing controller rules; the overlay never transfers an earlier name to them.
- The existing per-preview visual guard can neutralize a name before the next detector sample when contrasting replacement pixels appear. A rejected or stale model completion cannot restore a previous name.
- A detected no-face state removes the label even if the 450 ms dashed historical outline remains. Re-entry needs a fresh context and model completion.
- Multiple faces remove the single identity label entirely. A valid result is not assigned to the first/arbitrary box.
- Retry and camera failure clear/reset the overlay along with existing preview state. Registration/action reset removes a no-longer-current label on rendering.

Box continuity is not identity tracking; a visually similar replacement may evade the pre-existing coarse visual guard, and trained matching itself can be wrong. The label is therefore explicitly UNVERIFIED and never an authorization input. Liveness remains SIMULATED, genuine registration/attendance remain blocked, and `can_save=false` is unchanged. The 800 ms presentation freshness and 450 ms previous-box policy are unchanged. No model, preprocessing, threshold, embedding dimension, label mapping, smoothing, inference cadence or CPU-thread policy changed.

Standard and Developer Mode, the side panel, registration, Retry and every diagnostics control remain intact. Pausing diagnostics does not freeze this overlay or the current controller: the overlay reads live controller/presentation state, not a diagnostic history buffer.

## Tests and measurements

Before editing, the actual baseline passed **324 automated tests** with zero failures/errors/skips and all **71 native GUI checks**. Source and asset manifests were captured under ignored `artifacts/ui03-overlay-evidence/`, with a frozen source snapshot under ignored `artifacts/ui03-overlay-before`.

After implementation, all existing checks plus focused regressions pass:

- **350 automated tests:** all previous 324 plus 26 new cases for current Known/Unknown/Checking, source-context mismatch, historical/unconfirmed rejection, invalid states, replacement, loss/return, multiple faces, Retry, stale completion, both modes, motion, boundaries, resize/mirror transforms, long names, cached drawing and unchanged authorization.
- **93 native checks:** all previous 71 plus 22 mapped overlay checks using generated inputs and the actual supplied Face/Emotion models. The controlled gallery contains only generated fixture embeddings and never touches a genuine person's gallery. Checks verify image/box/text transforms, exact current model names, conservative invalidation, fixed geometry, two canvas items, unchanged source pixels/model/worker instances and blocked genuine saving.
- Zero native callback errors; all workers stopped. All five existing layout checks pass. The benchmark recorded no side-panel field writes after warm-up, no geometry transitions and no dropped preview frames before or after.

Equivalent serial ten-second comparisons used generated 1280×720 BGR input, actual CPU YuNet work plus controlled ±4 px boxes, both actual models, one Torch/OpenCV thread, the existing cadence and the same benchmark harness:

| Measurement | Before | With overlay |
| --- | ---: | ---: |
| Preview FPS | 26.90 | 26.57 |
| Mean GUI tick | 4.302 ms | 4.305 ms |
| Mean result-render pass, including overlay | 0.0181 ms | 0.0738 ms |
| Mean preview frame age | 22.422 ms | 21.933 ms |
| Recent p95 preview interval | 67.168 ms | 67.239 ms |
| Geometry transitions / dropped preview frames | 0 / 0 | 0 / 0 |

The overlay adds measurable drawing work. Single short windows do not prove universal performance equivalence, a causal FPS regression or physical improvement. No real webcam accuracy/FPS or owner acceptance is claimed. Baseline/after reports retain their source fingerprints.

## Files and integrity

| File | Change |
| --- | --- |
| `src/facial_attendance/ui/identity_overlay.py` | Pure context/label/coordinate helpers and cached two-item canvas renderer |
| `src/facial_attendance/ui/webcam_app.py` | Current-view overlay update, shared display transform, reset handling; existing pipeline/policies preserved |
| `tests/test_identity_overlay.py` | 26 new focused automated regressions |
| `scripts/check_ui03_identity_overlay.py` | 22 new generated-input native checks |
| `README.md` | Concise interface description/link |
| This document | Behavior, safety, measurements, limits and owner retest |

Generated logs/snapshots remain ignored in `artifacts/`; they are not staged. Existing uncommitted integration work and historical evidence remain intact. Before/after hashes confirm the supplied Face, Emotion and YuNet files and protected original `reports/environment/vy-mac-m1-review.json` are unchanged. Original SYS-05 remains at `798f2e7b9484a682e50c70b0b4689cd2fde316dc`. No staging, commit, push, merge, PR, Google Sheet, Notion evidence or reviewer-field change was made.

## Launch and physical owner retest

Close any old GUI and use the unchanged command:

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app \
  --camera 0 --detector yunet \
  --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx \
  --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' \
  --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' \
  --opencv-threads 1 --diagnostics
```

Without an existing compatible read-only gallery, expect Unknown, not a personal name. A real known-name check requires an already authorized compatible gallery supplied using the existing `--gallery` option. Genuine enrollment remains unavailable; the controlled generated registration demo is not physical identity evidence.

1. In Standard Mode, show one face and confirm Unknown/Checking... appears near its box while the side panel remains intact. With an authorized existing gallery, check the matched name is marked UNVERIFIED.
2. Move horizontally/vertically and toward all preview edges; check readable contrast, safe positioning, camera responsiveness and no side-panel/window movement. Repeat with/without glasses and indoor/backlit conditions.
3. Leave/re-enter, try a different person, and show multiple faces where practical. No prior name should remain on an invalid/replacement/multiple-face view. Fresh processing may show Checking...; names must not be held merely because a box remains.
4. Switch to Developer Mode and back. Confirm the same overlay, preserved controls and safety warnings. Pause/copy/export diagnostics and capture historical inference; these actions must not freeze the live overlay or authorize saving.
5. Check Retry and camera recovery. Confirm old labels disappear, the interface remains stable, and genuine registration/attendance remain blocked.

Record PASS/FAIL/NOT TESTED and visible defects. Physical behavior, recognition accuracy and independent approval remain pending; normal movement may deliberately clear the name under the unchanged conservative visual policy.
