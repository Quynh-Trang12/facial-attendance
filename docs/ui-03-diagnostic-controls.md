# Developer diagnostics: pause, copy and privacy-safe snapshots

This extension stays in the existing UI-03/UI-04 worktree and changes diagnostics only. All controls are inside the Developer-only window. Standard Mode still hides the diagnostics window and button; the mode switch and mixed-demo warnings remain available. The launch command is unchanged; see the [README](../README.md).

## Why inspection was difficult

Runtime JSON refreshed through `DeveloperDiagnostics.refresh`, while raw SYS-02 JSON was rewritten separately in `LiveWebcamApp._render`. Pausing only the runtime tab would have left the raw tab changing. Both presentation paths now honor the same diagnostic pause flag. No camera, worker, model, controller, gallery or authorization operation is paused.

## Controls

- **Pause refresh / Resume refresh:** freezes both current diagnostic text buffers and timestamps. The status shows PAUSED and identifies the data as a frozen snapshot, not current authorization. Resume requests fresh runtime and raw data on the next GUI tick. Closing/reopening diagnostics preserves an intentional pause.
- **Copy displayed JSON:** copies the complete JSON in the selected tab, exactly as displayed, without fetching a newer result. It does not redact raw on-screen identity information. The normal text selection/keyboard copy path also remains available.
- **Export privacy-safe snapshot:** opens a save dialog and writes a new allowlisted JSON summary of the selected displayed snapshot. Data and timestamp are captured at the export click, before the file dialog; subsequent inference cannot change that export. Cancellation writes nothing. Failed writes produce a concise message without exposing the destination path.
- **Capture latest completed inference (historical):** copies the latest current-context completed batch into a separate historical tab and pauses diagnostic refresh for inspection. It can be captured while the current controller is checking a newer sample. Resume returns to the current raw tab; the captured historical tab stays separate.

LIVE/PAUSED and the selected tab's UTC snapshot timestamp remain visible. A completion timestamp is when the GUI observed an accepted completed batch, not the camera exposure time. Each tab has its own timestamp; copied raw SYS-02 JSON keeps the contract schema, while an export includes snapshot metadata outside the sanitized data summary.

Both scroll axes are retained during text replacement. A selected substring is restored when it remains at the same indices or can be uniquely relocated; a removed/ambiguous selection is not silently applied to different content. Pausing gives a stable inspection/copy target. Controls remain exclusively in Developer Mode.

## Completion history and authorization

The GUI retains one frozen Envelope reference for the latest accepted completed worker batch. It contains no image/crop or embedding, is overwritten by the next accepted completion, and is cleared on application close. Stale/rejected completions cannot replace it. Capture retains one separate historical text snapshot. There is no unbounded history or queue.

Historical diagnostics may outlive a Retry, face loss or the main result's freshness interval. This is deliberate, explicitly marked diagnostic history: it is never assigned back to the controller, adapters, presentation cache or registration eligibility. Main results still invalidate on the original rules. Paused diagnostics must not be interpreted as the current state of a different face. Liveness remains SIMULATED, identity UNVERIFIED, and genuine enrollment/attendance remain disabled with `can_save=false`.

No model weights, architecture, preprocessing, thresholds, embedding generation, label mapping, dispatch/scheduling, display smoothing, 800 ms presentation freshness or 450 ms box retention policy changed.

## Export privacy boundary

`diagnostic_snapshot.py` constructs a new document from explicitly allowed scalars and enums. It never recursively dumps the displayed document or relies only on removing a few sensitive keys.

Exports retain snapshot timestamp/source/paused/history flags, anonymous raw decisions/scores, trained Emotion labels/confidence, allowed checkpoint digests/versions, dimensions/threshold, selected runtime booleans/counts/device/detector, known display reasons/ages and measured scalar timing summaries. Invalid/nonfinite values become null; unrecognized string fields are omitted or null.

Exports exclude names, person IDs, full contexts/attempt/capture/face identifiers, personal gallery records/templates, images, frames, crops, biometric vectors, credentials, arbitrary exception messages, arbitrary metadata and local paths. The destination path is not embedded in the file or success message. Copy is intentionally exact and is a distinct operation from privacy-safe export.

This export is a diagnostic file, not genuine attendance or enrollment persistence. The application does not automatically export, upload, send or add it to Git. Choose an appropriate local destination. Test exports use temporary directories and generated fixtures and are removed after validation.

## Verification

All **302 previous automated tests plus 22 new regressions pass: 324 tests**, with both supplied checkpoint tests enabled, zero failures/errors/skips. Tests cover both paused buffers, resume refresh requests, timestamps, selected-tab copy, canceled/failed/malformed export, adversarial sensitive fields, historical capture, scroll/selection retention, continued preview/inference, immediate raw face-loss invalidation and stale-completion rejection.

All **56 previous native checks plus 15 new checks pass: 71 native checks**. The new mapped checks verify actual control visibility, LIVE/PAUSED timestamps, selection retention across refresh, frozen text while new camera frames and inference results arrive, exact clipboard content, sanitized temporary export, separate completed history, resume and hiding all controls in Standard Mode. The test restores the previous text clipboard. All workers stopped, with zero callback errors and genuine saving blocked. These checks use generated frames and actual model checkpoints; they do not establish owner physical acceptance or webcam performance improvement.

Evidence remains ignored under `artifacts/ui03-diagnostic-controls-evidence/`. Model/checkpoint and protected-file integrity checks are recorded there. No original SYS-05 branch, protected review JSON, tracker, Google Sheet or reviewer field was changed. Nothing is staged, committed or pushed.

## Files changed in this extension

| File | Purpose |
| --- | --- |
| `ui/developer_view.py` | Developer-only controls, timestamped status, third historical tab, frozen raw/runtime views, copy and export |
| `ui/diagnostic_snapshot.py` | Immutable completion record, timestamp helpers, positive export allowlist and selection-preserving text updates |
| `ui/webcam_app.py` | Observe accepted completions without altering dispatch; honor diagnostic-only pause and raw refresh requests |
| `tests/test_diagnostic_controls.py` | 22 new automated regressions |
| `tests/test_view_modes.py` | Initialize the existing refresh-test fixture for the new diagnostic presentation fields; all old assertions retained |
| `scripts/check_ui03_modes.py` | Preserve the prior checks and add 15 mapped controls checks |
| `README.md`, `docs/ui-03-dual-mode.md`, this document | Controls, privacy distinction, historical-data semantics and verification |

## Owner retest

1. Enter Developer Mode. Confirm LIVE and a timestamp; select raw JSON and pause.
2. Stay/move in view. Camera and main results should continue while both diagnostic text buffers and the selected timestamp remain fixed.
3. Copy displayed JSON and paste locally to verify it matches the selected tab. Export a safe snapshot and check it contains no name, ID, images, vectors or local paths.
4. Capture the latest completed inference while another sample is checking. Confirm the HISTORICAL label and preserved completion timestamp; it must not change the main authorization state.
5. Resume and confirm current JSON updates again. Close/reopen, then switch to Standard Mode; all diagnostic controls must disappear and safety warnings must remain.

Physical acceptance remains pending owner confirmation. Existing deliberate main-display expiry under slow inference is unchanged; a paused or historical diagnostic record does not extend that deadline.
