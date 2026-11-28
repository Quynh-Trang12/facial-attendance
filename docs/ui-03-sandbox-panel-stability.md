# Recognition Sandbox status panel stability

Prepared 21 November 2026 on `feature/ui03-ui04-emotion-integration`. This is a presentation-only correction in the integration worktree. Physical owner acceptance remains pending. All new measurements use generated inputs; none measure webcam FPS or recognition accuracy.

## Current entry point

This record retains the behavior, commands and environment of its original UI revision. Use the [launch guide](attendance-demo.md) and [setup](setup.md) for the current bundled-model application.

## Cause and scope

The original 420 × 360 outer frame already disabled geometry propagation. However, its children were packed vertically at their natural sizes. The current-result label switched between a one-line checking/stopped message and a four-line completed result. That resized the label and moved checkpoint metadata, progress and both action buttons. In the controlled comparison, the result label changed geometry 65 times; metadata, progress, Start and Reset each changed 32 times: 193 widget geometry transitions in total. The outer window/panel did not need another geometry change.

The original multiline result and progress StringVars already skipped identical text. Redundant status text writes were **zero**. The three capture/commit buttons were nevertheless configured on every refresh: 2,733 state writes, 2,658 of them redundant. There is no widget rebuilding/repacking during inference in the original code. The controlled event-loop measurements do not establish blocking inference as the cause of this panel movement. Camera/inference performance under physical conditions has not been measured in this task.

The previous sandbox native checks validated capture, consent, matching, stale completion, reset, privacy and worker shutdown, but never asserted the positions of the panel's children. That allowed correct functional checks to coexist with the owner's physical layout defect.

## Change

`SandboxStatusPanel` follows the main webcam application's fixed pixel-cell pattern, with independent cells for camera, current decision, current cosine, checkpoint metadata, alias, draft reference count and gallery/trial counts. Each label fills its reserved cell. Start and Reset have separate fixed slots. The outer panel remains 420 × 360; existing main-GUI geometry is untouched.

Checkpoint metadata is static, created once. Dynamic fields cache their last text and only write changed values. Button states are changed only when necessary. A new analysis sample still clears the authoritative current decision; the score field explicitly says there is no current result until an accepted completion arrives. No identity history, score smoothing or delayed invalidation is introduced. Current Known/Unknown and six-decimal cosine rendering use the existing raw decision; the raw score, threshold and exports are unchanged. Camera failure messages remain visible and capture is blocked on invalid states.

## Verification

Baseline reproduced before product changes: **381 automated tests and 113 native checks passed**. Existing native checks were rerun before and after using generated pixels and the supplied checkpoints. Twelve new automated regressions cover idempotent text/state updates, per-field changes, current Known/Unknown/Checking, stopped camera, face loss/re-entry, multiple faces, truthful scores, empty gallery, reference counts 0–10 and reset.

Native sandbox checks additionally exercise ten distinct reference captures, commitment/alias updates, scores, trial counts, face loss/re-entry, reset, empty-gallery Unknown, camera failure, Retry, stopped camera, and unchanged geometry of every panel child. The existing functional checks remain present. A separate deterministic presentation fixture drives 910 updates across stopped, checking, Known, Unknown and unavailable states, reference counts 0–10 and varying cosine lengths, for more than 30 seconds. This fixture uses scalar states to isolate presentation; it is not model or enrollment accuracy evidence. Native capture/matching checks run separately with the real checkpoint and generated frames.

Raw evidence, source hashes, baseline source snapshot, automated logs and native reports are in ignored `artifacts/recognition-panel-evidence/`. Final measurement counts and timings are recorded in `verification.json` and `panel-after-final.json`. Timing covers the generated presentation workload only, not camera capture, trained inference or physical preview FPS. Different numbers of legitimate text writes are expected because current decision and score now occupy separate fields; zero redundant writes is the relevant caching check.

Final results: **393 automated tests and 142 native checks passed**, comprising the original 113 native checks, 21 added sandbox checks and eight long-run presentation checks. No callback errors were reported and native sandbox workers stopped cleanly.

| Controlled measurement | Before | After |
| --- | ---: | ---: |
| Driven updates | 910 | 910 |
| Duration (seconds) | 31.543 | 31.873 |
| Panel/child geometry transitions (sum across widgets) | 193 | 0 |
| Visible dynamic text writes / actual changes | 162 / 162 | 242 / 242 |
| Redundant text writes | 0 | 0 |
| Button state writes | 2,733 | 75 |
| Redundant button state writes | 2,658 | 0 |
| Mean render/control callback including idle layout (ms) | 0.580 | 0.832 |
| Maximum render/control callback (ms) | 1.967 | 3.555 |
| Mean event interval (ms) | 34.700 | 35.063 |
| Maximum event interval (ms) | 78.892 | 55.254 |

The after run was isolated from model test execution. This correction removes measured layout churn and duplicate state mutations; it does **not** establish a rendering-speed or webcam-FPS gain. New independent fields require more legitimate text writes, while all metadata and controls retain their geometry. Timing differences also include normal native event-loop variation.

## Owner launch and targeted physical retest

Close other applications using camera 0, then run:

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.recognition_sandbox \
  --camera 0 \
  --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' \
  --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx
```

1. Confirm consent and Start. Watch the right panel for at least 30 seconds, first stationary and then gently moving. Camera, model information, alias/counts and buttons must keep their positions; only current result/score text should change.
2. Collect distinct references from zero through ten if convenient. Check the counter, then commit and confirm the next alias changes and draft count returns to zero without shifting controls. Reconfirm consent for evaluation.
3. Observe current Known/Unknown decisions and cosine scores. Pending/invalid samples should show no current score; a cosmetic change must never retain a previous valid score. A match or rejection is the unchanged model's decision, not a promised test outcome.
4. With consent, test an unenrolled person; note the actual score/decision honestly. Leave/re-enter the frame, then test multiple faces. Capture must be disabled when invalid, with no retained identity or score.
5. Untick consent to stop; reconfirm and use Start/restart. Test camera recovery if feasible. Failures must remain visible, and the panel must stay fixed.
6. Check recorded trial counts and reset/delete after testing. Export only if desired; this task does not alter existing owner records. Confirm `UNVERIFIED`, liveness not evaluated, `can_save=false`, and no production enrollment/attendance throughout.

Report any remaining movement, clipped text, incorrect score retention, hidden failure or unresponsive control. Changing text on a genuine new result or invalidation remains intentional. Generated tests cannot prove physical flickering is resolved.

## Changed files and safety

Only these files were edited/added for this task:

- `src/facial_attendance/ui/recognition_sandbox.py`
- `src/facial_attendance/ui/sandbox_status.py`
- `tests/test_sandbox_status.py`
- `scripts/check_recognition_sandbox.py`
- `scripts/benchmark_recognition_panel.py`
- `docs/ui-03-sandbox-panel-stability.md`
- `docs/ui-03-recognition-sandbox.md` (link to this correction)

The existing model/domain/inference code, model selection, checkpoint files, preprocessing, cosine threshold, scheduling, consent, gallery isolation, exports, freshness policy and production gates are untouched. Existing source comparison confirms only the sandbox UI changed, plus its new presentation helper. SHA-256 checks confirm both supplied checkpoints, YuNet and the protected environment review JSON are unchanged. The original SYS-05 branch is untouched. No staging, commit, push, merge, Google Sheet or Notion operation was performed. Existing unrelated uncommitted changes are preserved.
