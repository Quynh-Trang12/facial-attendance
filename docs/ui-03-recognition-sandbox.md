# Que An Face Recognition Evaluation / SANDBOX

Prepared 21 November 2026 in `feature/ui03-ui04-emotion-integration`. This standalone owner-operated tool evaluates recognition only. **No physical evaluation has been performed by the implementation agent.** Generated-input checks establish mechanics, not webcam accuracy or independent approval.

The subsequent [status panel stability correction](ui-03-sandbox-panel-stability.md) documents fixed field positions, measured layout changes, regression checks and the targeted owner retest. Current result invalidation and model policy remain unchanged.

## Isolation and model verification

The normal webcam application, production registration/attendance gates and all existing source files are unchanged. The sandbox has its own process entry point, camera, bounded workers, session and newly constructed empty in-memory gallery. It does not accept a production gallery path and never constructs or modifies the production attendance/enrollment controller. There is no new normal-GUI liveness bypass or sandbox toggle.

Verified against the actual supplied checkpoint, the unchanged `FaceInferenceContract`, local FAC-04 implementation and pinned FAC-05 source audit/selection:

| Item | Verified value |
| --- | --- |
| Checkpoint | `/Users/elvyx/Downloads/best (Que An).pt` |
| File SHA-256 | `bc4502836f306c886323c444f11d6ddd7fe02515ba40a467d22f486fbfdcbc4f` |
| Architecture/approach | ResNet18, 3×3 stem/no maxpool, metric learning, no classification head |
| Model ID / weights version | `20261114-125242-metric-74a8657-s42` / `cb38ea47ceff` |
| Actual forward output | 512-dimensional embedding |
| Match | Existing normalized embeddings and cosine matcher; Known when score **>= 0.4619967254593038** |
| Preprocessing | Existing source-box square margin 1.2, clipping, RGB 64×64 bilinear, ImageNet mean/std; unchanged |

The checkpoint itself has no embedded selected threshold. The unchanged pinned FAC-05 decision supplies it, bound to this model/version. `FaceInference.load()` recomputes the weights version, verifies file SHA-256, architecture/input/dimensions/preprocessing, and the metric head. The sandbox neither recalibrates nor edits it. FAC-05's validation-pair operating point is not a guaranteed gallery/webcam FAR or accuracy, particularly after averaging reference embeddings.

Sources reviewed: `docs/ui-03-face-inference-registration-contract.md`, `docs/ui-03-integration-delivery.md`, `configs/ui03-recognition-selection.json`, `reports/ui03/remote-source-audit-20261118.json`, the actual FAC-04 model/inference code and existing gallery contracts. No PROJECT_CONTEXT.md was found in the workspace.

## Launch

Close any other GUI using camera 0 first. From the integration worktree:

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.recognition_sandbox \
  --camera 0 \
  --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' \
  --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx
```

The title and banner explicitly say **Recognition Evaluation / SANDBOX**, **UNVERIFIED**, **Liveness NOT evaluated**, **can_save=false** and no production enrollment/attendance. The model loads once before Tk. The camera does not open until the owner confirms informed consent and presses Start. macOS may present its normal camera permission prompt. Missing/incompatible models fail initialization rather than silently selecting another model or policy.

## Owner physical protocol

1. Obtain informed consent before collecting anyone's face samples, including unknown people and photographs. Use yourself if consent for other people is unavailable. Explain that transient face pixels and biometric reference embeddings are held in memory, and that this tool does not authenticate anyone. Each profile receives a generated test ID such as `test-001`. The optional display name appears in the Sandbox for the current session. Score exports use test IDs.
2. Tick the current-subject consent checkbox and press **Start / restart camera**. The **Next step** panel guides the basic flow. Keep exactly one subject visible. Face loss, multiple faces, camera errors and invalid/expired detector regions block capture and clear the current result. A small face may fail the unchanged minimum crop size; this is not automatically a recognition rejection.
3. In normal indoor lighting, collect **five reference captures**, using **Take reference photo** once for each distinct capture. Three to ten are supported by the existing gallery schema; five is the target. Start with reasonably consistent reference conditions and modest pose changes. Wait for each acceptance message. To annotate conditions, open **Settings and model details** before clicking. A repeated identical frame is rejected, even if its camera sequence number differs.
4. Optionally enter a short label in **Test profile name**, then press **Save test profile**. The existing normalize/average/normalize aggregation, consistency checks and duplicate-template guards are reused. An inconsistent/duplicate reference is an error, not silently accepted. Saving affects only the sandbox's new memory gallery. The camera stops and consent is cleared for the next phase. Repeat for another subject only with consent; the next test ID is generated, without overwriting an existing identity.
5. Freeze the reference set before measurement. Under **Expected person for recorded tests**, select an enrolled test ID, `unknown` for a genuinely unenrolled person, or `unspecified` if ground truth is unavailable. Changing it requires reconfirming consent/restarting the camera. Keep the subject stable until capture finishes: the sample is the most recent YuNet-confirmed frame, with an 800 ms stale limit, not a promise of instantaneous click-time exposure.
6. Restart to see the live match, then use **Record a test result** for distinct hold-out captures. These are separate from references. Both frame identity/content reuse and duplicate recorded evaluations are rejected. Live preview also suppresses scores for an exact reference frame, preventing a misleading resubstitution success.
7. Test the same person in separate captures: without/with glasses, normal indoor/backlit/dim lighting, frontal/left/right/up/down head positions, near/normal/far distance. Change one condition at a time, select its annotation and collect several captures without changing the threshold or retrospectively improving references to conceal failure. Different frames can still be highly correlated; they are not automatically independent test examples.
8. With informed consent, test people **not enrolled** with ground truth `unknown`. Record false accepts honestly. If multiple enrolled subjects are present in the session, record a wrong-identity match as a misidentification, not a correct accept.
9. Leave/re-enter the frame and test multiple faces. Confirm captures are blocked when appropriate; do not score a no-face/multiple-face/invalid input as a correct reject. Record such blocked/failed attempts as N/A with their visible reason in the manual template. They are excluded from the valid scored-trial counts, not hidden evidence of model success.
10. Present a photograph of a consenting subject and select `photograph`. If it matches, that demonstrates recognition without liveness; it is **not** genuine liveness or attendance success. Known-person photographs may be recognition-correct accepts, but the separate photograph-accept counter must be reported as a security limitation. These tests require actual owner execution; generated condition-tag tests do not establish photographic spoof behavior.
11. Optionally stop sampling by unticking consent. Under **Settings and model details**, press **Check gallery reload (memory only)** while idle. This validates schema/model-compatible serialization/deserialization and preserves matching without writing any biometric template to disk.
12. Inspect the frozen trial table and raw score export. **Export scores only** in Settings writes non-identifying metadata/decisions to the fixed ignored sandbox folder. Use the [recording template](recognition-evaluation-recording-template.md) to summarize counts and failed attempts. Avoid accuracy/FAR/TAR percentages from this small convenience sample.
13. Under Settings, press **Delete all sandbox data and exports** when finished. It stops the camera, drops references/trials, rejects stale jobs and deletes owned score exports from this sandbox root. It retires sample workers and recreates them only after old native work exits, releasing last-input references while keeping the model loaded. Recheck consent before any further sampling. Close to discard in-memory data; explicitly exported metadata can remain after close until reset/delete.

## Scores and outcomes

The live panel shows Known/Unknown and the matched test identity. **Settings and model details** shows the current cosine similarity and exact selected threshold. Similarity is not a confidence percentage. Displayed scores are rounded for readability; JSON retains the original float used by the unchanged matcher. Empty galleries yield Unknown with score N/A, not an invented zero score.

Only explicitly captured evaluation samples enter the trial table. Automatic preview is not a scored trial. Outcomes with operator-provided truth are:

- **correct_accept:** Known and the predicted test ID equals the enrolled ground truth.
- **false_reject:** enrolled ground truth, but the model says Unknown.
- **correct_reject:** unenrolled/unknown ground truth and the model says Unknown.
- **false_accept:** unknown truth accepted as a test identity, or an accepted wrong enrolled ID. Wrong enrolled-ID cases also set `wrong_identity=true`; they are not double-counted as correct accepts.
- **unscored:** no ground truth supplied; no correctness claim is made.

Live-person and photograph counts are exported separately. The tool prints counts rather than asserting reliable accuracy. Ground truth is an owner annotation, not something inferred from the frame.

## Reuse and data handling

The sandbox reuses `CameraFeed`, `SingleWorker`, `ScaledFaceDetector(YuNetFaceDetector)`, source-coordinate `FaceRequest`, `FaceInference`, `TrainedRecognitionAdapter`, `GalleryMatcher` and schema-0.2 `FaceGallery`. There is one camera feed, one detector worker and one model worker with no job backlog. Camera preview is independent of model work. Request contexts and exact source tokens remain associated; consent withdrawal, source change, expiry, reset or gallery revision reject late results.

The adapter's existing SYS-02 request action enum is reused for schema compatibility only; no attendance controller or writer is called. Research reference commitment uses the storage interface under its own explicit sandbox consent, never a relaxed production eligibility rule. Normal `GalleryMatcher.register_live` still raises PermissionError, and the normal `Envelope.can_save` remains false.

Memory holds at most ten test subjects, ten draft references, 500 recorded trials and bounded frame-hash history. No frame, crop, biometric embedding, personal name, private local path or exception message is exported. Frame fingerprints are memory-only duplicate guards, not published identifiers. There is no biometric gallery save/load-path option, so a sandbox file cannot be selected as a production enrollment output. The optional gallery round-trip is entirely in memory.

Score files are created only on explicit export at:

```text
artifacts/recognition-evaluation/<random-run-id>/scores.json
```

This root is separate and Git-ignored. Writes are atomic, use private temporary-file permissions and refuse linked output roots. Reset deletes only recognized score runs at that fixed root, never recursively deleting unexpected/unrelated content. It refuses unsafe links or unknown content and reports cleanup refusal while still clearing session memory. Do not put unrelated files there. Do not stage/export reference vectors or face imagery elsewhere.

Consent withdrawal stops camera sampling. Reset/close releases application references; a native call already running may retain its input until it exits. Python/OS memory or filesystem caches are not a cryptographic secure erase. Scores/pseudonyms and condition annotations should still be handled thoughtfully and retained only as agreed with participants. No network upload or message sending is implemented.

## Verification and unchanged production behavior

Actual pre-change baseline: **350 automated tests and 93 native checks passed**. The new full suite has **381 tests** (31 new), zero failures/errors/skips with actual supplied checkpoints enabled. All existing **93 native checks** pass again, plus **20 generated-input sandbox checks**, **113 native checks total**. The new checks cover consent, isolated capture/commit, distinct-frame evaluation, score/threshold display, honest wrong-ground-truth counting, photograph-condition separation, face loss/multiple faces, in-memory reload, score-only export, reset/delete, stale completion and model reuse. Orthogonal unit-vector fixtures exercise Unknown rejection; actual trained generated-pattern checks establish mechanics only.

The first sandbox native probe timed out because its harness compared Tk's state object directly with a string. Using the native `instate` check fixed the harness; final checks passed. This was not a physical camera or recognition finding. All generated reports mark physical evaluation false, including the photograph-condition plumbing test, which did not use an actual photograph.

Evidence/model provenance, source fingerprints, full test output and Git status remain ignored under `artifacts/recognition-sandbox-evidence/`. Existing production source and previous 350 test files are byte-for-byte unchanged. Supplied Face, Emotion and YuNet assets and protected original review JSON retain their before/after hashes. Original SYS-05 stays at `798f2e7b9484a682e50c70b0b4689cd2fde316dc`. No staging, commit, push, merge, tracker or reviewer-field update occurred.

## Complete changed-file inventory

| File | Purpose |
| --- | --- |
| `src/facial_attendance/recognition/evaluation.py` | Consent-gated private gallery/session, adapter worker, sample separation, outcomes, safe metadata export/delete |
| `src/facial_attendance/ui/recognition_sandbox.py` | Separate explicitly labeled owner-operated GUI and launch entry point |
| `tests/test_recognition_sandbox.py` | 31 new isolation, matching, privacy, reset/stale/model and production-safety tests |
| `scripts/check_recognition_sandbox.py` | 20 generated-input native checks with actual Face model |
| `README.md` | Link to the separate evaluation workflow; normal launch unchanged |
| `docs/ui-03-recognition-sandbox.md` | This verification, protocol, privacy/limitations and launch guide |
| `docs/recognition-evaluation-recording-template.md` | Concise copy-ready trial/count template |

Existing uncommitted integration work is retained. New generated evidence/data stay ignored. Physical model performance, unknown-person behavior, glasses robustness, photograph acceptance and independent review remain pending owner execution; no threshold or security gate was weakened to obtain the generated test results.
