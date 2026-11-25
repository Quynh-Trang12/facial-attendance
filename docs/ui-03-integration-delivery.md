# UI-03 trained Face + Emotion integration delivery

> Historical Face and Emotion integration evidence. The current app uses trained liveness and supports local registration and isolated attendance. See the [current demo guide](attendance-demo.md).

> Latest GUI extension: [Standard/Developer modes and diagnostics repair](ui-03-dual-mode.md) documents the shared application, visible scrollable inspector, 299 automated tests, 47 native checks and pending owner physical verification. Historical delivery evidence below remains unchanged.

> Current presentation follow-up: the owner physically failed the preceding 254-test presentation change. See [Emotion continuity investigation](ui-03-emotion-continuity.md) for the real-camera measurements, independent informational Emotion display, 284 passing tests, 20 passing native checks and pending physical retest. The 217-test results, 32-file inventory and older source fingerprints below describe the preceding integration delivery.

Prepared 18 November 2026, branch `feature/ui03-ui04-emotion-integration`, worktree `/Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration`. Continued the existing Emotion integration on SYS-05 baseline `798f2e7b9484a682e50c70b0b4689cd2fde316dc`; no branch merge, staging, commit, push, PR or tracker edit. Original SYS-05 source/environment and protected `reports/environment/vy-mac-m1-review.json` remain unchanged.

This is an owner-side implementation prepared for read-only review, not reviewer approval or Final Owner Evidence. Face and Emotion inference are trained. **Liveness is simulated and genuine enrollment/attendance remain disabled (`can_save=false`).** Displayed identities are provisional, explicitly UNVERIFIED. Controlled registration exercises mechanics using generated images and an isolated memory gallery, never a genuine person's enrollment.

## Actual checkpoint verification

Both original files remain outside the repository and were read without modification.

| Check | Face Recognition | Emotion |
| --- | --- | --- |
| Current local path | `/Users/elvyx/Downloads/best (Que An).pt` | `/Users/elvyx/Downloads/best(Dung).pt` |
| File SHA-256 | `bc4502836f306c886323c444f11d6ddd7fe02515ba40a467d22f486fbfdcbc4f` | `323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd` |
| Model ID | `20261114-125242-metric-74a8657-s42` | `20261114T180653198248Z-emotion-cnn-e161192-s42` |
| Recomputed weights version | `cb38ea47ceff` | `546da19237b1ed2549f657c535102071fa0c951d428412746819e71a137b7a20` |
| Architecture | Metric FaceEmbeddingNet: ResNet-18, 3×3 stride-1 stem, no max-pool or classification head | EmotionCNN, channels [32,64,128], dropout 0.3 |
| Epoch | File has only `state_dict` and `meta`, with **no epoch field**. Expected selection epoch 29 is user-reported, not independently established by this checkpoint. No invented epoch is displayed. | Epoch 9 verified in checkpoint |
| Load | Team `load_checkpoint`, CPU, `weights_only=True`, strict state load, selected metadata/version validation, finite tensors, eval mode; cached once before Tk construction | Existing hash-verified bytes, explicit architecture, CPU, `weights_only=True`, strict validation, eval/inference mode; cached before Tk in combined mode |
| Output | Finite float32 raw `[1,512]`; adapter L2-normalizes. Repeated generated-input inference exactly equal on this runtime. | Finite float32 logits `[1,3]`, softmax and explicit Negative=0 / Neutral=1 / Positive=2 |

File SHA-256 and weights version are different identifiers. Face uses the team's 12-digit hash over sorted tensor names/dtypes/bytes; Emotion uses its full hash including tensor shapes. Face file hash is pinned to this supplied file. Mismatched weights/hash/metadata stop the affected initialization with an error; there is no simulated recognition fallback.

The latest FAC-05 branch was re-read at `6212d86305f425dc4a4d3ce251966661e6d58bef`. Its [selection.json](https://github.com/Quynh-Trang12/facial-attendance/blob/6212d86305f425dc4a4d3ce251966661e6d58bef/evidence/FAC-05/selection.json), blob `620d5c1c8df61a451d24a8ba79fbb45cb318d9dc`, selects metric model/version above, **cosine >= 0.4619967254593038**, validation cosine AUC **0.9911137488198654**. The decision is unchanged from the prior inspection. AUC is validation-pair evidence, not webcam identity accuracy. The checkpoint's `threshold=None` is bound to that separately verified decision without editing the checkpoint.

## Implementation and source-to-contract traceability

Read [SYS-02 Final v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d), [UI-02](https://app.notion.com/p/412863094ebe80c09ffce13f894f16ae), the current Requirements rows 79–87, UI-01's embedding/enrollment schema and gallery revisions, and existing SYS-05 traceability/handover before editing. Sheet content was read only. The remote snapshots are recorded in `reports/ui03/remote-source-audit-20261118.json`.

| Source / contract | Reused implementation or boundary | Verification |
| --- | --- | --- |
| FAC-04 `5700e02ce42fbe65a316dc578998b47e996926d2` | `recognition/model.py` reused verbatim: FaceEmbeddingNet, load_checkpoint, model_version, eval_transform; torchvision uses `weights=None`, no model download | Actual selected weights load strictly, finite 512D output, deterministic eval, no identity head, mismatch rejection |
| FAC-05 selection above | `configs/ui03-recognition-selection.json`; `FaceInferenceContract`; cosine matcher uses exact published threshold | Equality/adjacent-score decisions, finite cosine, known/unknown/empty gallery; no recalibration |
| [UI-01 schema](https://github.com/Quynh-Trang12/facial-attendance/blob/4725d1121cbb2fffa08960d58f89a8311012856e/docs/embedding-schema.md) | Adapter-owned centered square margin 1.2, outward rounding/clipping, min box short side 64; RGB 64×64 bilinear /255 ImageNet normalization; normalized 512D output | Source-coordinate edge cases, distinct BGR channels versus authoritative eval_transform, immutable source, finite/unit vectors |
| UI-01 gallery `82a74f8c1ec8f8b769b541c3e0527dd28ca85c3e` | `ui/gallery.py` and 25 tests copied verbatim; schema 0.2, one averaged unit template, ID casefold uniqueness, name 1–64, count 3–10, atomic save/rollback | Upstream storage regressions plus actual-weight controlled enrollment/save/reload under temporary directories |
| UI-01 aggregation/guards | Five internally generated samples; `normalize(mean(normalize(e_i)))`; each sample >= selected threshold against template; duplicate-template/ID rejection | Controlled enrollment tests; zero-mean, inconsistent samples, duplicate and invalid input guards |
| SYS-02 §3/§7 | Original BGR uint8 and unsmoothed source box; `FaceRequest` / `Completion` echo full context; additive Recognition provenance | Capture/face/action/attempt/version correlation; stale, Retry, loss, multiple faces and camera/model errors |
| SYS-02 liveness / UI-02 authorization | Existing controller still rejects trained liveness input; simulated live only permits explicitly unverified preview, never genuine saving | Fixed false `can_save`; blocked genuine form/API; spoof/uncertain negative tests; no live gallery writer |
| UI-02 scheduling | Preload models before Tk, face/emotion inference on existing bounded worker; one additional worker only in controlled test mode prepares drafts; Tk applies current drafts | Slow-worker bounded submission/shutdown tests; actual-model native GUI transitions and clean worker stop |
| Emotion `284c48310e0335ed4b9ce4f641389c44b7c6841c` | Existing model/inference/preprocessing/label mapping retained; renamed file path in active instructions | All previous Emotion tests retained; both actual trained sources and original contexts checked together |

`docs/ui-03-face-inference-registration-contract.md` and `docs/ui-04-trained-adapter-contract.md` remain the detailed interfaces. Face crop margin/clipping is implemented consistently for live embedding and controlled samples; physical comparison with the training domain is still pending.

## Known, Unknown and registration behavior

Normal webcam mode uses an empty in-memory gallery unless `--gallery` supplies a compatible schema 0.2 file **read-only**. Corrupt/incompatible galleries fail initialization; no automatic migration or overwriting occurs. Matching normalizes raw embeddings, computes finite cosine, selects the maximum compatible template and uses casefold person-ID ordering for ties. An empty gallery yields valid Unknown with null score/identity; a valid below-threshold comparison also yields Unknown. No FAC-05 evaluation embeddings are used as enrolled identities.

The GUI displays trained Face matches as **UNVERIFIED**, trained Emotion separately, and **Liveness (simulated)**. Known output shows the stored name and canonical ID. Unknown shows `Unknown — Register a new person` and opens the ID/name form as a distinct registration action. In webcam mode the genuine submission button is disabled and explains the missing trained liveness prerequisite. `GalleryMatcher.register_live` always rejects; there is no live enrollment writer or attendance service.

`--controlled-registration-demo` requires `--demo-camera`, a real Face checkpoint and **no external gallery**. It uses internally generated pattern images, explicit controlled regions and a fresh isolated memory gallery. Five samples come from the fixture generator, never camera pixels. Entered test ID/name are validated; a bounded worker prepares a draft, and Tk checks the original registration attempt before committing it to memory. Shared gallery reads/writes are serialized. Repeated captures retain a registration attempt but receive fresh capture/face IDs; scene failure/Retry invalidates the attempt, and old completions cannot restore it. Spoof, uncertain or model error cancels a pending test draft even if its worker finishes after preview advances; a form left visible after Retry cannot submit from an attendance action. A successful test commit starts fresh attendance-preview recognition against the updated test gallery. Closing discards the memory gallery.

Only automated checks write generated templates to newly created temporary galleries to prove persistence/reload; those directories are removed after the checks. No genuine gallery or attendance file is changed. UI-01's normalized vectors/count do not establish genuine liveness or consent by themselves; the test mechanics do not claim live enrollment acceptance.

## Exact macOS setup and launch

Use the existing isolated environment, not the verified SYS-05 `.venv`. It now includes compatible torchvision; exact runtime used: Python 3.13.17, torch 2.13.0, torchvision 0.28.0, NumPy 2.2.6, Pillow 11.2.1, OpenCV 4.13.0.92. The first dependency install was blocked by network/DNS restrictions; the authorized retry installed torchvision into this environment only. No third-party model was downloaded.

From a fresh isolated checkout of this source with Python 3.13 and Tk:

```sh
python3.13 -m venv artifacts/ui04-runtime
artifacts/ui04-runtime/bin/python -m pip install -r requirements-ui03-ui04.txt
```

Current owner's webcam command (paths containing parentheses/spaces must stay quoted):

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' --diagnostics
```

This reuses the already present verified YuNet asset; it never silently downloads it. Grant the launching interpreter camera access if requested by macOS. With no existing compatible gallery, expect Unknown. An authorized existing compatible gallery can be added using `--gallery '/absolute/ignored/path/gallery.json'`; the UI reads it without writing. Missing/corrupt/incompatible checkpoints/gallery require correcting the path/file and relaunching, not pressing a bypass control. A transient inference or camera failure can be retried with the existing Retry control. UI-01 treats a missing gallery file as a compatible empty gallery and creates no file; a missing checkpoint is an initialization error.

Controlled registration demonstration, without physical camera or YuNet dependency:

```sh
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app --demo-camera --controlled-registration-demo --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt'
```

## Demonstration script

1. Say: “Face and Emotion inference use the supplied trained models. Liveness is simulated; every identity is unverified and genuine saving is disabled.” Show the banner and provenance.
2. In the **controlled generated-input** command, show Unknown and open Register a new person. Enter `test-demo-001` / `Generated Demo Person`.
3. Say: “This action embeds five generated patterns into an isolated memory gallery. It is not live enrollment.” Submit the test form. Show the name/ID on a subsequent fresh trained inference and `can_save=false` in debug JSON.
4. Show Retry and fresh context identifiers. Explain that leaving/relaunching this demo discards its memory gallery; automated temporary-file tests separately cover gallery save/reload.
5. Launch webcam mode separately. Show that the registration form cannot submit genuine enrollment. No real person was enrolled by the controlled demo and no attendance was saved.

Observed generated-input example: empty gallery Unknown → test enrollment of five generated samples → trained match score about **0.9996795679200035** → the same result after temporary gallery reload. This only establishes mechanics on a controlled fixture; it does not establish recognition of a real person, spoof resistance or expression accuracy.

## Automated and native results

Commands from the worktree:

```sh
FA_FACE_CHECKPOINT='/Users/elvyx/Downloads/best (Que An).pt' FA_EMOTION_CHECKPOINT='/Users/elvyx/Downloads/best(Dung).pt' PYTHONPATH=src artifacts/ui04-runtime/bin/python -m unittest discover -s tests -v
PYTHONPATH=src artifacts/ui04-runtime/bin/python scripts/check_ui03_integration.py --face-checkpoint '/Users/elvyx/Downloads/best (Que An).pt' --emotion-checkpoint '/Users/elvyx/Downloads/best(Dung).pt' --native-gui --output artifacts/ui03-native-retest.json
```

The report command refuses to overwrite an existing output. Omitting `--native-gui` runs actual-model headless checks only. Native checks use generated pixels, controlled face regions and controlled error injection, not a physical webcam. Runtime is tested on macOS only; Windows execution and CPU performance on other hardware remain unverified. Without local checkpoint environment variables the corresponding real-asset tests explicitly skip.

| Final executed check | Result |
| --- | --- |
| Existing suite with renamed Emotion file, before new edits | 160 passed, 0 failed, 0 skipped |
| Previous 160 + upstream gallery 25 + new recognition/integration 32 | **217 passed, 0 failures/errors, 0 skipped** |
| Native controlled GUI | **13 checks passed**: Unknown/form, test registration/subsequent trained Known, stale Retry rejection, fresh Retry result, loss clearing, multiple-face blocking, model error/recovery, camera error/recovery, detector failure, simulated spoof block, uncertain identity block with Emotion display |
| Native regular form | **2 checks passed**: Unknown/form and genuine submission disabled |
| Native callbacks / worker shutdown | No callback errors; all workers stopped |
| Actual trained synthetic Face/Emotion and temporary gallery reload | Passed; original source unchanged, deterministic Face embedding, context correlation, fixed false save eligibility |
| Physical webcam / real known-vs-unknown / expression accuracy / real spoof resistance / FPS | **Not tested or claimed** |

Reports: `reports/ui03/trained-model-native-20261118-final.json`, `reports/ui03/automated-tests-20261118-final.json`. Final native report includes hashes of the tested source/config/test/script files. Earlier failed exploratory checks are not counted as passes: an enrollment-test fixture initially expected orthogonal vectors to violate a threshold they actually exceeded; corrected the fixture to use inconsistent opposing vectors. The first native walkthrough found attendance resets preventing a registration attempt from spanning captures; fixed the operation context and added a targeted regression. Final counts above include those fixes. Earlier Emotion-only reports remain historical evidence, not current-source coverage.

## Physical validation checklist

- Confirm the actual source/worktree and both checkpoint paths/hashes; close other camera applications. Show trained Face/Emotion, simulated liveness, UNVERIFIED identity and disabled saving.
- Compare normal indoor light and bright backlight, without glasses and with glasses, small head turns and movement. Record detector box behavior, invalid-small-face guidance, preview responsiveness and model errors. Do not infer accuracy from a softmax score or validation AUC.
- Leave/re-enter, add a second person, Retry during work, interrupt/relaunch the camera and close the app. Current fields must clear, old results must not reappear, and recovery must use fresh contexts.
- With the normal webcam Unknown prompt, open the form: genuine enrollment must remain blocked even when simulated liveness says live. No real person should be enrolled for this test. Use only the separate generated-input command for registration mechanics.
- If an authorized pre-existing compatible gallery becomes available, compare known and unknown inputs and record actual results. Without such a gallery and trained liveness, genuine live enrollment/subsequent-person recognition cannot be accepted.

## Acceptance and remaining blockers

| Criterion | Implemented | Tested | Pending / blocked |
| --- | --- | --- | --- |
| UI-03.1 interfaces | Model/source/context/provenance, exact scoring policy, gallery/ID/name/aggregation rules, known/Unknown/form UI | Contract/runtime tests and controlled GUI | Owner/reviewer agreement and physical crop/domain validation |
| UI-03.2 trained model + gallery | Actual selected model, compatible gallery matching, explicit registration interface; isolated test mechanics | Actual embeddings, known/unknown, guards, generated enrollment/subsequent recognition/reload | **Full acceptance blocked:** trained liveness required for genuine live enrollment/attendance; physical known/unknown acceptance pending |
| UI-04.1 compatibility | Liveness-first dispatch, model-owned preprocessing, original contexts and typed errors, display states | Integration and negative-path regressions | Model-owner agreement; actual liveness architecture/config/policy values |
| UI-04.2 Emotion | Previously trained Emotion preserved and combined with trained Face display | Actual renamed file + all prior Emotion tests + native combined output | Physical Emotion validation; trained liveness integration and independent integrated review pending |
| Genuine persistence / spoof resistance | Disabled; no attendance service or live enrollment writer | Stub never authorizes, blocked form/API, simulated spoof/uncertain scenarios | **Blocked:** selected trained liveness checkpoint, architecture, input/crop/normalization, mapping, validated threshold, quality and decision-policy versions; separately tested genuine authorization/persistence |

No missing model is substituted or retrained. Expected Face epoch 29 additionally needs a matching training-run record if epoch provenance is required; the verified deployable weights themselves match the selected model version. Model matching on pre-cropped validation pairs does not validate webcam domain, margin, face quality, open-set multi-person gallery false accepts or glass/lighting behavior. These require owner/model-owner tests; no threshold adjustment is invented here.

## Read-only Copilot review and changed-file inventory

Open **this worktree**, not the original SYS-05 checkout, in VS Code. Ask Copilot to review without edits, staging, commits, downloads or tracker writes. Read both tracked diffs and untracked source files: `git diff` alone omits the new implementation. Focus on checkpoint/version binding, crop/source association, cosine/aggregation/duplicate handling, the separation of generated test enrollment from genuine enrollment, stub liveness never authorizing records, stale-result rejection, bounded workers and honest labels/evidence. Reproduce the commands above with actual local assets; do not reinterpret skipped tests as passes.

Review scope includes all pending integration files below. Exclude ignored environments/caches, model files, datasets, private frames/galleries, unrelated files, and especially the protected original review JSON. Historical SYS-05 source/documents are dependencies to read, not changes to stage. No code or report should claim independent approval.

```text
configs/emotion-training.toml
configs/emotion.toml
configs/ui03-recognition-selection.json
docs/ui-03-face-inference-registration-contract.md
docs/ui-03-integration-delivery.md
docs/ui-04-emotion-integration.md
docs/ui-04-trained-adapter-contract.md
reports/ui03/automated-tests-20261118-final.json
reports/ui03/remote-source-audit-20261118.json
reports/ui03/trained-model-native-20261118-final.json
reports/ui04/automated-tests-20261118.json
reports/ui04/emotion-checkpoint-native-20261118.json
requirements-ui03-ui04.txt
requirements-ui04-emotion.txt
scripts/check_ui03_integration.py
scripts/check_ui04_emotion.py
src/facial_attendance/emotion/inference.py
src/facial_attendance/emotion/model.py
src/facial_attendance/recognition/contracts.py
src/facial_attendance/recognition/gallery_matching.py
src/facial_attendance/recognition/inference.py
src/facial_attendance/recognition/model.py
src/facial_attendance/ui/emotion_adapter.py
src/facial_attendance/ui/gallery.py
src/facial_attendance/ui/live_preview.py
src/facial_attendance/ui/recognition_adapter.py
src/facial_attendance/ui/webcam_app.py
src/facial_attendance/ui/webcam_prototype.py
tests/test_emotion_integration.py
tests/test_gallery.py
tests/test_trained_recognition.py
tests/test_ui03_contracts.py
```

File purposes: upstream model/gallery/tests are verbatim team reuse; `recognition/inference.py` validates and runs selected weights; `gallery_matching.py` matches and isolates generated enrollment; `ui/recognition_adapter.py` preserves SYS-02 completions and initializes the mixed bundle; GUI/controller changes add honest provenance, forms and context handling without save permission. New tests/scripts/reports record actual results. Existing Emotion implementation/configs/tests and earlier reports are retained in this still-uncommitted integration delivery; setup pins and documents describe the current source.
