# Emotion-only integration record — 18 November 2026

> Historical Emotion-only scope and reports below. Current Face + Emotion implementation, launch commands, acceptance status and review inventory are in [the UI-03 delivery](ui-03-integration-delivery.md). The local Emotion path below is updated to its renamed file; old reports retain the paths actually used at their creation.

## Current entry point

This record retains the behavior, commands and environment of its original UI revision. Use the [launch guide](attendance-demo.md) and [setup](setup.md) for the current bundled-model application.

## Scope and isolation

Implementation branch: `feature/ui03-ui04-emotion-integration`, isolated worktree `/Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration`, based on SYS-05 `798f2e7b9484a682e50c70b0b4689cd2fde316dc`. The original verified SYS-05 branch/environment and protected `reports/environment/vy-mac-m1-review.json` are untouched. No changes are staged, committed, pushed or merged; no PR or Google Sheet update is made.

| Criterion | Delivered | Still pending |
| --- | --- | --- |
| UI-03.1 | Face inference/enrollment design, pinned FAC-05 selection, 8 pure contract tests | Owner/reviewer agreement; crop/quality validation with selected weights |
| UI-04.1 | Adapter fields, preprocessing, call order, states, provenance and authorization design | Model-owner agreement; trained liveness configuration |
| UI-04.2 Emotion portion | Actual supplied EmotionCNN load/inference, context-bound adapter, opt-in GUI display, automated/native verification | Current-version physical webcam/owner validation and independent integrated review |
| UI-03.2 / full UI-04.2 | Not complete | Selected trained face and liveness checkpoints/adapters, compatible gallery integration and genuine authorization flow |

Designs: [UI-03 face/registration contract](ui-03-face-inference-registration-contract.md), [UI-04 trained adapter contract](ui-04-trained-adapter-contract.md). Current [acceptance criteria](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit), Requirements rows 79–87, were read on 18 November without editing. No Final Owner Evidence or independent approval is generated for incomplete criteria.

## Actual Emotion checkpoint validation

Located the supplied file at `"/Users/elvyx/Downloads/best(Dung).pt"`; no `/mnt/data` path is assumed and no checkpoint is downloaded or committed.

| Field | Verified value |
| --- | --- |
| File SHA-256 | `323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd` |
| Architecture | EmotionCNN, channels [32,64,128], dropout 0.3 |
| Model ID | `20261114T180653198248Z-emotion-cnn-e161192-s42` |
| Full weights version | `546da19237b1ed2549f657c535102071fa0c951d428412746819e71a137b7a20` |
| Selected epoch | 9 |
| Input / output | float32 `[1,1,48,48]` / finite float32 logits `[1,3]` |
| Mapping | 0 Negative, 1 Neutral, 2 Positive; explicit lowercase training keys verified |
| Policy | Argmax softmax; no decision threshold or extra mean/std |
| Load | Immutable hash-verified bytes, `torch.load(...,map_location="cpu",weights_only=True)`, explicit architecture, strict state validation, eval/inference mode |

The architecture/configs are copied verbatim from [feature/emotion-eda at `284c48310e0335ed4b9ce4f641389c44b7c6841c`](https://github.com/Quynh-Trang12/facial-attendance/tree/284c48310e0335ed4b9ce4f641389c44b7c6841c). Model Git blob: `4bf49f3575db016d36a284a091758439c4430462`. Local model source SHA-256 `720cbf2cc536819d545302f801760eb8f09b8a6d46d9c34ddf8e784dc8b6e220` matches the checkpoint's recorded source hash. Training's sorted name/dtype/shape/tensor-byte digest was recomputed and matches the full weights version. Optimizer/RNG state is never restored. Training code/datasets are not copied into this runtime.

[Upstream tensor conversion](https://github.com/Quynh-Trang12/facial-attendance/blob/284c48310e0335ed4b9ce4f641389c44b7c6841c/src/facial_attendance/emotion/tensor_data.py) confirms Pillow grayscale, bilinear 48×48 and float32 /255. The webcam adapter copies BGR to RGB before that conversion, preserving the source crop. It validates keys, tensor dimensions/dtypes/finiteness, label mapping, epoch and metadata before accepting a model.

Actual synthetic source BGR `(11,22,33)`, original-frame box `(10,20,50,60)`: logits approximately `[1.3285125,0.2952696,-2.3293366]`; softmax `[0.7237781,0.2575572,0.0186647]`, class 0 Negative. This is genuine checkpoint execution on generated pixels, **not expression accuracy or physical webcam evidence**. Values can vary slightly by runtime/platform.

`reports/ui04/emotion-checkpoint-native-20261118.json` records the checkpoint identity, exact source hashes, runtime, synthetic outputs and native Tk checks. It contains no webcam images. GUI metadata shows model ID, weights prefix, epoch and file-hash prefix; debug JSON includes full provenance. Old results/provenance clear on a fresh sample, loss, multiple faces, error or Retry.

## Installation and launch

Use a separate Python 3.13 environment with Tk; do not install these pins into the verified SYS-05 `.venv`. Current machine verification used Python 3.13.17 on macOS arm64, Torch 2.13.0, NumPy 2.2.6, Pillow 11.2.1 and OpenCV 4.13.0.92. Windows setup below is reproducible guidance, not a Windows execution claim. No torchvision dependency is needed for EmotionCNN.

From the isolated worktree root on macOS:

```sh
python3.13 -m venv artifacts/ui04-runtime
artifacts/ui04-runtime/bin/python -m pip install -r requirements-ui04-emotion.txt
FA_EMOTION_CHECKPOINT="/Users/elvyx/Downloads/best(Dung).pt" PYTHONPATH=src artifacts/ui04-runtime/bin/python -m unittest discover -s tests -v
```

The environment already exists on the owner's current machine. Exact current-machine launch command (automatic YuNet preview + trained Emotion, simulated identity/liveness):

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --yunet-model /Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx --emotion-checkpoint "/Users/elvyx/Downloads/best(Dung).pt" --diagnostics
```

The YuNet path deliberately reuses the owner's existing verified local ONNX asset; this isolated worktree does not silently obtain a model. Elsewhere supply the locally verified asset with `--yunet-model`; follow the existing `docs/sys-05-handover.md` setup/provenance instructions. `--detector haar` needs no ONNX but has the previously documented glasses limitation. Neither option changes Emotion weights.

Without `--emotion-checkpoint`, SYS-05 stays entirely simulated. `--dev-manual` cannot be combined with trained Emotion because that legacy display is the separate stub developer harness. Missing/wrong checkpoint errors appear as Emotion Unavailable; identity/liveness simulations are not promoted to trained results.

Windows PowerShell, from this branch's repository root with a local checkpoint and YuNet asset:

```powershell
py -3.13 -m venv artifacts/ui04-runtime
.\artifacts\ui04-runtime\Scripts\python.exe -m pip install -r requirements-ui04-emotion.txt
$env:PYTHONPATH = "src"
$env:FA_EMOTION_CHECKPOINT = "C:\Models\emotion\best(Dung).pt"
.\artifacts\ui04-runtime\Scripts\python.exe -m unittest discover -s tests -v
.\artifacts\ui04-runtime\Scripts\python.exe -m facial_attendance.ui.webcam_app --camera 0 --detector yunet --yunet-model C:\Models\face_detection_yunet_2023mar.onnx --emotion-checkpoint "C:\Models\emotion\best(Dung).pt" --diagnostics
```

Use the actual local asset paths. Python must include Tk; macOS camera permission and Windows privacy permission must allow that interpreter. No webcam access is needed for automated checkpoint verification. A fresh native GUI report can be generated with:

```sh
PYTHONPATH=src artifacts/ui04-runtime/bin/python scripts/check_ui04_emotion.py --checkpoint "/Users/elvyx/Downloads/best(Dung).pt" --native-gui --output artifacts/ui04-native-retest.json
```

The report writer refuses existing output files. Omitting `--native-gui` performs headless checkpoint/inference/context verification only. Native tests use generated pixels and controlled regions, never an actual person.

## Verification results

| Executed check | Result |
| --- | --- |
| Verified baseline regression suite before edits | 131 passed, 0 failed, 0 skipped |
| UI-03 contract tests | 8 passed |
| Emotion tests, supplied checkpoint enabled | 21 passed, including 3 using actual `best.pt` |
| Full modified suite | **160 passed, 0 failed, 0 skipped** |
| Native Tk with actual Emotion weights | 5 checks passed: trained result/labels, Retry stale rejection, fresh result after Retry, face-loss clearing, multiple-face blocking/clearing; no callback errors, all workers stopped |
| Physical webcam for this new implementation | **Not tested**; prior SYS-05 owner passes do not cover trained Emotion |
| Windows / real spoof resistance / expression accuracy / real-world FPS | Not measured or claimed |

Tests cover hash rejection before deserialization, metadata/state mismatch, exact training preprocessing, label mapping, typed errors, display-only Emotion failure, preserved liveness gate, current context, reset/camera-error/face-loss stale rejection, off-Tk bounded worker inference and shutdown. All 131 previous tests are retained without modification. Without `FA_EMOTION_CHECKPOINT`, the three local-asset tests skip explicitly; without optional Torch, other trained-runtime tests skip rather than pretending to execute a model.

## Files prepared for review

| Files | Purpose |
| --- | --- |
| `configs/emotion.toml`, `configs/emotion-training.toml`, `src/facial_attendance/emotion/model.py` | Authoritative architecture/config reuse and provenance |
| `src/facial_attendance/emotion/inference.py` | Hash-pinned CPU load, validation, copied preprocessing, finite logits/softmax |
| `src/facial_attendance/ui/emotion_adapter.py` | Original-context trained Emotion completion and explicit mixed adapter bundle |
| `src/facial_attendance/ui/webcam_prototype.py` | Additive Emotion provenance validation; fixed false save eligibility retained |
| `src/facial_attendance/ui/live_preview.py`, `src/facial_attendance/ui/webcam_app.py` | Optional mixed GUI/CLI labels and model information within existing workers |
| `configs/ui03-recognition-selection.json`, `src/facial_attendance/recognition/contracts.py` | FAC-05 decision binding and pure face/enrollment design guards |
| `tests/test_ui03_contracts.py`, `tests/test_emotion_integration.py` | Contract/runtime/error/regression tests, including actual local weights |
| `requirements-ui04-emotion.txt` | Isolated optional inference runtime pins |
| `scripts/check_ui04_emotion.py`, `reports/ui04/emotion-checkpoint-native-20261118.json`, `reports/ui04/automated-tests-20261118.json` | Reproducible checkpoint/native verification and actual test reports |
| Three `docs/ui-03-*` / `docs/ui-04-*` documents in this delivery | Design, scope, setup, blockers, tests and acceptance guidance |

Exclude local environments, checkpoint/model files, datasets, personal frames, caches, private files, unrelated modifications and the protected review JSON from any future commit. The current source changes remain unstaged and uncommitted.

## Owner acceptance checklist and handover

1. **UI-03.1 design:** confirm the exact selected model/threshold; review raw embedding normalization, crop/edge policy, known name/ID display, explicit Unknown registration prompt, ID/name/sample/duplicate rules and gallery compatibility. Confirm this is contract preparation only.
2. **UI-04.1 design:** agree with chị Dung the fields, trained/simulated labels, preprocessing, original-context association, call order, fail-closed spoof/uncertain behavior and display-only Emotion errors. Obtain the real liveness model's numeric configuration before implementation.
3. **UI-04.2 Emotion physical test:** launch the command above; check MIXED DEMO banner, simulated identity/liveness, trained Emotion label/softmax and matching model provenance. Try expressions without treating label changes as validated accuracy. Compare glasses/no glasses and indoor/backlit conditions; inspect preview responsiveness during movement.
4. Leave and re-enter frame, add a second face, press Retry during processing, test camera interruption/relaunch, and supply a missing or wrong checkpoint. Old Emotion/identity/liveness must clear, stale callbacks must not restore them, model errors must be visible, and saving must remain disabled. Close and confirm no hanging worker/application. Record actual results against this branch/source, not the earlier SYS-05 candidate.
5. Handover these documents, changed source/test files, pins and verification report; transfer the supplied weights separately under the model owner's access policy and verify its full hash. Do not include checkpoint bytes, the local runtime, personal frames or protected environment JSON in Git. No demonstration, agreement or reviewer approval is recorded by providing these instructions.

Remaining prerequisites: selected Face Recognition `best.pt` with ID `20261114-125242-metric-74a8657-s42` / version `cb38ea47ceff`, validated face crop/quality policy and compatible gallery wiring; trained liveness checkpoint, architecture, input preprocessing, mapping, validated threshold and quality/decision policy with versions; separately tested genuine authorization/persistence integration. The Emotion file is not a substitute for either missing model. No liveness bypass or genuine enrollment/attendance writes exist in this delivery.
