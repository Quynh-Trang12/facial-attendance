# Emotion experiment plan

Owner: chị Dung. Status: baseline implementation ready for review; full training has not started. These settings are starting values, not measured results. Record any change and its reason in the run manifest before comparing results.

## Data and labels

Use the fixed FER2013 partitions under `data/processed/emotion_3class/`. The [dataset protocol](emotion-3class.md) defines filtering and split creation; the [loader guide](emotion-loader.md) defines loading. Do not create a new split for each run.

| Model ID | Display label | Source expressions |
| --- | --- | --- |
| 0 | Negative | angry, disgust, fear, sad |
| 1 | Neutral | neutral |
| 2 | Positive | happy |

Surprise is excluded. Original expression annotations are retained without manual relabelling.

| Split | Negative | Neutral | Positive | Total |
| --- | ---: | ---: | ---: | ---: |
| Train | 10,089 | 3,858 | 5,626 | 19,573 |
| Validation | 2,521 | 964 | 1,406 | 4,891 |
| Test | 3,293 | 1,224 | 1,766 | 6,283 |

The verified dataset manifest SHA-256 is `79490d419bc9f4c05042dbdd3fcf76921bffdd73d22f855cd565ce9631c8386b`. Verify it before training. Validation was selected within each source expression with seed 42. Exact duplicate pixels do not cross splits; identity and near-duplicate separation are not established.

## Baseline configuration

| Setting | Proposed value |
| --- | --- |
| Input | Grayscale, 48 × 48; float32 tensor `[B, 1, 48, 48]` |
| Normalization | Divide pixel values by 255; no additional mean/std normalization |
| Augmentation | None for this baseline, in every split |
| Architecture | Three convolution blocks, with 32, 64 and 128 output channels |
| Each block | Conv2d (3 × 3, stride 1, padding 1, bias false), BatchNorm2d, ReLU, MaxPool2d (2 × 2, stride 2) |
| Head | Adaptive average pooling to 1 × 1, flatten, dropout 0.3, linear 128 → 3 |
| Initialization | Train from scratch; no pretrained checkpoint or frozen layers |
| Output and loss | Three raw logits; unweighted cross-entropy on integer class IDs |
| Optimizer | Adam, learning rate 0.001, betas (0.9, 0.999), epsilon 1e-8, weight decay 0.0001 |
| Learning-rate schedule | Constant for this baseline |
| Batch size | 256; retain the final incomplete batch |
| Training length | Up to 30 epochs; early stopping after 5 consecutive epochs without a checkpoint improvement |
| Seed | 42 for Python, NumPy, PyTorch and loader sampling; seed CUDA when used |
| Sampling | Shuffle training data only; no class weighting or oversampling |
| Evaluation | `model.eval()` with gradients disabled; fixed validation/test order |

Training batch size is 256, set explicitly in `configs/emotion-training.toml`. The loader verification in `configs/emotion.toml` used 32; its recorded results describe that loader check, not the training batch size. A real batch-256 CPU training step, validation batch and checkpoint round trip passed. CUDA memory usage has not been checked.

Use the verified preprocessing in `configs/emotion.toml`. The runnable training configuration is [emotion-training.toml](../configs/emotion-training.toml); the CNN is implemented in [model.py](../src/facial_attendance/emotion/model.py). The training command overrides batch size and seed from that configuration while preserving the preprocessing contract. Use deterministic operations where supported and record any exception. A seed alone does not guarantee identical results across devices or framework versions.

## Checkpoint selection and final evaluation

Select the epoch with the highest validation macro F1 across all three classes. Break an exact tie using lower validation cross-entropy, then the earlier epoch. Use this same ordering for early stopping. Do not use test results to choose an epoch, architecture or training settings.

Save a resumable checkpoint after every epoch, including model, optimizer, epoch, best selection result and random generator states. Keep the best checkpoint separately. Store checkpoints outside Git under `artifacts/checkpoints/<run_id>/` and back them up to the shared Drive location when available.

For inference, take the argmax of the three logits and map its ID to the display label above. Softmax supplies the three class probabilities; the maximum is the model confidence, without a calibration claim. No confidence threshold or uncertain class is defined for this baseline. Record `threshold: null` and `threshold_source: "not_applicable_argmax"`. Any later rejection threshold must be selected on validation data and documented before final evaluation.

Report validation and final test macro F1, accuracy, per-class precision/recall/F1 and support, and a confusion matrix with rows = true labels and columns = predicted labels, ordered Negative, Neutral, Positive. Use zero_division=0 for undefined precision/recall/F1. Report test results once after selecting the final configuration. Save per-image predictions so metrics can be reproduced.

These are three-class FER2013 results, not seven-class benchmark scores. The aggregated expressions do not establish a person's mood. Performance on attendance-camera images requires a separate integration check.

## Run records

Use `YYYYMMDDTHHMMSSffffffZ-emotion-cnn-<shortcommit>-s42` as the emotion run ID; create its output directory exclusively and fail if it already exists. Record the full effective settings, dataset manifest hash, exclusion records/hash, code commit and dirty state, seed, dependency versions, device, selected epoch, validation selection metrics, checkpoint location/hash and label mapping in `reports/runs/<run_id>/manifest.json`. Record hashes of the exact split records used and distinguish validation metrics from final test metrics. Pair-file hashes are not applicable to this classifier.

Commit settings, code, aggregate metrics and plots. Keep raw images, per-image records that expose source data, credentials and checkpoint binaries out of Git. Link the final results to the run ID and checkpoint. The GUI input crop and adapter must use the same grayscale, size, scaling and class mapping as training.

## Commands and preflight evidence

Use the project Python 3.13 environment from the repository root. These commands create new output directories and refuse to overwrite existing runs.

```powershell
# One real train batch and validation batch; no full training or final test.
.venv/Scripts/python.exe scripts/train_emotion.py --smoke-test --device cpu

# Full training: choose a CUDA-enabled environment before selecting cuda.
.venv/Scripts/python.exe scripts/train_emotion.py --device cuda

# CPU training is supported, but may be slower.
.venv/Scripts/python.exe scripts/train_emotion.py --device cpu

# Resume a trusted epoch checkpoint in its original run directory.
.venv/Scripts/python.exe scripts/train_emotion.py --device cuda --output artifacts/checkpoints/YOUR_RUN_ID --resume artifacts/checkpoints/YOUR_RUN_ID/epoch-005.pt
```

For another checkout or Colab, use that environment's Python executable instead of the Windows .venv path. Follow the shared setup guide and reproduce the prepared dataset first. Resume requires matching training configuration, dataset manifest and canonical source hashes, plus the original best.pt. Epoch checkpoints retain optimizer, model, early-stopping history, Python/NumPy/PyTorch random states and the training shuffle state. Reproducing a run on different hardware/framework versions is not guaranteed.

The [CPU preflight report](../reports/emotion/training/baseline-preflight-20261114.json) records one batch of 256 real training images, one validation batch, finite loss/gradients, updated weights and identical logits after checkpoint reload. The synthetic training tests separately verify exact continuation after resume and checkpoint-selection ties. The smoke checkpoint stays in ignored artifacts/, is labelled epoch 0, and is not a trained final model.

Full training writes epoch checkpoints and best.pt under artifacts/checkpoints/<run_id>/, with aggregate epoch history and selection metadata at reports/runs/<run_id>/manifest.json. Both training and validation losses are sample-weighted. The training command never evaluates the test split. Final test evaluation, a selected trained model, webcam inference and learning-curve plots belong to the next training/evaluation work; none is claimed by this preflight.
