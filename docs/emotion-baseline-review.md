# Emotion baseline review package

Owner: chị Dung. Independent reviewer: Vỹ. Status: submitted for EMO-04.3 review; not independently accepted. No final test or webcam results are claimed.

## Files

| Item | Location |
| --- | --- |
| Training code/configuration | [Training commit e161192](https://github.com/Quynh-Trang12/facial-attendance/tree/e161192dfbe92b35fecb84a8a2106d63cf2b11e3), [configuration](https://github.com/Quynh-Trang12/facial-attendance/blob/e161192dfbe92b35fecb84a8a2106d63cf2b11e3/configs/emotion-training.toml), [baseline plan](emotion-experiment-plan.md) |
| Full run manifest and epoch history | [manifest.json](../reports/emotion/training/20261114T180653198248Z-emotion-cnn-e161192-s42/manifest.json) |
| Learning curves | [learning-curves.png](../reports/emotion/training/20261114T180653198248Z-emotion-cnn-e161192-s42/learning-curves.png) |
| Selected validation confusion matrix | [validation-confusion-matrix.png](../reports/emotion/training/20261114T180653198248Z-emotion-cnn-e161192-s42/validation-confusion-matrix.png) |
| Checkpoint and original run files | [Drive run folder](https://drive.google.com/drive/folders/15s3xeG4bdCjoF2xb7UG0TrLX-6MoVFgF), [best.pt](https://drive.google.com/file/d/1KGD6saUwDZ6-ouJxWGVwUeai8_VeAKL6/view) |
| Owner checkpoint integrity check | [handover-check.json](../reports/emotion/training/20261114T180653198248Z-emotion-cnn-e161192-s42/handover-check.json) |

The Drive folder is private at handover preparation. The owner must grant Vỹ access using his Google account before checkpoint download. Checkpoint binaries and raw face images stay outside Git. The manifest contains the effective configuration as well as links to its exact source commit.

## Run and results

Run ID: `20261114T180653198248Z-emotion-cnn-e161192-s42`.

Python 3.13.16, PyTorch 2.13.0+cu126, Tesla T4. Three convolution blocks (32/64/128 channels), trained from scratch with Adam, learning rate 0.001, unweighted cross-entropy, batch 256, no augmentation and seed 42. Dataset: 19,573 train / 4,891 validation / 6,283 test. IDs: Negative=0, Neutral=1, Positive=2; Surprise excluded.

Training ended at epoch 14 after five epochs without checkpoint improvement. Epoch 9 has the highest validation macro F1 and is the selected checkpoint. Epoch 13 has lower validation loss but lower macro F1, so it is not selected. The manifest records a clean training checkout and source hashes matching e161192.

| Validation measure | Epoch 9 |
| --- | ---: |
| Images | 4,891 |
| Correct | 3,067 |
| Accuracy | 0.6270701288 |
| Macro F1 | 0.5920539289 |
| Cross-entropy loss | 0.8323566582 |

| Class | Precision | Recall | F1 | Support |
| --- | ---: | ---: | ---: | ---: |
| Negative | 0.734397 | 0.620785 | 0.672829 | 2,521 |
| Neutral | 0.461001 | 0.410788 | 0.434449 | 964 |
| Positive | 0.581799 | 0.786629 | 0.668884 | 1,406 |

Confusion matrix rows are true labels and columns are predicted labels, ordered Negative, Neutral, Positive. It is [[1565,392,564],[337,396,231],[229,71,1106]]. Neutral is the weakest class. Validation fluctuates substantially; these plots alone do not establish the cause or prove overfitting. Training metrics include dropout and changing weights, whereas validation uses evaluation mode.

Checkpoint SHA-256: `323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd`.

Dataset manifest SHA-256: `79490d419bc9f4c05042dbdd3fcf76921bffdd73d22f855cd565ce9631c8386b`.

## Reviewer procedure

1. Read the plan and effective configuration. Check that labels, preprocessing, batch size, trainable layers, loss, optimizer and selection rule agree with the run manifest.
2. Check that history has 14 completed epochs, that every train/validation count is correct, and that epoch 9 wins validation macro F1. Confirm the five following epochs did not improve the selection result.
3. Inspect both plots and per-class results. Describe observed errors without diagnosing their cause from the curves alone. Keep original FER2013 annotation, near-duplicate, identity-overlap and webcam limitations explicit.
4. Download best.pt from Drive into ignored artifacts/checkpoints/emotion-review/. Verify its SHA-256 before loading. In a checkout of e161192, use the project Python 3.13 environment. The following example verifies metadata and reproduces validation without touching test images:

```python
from pathlib import Path
import hashlib
import json
import sys

root = Path.cwd()
sys.path.insert(0, str(root / 'src'))
from facial_attendance.emotion.training import load_checkpoint, build_data, run_epoch

checkpoint = root / 'artifacts/checkpoints/emotion-review/best.pt'
expected = '323718881b67576a854fe714b2ba136c5aadbadc9b6676e9eb2cf16d2a86cfcd'
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected
model, _, saved = load_checkpoint(checkpoint, 'cpu')
assert saved['epoch'] == 9
assert saved['manifest_sha256'] == '79490d419bc9f4c05042dbdd3fcf76921bffdd73d22f855cd565ce9631c8386b'
layout, loaders = build_data(saved['config'], root)
metrics = run_epoch(model, loaders['val'], 'cpu')
print(json.dumps(metrics, indent=2))
```

Prepare the same dataset locally using [the dataset guide](emotion-3class.md). Only the validation loader is iterated above. CPU/GPU numerical differences can affect borderline predictions: record the actual device, versions and any metric difference; do not silently substitute the final epoch or tune on test data. The owner verified file hash/metadata and finite synthetic CPU inference, not an independent full validation reproduction on Vỹ's machine.

5. Record the reviewed commit, device/versions, commands, reproduced results and findings in EMO-04.3's review evidence. Training corrections belong to EMO-04.4. Do not confirm the review until its checks are performed. The test split remains untouched while configuration changes are considered.
