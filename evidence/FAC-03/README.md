# FAC-03 — Classification-based face embeddings

Task requirements, assignments and review records: [Project Tracker](../../README.md#project-tracker), task `FAC-03`. The network, training rules and checkpoint use are in [face models](../../docs/face-models.md); data cleaning is FAC-02.

**Result.** ResNet-18 trained from scratch for 30 epochs on the filtered `train_data` (372,057 images, 4,000 identities). Selected epoch 30: cosine AUC 0.9884 on the validation pairs, top-1 accuracy **85.0 %** on `val_data` and **85.9 %** on `test_data`.

## Run

| Item | Value |
| --- | --- |
| `run_id` / `model_id` | `20261113-160440-classification-78122af-s42` |
| `model_version` | `3fd31562478d` |
| Code | commit `78122af`, `git_dirty` false in both sessions |
| Runtime | Colab, Tesla T4, Python 3.13.16, torch 2.13.0+cu126; about 6 minutes per epoch |
| Sessions | Epochs 1–25 on 13–14 Nov (the runtime was then reclaimed), epochs 26–30 resumed from `last.pt` on 14 Nov |
| Settings | Exactly [`configs/face-classification.toml`](../../configs/face-classification.toml); `changes.json` is empty |
| Data | Exclusion list `06d4251d…` (FAC-02); 372,057 / 7,804 / 7,832 images in train / val / test |
| Checkpoint | `best.pt` and `last.pt` in the owner's Drive, `MyDrive/fac-03-runs/<run_id>/`; not in git |

## Criteria

| Criterion | Evidence |
| --- | --- |
| Train a CNN with an identity-classification head on `train_data`; save architecture, trainable layers, loss, hyperparameters and checkpoint | [`manifest.json`](20261113-160440-classification-78122af-s42/manifest.json) records the architecture (13,483,488 parameters, all trainable, no pre-trained weights), cross-entropy loss, the full configuration and its hash, data hashes and the checkpoint location. Per-epoch values are in [`history.json`](20261113-160440-classification-78122af-s42/history.json). |
| Fixed-length embeddings from the layer before softmax; classification accuracy on validation/test | `embed()` returns `float32[N, 512]` before the 4,000-way head. Top-1 on `best.pt`: 0.8498 on `val_data`, 0.8588 on `test_data` (manifest). |
| Two-image embedding scoring | [`two-image-demo.json`](20261113-160440-classification-78122af-s42/two-image-demo.json): the first same-person official trial scores cosine 0.649 (distance 50.4); the first different-person trial scores 0.318 (distance 70.3). No threshold is applied; FAC-05 sets it. |
| Model for the FAC-05 ROC/AUC comparison | [`export-summary.json`](20261113-160440-classification-78122af-s42/export-summary.json): raw embeddings of the 9,355 images in the 8,805 official trials and of the 7,777 images in the validation pairs, plus `val_data`/`test_data` prediction CSVs, all in the shared evaluator's input format, stored next to the checkpoint in Drive. |

## Training curve

| Epoch | Learning rate | Train top-1 | `val_data` top-1 | Validation-pair AUC, cosine | Validation-pair AUC, Euclidean |
| --- | --- | --- | --- | --- | --- |
| 1 | 0.100 | 0.017 | 0.021 | 0.884 | 0.889 |
| 5 | 0.096 | 0.418 | 0.287 | 0.947 | 0.926 |
| 10 | 0.079 | 0.571 | 0.431 | 0.959 | 0.939 |
| 15 | 0.055 | 0.668 | 0.502 | 0.969 | 0.961 |
| 20 | 0.030 | 0.771 | 0.671 | 0.976 | 0.973 |
| 25 | 0.010 | 0.889 | 0.773 | 0.984 | 0.984 |
| 30 | 0.0003 | 0.966 | 0.850 | **0.988** | 0.984 |

Cosine AUC kept rising, with small dips, as the learning rate decayed, so the last epoch was selected. The gap between train (96.6 %) and `val_data` (85.0 %) top-1 shows some overfitting at the end of the schedule; it did not lower the validation-pair AUC.

The [validation pairs](20261113-160440-classification-78122af-s42/val-pairs.txt) (`772e9983…`) are 3,806 same-identity and 3,806 distinct different-identity pairs from the filtered `val_data`, seed 42. FAC-04 and FAC-05 reuse this exact file.

## Checks

- **Unit tests:** 114 of 114 passed on the T4 runtime in both sessions, none skipped ([log](20261113-160440-classification-78122af-s42/logs/20261114T044327Z-unit-tests.log)).
- **Smoke run:** one 30-step epoch on the real data completed the whole path (training, validation AUC, selection, test accuracy) before the full run ([log](20261113-160440-classification-78122af-s42/logs/20261113T155659Z-smoke.log)).
- **Resume:** the second session continued at epoch 26 with the same settings, data and validation pairs; the manifest lists both sessions.
- **Accuracy recomputed:** top-1 recomputed from the exported CSVs gives 0.8494 (val) and 0.8587 (test). The manifest values (0.8498, 0.8588) come from mixed-precision evaluation during training, the CSVs from full precision; the net difference is 3 and 1 correctly classified images.

## Limits

- The validation-pair AUC selects the epoch. The validation pairs contain identities seen in training, so it does not estimate verification on new people, and it is not the official result. FAC-05 reports ROC/AUC once on `verification_pairs_val.txt` with the shared evaluator.
- Official-trial images may show training identities (see FAC-02 limits), so that AUC is not a result on unseen people either.
- Augmentation after the resume is not bit-identical to an uninterrupted run; the data order per epoch is.
- Prediction CSVs and embedding NPZ files stay in Drive, as the evaluator guide asks for real outputs.

## Re-run

Run the team notebook `fac-03-train-classification.ipynb` on a Colab GPU (T4, `KAGGLE_API_TOKEN` secret). It mounts Drive, resumes an unfinished run automatically, then exports and downloads these records.
