# FAC-04 — Metric-learning face embeddings

Task requirements, assignments and review records: [Project Tracker](../../README.md#project-tracker), task `FAC-04`. The shared network and checkpoint format are in [face models](../../docs/face-models.md) (FAC-03); data cleaning and the P×K sampler are [FAC-02](../FAC-02/README.md).

**Result.** The same ResNet-18 as FAC-03, trained from scratch for 30 epochs with triplet loss on the filtered `train_data` (372,057 images, 4,000 identities). Training did not collapse. Selected epoch 29: cosine AUC **0.9911** on the validation pairs, against 0.9884 for the FAC-03 classification model on the same pairs. On the two demo trials, the metric model separates the same-person and different-person pair more widely than FAC-03 (cosine 0.632 vs 0.102, against 0.649 vs 0.318).

## Loss

Triplet loss (FaceNet, Schroff et al., 2015) on L2-normalized embeddings, with squared Euclidean distance `d(a, b) = ‖â − b̂‖² = 2 − 2·cos(a, b)`:

```text
L = mean over anchor–positive pairs of max(0, d(a, p) − d(a, n) + 0.2)
```

Every same-identity pair in a batch is an anchor–positive pair. Its negative is mined in the batch with the semi-hard rule of `tfa.losses.TripletSemiHardLoss`: the closest negative still farther than the positive, or the farthest negative when none is. Semi-hard mining is used because taking the hardest negative from the start tends to collapse all embeddings to one point. The loss is in [`metric.py`](../../src/facial_attendance/recognition/metric.py).

## Run

| Item | Value |
| --- | --- |
| `run_id` / `model_id` | `20261114-125242-metric-74a8657-s42` |
| `model_version` | `cb38ea47ceff` |
| Code | commit `74a8657`, `git_dirty` false in both sessions |
| Runtime | Colab, Tesla T4, Python 3.13.16, torch 2.13.0+cu126; about 6–7 minutes per epoch |
| Sessions | Epochs 1–20 on 14 Nov (the runtime then disconnected), epochs 21–30 resumed from `last.pt` on 15 Nov |
| Settings | Exactly [`configs/face-metric.toml`](../../configs/face-metric.toml); `changes.json` is empty |
| Batches | 64 identities × 4 images = 256 images from `IdentityBatchSampler`, 1,453 batches per epoch |
| Optimiser | SGD, learning rate 0.05, momentum 0.9, weight decay 5e-4, cosine decay; mixed precision, seed 42 |
| Network | ResNet-18, 3×3 stride-1 stem, no max-pool, 64×64 input, 512-d embedding, no classification head; 11,431,488 parameters, all trainable, no pre-trained weights |
| Data | Exclusion list `06d4251d…` (FAC-02); 372,057 / 7,804 images in train / val |
| Checkpoint | `best.pt` and `last.pt` in the Drive of the Colab account that ran the notebook, `MyDrive/fac-04-runs/<run_id>/`; not in git |

## Criteria

| Criterion | Evidence |
| --- | --- |
| Train a CNN producing fixed-length embeddings with a documented metric-learning loss on `train_data` | [`manifest.json`](20261114-125242-metric-74a8657-s42/manifest.json) records the architecture, the loss text above, the full configuration and its hash, data hashes, both sessions and the checkpoint location. Per-epoch values are in [`history.json`](20261114-125242-metric-74a8657-s42/history.json). The model outputs `float32[N, 512]`. |
| Verify the positive/negative identity relationships the loss needs | Every batch is checked before the loss is computed: each identity has 2+ images, no anchor is its own positive, every positive shares the anchor's identity and no negative does. The check passed on all 43,590 batches (1,453 × 30) and 33,477,120 anchor–positive pairs; each epoch in `history.json` records `"passed on all 1453 batches"`. A failed check raises an error and stops training; [unit tests](../../tests/test_metric.py) confirm that a wrong positive, a negative of the same identity, an anchor used as its own positive, a one-identity batch and an identity with a single image are each rejected. |
| Save loss, configuration and checkpoint; two-image scoring for comparison with FAC-03 | Loss and configuration are in the manifest; checkpoints are in Drive. [`two-image-demo.json`](20261114-125242-metric-74a8657-s42/two-image-demo.json) scores the FAC-03 demo trials with FAC-03's `scripts/score_face_pair.py`, unchanged. See the table below. |

### Two-image demo

| Official trial | Label | FAC-04 cosine | FAC-03 cosine |
| --- | --- | --- | --- |
| `00007133.jpg` vs `00060449.jpg` (first same-person trial) | 1 | 0.632 | 0.649 |
| `00041961.jpg` vs `00044353.jpg` (first different-person trial) | 0 | 0.102 | 0.318 |

The gap between the two pairs is 0.530 for FAC-04 and 0.331 for FAC-03. Two pairs show that scoring works; they do not rank the models. No threshold is applied; FAC-05 sets it and reports ROC/AUC on all trials.

## Training curve

| Epoch | Learning rate | Train loss | Active triplets | Mean d(a, p) | Mean d(a, n) | Validation-pair AUC, cosine | Validation-pair AUC, Euclidean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.050 | 0.177 | 98.6 % | 0.751 | 0.774 | 0.906 | 0.887 |
| 2 | 0.050 | 0.150 | 92.6 % | 0.741 | 0.798 | 0.939 | 0.902 |
| 5 | 0.048 | 0.120 | 80.2 % | 0.737 | 0.847 | 0.954 | 0.906 |
| 10 | 0.040 | 0.103 | 71.8 % | 0.725 | 0.872 | 0.970 | 0.936 |
| 15 | 0.028 | 0.089 | 64.2 % | 0.719 | 0.901 | 0.973 | 0.940 |
| 20 | 0.015 | 0.074 | 55.4 % | 0.700 | 0.928 | 0.984 | 0.956 |
| 25 | 0.005 | 0.053 | 42.7 % | 0.660 | 0.959 | 0.989 | 0.957 |
| 29 | 0.0005 | 0.040 | 33.7 % | 0.628 | 0.985 | **0.9911** | 0.956 |
| 30 | 0.0001 | 0.039 | 32.7 % | 0.625 | 0.989 | 0.9911 | 0.955 |

"Active triplets" is the share of anchor–positive pairs whose loss is above zero. Distances are on the unit sphere, measured on the mined triplets in training batches. The mined negatives moved away from the anchors while the positives moved closer, so fewer triplets violated the margin. Epoch 29 was selected; epoch 30 is lower only in the seventh decimal place. The Euclidean AUC is lower because it is measured on the raw embeddings, while the loss trains their direction.

**Collapse check.** The plan required switching to a warm start from FAC-03 if, after epoch 2, the validation-pair AUC was ≤ 0.6 and the embedding spread was near 0. At epoch 2 the AUC was 0.939 and the spread (mean per-dimension standard deviation of the unit embeddings) was 0.042. That is close to 1/√512 ≈ 0.044, the value for embeddings spread evenly over the sphere. The run therefore trained from scratch, and no warm-start run was made.

The validation pairs are FAC-03's [file](../FAC-03/20261113-160440-classification-78122af-s42/val-pairs.txt): the copy rebuilt in this run has the same hash, `772e9983…`. That is why the two AUC values above compare like for like.

## Checks

- **Unit tests:** 125 of 125 passed on the T4 runtime in both sessions, none skipped ([log](20261114-125242-metric-74a8657-s42/logs/20261115T024339Z-unit-tests.log)).
- **Smoke run:** one 30-step epoch on the real data ran the whole path before the full run. It covered training, the identity check, validation AUC and selection, with 23,040 anchor–positive pairs as expected for 30 × 64 × 4 × 3 ([log](20261114-125242-metric-74a8657-s42/logs/20261114T124753Z-smoke.log), [summary](20261114-125242-metric-74a8657-s42/smoke-passed.json)).
- **Resume:** the second session continued at epoch 21 with the same settings, data and validation pairs; the manifest lists both sessions.
- **Demo images:** the notebook checked the trial file names before scoring; both demo logs show the model ID and version of `best.pt`.

## Limits

- The validation-pair AUC selects the epoch. The validation pairs contain identities seen in training, so it does not estimate verification on new people, and it is not the official result. FAC-05 reports ROC/AUC once on `verification_pairs_val.txt` with the shared evaluator.
- Official-trial images may show training identities (see FAC-02 limits), so that AUC is not a result on unseen people either.
- Augmentation after the resume is not bit-identical to an uninterrupted run; the data order per epoch is.
- The checkpoint is in the Drive of the Google account that ran the notebook. FAC-05 needs `best.pt` from `MyDrive/fac-04-runs/20261114-125242-metric-74a8657-s42/`.

## Re-run

Run the team notebook `fac-04-metric-training.ipynb` on a Colab GPU (T4, `KAGGLE_API_TOKEN` secret). It mounts Drive, runs a smoke test once, resumes an unfinished run automatically, scores the two demo trials and downloads these records.
