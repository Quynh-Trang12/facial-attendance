# Face data protocol

This protocol fixes how the assigned [11-785 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation) data is split, paired and scored for both face-embedding approaches (FAC-01.1). Loading, sampling code and recorded counts are in [data loading](data-loading.md). Rules here apply to every training, selection and evaluation run; a change needs a revised protocol before results are compared.

## Partition roles

| Partition | Content (recorded counts) | Use | Never used for |
| --- | --- | --- | --- |
| `classification_data/train_data` | 380,638 images, 4,000 identities | Training for **both** approaches: classification head targets and metric-learning pairs/triplets | Selection or reported scores |
| `classification_data/val_data` | 8,000 images, the same 4,000 identities | Checkpoint, hyperparameter and verification-threshold selection for both approaches | Training |
| `classification_data/test_data` | 8,000 images, the same 4,000 identities | One final closed-set classification accuracy per submitted model | Training or any selection |
| `verification_pairs_val.txt` + `verification_data/` | 8,805 labelled trials: 4,315 same, 4,490 different | The required verification result: ROC curve and AUC for each approach | Training, checkpoint choice or threshold tuning |

Classification target IDs come from `train_data` alone; validation and test reuse that mapping and reject unknown identities. For selection, the classification model uses validation top-1 accuracy. Both approaches also report verification AUC on a fixed, seeded pair list built only from `val_data` (same-identity pairs within an identity folder, different-identity pairs drawn uniformly), so the two models are selected by the same measure. The attendance decision threshold is chosen on that validation pair list too. The official trials are then scored once per final model. If any run reuses the official trials for a selection step, the report must say which step and present that result separately, as SYS-04 requires.

## Pair rules

- **Official trials.** Each line is `left right label`, with `1` = same identity and `0` = different. Trials are read in file order, never resampled, filtered or relabelled, and every reported AUC uses all 8,805.
- **Training pairs and triplets.** These come from `train_data` only. The baseline is `sample_training_pairs` with an explicit `seed` (default 42) and an equal positive/negative count: identities are drawn uniformly, positives use two different files of one identity, negatives use two different identities, and pairs are drawn with replacement. Triplets follow the same rules: anchor and positive are two different files of one identity, and the negative comes from another identity. Hard-negative mining may replace uniform sampling later if it still draws only from `train_data` and records its seed.
- **Validation pairs.** These are built once with a recorded seed and reused unchanged for every checkpoint of both approaches. `sample_training_pairs` refuses held-out splits on purpose, so this list needs its own builder, written with the training code.

## Score direction

Every verification score is oriented so that **higher means more likely the same person**, with label `1` as the positive class for ROC/AUC.

| Measure | Score passed to ROC/AUC |
| --- | --- |
| Cosine similarity of the two embeddings | `cos(a, b)` unchanged |
| Euclidean distance | `-‖a − b‖` (negated) |

Passing a raw distance would invert the ROC curve and report `1 − AUC`. Thresholds are stored in the same orientation as the score: accept when `score ≥ threshold`.

## Data checks

The FAC-01.3 audit checks the full data before the protocol is relied on:

- Images per identity in each split, and that `val_data` and `test_data` identities are a subset of `train_data`.
- Exact duplicates by SHA-256 content hash: within one identity, across identities (possible label error), between `train_data` and `val_data`/`test_data`, and between every classification split and `verification_data`.
- Trial label balance, self-pairs, repeated trials, conflicting labels for the same pair, and the number of distinct images the trials use.
- That every image decodes, with a histogram of image sizes and colour modes.
- A seeded spot-check list of training paths with identities and trials with labels, for manual inspection.

Undecodable images, conflicting trial labels and duplicates across partitions fail the audit. Duplicates within one identity are warnings, because they can make a "positive" pair of identical bytes. If any are found, FAC-01.4 decides whether to remove them before sampling. The [full-data audit](../reports/data/colab-audit-2026-11-05.json) failed on cross-partition duplicates; the finding and the open decision are in [face data leakage](face-data-leakage.md).

## Known limits

- `val_data` and `test_data` hold the same 4,000 identities as training. Their scores measure closed-set recognition of seen people and are likely to be higher than results on unseen identities.
- Verification images have no identity labels, so overlap between trial identities and training identities cannot be measured. Only exact duplicate images across partitions can be detected.
- Do not open the split folders in the Jupyter/Colab file browser. A `.ipynb_checkpoints` folder becomes an extra identity, which the loader rejects as unknown or empty.
- Image size, crop and normalization for the models are not fixed here; they belong to the [embedding schema](embedding-schema.md) under SYS-02.
