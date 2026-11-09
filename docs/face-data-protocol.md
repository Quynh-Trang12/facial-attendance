# Face data protocol

This protocol defines how the assigned [11-785 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation) data is used by both face approaches. The data serves three separate purposes: **training** uses `train_data`, **classification evaluation** uses `val_data` and `test_data`, and the **required verification result** uses the supplied trial file. Download steps and loader behavior are in [data loading](data-loading.md).

## Partitions by approach

| Partition | Classification-based | Metric learning |
| --- | --- | --- |
| `classification_data/train_data` | Trains the identity classifier | Source of all training pairs and triplets |
| `classification_data/val_data` | Top-1 accuracy during training; seeded validation pairs for selection | Not required by the brief; seeded validation pairs for selection |
| `classification_data/test_data` | Final top-1 accuracy, once per final model | Not used |
| `verification_data/` + `verification_pairs_val.txt` | ROC and AUC once per final model; never training or selection | ROC and AUC once per final model; never training or selection |

Validation pairs are a team design choice under SYS-04 and step 1 of the team's data plan, not a requirement of the brief. They keep the supplied trial file out of epoch, model and threshold selection.

## Trial file

Each line of `verification_pairs_val.txt` is `left_path right_path label`. The first two columns are image paths. Label `1` means the same identity and `0` means different identities. Trials are read in file order and are never resampled, filtered or relabelled. Verification AUC uses exactly this file and is computed in FAC-05.

## Identity relationships

Identity is the folder name under `train_data`. Target IDs come from the sorted `train_data` identities, and `val_data` and `test_data` reuse that mapping.

- **Positive pair:** two different image files from one identity.
- **Negative pair:** one image file from each of two different identities.
- **Triplet:** the anchor and positive are two different image files from one identity, and the negative is from another identity.

## Sampling

Training pairs and triplets come from `train_data` only, with an explicit recorded seed. A file is never paired with itself. The baseline sampler is described in [data loading](data-loading.md). The metric-learning loss, and therefore pairs or triplets, is chosen in SYS-04. Validation pairs are built once from `val_data` with a recorded seed and reused unchanged.

## Keeping partitions separate

One image belongs to one partition only. The same identity appearing in `train_data`, `val_data` and `test_data` is intentional, because closed-set classification needs it. FAC-02 checks this rule on the full data.

## Checks required

- Every listed path resolves inside its own partition, and every image decodes.
- `val_data` and `test_data` identities exist in `train_data` and use its mapping.
- Content hashes find identical images across partitions, across identities and inside one identity.
- Trial lines have exactly two paths and a `0` or `1` label, with no self-pairs, repeated trials or conflicting labels.
- A seeded sample of paths, identities and labels is inspected by hand against the source folders and trial file.

## Sample records

These records come from the seed-42 spot-check of the full-data audit. Check them against the source folders and the trial file.

| `train_data` path | Identity | Target ID |
| --- | --- | --- |
| `classification_data/train_data/n008163/0076_01.jpg` | `n008163` | 3515 |
| `classification_data/train_data/n001451/0197_01.jpg` | `n001451` | 610 |
| `classification_data/train_data/n000334/0064_01.jpg` | `n000334` | 139 |

| Left path | Right path | Label |
| --- | --- | --- |
| `verification_data/00056047.jpg` | `verification_data/00026947.jpg` | 1 |
| `verification_data/00018396.jpg` | `verification_data/00029906.jpg` | 1 |
| `verification_data/00020442.jpg` | `verification_data/00004467.jpg` | 1 |
| `verification_data/00060107.jpg` | `verification_data/00026353.jpg` | 0 |
| `verification_data/00041961.jpg` | `verification_data/00037084.jpg` | 0 |
| `verification_data/00037063.jpg` | `verification_data/00043734.jpg` | 0 |

## Score direction

Higher scores mean more likely the same person, and label `1` is the positive class. Cosine similarity is used unchanged. Euclidean distance is negated, as `-‖a − b‖`.

## Known risk

Some identical images exist in more than one partition; the expected correction is to remove the duplicates from `train_data` while keeping the identity mapping and the trial file unchanged, and FAC-02 settles the `val_data`/verification overlap and records the result.

## Next tasks

| Task | Work |
| --- | --- |
| FAC-02 | Checks and loads the data; builds the samples |
| SYS-06 | Writes the evaluator |
| FAC-03 | Trains the classification-based approach |
| FAC-04 | Trains the metric-learning approach |
| FAC-05 | Computes ROC and AUC on the trial file |
