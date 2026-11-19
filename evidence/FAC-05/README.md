# FAC-05 — Face approach comparison

Task requirements, assignments and review records: [Project Tracker](../../README.md#project-tracker), task `FAC-05`. The comparison rules (the two models, the score, what each data set decides, the order of the steps) are fixed in [face comparison](../../docs/face-comparison.md) (FAC-05.1) and are not repeated here.

## FAC-05.2 — ROC/AUC, cosine and Euclidean scores

**Result.** The metric-learning model (FAC-04) is selected, scored by cosine similarity with threshold **τ = 0.4620**. On all 8,805 official trials its cosine AUC is **0.9687**, against 0.9618 for the classification model (FAC-03); Euclidean AUC is 0.9359 and 0.9445. At the fixed τ it accepts 76.8 % of same-person trials and 2.5 % of different-person trials.

**Status.** All three steps are done; FAC-05.3 to FAC-05.5 below recheck and record them. The selection ([`selection.json`](selection.json), SHA-256 `7255986a…`) was committed in `cddc070` before the official trials were scored, and [`official-summary.json`](official-summary.json) confirms the same selection, exports and threshold.

### How it is run

The team notebook `fac-05-2-compare-approaches.ipynb` runs on Colab, where both checkpoints and the FAC-03 export are in Drive.

| Step | What runs | Output |
| --- | --- | --- |
| 1. Export | `scripts/export_face_embeddings.py` on FAC-04's `best.pt`: raw embeddings of the official-trial images and of the validation-pair images. FAC-03's export is reused, as the comparison allows. | `MyDrive/fac-04-runs/<run_id>/exports/` |
| 2. Validation | `scripts/compare_face_approaches.py validation`: checks both exports, runs the SYS-06 evaluator (commit `2d14220`) on the validation pairs for both models, computes the paired bootstrap of the cosine AUC difference, selects the approach and its threshold | `selection.json`, committed here before step 3 |
| 3. Official | `scripts/compare_face_approaches.py official`: refuses unless `selection.json` matches the committed file and the exports are unchanged, then scores all 8,805 official trials once per model, cosine and Euclidean, and reads the fixed threshold's operating point | `official-summary.json` |

The evaluator's full outputs (`metrics.json` with every ROC point, `pair_scores.csv`, `roc_*.csv`) stay in Drive under `MyDrive/fac-05-runs/`. This folder keeps the two summaries, which record the SHA-256 of every one of those files, and the ROC plots.

### Validation result and selection (steps 1–2)

Colab run of 18 Nov 2026 (session `20261118T025253Z`), T4, Python 3.13.16, torch 2.13.0+cu126, scikit-learn 1.9.1. Comparison code `190657c`, evaluator `2d14220`, both clean. Unit tests 137 of 137 passed ([log](logs/20261118T025253Z-unit-tests.log)).

**Exports.** FAC-04's `best.pt` was exported for the 9,355 official-trial images and the 7,777 validation-pair images, without class predictions ([summary](fac04-export-summary.json), [log](logs/20261118T025253Z-export-fac04.log)). Both exports passed the checks before scoring: the expected `model_id` and `model_version`, and exactly the image paths of each trial file. The SHA-256 of all four NPZ files is in `selection.json`.

**Scores on the 7,612 validation pairs** (3,806 same-identity, 3,806 different-identity):

| Approach | `run_id` | Cosine AUC | Euclidean AUC | ROC |
| --- | --- | --- | --- | --- |
| Classification (FAC-03) | `20261113-160440-classification-78122af-s42` | 0.9884 | 0.9836 | [plot](validation/roc-classification.png) |
| Metric learning (FAC-04) | `20261114-125242-metric-74a8657-s42` | **0.9911** | 0.9561 | [plot](validation/roc-metric.png) |

Both cosine AUCs equal the values the training runs logged at the selected epoch (0.9884 and 0.9911), so the exports reproduce the selected checkpoints. The Euclidean AUC uses `−‖a − b‖` on the raw vectors, so higher is still better.

**Uncertainty.** Paired bootstrap, 2,000 replicates, seed 42: `ΔAUC = AUC(metric) − AUC(classification)` = 0.0027, 95 % interval [0.0014, 0.0040]. The interval excludes 0, so the metric model scored higher on these pairs. The pairs share images and contain training identities, so the interval is likely too narrow and does not describe unseen people.

**Decision.**

| Item | Value |
| --- | --- |
| Selected approach | Metric learning, `20261114-125242-metric-74a8657-s42`, `model_version` `cb38ea47ceff` |
| Distance measure | Cosine similarity of the raw embeddings; accept as the same person when `cos(a, b) ≥ τ`. Euclidean distance is reported only. |
| Threshold `τ` | **0.4620** (0.4619967254593038), the lowest validation cosine score with a false-accept rate of at most 1 % |
| At `τ` on the validation pairs | False-accept rate 0.998 % (38 of 3,806), true-accept rate 87.97 % (3,348 of 3,806) |

The numbers above were checked again outside the notebook: the AUCs recomputed from the saved pair scores with a rank formula match to six decimals, and the next lower observed score would raise the false-accept rate to 1.03 %, so `τ` is the lowest one that meets the rule. For reference only, the classification model's own threshold by the same rule would be 0.4462, with a true-accept rate of 85.37 %.

### Official trials (step 3)

Colab run of 18 Nov 2026 (session `20261118T030458Z`), same runtime and versions. Comparison code `cddc070`, whose code is unchanged from `190657c` (only evidence was added); evaluator `2d14220`; both clean. Unit tests 137 of 137 passed ([log](logs/20261118T030458Z-unit-tests.log)); the run log is [here](logs/20261118T030458Z-official.log).

Before scoring, the script checked that `selection.json` has the committed SHA-256, that all four NPZ files have the hashes recorded at validation, and that the trial file is `verification_pairs_val.txt` (`55ecf149…`). The evaluator ran in `--mode official`, which requires exactly 8,805 trials and allows no threshold selection. Each model was scored once.

**Scores on the 8,805 official trials** (4,315 same-person, 4,490 different-person):

| Approach | Cosine AUC | Euclidean AUC | ROC |
| --- | --- | --- | --- |
| Classification (FAC-03) | 0.9618 | 0.9445 | [plot](official/roc-classification.png) |
| Metric learning (FAC-04), selected | **0.9687** | 0.9359 | [plot](official/roc-metric.png) |

Paired bootstrap of `AUC(metric) − AUC(classification)`, 2,000 replicates, seed 42, reported only:

| Score | Difference | 95 % interval |
| --- | --- | --- |
| Cosine | +0.0069 | [+0.0040, +0.0096] |
| Euclidean | −0.0085 | [−0.0138, −0.0030] |

Both intervals exclude 0. With cosine, the deployed measure, the metric model ranks the official trials better, as it did on the validation pairs. With raw Euclidean distance the order reverses; the triplet loss trained only the direction of the embeddings, not their length, so this does not show a worse embedding space.

**Operating point of the fixed τ = 0.4620** (metric model, cosine, accept when score ≥ τ):

| Data | False-accept rate | True-accept rate |
| --- | --- | --- |
| Validation pairs, where τ was chosen | 0.998 % (38 of 3,806) | 87.97 % (3,348 of 3,806) |
| Official trials | **2.54 %** (114 of 4,490) | **76.76 %** (3,312 of 4,315) |

The official rates were counted from the pair scores and match the evaluator's ROC point at τ exactly. τ was not changed after the official scoring.

### Checks outside the notebook

- The SHA-256 of every `metrics.json` and `pair_scores.csv` matches the summaries, and every evaluator output records commit `2d14220` with a clean tree.
- AUCs recomputed from the saved pair scores with a rank formula match the evaluator's to six decimals, for both models, both scores and both data sets.
- Both models scored the same trials in the same order.

### Limits

- The official false-accept rate at τ is 2.5 times the 1 % target. τ was chosen on validation pairs of training identities, which are easier to separate than the official trials (cosine AUC 0.9911 against 0.9687), so it is too permissive here, as the comparison rules warned. SYS-07.6 checks τ on the webcam.
- The official trials are not shown to be identity-disjoint from training, so their AUC is not a result on unseen people.
- Trials share images, so the bootstrap intervals are likely too narrow.
- `val_data` was used to select the metric model too, so the official trials stay out of every choice.

## FAC-05.3 — Recheck of the rankings and the threshold

Every recorded number was recomputed from the evaluator's saved outputs with [`check_face_comparison.py`](../../scripts/check_face_comparison.py), without the models or the evaluator; the checks are listed in [face comparison](../../docs/face-comparison.md#checking-the-recorded-results).

| Step | Inputs | Result |
| --- | --- | --- |
| Validation | `MyDrive/fac-05-runs/validation-20261118T025253Z/` | **17 of 17 pass**: hashes, evaluator and model identity, AUCs, bootstrap, the selected approach, and `τ` for both models (the next lower score gives a false-accept rate of 1.02 %) |
| Official | `MyDrive/fac-05-runs/official-20261118T030458Z/` | **18 of 18 pass**; `selection.json` is in `cddc070`, the commit the official run names, and the decision is unchanged |

Run of 19 Nov 2026 on a Mac, Python 3.13.0, numpy 2.2.6, check script at `b548aae` (the only uncommitted change was this README): [log](logs/20261119T055311Z-recheck.log), [all 35 checks](recheck/checks.json).

Points a reviewer may ask about; none is a calculation error:

- The official operating point is read from the ROC point with threshold 0.46206, not 0.46200. That point is the lowest official score at or above `τ`, so it accepts exactly the same trials; `τ` itself is unchanged.
- The metric model's raw embeddings are short: the median official-pair Euclidean distance is 0.21, against 69.3 for the classification model. The triplet loss trained only the direction, so raw Euclidean distance ranks pairs worse for this model, while cosine does not depend on the length. This is why the Euclidean ranking of the two models reverses.

## FAC-05.4 — Corrections

The recheck found no calculation error, so no result was regenerated and `selection.json`, `official-summary.json` and the ROC plots are unchanged. The reviewer verifies the comparison or records that it passed without changes.

## FAC-05.5 — Selected model and limitations

**Selection.** The metric-learning model `20261114-125242-metric-74a8657-s42` (`model_version` `cb38ea47ceff`), scored by cosine similarity of L2-normalized embeddings, accepting the same person when the score is at least **τ = 0.4620**. The checkpoint, preprocessing and decision rule for the GUI are in [face models](../../docs/face-models.md#deployed-model).

**Reasons, from the recorded results.**

- It has the higher validation-pair cosine AUC, the rule fixed before scoring: 0.9911 against 0.9884, paired bootstrap difference +0.0027 [+0.0014, +0.0040].
- The official trials, scored once after the selection was committed, rank the models the same way: cosine AUC 0.9687 against 0.9618, difference +0.0069 [+0.0040, +0.0096].
- Cosine is the geometry the triplet loss trains and the standard measure for softmax-trained face features, and it ignores the vector length, which the metric model never learned. Its cosine AUC is higher than its Euclidean AUC on both data sets.
- `τ` follows the SYS-06 rule on the validation pairs: false-accept rate 0.998 %, true-accept rate 87.97 %.

**Deployable checkpoint.** `deploy.pt` was written from the selected `best.pt` with [`set_face_threshold.py`](../../scripts/set_face_threshold.py) ([log](logs/20261119T055311Z-set-threshold.log)) and stored beside it in `MyDrive/fac-04-runs/20261114-125242-metric-74a8657-s42/`.

| File | SHA-256 | `meta` |
| --- | --- | --- |
| `best.pt` (source, unchanged) | `bc450283…dcbc4f` | `threshold` empty |
| `deploy.pt` | `76f12338…08145b` | `threshold` 0.4619967254593038, `threshold_source` naming the rule, the validation-pair hash and `selection.json` |

Both files reload with `model_version` `cb38ea47ceff` and give identical embeddings; only the two threshold fields differ.

**Limitations.**

- **Too permissive on harder data.** On the official trials `τ` accepts 2.54 % of different-person pairs, 2.5 times the 1 % target, and 76.8 % of same-person pairs. `τ` was not tuned on the official trials, because they would then no longer be an evaluation. SYS-07.6 checks `τ` on the webcam; if it must be raised, the new value comes from a stated rule on non-evaluation data and is disclosed.
- **One-to-many matching multiplies false accepts.** The GUI compares a face with every enrolled person. If the comparisons were independent, an unknown person would match someone with probability about `1 − (1 − FAR)^N`: about 10 % for 4 enrolled people and 23 % for 10 at the official rate, or 4 % and 10 % at 1 %. The duplicate guard at enrollment uses the same `τ`.
- **Pairs, not templates.** `τ` was chosen on single-image pairs; the GUI compares a live frame with an averaged template of several frames, whose score distribution was not measured.
- **Domain shift.** Both models saw only 64×64 dataset crops. Webcam crops differ in detector, alignment, lighting and camera.
- **Identities and dependence.** The validation pairs contain training identities and the official trials are not shown to be identity-disjoint, so neither AUC describes unseen people. Trials share images, so the bootstrap intervals are likely too narrow.
- **One run per approach.** Each approach was trained once, with seed 42. The bootstrap covers trial sampling only, not run-to-run variation, and the AUC difference is small (0.7 points on the official trials), so the report says the metric model scored higher here, not that metric learning is better in general.
