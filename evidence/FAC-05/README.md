# FAC-05 — Face approach comparison

Task requirements, assignments and review records: [Project Tracker](../../README.md#project-tracker), task `FAC-05`. The comparison rules (the two models, the score, what each data set decides, the order of the steps) are fixed in [face comparison](../../docs/face-comparison.md) (FAC-05.1) and are not repeated here.

## FAC-05.2 — ROC/AUC, cosine and Euclidean scores

**Status.** Steps 1–2 are done and the selection is committed ([`selection.json`](selection.json), SHA-256 `7255986a…`). Step 3, the official trials, is pending.

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
