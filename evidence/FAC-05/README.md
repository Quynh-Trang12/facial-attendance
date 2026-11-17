# FAC-05 — Face approach comparison

Task requirements, assignments and review records: [Project Tracker](../../README.md#project-tracker), task `FAC-05`. The comparison rules (the two models, the score, what each data set decides, the order of the steps) are fixed in [face comparison](../../docs/face-comparison.md) (FAC-05.1) and are not repeated here.

## FAC-05.2 — ROC/AUC, cosine and Euclidean scores

**Status.** The scoring code is ready and tested; the Colab runs on the real exports are pending. The results below are filled in from those runs.

### How it is run

The team notebook `fac-05-2-compare-approaches.ipynb` runs on Colab, where both checkpoints and the FAC-03 export are in Drive.

| Step | What runs | Output |
| --- | --- | --- |
| 1. Export | `scripts/export_face_embeddings.py` on FAC-04's `best.pt`: raw embeddings of the official-trial images and of the validation-pair images. FAC-03's export is reused, as the comparison allows. | `MyDrive/fac-04-runs/<run_id>/exports/` |
| 2. Validation | `scripts/compare_face_approaches.py validation`: checks both exports, runs the SYS-06 evaluator (commit `2d14220`) on the validation pairs for both models, computes the paired bootstrap of the cosine AUC difference, selects the approach and its threshold | `selection.json`, committed here before step 3 |
| 3. Official | `scripts/compare_face_approaches.py official`: refuses unless `selection.json` matches the committed file and the exports are unchanged, then scores all 8,805 official trials once per model, cosine and Euclidean, and reads the fixed threshold's operating point | `official-summary.json` |

The evaluator's full outputs (`metrics.json` with every ROC point, `pair_scores.csv`, `roc_*.csv`) stay in Drive under `MyDrive/fac-05-runs/`. This folder keeps the two summaries, which record the SHA-256 of every one of those files, and the ROC plots.
