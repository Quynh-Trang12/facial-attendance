# SYS-06.3 review guide

Owner: chị Dung. Assigned independent reviewer: Quế An.

Status: owner precheck passed; independent review pending. This guide supports review of [implementation 2d14220](https://github.com/Quynh-Trang12/facial-attendance/commit/2d14220e912acfbdda8f378e94f81c9bafbbf3a6). It does not record acceptance on the reviewer's behalf.

## Evidence to inspect

- [Evaluation protocol](evaluation-protocol.md): metrics, selection versus final-test data and limitations.
- [Commands and input contracts](evaluation-scripts.md).
- [Metrics implementation](../src/facial_attendance/evaluation/metrics.py), [CLI](../scripts/evaluate.py), [known-answer tests](../tests/test_evaluation.py).
- [Synthetic inputs](../tests/fixtures/evaluation) and [saved outputs](../reports/evaluation/sys06-fixtures).
- [Owner's manual calculation check](../reports/evaluation/sys06-owner-precheck.json).
- [GitHub CI run](https://github.com/Quynh-Trang12/facial-attendance/actions/runs/43041595263): source, Windows and macOS jobs passed for implementation 2d14220.

## Reviewer checks

| Check | Expected behavior |
| --- | --- |
| Trial labels and alignment | 1=same, 0=different. Saved score paths, labels and order match every supplied trial. Missing/extra/reordered score rows fail. Official mode requires 8,805 rows. |
| Score direction | Cosine is unchanged; Euclidean distance is negated for ROC. Euclidean uses raw embeddings, before L2 normalization. |
| AUC calculation | Classification fixture: 0.75 for both measures; metric fixture: 1.0. Tied scores: 0.5; fully inverted example: 0. |
| Threshold selection | Validation only; accept on score >= threshold. Ties are evaluated together. An unusable operating point produces a documented reject-all threshold, not selection on test. |
| Emotion and face classification | Emotion accuracy=2/3 and macro F1=5/9, with all three classes included. Face fixture top-1=2/3. Confusion rows=true, columns=predicted. |
| Liveness failures | 6 attempts remain in totals, including 2 failures. Four attempts have scores. Spoof acceptance=1/3; genuine-user blocking=2/3. No invented liveness threshold. |
| Data reuse and limitations | Final test/trials do not select checkpoints or thresholds. Face filtering's full-data audit remains pending in the referenced protocol; do not claim unseen identities or independent real-model accuracy from these fixtures. |

For the classification fixture there are two positive and two negative trials: four positive-negative comparisons. The positive score wins three comparisons and loses one, giving AUC=3/4. Negated distances have the same ordering. For the metric fixture all four comparisons win, giving AUC=1. These results were recomputed directly using pairwise comparisons, separately from sklearn's ROC calculation.

## Run the review

Use the project Python 3.13 environment from the repository root:

```powershell
$env:PYTHONPATH = 'src'
.venv/Scripts/python.exe -m unittest discover -s tests -p test_evaluation.py -v
```

Then reproduce the five commands in the scripts guide, using new output directories. Compare metrics.json and numeric ROC CSV files with the known answers above. Check that invalid/misaligned inputs fail and that existing output folders are not overwritten. Actual image inference and the full official trial run require the corresponding trained models and data; they are not claimed here.

Record the reviewed commit, commands/runtime, findings and review outcome in SYS-06.3's review-evidence cell. If a problem is found, identify the input, expected result and observed result so the owner can correct it under SYS-06.4. If the checks pass, record that result without inventing corrections. FAC-02/LIV-03/EMO-03 dependency evidence must be considered before final acceptance of integration/data claims.
