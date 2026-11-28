# Face approach comparison

This page fixes how the classification-based model (FAC-03) and the metric-learning model (FAC-04) are compared on face verification, before any official score is computed (FAC-05.1). It names the two runs, the conditions they must share, the score and its direction, and which data may select what. FAC-05.2 computes the numbers; FAC-05.3 to FAC-05.5 review and report them.

It builds on rules that already exist and does not repeat them:

| Source | What it fixes |
| --- | --- |
| [Face data protocol](https://github.com/Quynh-Trang12/facial-attendance/blob/7fd99d6645b5bc3e759b856323cb4f750d12402c/docs/face-data-protocol.md) (FAC-01) | Partition roles, pair rules, score direction |
| [Experiment settings](https://github.com/Quynh-Trang12/facial-attendance/blob/4725d1121cbb2fffa08960d58f89a8311012856e/docs/experiment-settings.md) (SYS-04) | Shared training settings, selection rules, disclosures |
| [Evaluation protocol](https://github.com/Quynh-Trang12/facial-attendance/blob/c6e4f6754a4cadd479c32f8774be1ecb81f2583f/docs/evaluation-protocol.md) and [evaluation scripts](https://github.com/Quynh-Trang12/facial-attendance/blob/2d14220e912acfbdda8f378e94f81c9bafbbf3a6/docs/evaluation-scripts.md) (SYS-06) | ROC/AUC calculation, threshold rule, evaluator input checks |
| [Face models](face-models.md), [FAC-03 evidence](../evidence/FAC-03/README.md), [FAC-04 evidence](../evidence/FAC-04/README.md) | Network, checkpoints and exports |

The links to unmerged documents point to fixed commits. If one of those rules changes before FAC-05.2 runs, this page is revised first.

## The two models

| | Classification (FAC-03) | Metric learning (FAC-04) |
| --- | --- | --- |
| `run_id` / `model_id` | `20261113-160440-classification-78122af-s42` | `20261114-125242-metric-74a8657-s42` |
| `model_version` (weights hash) | `3fd31562478d` | `cb38ea47ceff` |
| Selected epoch | 30 of 30 | 29 of 30 |
| Validation-pair cosine AUC at selection | 0.9884 | 0.9911 |
| Checkpoint | `MyDrive/fac-03-runs/<run_id>/best.pt` | `MyDrive/fac-04-runs/<run_id>/best.pt` |

These are the final models of each approach. No other checkpoint enters the comparison, and neither model is retrained or re-selected after the official trials are scored.

## Like-for-like conditions

The comparison measures the training approach, so everything else must match. Values come from the two committed manifests ([FAC-03](../evidence/FAC-03/20261113-160440-classification-78122af-s42/manifest.json), [FAC-04](../evidence/FAC-04/20261114-125242-metric-74a8657-s42/manifest.json)).

| Condition | Both runs | Status |
| --- | --- | --- |
| Network | `resnet18-3x3-stem-no-maxpool`, trained from scratch, all layers trainable, 11,431,488 embedding parameters. FAC-03's extra 2,052,000 are its 4,000-way head, which verification never uses. | Same |
| Input and embedding | 64×64 RGB, the same `eval_transform`, raw `float32[512]` from `embed()` | Same |
| Training data | Filtered `train_data`, 372,057 images, 4,000 identities; exclusion list `06d4251d…`; identity mapping `4ce1e173…` | Same |
| Training budget | 30 epochs, seed 42, about 1,453 batches of 256 images per epoch, same augmentation | Same |
| Epoch selection | Highest cosine AUC on the same validation pairs, `772e9983…` (3,806 same-identity and 3,806 different-identity pairs from filtered `val_data`) | Same |
| Official trials | `verification_pairs_val.txt`, `55ecf149…`, all 8,805 trials in file order | Same |
| Export | `scripts/export_face_embeddings.py` with the same exclusion list. The export code and the model, loader and exclusion code are unchanged between FAC-03's commit `78122af` and FAC-04's branch tip, so FAC-03's existing export may be reused. | Same; FAC-04 export still to run |
| Scoring | One evaluator commit for both models, `--embeddings` on raw vectors, `--mode validation` for the validation pairs and `--mode official` for the trials | Required in FAC-05.2 |
| Loss, head, sampler, learning rate | Cross-entropy with a head, shuffled images, LR 0.1 against triplet loss, P×K batches with semi-hard mining, LR 0.05 | Different on purpose: this is the approach being compared |

FAC-05.2 checks before scoring that each export names the expected `model_id` and `model_version`, that both NPZ files list exactly the paths of the trial file, and that the trial-file SHA-256 saved by the evaluator equals `55ecf149…`. A mismatch stops the comparison.

`export_face_embeddings.py` also writes `val_data`/`test_data` predictions, which needs an identity head. The metric model has none, so its export skips that step and records `"classification": null`; the embedding export itself is unchanged. Classification accuracy is reported for FAC-03 only, as SYS-06 already states.

## Verification score

**Measure.** The deployed and selecting score is cosine similarity of the two raw embeddings:

```text
cos(a, b) = a·b / (‖a‖ ‖b‖),  range −1 to 1
```

Higher means more likely the same person. A trial is accepted as the same person when `cos(a, b) ≥ τ`. Euclidean distance `‖a − b‖` on the same raw vectors is reported beside it. Lower distance means more likely the same person, so the evaluator passes `−‖a − b‖` to ROC/AUC. It is never used for selection or for the attendance decision.

**Why cosine.**

- It is the geometry FAC-04 was trained in. The triplet loss compares L2-normalized embeddings with `‖â − b̂‖² = 2 − 2·cos(a, b)` (Schroff et al., 2015), so cosine ranks pairs exactly as the training objective does. Raw Euclidean distance also depends on the vector length, which the loss never constrains.
- It is the standard comparison for softmax-trained face features. NormFace (Wang et al., 2017) showed that normalizing features before comparing them, which is cosine similarity, improves verification for networks trained with a softmax classifier like FAC-03. SphereFace (Liu et al., 2017), CosFace (Wang et al., 2018) and ArcFace (Deng et al., 2019) all define the identity decision by angle and verify with cosine similarity.
- One measure serves both models, so the comparison and the deployed rule use the same score. Cosine is bounded and ignores vector scale, so a threshold keeps its meaning across inputs.
- The training logs agree. On the validation pairs, cosine AUC at the selected epoch was 0.9884 against 0.9836 Euclidean for FAC-03, and 0.9911 against 0.9561 for FAC-04 (`history.json` of each run).

**AUC.** The area under the ROC curve is the probability that a randomly chosen same-person trial scores higher than a randomly chosen different-person trial (Hanley & McNeil, 1982). It needs no threshold and does not depend on the class balance (Fawcett, 2006). Label `1` is the positive class. An AUC of 0.5 is chance; an AUC below 0.5 usually means a distance was passed without negation.

## What each data set decides

| Decision | Rule | Data | Used for |
| --- | --- | --- | --- |
| Epoch per approach | Highest cosine AUC | Validation pairs | Done in FAC-03 and FAC-04 |
| Approach and deployed model | Higher cosine AUC of the two final models, as SYS-04 fixes | Validation pairs | Selection |
| Uncertainty of that choice | Paired bootstrap interval, below | Validation pairs | Reported beside the choice; does not change it |
| Threshold `τ` | Lowest observed cosine score whose false-accept rate is at most 1%, ties evaluated together, accept when score ≥ `τ` (SYS-06 rule, `--select-threshold`) | Validation pairs, selected model only | Attendance decision in the GUI |
| Reported verification result | ROC curve and AUC, cosine and Euclidean, four values in total | All 8,805 official trials, scored once per model | Evaluation only |
| Operating point on the trials | False-accept and true-accept rates at the fixed `τ`, read from the official ROC points | Official trials | Evaluation only; `τ` is not changed |
| Classification accuracy | Top-1, FAC-03 only | `test_data` | Evaluation only |

**Order.** FAC-05.2 works in this order, so the official trials cannot influence a choice:

1. Export the FAC-04 embeddings and check both exports as above.
2. Score both models on the validation pairs. Record both AUCs, the bootstrap interval, the selected approach and `τ` with its validation false-accept and true-accept rates.
3. Commit that record before step 4.
4. Score both models once on the official trials with `--mode official`.

[`scripts/compare_face_approaches.py`](../scripts/compare_face_approaches.py) runs steps 1–2 as `validation` and step 4 as `official`. It runs the evaluator unchanged from its own checkout at commit `2d14220` and refuses to continue if that commit, a trial-file hash, a model ID or an export's path list differs. The `official` step also needs the SHA-256 of the committed `selection.json` and refuses an export that changed after validation. Results are in [FAC-05 evidence](../evidence/FAC-05/README.md).

After step 4 only calculation errors may be fixed (FAC-05.4), and the fix must reproduce the step 2 choices unchanged. Any change of model, measure or `τ` after official scoring is disclosed with the step it changed, and the result is not called independent.

## Uncertainty of the difference

The two validation AUCs differ by 0.0027, so the report states how much of a difference sampling alone could explain. The interval is computed on the evaluator's saved pair scores and needs no model.

| Setting | Value |
| --- | --- |
| Method | Paired percentile bootstrap (Efron & Tibshirani, 1993). Each replicate draws trial indices with replacement and uses the **same** indices for both models, because both scored the same trials. |
| Statistic | `ΔAUC = AUC(metric) − AUC(classification)`, with the evaluator's AUC calculation |
| Replicates and seed | 2,000 replicates, seed 42 |
| Interval | 95 %, the 2.5th and 97.5th percentiles of `ΔAUC` |
| Where | Validation pairs, cosine. Official trials, cosine and Euclidean, reported only. |

If the interval excludes 0, the report may say one approach scored higher on these trials. If it includes 0, the report says the two are not clearly separated. In both cases the selection rule above still decides. DeLong's test (DeLong et al., 1988) is the analytic alternative for correlated ROC curves; the bootstrap is used because it needs no extra assumptions and reuses the evaluator's AUC directly.

Trials share images: the 8,805 official trials use 9,355 distinct images, and the 7,612 validation pairs use 7,777. Resampling trials treats them as independent, so the interval is likely too narrow. The report states this limit.

## Checking the recorded results

[`scripts/check_face_comparison.py`](../scripts/check_face_comparison.py) rechecks `selection.json` and `official-summary.json` from the evaluator's saved outputs alone, so a reviewer needs neither the models nor the evaluator. For each step it checks:

| Check | How |
| --- | --- |
| The outputs are the recorded ones | SHA-256 of every `pair_scores.csv` and `metrics.json` equals the summary |
| Right evaluator, mode, trial file and model | `metrics.json` names commit `2d14220` with a clean tree, the step's mode and trial hash, and the expected `run_id` and `model_version` |
| Score direction | Cosine lies in [−1, 1]; the Euclidean AUC input is exactly minus the distance; every AUC is above 0.5 |
| AUC | Recomputed with the rank (Mann-Whitney) formula, ties counted one half, against both the evaluator and the summary |
| Same trials | Both approaches scored the same trials in the same order |
| Bootstrap | The paired bootstrap reproduces with the rank-formula AUC |
| Selection | The selected approach has the higher validation cosine AUC |
| Threshold | `τ` is the lowest observed validation cosine with a false-accept rate of at most 1 %, ties moving together, and the next lower score exceeds 1 % |
| Order | `official-summary.json` used the committed `selection.json`, which is in the commit the official run names, with an unchanged decision |
| Operating point | The counts at `τ` on the official trials, and the evaluator's ROC point there |

```bash
python scripts/check_face_comparison.py --selection evidence/FAC-05/selection.json \
    --validation-dir <validation run> --official-summary evidence/FAC-05/official-summary.json \
    --official-dir <official run> --output checks.json
```

Each check prints PASS or FAIL and the exit code is 1 on any failure. The run directories are the ones under `MyDrive/fac-05-runs/` that the summaries name.

## What the report must disclose

The disclosures in SYS-04 and SYS-06 apply unchanged. For this comparison they mean:

- The validation pairs contain only training identities. Their AUC selects the model, and `τ` chosen on them may be too permissive for unseen people. SYS-07.6 checks `τ` on the webcam.
- The official trials are not shown to be identity-disjoint from training, so their AUC is not a result on unseen people.
- `val_data` is used for selection by the metric-learning model too, although the brief says metric learning does not need it, because the official trials must stay out of selection.
- The Euclidean AUC is measured on raw vectors. It is lower for FAC-04 partly because the triplet loss never trained vector length, so it does not show that FAC-04's embedding space is worse.
- Every number and plot names its `run_id`, and pair scores are kept so ROC and AUC can be recomputed without the models.

## References

- DeLong, E. R., DeLong, D. M., & Clarke-Pearson, D. L. (1988). Comparing the areas under two or more correlated receiver operating characteristic curves: A nonparametric approach. *Biometrics, 44*(3), 837–845.
- Deng, J., Guo, J., Xue, N., & Zafeiriou, S. (2019). ArcFace: Additive angular margin loss for deep face recognition. *CVPR*.
- Efron, B., & Tibshirani, R. J. (1993). *An introduction to the bootstrap*. Chapman & Hall.
- Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874.
- Hanley, J. A., & McNeil, B. J. (1982). The meaning and use of the area under a receiver operating characteristic (ROC) curve. *Radiology, 143*(1), 29–36.
- Liu, W., Wen, Y., Yu, Z., Li, M., Raj, B., & Song, L. (2017). SphereFace: Deep hypersphere embedding for face recognition. *CVPR*.
- Schroff, F., Kalenichenko, D., & Philbin, J. (2015). FaceNet: A unified embedding for face recognition and clustering. *CVPR*.
- Wang, F., Xiang, X., Cheng, J., & Yuille, A. L. (2017). NormFace: L2 hypersphere embedding for face verification. *ACM Multimedia*.
- Wang, H., Wang, Y., Zhou, Z., Ji, X., Gong, D., Zhou, J., Li, Z., & Liu, W. (2018). CosFace: Large margin cosine loss for deep face recognition. *CVPR*.
