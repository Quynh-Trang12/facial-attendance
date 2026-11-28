# Evaluation protocol

Owner: chị Dung. Reviewer: Quế An. Date: 11 November 2026.
Scope: SYS-06.1 evaluation design. This file defines the calculations and data roles; it does not claim scripts, trained-model results or webcam tests have been completed. SYS-06.2 implements and checks the calculations on known examples.

## Metrics and data roles

| Task | Metrics | Selection data | Final evaluation data |
| --- | --- | --- | --- |
| Face verification: classification-based and metric-learning embeddings | ROC curve and AUC, separately for cosine similarity and Euclidean distance | One fixed, seeded pair list from classification_data/val_data | All 8,805 supplied trials in verification_pairs_val.txt, in original order |
| Face classification | Top-1 accuracy and number of evaluated images | Monitor val_data; checkpoint selection follows validation-pair AUC, not top-1 accuracy | classification_data/test_data, once for the selected classifier |
| Liveness | ROC/AUC, spoof acceptance rate, live rejection rate and confusion matrix; attack errors by family | Official validation subjects select checkpoint and decision policy | Official test subjects; separate physical webcam cases |
| Emotion | Accuracy, macro F1, per-class precision/recall/F1 and support, confusion matrix | Fixed prepared validation split selects checkpoint | Fixed prepared test split |

Training data fits weights and any learned preprocessing. Validation selects hyperparameters, checkpoints and thresholds. Lock those choices before final testing. If final-test results influence a change, disclose the reuse and do not call the revised result an untouched independent test. Do not apply random training augmentation during evaluation.

## Face verification and classification

- Read each official trial as left_path, right_path, label. Label 1 means same identity; label 0 means different. Preserve paths, labels and order. Missing images, invalid labels or non-finite scores are errors; never silently discard trials. The required ROC/AUC uses all 8,805 rows.
- ROC treats label 1 as positive and higher scores as stronger matches. Use cosine similarity directly and use negative Euclidean distance, -||a-b||, for distance scores.
- Compare both approaches using the same trials, preprocessing contract and scoring definitions. Compute cosine on the raw model embeddings (equivalently on their L2-normalized versions). Compute Euclidean distance on the raw embeddings before L2 normalization. Distances between normalized embeddings rank pairs identically to cosine, so they do not provide the comparison specified in SYS-04. Record which representation was scored.
- Follow SYS-04: select the epoch and face approach using highest cosine AUC on the fixed validation pairs. Build the validation list once with seed 42; keep and hash the actual list used after filtering. Never select on the official trials.
- Select the deployed cosine threshold on those validation pairs: among observed scores, choose the lowest threshold whose empirical false-accept rate is at most 1%, using accept when score >= threshold. Evaluate every tied score together. If no observed threshold qualifies, use a threshold strictly above the maximum score (reject all), and report that outcome. Save the threshold and measured validation rates with its checkpoint. This pairwise operating point is not a guarantee about gallery-wide or webcam false accepts.
- Report top-1 accuracy as correct predictions / evaluated images. Reuse the train-derived identity mapping. Report validation and final test separately, including retained and excluded counts; do not invent an identity-classification head for the metric-learning model.

### Face overlap and limitations

The raw-data audit found identical files across partitions. FAC-01.4 selected option C: filter classification partitions without moving or refilling images, preserve all official trials, and record shared-identity evidence. At the referenced revision, fixture checks pass but full-data filtering/audit results and reviewer confirmation are pending. Record the actual exclusion-list hash and measured partition counts for every evaluated run. Do not claim zero overlap before the full-data checks pass. Without verified filtering, disclose known overlap. Even zero file-hash overlap does not establish unseen identities; validation classification identities are shared with training.

## Liveness

Project labels are live=0 and spoof=1. Use p_spoof directly for ROC/AUC with spoof as the positive class. The score is not assumed calibrated.

At the locked decision policy, report:

- Spoof acceptance rate: spoof attempts returned as live / all spoof attempts.
- Live rejection rate: live attempts returned as spoof / all live attempts.
- Uncertain and capture/processing failure counts and rates separately, with explicit denominators. Also report genuine-user blocking as all live attempts that did not produce live / all live attempts. Do not hide failures by dropping them from operational results.

Keep a true-label versus predicted-status table, including uncertain/failure outcomes. Report scored-image ROC/AUC separately from operational outcomes and state how many samples had no usable score. Break down spoof results by print and screen; webcam results additionally distinguish screen_photo and video_replay. Each webcam attempt evaluates one captured image, not motion or averaged frame predictions.

Follow LIV-02: p_spoof >= t_spoof means spoof; below it, live requires the recorded quality/confidence checks, otherwise uncertain. Missing models or policy settings cannot authorize genuine attendance. Preserve official subject partitions and keep known related media together.

**Pending owner decision:** LIV-04 must specify the checkpoint-selection metric, threshold-selection objective, quality/confidence rules and any tie-breaking rule using validation data before final testing. No numeric threshold or optimal checkpoint is claimed here. Freeze and version the complete policy for test and webcam use.

## Emotion

Use the [three-class dataset protocol](emotion-3class.md) and [loader contract](emotion-loader.md). Labels: 0=Negative (angry, disgust, fear, sad), 1=Neutral, 2=Positive (happy); Surprise is excluded. Preserve the prepared 19,573 train / 4,891 validation / 6,283 test images. Manifest SHA-256: 79490d419bc9f4c05042dbdd3fcf76921bffdd73d22f855cd565ce9631c8386b.

Select the checkpoint with highest validation macro F1 across all three classes; break exact ties using lower validation cross-entropy, then earlier epoch. Macro F1 is the unweighted mean of the three class F1 scores. Use zero_division=0 for undefined precision/recall/F1, and list all three classes explicitly.

Predict the argmax of the three logits. Report maximum softmax probability as confidence without a calibration claim. There is no confidence threshold or uncertain class in this baseline; record threshold=null. Any later rejection policy needs a documented validation-only selection rule before final testing.

Report the selected model's final test metrics once, with confusion-matrix rows=true and columns=predicted in Negative, Neutral, Positive order. Original label errors, near-duplicate and identity overlap are not ruled out. These scores describe a three-class FER2013 task, not the original seven-class benchmark or a person's mood.

## Reproducible outputs

Save run ID, checkpoint ID/version/hash, code commit, effective configuration, preprocessing/label mapping, split and trial/exclusion hashes, counts, seed, dependency versions and selection source. Save final metrics and plots with the same run ID. Keep numeric scores/predictions so metrics can be recomputed; detailed image/identity records and raw face media stay outside Git. Use synthetic known-label examples to test scripts without exposing face data. Record all missing/invalid inputs and fail an official-trial evaluation rather than silently changing its denominator.

## Related specifications

- [SYS-04 face settings and selection](https://github.com/Quynh-Trang12/facial-attendance/blob/71f3b56fd7fdc6e90106ad6fdeae8e8b23749237/docs/experiment-settings.md)
- [FAC-01.4 filtering decision and pending evidence](https://github.com/Quynh-Trang12/facial-attendance/blob/7fd99d6645b5bc3e759b856323cb4f750d12402c/docs/face-data-leakage.md)
- [LIV-01 source and split protocol](https://github.com/Quynh-Trang12/facial-attendance/blob/ce6a635f8491a3b0f4fc0cff35bcf755b8b2cdff/docs/liveness-protocol.md)
- [LIV-02 decision policy](https://github.com/Quynh-Trang12/facial-attendance/blob/25eafa26091be2f9dcd6b17aeffb75d801448435/docs/liveness-gating.md)
