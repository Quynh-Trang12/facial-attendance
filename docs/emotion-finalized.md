# Final emotion model: Sad / Happy / Neutral

Owner: Dung. Independent reviewer: Vy. This is the canonical final configuration on Emotion-finalized. Older Positive/Negative/Neutral, six-class and 48x48 baseline documents are historical and do not describe this checkpoint. Earlier reviewer passes do not automatically approve the changed dataset, labels or model.

## Evidence
- [Data and loader notebook](../notebooks/emotion_final_data.ipynb): recorded counts, archive checks and runnable train-only augmentation examples.
- [Training reproduction notebook](../notebooks/emotion_final_training.ipynb): readable GPU runner; creates a new run; requires original Drive sample-manifest records.
- [Executed final results](../notebooks/emotion_final_results.ipynb): actual saved learning history, validation and held-out reports.
- [Training configuration](../reports/emotion/final/training-config.json), [dataset record](../reports/emotion/final/dataset-info.json), [held-out report](../reports/emotion/final/held-out-test.json).
- [Integration contract](emotion-final-integration.md) and [review checklist](emotion-final-review.md).

## Dataset and labels
FER2013 source: https://www.kaggle.com/datasets/msambare/fer2013 . Use original Sad=0, Happy=1, Neutral=2. Other expressions are excluded; no manual relabeling was applied. Preparation removed exact duplicate copies, exact training/test overlap and conflicting duplicate labels. The fixed 20% per-source-class validation selection uses seed 42. The later geometry review removed 179 severe crops and 284 extreme profiles from the three-class subset; remaining slight turns and borderline cases were retained. These were subjective visual decisions, saved in manifests, not a guaranteed clean dataset.

| Split | Sad | Happy | Neutral | Total |
|---|---:|---:|---:|---:|
| Train | 3581 | 5541 | 3781 | 12903 |
| Validation | 890 | 1384 | 943 | 3217 |
| Test | 1186 | 1749 | 1198 | 4133 |

Raw 48x48 grayscale images are preserved. Model input is grayscale repeated to RGB channels, 224x224 bilinear resize, float [0,1], ImageNet mean [0.485,0.456,0.406] / std [0.229,0.224,0.225]. Horizontal flip p=0.5 is training-only. Labels are integer IDs, not alphabetical folder indices. SHA256 checks verify bytes and membership, not expression correctness. Full manifests, visual-review records and datasets remain external, not committed. FER2013 source rights remain those of the provider; public availability does not establish unrestricted redistribution rights.

## Training and selection
ImageNet-initialized ResNet-18; Sequential(Dropout(0.3), Linear(512,3)) head. Adam, unweighted cross-entropy, weight decay 0.0001, seed 42. Three classifier-only epochs at LR 0.001; seven layer4+classifier epochs at LR 0.0001/0.0003. Earlier layers and BatchNorm running statistics frozen. Micro-batch 32, accumulation 8 (effective 256 for full groups). Final partial groups normalized by actual sample count. Colab Tesla T4 training, no mixed precision. Select by validation Neutral F1, ties: macro F1, lower loss, earlier epoch. Selected epoch 10 was frozen before test scoring.

## Results
Validation: accuracy 77.49%, macro F1 75.08%, Neutral F1 69.97%. Held-out test: accuracy 77.57%, macro F1 75.20%, Neutral F1 70.34%. CPU checkpoint reload reproduced all 3217 saved validation predictions before testing. Test used every retained test image with no subsequent quality removals or relabeling. This is a quality-filtered FER2013 result, not the original seven-class benchmark or a measured webcam accuracy.

To reproduce metrics with the frozen checkpoint (no model selection), run:
```text
python scripts/emotion_final/evaluate.py --checkpoint artifacts/checkpoints/emotion-final/best.pt --data-root PATH_TO_EXTRACTED_DATA --split test --output artifacts/emotion-final-test-recheck
```
Use a new output folder. Per-image predictions are local artifacts, not Git evidence.

## External artifacts
- [Tested best.pt](https://drive.google.com/file/d/1s5wockz4DTkKWMspzrOVTEbsaxK2zyY0/view). SHA256: `0dcecbdbc7195bfb8cfd4529e0aa287d9b40c43da323311ad3803d687fcbfd24`.
- [Approved dataset ZIP](https://drive.google.com/file/d/1Anse0tdcSrXPJdd1QP-voeYksrA-G3sv/view). SHA256: `f94320ce10a3403722d4b693c0ca0f32786ad41f0683a31bc1a619790e3cd039`.
- [Original run and detailed records](https://drive.google.com/drive/folders/1Q-BDoMvHr3yFwm3ZDRmbxjIy13KbjnuP).
These existing links do not grant access automatically; owners must share access through the team. No new upload or sharing change is claimed. Checkpoints remain outside Git under the repository policy. The checkpoint's original training config says test_accessed=false; the later test record is separate and its weights were not modified.

## Limitations and current status
One final seed, unequal class sizes, low-resolution images and known content/annotation errors. Identity and near-duplicate separation are not established. Geometry screening included the test images before model scoring. Cleaning changes validation membership, so historical before/after scores are not a controlled comparison. The owner reported useful predictions in one informal webcam trial; independent multi-user evaluation is pending. Expressions do not establish actual feelings and must not determine attendance. Prepared evidence is not independent reviewer approval or shared-GUI integration completion.
