# Course requirements and coverage

This page maps the supplied [project brief](reference/project-brief.pdf) to repository components and recorded progress. The assigned face-recognition dataset is the [11-785 Fall 2020 Homework 2 Part 2 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation).

| Requirement | Implementation location | Current coverage |
| --- | --- | --- |
| CNN embeddings using classification-based and metric-learning approaches, with a verification-performance comparison | `recognition/` | Data/pair loaders verified on the assigned competition data; models not trained |
| ROC/AUC evaluation on `verification_pairs_val.txt` and investigation of similarity metrics | `recognition/` evaluation | Trial loader and all listed paths/labels validated; model evaluation pending |
| Trained and integrated anti-spoofing for real, printed and screen/replay faces | `liveness/`, `ui/` | [Full-dataset inventory audit and loader](liveness-loader.md); trained model and integration unverified |
| Emotion recognition trained on public data, with displayed emotion labels/icons | `emotion/`, `ui/` | Scaffold only |
| Real-time webcam, display registered identity, prompt unknown users and support registration | `ui/`, `recognition/` | Scaffold only |
| Versioned shared setup and a training/inference check | Dependencies, `environment.py`, setup guide, execution reports | Linux GPU execution and independent Windows 11 / Mac M1 CPU review recorded |
| Group report with Methodology, Results and Discussion, at most 12 pages | Later report deliverable | Not implemented in this checkout |
| Continuous-take demonstration, 5–10 minutes, balanced member participation with visible faces | Later video deliverable | Not implemented in this checkout |
| Individual contribution report, at most 5 pages; peer evaluation; individual innovation where applicable | Later individual deliverables | Separate individual deliverables |

## Methodological scope

- **Pretraining:** the brief excludes fully pretrained models used without further training; partial fine-tuning is allowed.
- **Data splits:** both embedding approaches use the assigned training data. Validation and test splits retain their designated roles.
- **Verification:** meaningful ROC/AUC comparison depends on consistent trial labels and score direction. Raw Euclidean distance indicates dissimilarity, whereas higher similarity scores indicate a stronger identity match.
- **Model selection:** the comparison protocol includes the model/threshold selection procedure and disclosure of any verification-trial reuse during selection.
- **Additional datasets:** liveness and emotion depend on appropriately labelled data; separate sources may be needed where the assigned face dataset lacks those labels.

## Evidence and limits

The assigned face archive was downloaded through KaggleHub and inspected by the FAC-01.2 loader. [Recorded counts and batch checks](data-loading.md#observed-evidence-and-acceptance) document dataset inspection; full image-integrity, duplicate and leakage audits and model accuracy remain unassessed.

The one-step synthetic environment check demonstrates infrastructure behavior. It does not establish model performance or completion of the application requirements above.

Task wording and owner/reviewer assignments come from the group's [Project Tracker](../README.md#project-tracker). Task status is maintained there; the repository retains the [independent review record](../evidence/SYS-03/README.md#sys-03-independent-review).
