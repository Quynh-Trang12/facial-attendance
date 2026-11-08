# Course requirements and coverage

The governing source is the supplied [project brief](reference/project-brief.pdf). The assigned face-recognition source is the [11-785 Fall 2020 Homework 2 Part 2 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). Use that dataset; the environment task does not authorize substituting another one.

| Requirement | Implementation location | Current coverage |
| --- | --- | --- |
| Train CNN embeddings with both classification-based and metric-learning approaches and compare verification performance | `recognition/` | Data/pair loaders verified on the assigned competition data; models not trained |
| Compute ROC/AUC on `verification_pairs_val.txt`; research similarity metrics | `recognition/` evaluation | Trial loader and all listed paths/labels validated; model evaluation pending |
| Train and integrate anti-spoofing for real, printed and screen/replay faces | `liveness/`, `ui/` | [LIV-02 gating design](liveness-gating.md) ready for GUI-owner review; model and GUI remain scaffold only |
| Train emotion recognition on public data and display emotion labels/icons | `emotion/`, `ui/` | Scaffold only |
| Real-time webcam, display registered identity, prompt unknown users and support registration | `ui/`, `recognition/` | Scaffold only |
| Versioned shared setup and a training/inference check | Dependencies, `environment.py`, setup guide, execution reports | Owner GPU check implemented; independent review pending |
| Group report with Methodology, Results and Discussion, at most 12 pages | Later report deliverable | Not started here |
| Continuous-take demonstration, 5–10 minutes, balanced member participation with visible faces | Later video deliverable | Not started here |
| Individual contribution report, at most 5 pages; peer evaluation; individual innovation where applicable | Later individual deliverables | Outside this setup task |

The brief disallows fully pretrained models used without further training; partial fine-tuning is allowed. Classification validation/test splits retain their specified roles. Both embedding approaches train on the given training data. Verification trials, score direction and ROC/AUC must be handled consistently; do not treat raw Euclidean distance as a higher-is-more-similar score. Model/threshold selection must be specified before comparison, and any selection reuse of verification trials must be disclosed.

The assigned face archive has been downloaded through KaggleHub and inspected by the FAC-01.2 loader. [Recorded counts and batch checks](data-loading.md#observed-evidence-and-acceptance) are actual observations; full image/duplicate/leakage audits and model accuracy remain unassessed. Liveness and emotion require appropriately labelled data; select additional sources only when the supplied data does not cover those labels. The one-step synthetic environment check reports infrastructure behavior, not model performance or rubric completion.

Task wording and owner/reviewer assignments come from the group's [WBS and Rubric Requirements](https://docs.google.com/spreadsheets/d/1ItpF11JBlpcTnNiADI-Qm-cDmexBuLRzuHBW5uDChLw/edit?usp=sharing). Review evidence is required before SYS-03 can be closed.
