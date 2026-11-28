# Course requirements and coverage

This page maps the supplied [project brief](reference/project-brief.pdf) to implementation, recorded results and verification limits. The assigned face dataset is the [11-785 Fall 2020 Homework 2 Part 2 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). Task assignments and reviewer confirmation are maintained in the [Project Tracker](../README.md#project-tracker).

## System requirements and evidence

| Brief requirement | Implementation and reproduction | Observed evidence and limits |
| --- | --- | --- |
| Train CNN embeddings using both classification and metric learning; compare face verification | [Face models](face-models.md), [classification training](../scripts/train_classification.py), [metric training](../scripts/train_metric.py), [comparison protocol](face-comparison.md) | Both models trained from scratch. [Recorded official results](../evidence/FAC-05/official-summary.json) cover the same 8,805 supplied trials. The selected metric model is bundled for inference; the classification checkpoint and full run exports are acquired separately. |
| ROC/AUC on the supplied verification trials; investigate a suitable similarity measure | [Evaluator](../scripts/evaluate.py), [comparison](../scripts/compare_face_approaches.py), [saved-result checks](../scripts/check_face_comparison.py) | Cosine and negative raw Euclidean distance evaluated for both models. Validation selects model and threshold; official trials assess the frozen choice. Trial identity independence from training remains unverified. |
| Train and integrate anti-spoofing for live, print and screen presentations | [Training/evaluation guide](liveness-training-and-evaluation.md), [liveness adapter](../src/facial_attendance/liveness/inference/liveness_adapter.py), [webcam integration](../src/facial_attendance/ui/liveness_preview.py) | Fine-tuned AENet-Full epoch 18 is bundled. Retained public-test and recorded camera results are available. Public-test exposure, pre-review training and unavailable source-video IDs limit independence claims. |
| Train emotion recognition on public data and display real-time labels/icons | [Final emotion record](emotion-finalized.md), [ResNet adapter](../src/facial_attendance/emotion/resnet_inference.py), [display controller](../src/facial_attendance/ui/emotion_presentation.py) | ResNet-18 was fine-tuned on quality-filtered FER2013. Sad/Happy/Neutral labels are sampled every three seconds while preview continues. This three-class scope excludes other expressions. Held-out image accuracy does not measure webcam accuracy or actual mood. |
| Automatic webcam recognition, registered names, unknown-user registration and immediate model feedback | [Demo launch and walkthrough](attendance-demo.md), [application](../src/facial_attendance/ui/webcam_app.py), [registration](../src/facial_attendance/ui/trial_profiles.py) | Developer Mode intentionally owns new-person registration; Standard Mode serves recognition and attendance. Local profiles persist five individually live references as one averaged template. Tight recognition crops use a separate tagged database; old square-crop profiles are rejected. Physical macOS checks are recorded; the revised workflow still needs physical Windows/Linux and independent review. |
| End-to-end attendance integration | [Verification gate](../src/facial_attendance/attendance/verification.py), [service](../src/facial_attendance/attendance/service.py), [SQLite storage](../src/facial_attendance/attendance/store.py) | Recognition and liveness must refer to the same fresh capture; temporal liveness and duplicate protection precede writes. Historical replay checks recorded false accepts; current webcam error rates have not been remeasured. Writes are confined to the isolated test mode described in the demo guide. |
| Fully pretrained task models without further training are prohibited; partial fine-tuning is permitted | [Face models](face-models.md), [liveness training](liveness-training-and-evaluation.md#model-and-training-procedure), [emotion training](emotion-finalized.md#training-and-selection) | Face models trained from scratch; liveness/emotion fine-tuned. YuNet is an auxiliary face locator, not the face-verification, liveness or emotion task model. |

## Recorded model results

These are attributed saved results, not new training runs. Higher AUC means better ranking; a high AUC does not guarantee a reliable fixed operating threshold.

| Model and evaluation input | Recorded result | Evidence |
| --- | --- | --- |
| Face classification model, 8,805 official trials | Cosine AUC 0.961800; negative raw Euclidean AUC 0.944473 | [Official summary](../evidence/FAC-05/official-summary.json), [ROC figures](../evidence/FAC-05/README.md) |
| Face metric model, same official trials | Cosine AUC 0.968652; negative raw Euclidean AUC 0.935946 | [Official summary](../evidence/FAC-05/official-summary.json) |
| Selected face metric model at validation cosine threshold 0.4619967254593038 | Official FAR 2.539%, TAR 76.756%. Gallery naming additionally requires minimum score 0.69 and margin 0.10; these are distinct from the pair threshold. | [Selection](../evidence/FAC-05/selection.json), [operating point](../evidence/FAC-05/official-summary.json), [gallery decision configuration](../configs/recognition.toml) |
| Liveness AENet-Full, 67,129 retained public-test images | ROC AUC 0.998578; APCER 12.222%, BPCER 0.131%, ACER 6.176% at the frozen validation threshold | [Retained-test report](../evidence/liveness/crop-review/filtered-view-v4-20261126/filtered-test-evaluation.json) |
| Emotion ResNet-18, 4,133 retained FER2013 test images | Accuracy 77.57%, macro F1 75.20%, Neutral F1 70.34% | [Held-out report](../reports/emotion/final/held-out-test.json), [results notebook](../notebooks/emotion_final_results.ipynb) |

The face gallery threshold study used square-crop LFW images and may overlap training identities. The local webcam workspace now uses tight boxes; the threshold study does not establish calibration for that crop or a new classroom. [Replay evidence](attendance-demo.md#verification) records wrong-person attendance, including a historical paired crop comparison in which both crop recipes admitted a stranger. Those records do not establish a failure in the current application. No threshold was retuned from those test outcomes in this revision.

Liveness recorded clips, single images and the application's temporal gate use different aggregation rules. Source-video independence remains unverified due to unavailable source-video IDs. Emotion geometry screening included test images before scoring, and identity/near-duplicate independence is not established. These limits must accompany the results in the report.

## Reproduction and acceptance boundaries

Follow [setup](setup.md) for the existing Python 3.13 policy, then use the [README entry points](../README.md#start-here). [Local webcam execution](../README.md#run-the-demo) and [Colab model execution](setup.md#google-colab-cpu-or-t4-gpu) serve different checks. Raw datasets and full training/resume states are external inputs; bundled models suffice for inference.

Source checks cover lint, format, syntax, generated-fixture behavior, bundled-model inference and saved liveness inspection. These checks do not certify physical webcam recognition, resistance to new attacks, model accuracy on a new population, native OS behavior on unrun targets or final report/video compliance. Evaluate unknown people, enrolled people, print/screen attacks, lighting changes, interrupted captures and duplicate recording through the [reviewer walkthrough](attendance-demo.md#reviewer-walkthrough). Record the actual revision, crop, checkpoints, inputs and failures.

## Separate submission deliverables

| Required submission | Brief constraint | Repository boundary |
| --- | --- | --- |
| Source code | Submitted by one group member | This repository contains implementation, reproduction guides and selected evidence. |
| Group report | At most 12 pages; Methodology, Results and Discussion; framework, training schemes, losses, hyperparameters, comparisons and selection justification | A separate formatted deliverable. Module Markdown guides do not by themselves establish a complete, page-limited report. |
| Demonstration video link | 5–10 minutes, one continuous take with no edits; balanced speaking time; each speaker's face visible; accessible hosted link | A separate recording and accessibility check. Replay clips and synthetic test logs do not satisfy this requirement. |
| Individual report | At most five pages; contributions, key methods, resolved challenges and innovation where applicable | Each member supplies their own report. |
| Peer evaluation | One form per member | Separate individual submission. |
| Individual innovation and interview | Relevant innovation is encouraged and, when included, explained in the individual report and demonstrated; Week 12 interview assesses understanding | No individual contribution, reviewer confirmation or interview outcome is inferred from automated checks. |

Source readiness and complete Canvas submission are separate decisions. Historical replay findings require attribution to their tested inputs and revision. Unmeasured current error rates and unverified integrated physical cases prevent an unconditional claim that every system requirement has passed. Report/video/individual deliverables require their own review against the brief.
