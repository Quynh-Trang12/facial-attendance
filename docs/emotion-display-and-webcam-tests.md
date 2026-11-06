# Emotion display and webcam test plan

EMO-02 owner: chị Dung. Reviewer: Vỹ.

This document defines the emotion display and the tests planned for webcam integration.
No webcam tests have been run for this document. Model training and working integration
are later steps. The shared interface is still a draft under SYS-02 / ISS-001.

## Emotion labels

Use the model IDs below for training, inference and display. Do not use alphabetical
folder order. Always show the text label; the icon is an optional visual aid.

| Model ID | Aggregated target | Display text | Suggested icon |
|---|---|---|---|
| 0 | negative | Negative | 😟 |
| 1 | neutral | Neutral | 😐 |
| 2 | positive | Positive | 🙂 |

Negative groups Angry, Disgust, Fear and Sad. Positive contains Happy. Neutral contains Neutral. Surprise is excluded. These are expression groups, not a direct measurement of a person's mood.

If an icon is unavailable on a device, the text label is sufficient. The mapping matches
`data/processed/emotion_3class/label_mapping.json` and the EDA notebook.

## Display rules

The following display behavior is proposed for agreement with Vỹ under SYS-02.

| Situation | Emotion display | Expected behavior |
|---|---|---|
| Valid prediction for the current face | For example, `Positive 🙂` | Map the predicted ID to the table above. |
| Face detected, first result pending | `Emotion: Processing` | Do not show a result from an earlier face or frame as current. |
| No face detected | `Emotion: No face detected` | Clear the previous emotion label and confidence. |
| Model missing, loading failed, or inference failed | `Emotion: Unavailable` | Clear the previous result; keep this state separate from Neutral. |
| Invalid ID or invalid prediction values | `Emotion: Unavailable` | Do not guess a label or display an invalid result. |
| Camera unavailable | `Emotion: Unavailable` | Let the GUI report the camera problem; do not retain an old emotion. |

A valid result uses a model ID from 0 to 2. If confidence is provided, it must be a finite
number from 0 to 1. A proposed result contains `class_id`, `label`, and optional `confidence`.
The adapter fields and how it reports pending, missing, or failed predictions still need
confirmation with Vỹ. This document does not introduce a new shared API implementation.

Confidence is the predicted class's softmax probability, not model accuracy or a guarantee
that the label is correct. Showing confidence is optional. No low-confidence threshold has
been chosen; do not invent one for the display. If the team later chooses a threshold,
select it using validation data and document its behavior.

When the visible face changes or no current result is available, do not present an earlier
prediction as current. Scheduling and any frame/result matching belong to the shared GUI
design. Smoothing, multiple-face support and a fixed FPS target are not required here.

Emotion is displayed as additional information. It does not determine identity or replace
the liveness checks used for attendance and registration.

## Planned checks

All checks below are **Not run**. Use controlled outputs to test display mapping before
the trained model is available. Later, repeat the relevant checks in the integrated GUI.

| ID | Setup or action | What to check |
|---|---|---|
| E01 | Supply each model ID from 0 through 2 as a controlled output. | All three text labels and any icons match the mapping table. |
| E02 | Supply a valid controlled prediction with confidence 0.75. | Correct label; if confidence is shown, it appears as 75%. |
| E03 | Supply an invalid ID, NaN, infinity, or confidence outside 0–1. | Unavailable state; no incorrect label or stale confidence. |
| E04 | Open the webcam with no face in view. | No-face state; no previous emotion is displayed. |
| E05 | Enter the frame, then leave it after a prediction appears. | Current prediction appears when available and clears when the face disappears. |
| E06 | Show a neutral expression, a happy expression and an expression from the Negative group, such as sadness. | Record the actual model label and displayed label. They must agree; a model mistake is recorded separately. |
| E07 | Change expression while the previous result is still pending. | Earlier results are not presented as current; record any stale-display problem. |
| E08 | Try dim lighting, a partial face covering and a turned face. | Record wrong or missing predictions and whether the UI represents the returned state correctly. Correct classification is not assumed. |
| E09 | Start without a usable checkpoint, then simulate an inference failure. | Unavailable state, cleared result, and a useful error record for debugging. |
| E10 | Deny camera access or disconnect the camera. | Camera problem reported by the GUI; emotion unavailable and no old label remains. |
| E11 | Restore camera/model availability after an error. | A new valid prediction can replace the unavailable state. |

For each run, record the test ID, date, model/checkpoint, software version, lighting or
input conditions, expected display, actual model output, actual display, and pass/fail
or not-run result. Link evidence where available. Do not mark a planned test as passed.

E06 checks two different things: whether the GUI shows the model's prediction correctly,
and whether the model recognizes an attempted expression. The first is a mapping check.
The second is an informal webcam observation, not a measured accuracy score or ground truth.

## Limitations and handover

FER2013 contains small grayscale face images. Webcam lighting, pose, crop and image quality
can differ from the training data. Some expressions are ambiguous, and the classes are
imbalanced. Real-world performance has not been measured yet.

An expression label does not establish a person's mood, intent, mental-health condition or
diagnosis. Do not make these claims in the interface or report.

Use validation data for model selection. Report final model results on the revised held-out
test set, including accuracy, macro-F1 and per-class results. The EDA notebook documents aggregation and exclusions; scores are not directly comparable to the original seven-class benchmark.
Webcam observations should be reported separately from dataset evaluation.

For Vỹ: confirm the display states and adapter fields with SYS-02, then use this plan during
UI-04 integration and SYS-07 webcam testing. Input crop, normalization, tensor layout and
checkpoint loading must match the eventual training pipeline; they are not finalized here.

Related files: `notebooks/emotion_eda.ipynb`, `docs/emotion-data.md`, and
`reports/emotion/review_decisions.csv`. This document supplies the owner output for EMO-02;
independent review and running integration remain pending.
