# Architecture

[facial_attendance](../src/facial_attendance/) separates recognition, liveness, emotion and UI code. Configuration/runtime checks are shared.

- [Recognition](../src/facial_attendance/recognition/): Identity recognition and verification; classification/pair loaders and acquisition
- [Liveness](../src/facial_attendance/liveness/): Dataset audit, model training, frozen evaluation and image/frame inference
- [Emotion](../src/facial_attendance/emotion/): Expression inference
- [UI](../src/facial_attendance/ui/): Webcam display, registration and attendance integration
- [Configuration](../src/facial_attendance/config.py): Resolve paths from [project.toml](../configs/project.toml), including external storage
- [Environment checks](../src/facial_attendance/environment.py): Version/device policy; lazy loading of numerical checks keeps help and failed-install reports usable

The attendance flow detects/crops a face, checks liveness and identity, predicts emotion and updates the UI. Recognition uses tight detector-box crops in the local workspace; liveness/emotion keep their own training preprocessing. Tagged local profile storage rejects square-crop databases. Preview results are informational; dedicated same-capture temporal verification supplies the attendance policy only in the explicitly isolated test mode. Only a verified identity with accepted liveness may record attendance. The application owns tracking, registration and the final action; model scores alone do not implement that flow.

## Liveness architecture and code reading route

- [models](../src/facial_attendance/liveness/models/): Backbone, task heads/losses, teachers and detector
- [datasets](../src/facial_attendance/liveness/datasets/): Source crops/labels, frozen row selection and cache alignment
- [training](../src/facial_attendance/liveness/training/): Seeded batches, optimization, validation and resumable state
- [experiments](../src/facial_attendance/liveness/experiments/): Construct matched configurations and dispatch training
- [audit](../src/facial_attendance/liveness/audit/): Integrity, duplicate/crop review and selection checks
- [evaluation](../src/facial_attendance/liveness/evaluation/): Metrics, saved-score diagnostics and plots
- [inference](../src/facial_attendance/liveness/inference/): Face/frame predictions and temporal admission

UI, attendance and CLI import the model and detector from models and the adapter and temporal gate from inference. Each component has one implementation; the old root-level duplicates have been removed.

Models do not import datasets/training/experiments; datasets do not import training/experiments; training does not import experiment policy. [dataset_files.py](../src/facial_attendance/liveness/dataset_files.py) shares path, checksum and snapshot helpers. Tests protect these dependency boundaries.

### Run the supported entry point

[run_liveness_notebook.py](../scripts/run_liveness_notebook.py) dispatches inspect, audit, inference and specialized training actions. Use the [inference/inspection guide](liveness-training-and-evaluation.md#portable-inference-and-saved-inspection) or [frozen-data audit](liveness-data-preparation-and-audit.md#audit-the-current-frozen-selection) for complete commands and prerequisites.

| Action | CLI | Experiment | Training engine |
| --- | --- | --- | --- |
| train-source | [AENet CLI](../scripts/training/train_liveness_aenet.py) | [AENet experiment](../src/facial_attendance/liveness/experiments/aenet_experiment.py) | [AENet training](../src/facial_attendance/liveness/training/aenet_training.py) |
| train-baseline | [Binary CLI](../scripts/training/train_liveness_binary.py) | [binary experiment](../src/facial_attendance/liveness/experiments/binary_experiment.py) | [binary training](../src/facial_attendance/liveness/training/binary_training.py) |
| train-multitask | [Multitask CLI](../scripts/training/train_liveness_multitask.py) | [multitask experiment](../src/facial_attendance/liveness/experiments/multitask_experiment.py) | [multitask training](../src/facial_attendance/liveness/training/multitask_training.py) |

AENet owns the current four-configuration comparison and 15→25 continuation. The other two are bounded baseline experiments. Training options follow `--` on the shared CLI.

### Interview reading route

Start with the [notebook results and error gallery](../notebooks/liveness-training-and-evaluation.ipynb), then follow the path you need:

| Path | Read in order | What to explain |
| --- | --- | --- |
| Attendance decision | [liveness_adapter.py](../src/facial_attendance/liveness/inference/liveness_adapter.py), [face_detector.py](../src/facial_attendance/liveness/models/face_detector.py), [binary_model.py](../src/facial_attendance/liveness/models/binary_model.py), [temporal_liveness.py](../src/facial_attendance/liveness/inference/temporal_liveness.py) | RGB/BGR conversion, locator boxes, checkpoint preprocessing/threshold and consecutive-frame admission |
| Model and supervision | [multitask_model.py](../src/facial_attendance/liveness/models/multitask_model.py): `forward_tasks`, `multitask_loss` | Shared features, enabled heads, weighted losses and binary-only deployment |
| Experiment assembly | [aenet_experiment.py](../src/facial_attendance/liveness/experiments/aenet_experiment.py): `FullTrainingConfig`, `run_full_training` | Frozen row selection, teacher alignment, matched initialization and enabled supervision |
| Weight updates and resume | [aenet_training.py](../src/facial_attendance/liveness/training/aenet_training.py): `train_aenet`; [multitask_training.py](../src/facial_attendance/liveness/training/multitask_training.py): `train_multitask_epoch` | Completed-update cursor, optimizer/RNG restoration and gradients for selected tasks |
| Dataset and replay | [source_dataset.py](../src/facial_attendance/liveness/datasets/source_dataset.py), [reviewed_cache.py](../src/facial_attendance/liveness/datasets/reviewed_cache.py), [batching.py](../src/facial_attendance/liveness/training/batching.py) | Source crop recipe, original row identities and worker-independent epoch replay |
| Selection and archives | [validation.py](../src/facial_attendance/liveness/training/validation.py), [checkpoints.py](../src/facial_attendance/liveness/training/checkpoints.py) | Validation-only operating point, tie-breaking, deployment export and complete resume archives |

Recorded clips use median sampled scores; tracked attendance streams use the temporal gate. Teachers supply pseudo-targets during training, while deployment consumes the binary classifier.

### Source snapshots and evidence

Snapshots retain package-relative source paths and verify copied bytes. Resume accepts only explicit before/after fingerprints and matching data, weights, optimizer/RNG state and cursor. Historical reports retain their original producing source and inputs. The [release inventory](../configs/liveness-release.json) and [verification receipt](../evidence/liveness/package-refactor/verification.json) distinguish current implementation checks from original model results.

### Reading comments and contracts

Docstrings explain responsibility and input/output/failure contracts. Inline comments explain consequential reasons, such as keeping image/teacher-map row alignment, restoring RNG after epoch seeding, or preventing a fast frame burst from satisfying elapsed-time admission. Conventional assignments and accessors need no syntax narration.

### Specialized scripts and verification limits

Scripts share phase folders; feature-qualified names such as [train_liveness_aenet.py](../scripts/training/train_liveness_aenet.py) distinguish module ownership. Shared environment/notebook helpers stay at the script root. Tests use liveness/recognition packages with shared configuration/runtime tests at the test root; discovery runs from the repository root.

Native current checks are Linux CPU tests/inspection and selected-checkpoint CPU inference, plus one CUDA face-crop inference run. They do not establish a new full GPU training result, native Windows/macOS/Colab execution, live webcam behavior or unchanged speed. [Setup](setup.md#platform-verification) separates configured targets and historical platform evidence.
