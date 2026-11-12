# Facial Attendance

Facial recognition with emotion and liveness for COS30082.

COS30082 group project combining face verification, anti-spoofing and emotion recognition. The first required application is webcam attendance with employee registration. The feature modules can later support workplace sentiment analysis or mental-health monitoring workflows.

The repository contains the application scaffold, an explicit CPU/CUDA environment check, classification/pair loaders and an entry point for the assigned Kaggle competition. Authenticated Kaggle acquisition and classification/verification/training-pair loader checks pass on the assigned dataset. Face models, model-training pipelines, registration and the webcam UI remain to be implemented.

## Project Tracker

The team's [Google Sheets workbook](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit?usp=sharing), referred to as the **Project Tracker**, supports planning and coordination throughout the project. It contains the Gantt chart, WBS, requirements, blockers and product backlog, including task assignments, evidence and review records. Task status and reviewer confirmation are maintained in the Project Tracker; repository documentation links to those records.

## Start here

1. [Setup](docs/setup.md): select your machine/device, prepare the environment and verify execution.
2. [Unit tests](docs/setup.md#run-unit-tests): check for regressions after setup or changes to code/configuration.
3. [Data loading](docs/data-loading.md#download-on-your-own-machine): download and check the dataset after setup passes.

Follow only the branch for your machine. Verification evidence and current limitations are in [evidence](evidence/README.md); reading them is not required to start setup.

## Repository layout

```text
facial-attendance/
├── .github/workflows/checks.yml    # Source checks
├── pyproject.toml                 # Package metadata
├── requirements.txt               # Shared dependencies
├── requirements-cpu.txt           # CPU wheels
├── requirements-cuda.txt          # CUDA 12.6 wheels
├── requirements.lock              # Archived dependency snapshot
├── configs/project.toml           # Dataset and output paths
├── src/facial_attendance/
│   ├── config.py                  # Storage configuration
│   ├── environment.py             # Version checks and reporting
│   ├── runtime_checks.py          # CPU/CUDA computation checks
│   ├── recognition/               # Loaders; models pending
│   ├── liveness/                  # Anti-spoofing scaffold
│   ├── emotion/                   # Emotion scaffold
│   └── ui/                        # Webcam UI scaffold
├── scripts/check_environment.py   # Environment check entry point
├── scripts/check_face_data.py     # Dataset check entry point
├── tests/                        # Configuration and loader tests
├── data/README.md                 # Dataset layout
├── artifacts/                     # Local checkpoints and attendance
├── evidence/                       # Execution evidence
└── docs/
    ├── setup.md                   # Setup and device checks
    ├── architecture.md            # Module interfaces
    ├── data-loading.md            # Loader contracts and evidence
    ├── face-data-protocol.md      # Partition roles, pair rules and score direction
    ├── requirements.md            # Requirements and coverage
    └── reference/project-brief.pdf
```

## Project delivery

[Requirements coverage](docs/requirements.md) maps the brief to the implementation areas. [Task evidence](evidence/README.md) links selected SYS-03 and FAC-01 artifacts. The [test suite](docs/setup.md#run-unit-tests) checks configuration, runtime policy and loader behavior; [data evidence](docs/data-loading.md#observed-evidence-and-acceptance) records authenticated acquisition, actual dataset counts and loader checks separately from fixture tests. Independent setup test results are documented in the [review record](evidence/SYS-03/README.md#sys-03-independent-review). Routine outputs stay under ignored `evidence/runs/`; selected artifacts are organized by parent task. Datasets, credentials, checkpoints and personal attendance records stay local.

