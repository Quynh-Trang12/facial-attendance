# Facial Attendance

Facial recognition with emotion and liveness for COS30082.

COS30082 group project combining face verification, anti-spoofing and emotion recognition. The first required application is webcam attendance with employee registration. The feature modules can later support workplace sentiment analysis or mental-health monitoring workflows.

The repository contains the application scaffold, an explicit CPU/CUDA environment check, classification/pair loaders, anti-spoofing data auditing/loading and an entry point for the assigned Kaggle competition. Authenticated Kaggle acquisition and classification/verification/training-pair loader checks pass on the assigned dataset. Face models, model-training pipelines, registration and the webcam UI remain to be implemented.

## Project Tracker

The team's [Google Sheets workbook](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit?usp=sharing), referred to as the **Project Tracker**, supports planning and coordination throughout the project. It contains the Gantt chart, WBS, requirements, blockers and product backlog, including task assignments, evidence and review records. Task status and reviewer confirmation are maintained in the Project Tracker; repository documentation links to those records.

## Start here

Choose the execution route for your machine or cloud runtime: Complete the numbered steps within your chosen route in order; continue only after the preceding checks pass.

| Execution route | Steps in order |
| --- | --- |
| **Local execution** | 1. From your cloned repository, [prepare the environment and check the device](docs/setup.md#1-open-a-terminal-and-select-the-environment).<br>2. Run the [unit tests](docs/setup.md#run-unit-tests); resolve failures before continuing.<br>3. Choose the module you want to run and follow its [dataset preparation and loader checks](docs/setup.md#4-download-and-check-the-dataset). |
| **Google Colab execution** | 1. Open [Colab setup](docs/setup.md#google-colab-cpu-or-t4-gpu) and expand the section for the module you want to run.<br>2. Complete that route's prerequisites and environment setup. Anti-spoofing uses the [shared notebook](docs/liveness-loader.md#google-colab-execution); face recognition uses the listed setup cells.<br>3. Run the tests and dataset checks in the order provided; inspect and retain the generated results. |

Both routes run the implemented modules; the attendance UI remains unimplemented. [Execution evidence](evidence/README.md) records tested behavior and limitations.

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
│   ├── liveness/                  # Anti-spoofing inventory audit and loader
│   ├── emotion/                   # Emotion scaffold
│   └── ui/                        # Webcam UI scaffold
├── scripts/check_environment.py   # Environment check entry point
├── scripts/check_face_data.py     # Dataset check entry point
├── scripts/check_liveness_data.py # Anti-spoofing audit and sample batches
├── scripts/prepare_liveness_data.py # Full archive extraction and subject splits
├── tests/                        # Configuration and loader tests
├── data/README.md                 # Dataset layout
├── artifacts/                     # Local checkpoints and attendance
├── evidence/                       # Execution evidence
└── docs/
    ├── setup.md                   # Setup and device checks
    ├── architecture.md            # Module interfaces
    ├── data-loading.md            # Loader contracts and evidence
    ├── requirements.md            # Requirements and coverage
    └── reference/project-brief.pdf
```

## Project delivery

[Requirements coverage](docs/requirements.md) maps the brief to the implementation areas. [Task evidence](evidence/README.md) links selected SYS-03, FAC-01 and LIV-03 artifacts. The [test suite](docs/setup.md#run-unit-tests) checks configuration, runtime policy and loader behavior; [data evidence](docs/data-loading.md#observed-evidence-and-acceptance) records authenticated acquisition, actual dataset counts and loader checks separately from fixture tests. Independent setup test results are documented in the [review record](evidence/SYS-03/README.md#sys-03-independent-review). Routine outputs follow the [evidence storage layout](evidence/README.md#scope); selected artifacts are organized by parent task. Keep datasets, credentials, checkpoints and personal attendance records out of Git.

[Anti-spoofing review and reproduction](docs/liveness-loader.md#reproduce-the-results) covers storage planning, archive/extracted-data locations, external-drive paths, fresh preparation, full audit/loader commands and expected results. [Experimental readiness](docs/liveness-loader.md#experimental-readiness-and-candidate-handling) prepares the bound subset used for training while retaining documented data and method limitations. [Dataset storage](data/README.md#anti-spoofing-data-liv-03) separates LIV-03 from face-recognition acquisition. Fixture tests and full-data review have different evidence scopes.

