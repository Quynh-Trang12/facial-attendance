# Facial Attendance

Facial recognition with emotion and liveness for COS30082.

COS30082 group project combining face verification, anti-spoofing and emotion recognition. The first required application is webcam attendance with employee registration. The feature modules can later support workplace sentiment analysis or mental-health monitoring workflows.

The repository contains the application scaffold, a GPU environment check, classification/pair loaders and an entry point for the assigned Kaggle competition. Authenticated Kaggle acquisition and classification/verification/training-pair loader checks pass on the assigned dataset. Face models, model-training pipelines, registration and the webcam UI remain to be implemented.

## Repository layout

```text
facial-attendance/
├── .github/workflows/checks.yml    # Read-only source/configuration CI
├── pyproject.toml                 # Python package metadata
├── requirements.txt               # Direct project dependency pins
├── requirements.lock              # Observed Linux CUDA dependency closure
├── configs/project.toml           # Dataset and generated-output locations
├── src/facial_attendance/
│   ├── config.py                  # Portable storage-path configuration
│   ├── environment.py             # Executable dependency and CUDA checks
│   ├── recognition/               # Working data/pair loaders; future embedding models
│   ├── liveness/                  # Trained anti-spoofing module
│   ├── emotion/                   # Trained expression-recognition module
│   └── ui/                        # Webcam, registration and integration
├── scripts/check_environment.py   # Source-checkout entry point
├── scripts/check_face_data.py     # Kaggle/local acquisition and loader inspection
├── tests/                        # Configuration and generated-image loader tests
├── data/README.md                 # Assigned dataset layout; raw files ignored
├── artifacts/                     # Local checkpoints/attendance; ignored
├── reports/                       # Environment and data-loader execution evidence
└── docs/
    ├── setup.md                   # Setup, device checks and independent review
    ├── architecture.md            # Module boundaries and extension rules
    ├── data-loading.md            # Loader contracts, sampling and acceptance evidence
    ├── requirements.md            # Course requirements and current coverage
    └── reference/project-brief.pdf
```

## Development setup

Use **Python 3.13.16** and the pinned PyTorch CUDA environment. On Trang's machine, reuse the existing shared runtime; do not reinstall it:

```bash
~/.local/share/ml-frameworks/run_gpu.sh scripts/check_environment.py \
  --operator "owner-host" \
  --output reports/environment/owner-check.json
```

Run from this repository root in a host terminal. Choose a new evidence filename for each subsequent run. The source entry point also works from another directory using its absolute path. GPU checks stop on device or dependency errors; they do not fall back to CPU.

For another developer's installation, platform limits and the review procedure, use [the setup guide](docs/setup.md). The assigned face dataset is documented in [data/README.md](data/README.md).

Run the small source/configuration suite without installing the project or changing the shared environment:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  ~/.local/share/ml-frameworks/.venv/bin/python -m unittest discover -s tests -v
```

GitHub CI runs source and generated-image tests on Python 3.13 with Pillow; it neither downloads competition data nor certifies CUDA. The workflow has read-only repository permissions and never commits generated results.

## Project delivery

[Requirements coverage](docs/requirements.md) maps the brief to the implementation areas. [Environment evidence](docs/setup.md#evidence-and-review) records the SYS-03 run at its recorded revision. The suite now has 26 passing configuration/loader tests; [data evidence](docs/data-loading.md#observed-evidence-and-acceptance) records authenticated acquisition, actual dataset counts and loader checks separately from fixture tests. Independent reviewer execution remains pending. Small reports may be committed after inspection; datasets, credentials, checkpoints and personal attendance records stay local.

