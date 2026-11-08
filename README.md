# Facial Attendance

Facial recognition with emotion and liveness for COS30082.

COS30082 group project combining face verification, anti-spoofing and emotion recognition. The first required application is webcam attendance with employee registration. The feature modules can later support workplace sentiment analysis or mental-health monitoring workflows.

The repository contains the application scaffold, an explicit CPU/CUDA environment check, classification/pair loaders and an entry point for the assigned Kaggle competition. Authenticated Kaggle acquisition and classification/verification/training-pair loader checks pass on the assigned dataset. Face models, model-training pipelines, registration and the webcam UI remain to be implemented.

[LIV-01 data and attack protocol](docs/liveness-protocol.md) selects CelebA-Spoof and subject-disjoint partitions for one RGB live/spoof CNN, with webcam checks for live, print, screen-photo and video-replay cases. The baseline blocks spoof attendance and displays liveness; it requires no motion analysis, frame aggregation or fixed capture matrix. Its [split specification](configs/liveness-protocol.json) is ready for independent review; raw-data audit, model implementation and reviewer acceptance remain pending.

## Repository layout

```text
facial-attendance/
├── .github/workflows/checks.yml    # Read-only source/configuration CI
├── pyproject.toml                 # Python package metadata
├── requirements.txt               # Shared public package versions
├── requirements-cpu.txt           # CPU wheel selection
├── requirements-cuda.txt          # CUDA 12.6 wheel selection
├── requirements.lock              # Owner-only Linux/CUDA snapshot
├── configs/project.toml           # Dataset and generated-output locations
├── src/facial_attendance/
│   ├── config.py                  # Portable storage-path configuration
│   ├── environment.py             # Version policy, CLI and evidence reporting
│   ├── runtime_checks.py          # ML imports and actual CPU/CUDA computation checks
│   ├── recognition/               # Working data/pair loaders; future embedding models
│   ├── liveness/                  # Reserved for the trained anti-spoofing module
│   ├── emotion/                   # Reserved for the trained emotion module
│   └── ui/                        # Reserved for webcam/registration integration
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

The project targets **Python 3.13.x**, Linux/Windows x64 CPU or NVIDIA CUDA, and macOS 14+ Apple Silicon CPU. See the [OS-specific setup and verification status](docs/setup.md#choose-the-operating-system-and-runtime-first); Windows/macOS native execution is pending. On Trang's machine, retain the existing **Python 3.13.16** GPU environment; do not reinstall it:

```bash
~/.local/share/ml-frameworks/run_gpu.sh scripts/check_environment.py \
  --device cuda --strict-owner --operator "owner-host" \
  --output reports/environment/owner-check.json
```

Run the GPU command in a host terminal. For an intentional CPU check, use the same environment's Python directly with `--device cpu` and omit `--strict-owner`. Requested CUDA still fails if unavailable; it never falls back silently. Choose a new evidence filename per run. Source entry points also work from another directory using an absolute script path.

For new CPU/CUDA installations and step-by-step Colab cells, use [the setup guide](docs/setup.md). It distinguishes portable direct pins from the optional owner snapshot. Colab [CPU](reports/environment/colab-cpu-real-2026-11-04.json) and [Tesla T4 GPU](reports/environment/colab-cuda-real-2026-11-04.json) checks passed: 39 tests per runtime, loader checks on the same 56 original competition images, and optimizer/inference with a real-image batch. This is a subset compatibility check, not full-dataset training or accuracy evaluation. To download the full competition with your own account, follow [data loading](docs/data-loading.md#download-on-your-own-machine). Dataset paths and external-storage overrides are documented in [data/README.md](data/README.md).

Run the source/configuration/loader suite without installing the project or changing the shared environment:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 \
  ~/.local/share/ml-frameworks/.venv/bin/python -m unittest discover -s tests -v
```

GitHub CI uses only the Python version in `.python-version` (3.13.16): one Linux source check and CPU installation/test jobs on Windows x64 and macOS Apple Silicon; it neither downloads competition data nor certifies CUDA. The workflow has read-only repository permissions and never commits generated results. Members use one Python 3.13 environment on their own OS; they do not need to test other Python versions or every device profile.

## Project delivery

[Requirements coverage](docs/requirements.md) maps the brief to the implementation areas. [Current verification evidence](reports/README.md) links the configured-data, CPU/IPython and strict CUDA runs; older runs retain their recorded source hashes. The suite now has 46 passing configuration, runtime-policy and loader tests; [data evidence](docs/data-loading.md#observed-evidence-and-acceptance) records authenticated acquisition, actual dataset counts and loader checks separately from fixture tests. Independent reviewer execution remains pending. Small reports may be committed after inspection; datasets, credentials, checkpoints and personal attendance records stay local.

