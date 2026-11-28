# Facial Attendance

COS30082 group project for face recognition, liveness, emotion and webcam attendance. The modules form the team’s attendance system.

## Project Tracker

The team's full-project [Google Sheets workbook](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit?usp=sharing), called the **Project Tracker**, maintains assignments, task updates and reviewer confirmation. Repository files retain implementation and dated execution evidence.

<a id="run-and-verify"></a>

## Start here

Extract the GitHub source ZIP or clone the repository, then use that folder as the working directory. Choose a route and a module below.

- **Local execution:** [Set up Python and dependencies for your OS](docs/setup.md#1-open-a-terminal-and-select-the-environment), then choose a module below. [Tests](docs/setup.md#run-unit-tests) and the [device check](docs/setup.md#3-check-the-device) check implementation and runtime behavior.
- **Google Colab execution:** Follow [Colab setup](docs/setup.md#google-colab-cpu-or-t4-gpu), then open the module notebook. Liveness training uses Colab; saved inspection and CPU inference also run locally.

### Face recognition

[Acquire and check face data](docs/data-loading.md) with [check_face_data.py](scripts/check_face_data.py). Available checks cover classification images, verification pairs and training-pair sampling.

### Liveness

[Inspect results or try your own image/MP4](docs/liveness-training-and-evaluation.md#portable-inference-and-saved-inspection). The guide covers checkpoint download/checksum verification, detector setup, input naming and outputs. No personal team recording or training dataset is needed for inference. [Colab reproduction](docs/liveness-training-and-evaluation.md#execution-and-resume) and [data audit](docs/liveness-data-preparation-and-audit.md#audit-the-current-frozen-selection) are separate routes. The [artifact index](evidence/liveness/submission-artifacts.json) identifies models, large inputs and evidence.

### Emotion

[Emotion module](src/facial_attendance/emotion/).

### Attendance UI

[Attendance UI module](src/facial_attendance/ui/). See the [course requirements](docs/requirements.md) for the project scope.

## Repository layout

```text
facial-attendance/
├── src/facial_attendance/  # Feature modules and shared runtime/configuration
├── scripts/               # CLI entry points grouped by responsibility
├── notebooks/             # Executable experiments and saved results
├── tests/                 # Feature suites and shared infrastructure checks
├── configs/               # Project paths and release checksums
├── docs/                  # Setup, module guides and architecture
├── data/                  # External datasets and caches; guide retained in Git
└── evidence/              # Selected results and verification records
```

The [architecture/code reading route](docs/architecture.md) explains responsibilities and integration boundaries. [Setup review](evidence/SYS-03/README.md#sys-03-independent-review), [face-loader evidence](evidence/FAC-01/README.md) and [liveness data evidence](evidence/LIV-03/README.md) retain their original execution attribution.
