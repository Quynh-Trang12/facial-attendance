# Facial Attendance

COS30082 group project integrating CNN face verification, trained anti-spoofing, Sad/Happy/Neutral expression recognition and webcam registration. Classification and metric-learning face models are compared using ROC/AUC. The source includes the selected inference checkpoints and face detector.

The application is a coursework demonstration. Attendance writes require explicitly isolated test databases. Historical replay checks recorded recognition errors; current webcam error rates have not been remeasured. See [requirements, results and remaining verification limits](docs/requirements.md) before assessing submission completeness.

## Project Tracker

The team's full-project [Google Sheets workbook](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit?usp=sharing), called the **Project Tracker**, maintains assignments, task updates and reviewer confirmation. Repository files retain implementation and dated execution evidence.

<a id="run-and-verify"></a>

## Start here

Extract the GitHub source ZIP or clone the repository, then use that folder as the working directory. Choose a route and a module below.

- **Local execution:** [Set up Python and dependencies for your OS](docs/setup.md#1-open-a-terminal-and-select-the-environment), then choose a module below. [Tests](docs/setup.md#run-unit-tests) and the [device check](docs/setup.md#3-check-the-device) check implementation and runtime behavior.
- **Google Colab execution:** Follow [Colab setup](docs/setup.md#google-colab-cpu-or-t4-gpu), then open the module notebook. Liveness training uses Colab; saved inspection and CPU inference also run locally.

### Face recognition

[Models and reproduction](docs/face-models.md), [approach comparison](docs/face-comparison.md) and [recorded ROC/AUC results](evidence/FAC-05/README.md). [Acquire and check face data](docs/data-loading.md) for training/evaluation; inference uses the bundled metric-learning checkpoint.

### Liveness

[Inspect results or try your own image/MP4](docs/liveness-training-and-evaluation.md#portable-inference-and-saved-inspection). The guide covers checkpoint download/checksum verification, detector setup, input naming and outputs. No personal team recording or training dataset is needed for inference. [Colab reproduction](docs/liveness-training-and-evaluation.md#execution-and-resume) and [data audit](docs/liveness-data-preparation-and-audit.md#audit-the-current-frozen-selection) are separate routes. The [artifact index](evidence/liveness/submission-artifacts.json) identifies models, large inputs and evidence.

### Emotion

[Final model, training and held-out results](docs/emotion-finalized.md), [adapter usage](docs/emotion-final-integration.md) and [saved results notebook](notebooks/emotion_final_results.ipynb). The deployed classes are Sad, Happy and Neutral; the older aggregate and six-class experiments are historical alternatives.

### Attendance UI

[Launch and reviewer walkthrough](docs/attendance-demo.md). The current application uses tight detector-box recognition crops and refuses profile databases created for the earlier square crop.

## Run the demo

From the repository root, activate the Python 3.13 environment described in [setup](docs/setup.md). Tk, camera access and a graphical desktop are required. These files are bundled; inference does not download model weights or require training datasets:

| Component | Bundled file |
| --- | --- |
| Face metric ResNet-18 | [Face checkpoint](artifacts/models/recognition/face-recognition-resnet18.pt) |
| Emotion ResNet-18, epoch 10 | [Emotion checkpoint](artifacts/models/emotion/best.pt) |
| Liveness AENet-Full, epoch 18 | [Liveness checkpoint](artifacts/models/liveness/liveness-inference.pt) |
| YuNet auxiliary face locator | [Detector](artifacts/detectors/face_detection_yunet_2023mar.onnx), [MIT license](artifacts/detectors/LICENSE-YuNet.txt) |

Verify the detector locally with `python scripts/setup_sys05_yunet.py`. The adapters verify the selected model checksums when loading. YuNet supplies face locations; the team's trained models supply identity, liveness and expression predictions.

<details>
<summary>Local execution — Linux/macOS shell</summary>

```sh
PYTHONPATH=src python -m facial_attendance.ui.webcam_app \
  --local-workspace --attendance --attendance-verification-test \
  --attendance-auto-retry-test --attendance-auto-record-test \
  --camera 0 --detector yunet \
  --face-checkpoint artifacts/models/recognition/face-recognition-resnet18.pt \
  --emotion-checkpoint artifacts/models/emotion/best.pt \
  --local-profile-db artifacts/attendance-verification-test/demo-box/profiles.sqlite3 \
  --attendance-db artifacts/attendance-verification-test/demo-box/attendance.sqlite3 \
  --opencv-threads 1
```

</details>

<details>
<summary>Local execution — Windows PowerShell</summary>

```powershell
$env:PYTHONPATH = "src"
python -m facial_attendance.ui.webcam_app `
  --local-workspace --attendance --attendance-verification-test `
  --attendance-auto-retry-test --attendance-auto-record-test `
  --camera 0 --detector yunet `
  --face-checkpoint artifacts/models/recognition/face-recognition-resnet18.pt `
  --emotion-checkpoint artifacts/models/emotion/best.pt `
  --local-profile-db artifacts/attendance-verification-test/demo-box/profiles.sqlite3 `
  --attendance-db artifacts/attendance-verification-test/demo-box/attendance.sqlite3 `
  --opencv-threads 1
```

</details>

<details>
<summary>Google Colab execution — models and saved evidence</summary>

Use [Colab setup](docs/setup.md#google-colab-cpu-or-t4-gpu) and the module guides above for training, evaluation and saved-result inspection. The Tk webcam application runs on a local graphical desktop; these Colab routes do not establish a physical webcam GUI test.

</details>

## Walk through the app

Registration is intentionally restricted to **Developer Mode** under the team's role design; **Standard Mode** serves recognition and attendance. Open **Developer Mode**, choose **Register new person**, enter a consenting participant's name and collect the five guided live references. Save the profile, return to **Standard Mode** and confirm its name and sampled expression. Start a session and retain a known live face continuously for automatic test attendance. Unknown, spoof, stale and multiple-face captures must not write a record. Repeated checks of the same person in one session must leave one row.

The [full walkthrough](docs/attendance-demo.md#reviewer-walkthrough) covers rejection, camera interruptions, duplicate prevention and SQLite verification. Treat a wrong-person record as a failed check even if the application displays success. Square-crop profiles cannot be reused in this workspace; use a separate database and register fresh references.

## Repository layout

```text
facial-attendance/
├── src/facial_attendance/  # Feature modules and shared runtime/configuration
├── scripts/               # CLI entry points grouped by responsibility
├── notebooks/             # Executable experiments and saved results
├── tests/                 # Feature suites and shared infrastructure checks
├── configs/               # Project paths and release checksums
├── artifacts/models/      # Selected inference checkpoints, grouped by module
├── artifacts/detectors/   # Bundled auxiliary locator and license
├── docs/                  # Setup, module guides and architecture
├── data/                  # External datasets and caches; guide retained in Git
└── evidence/              # Selected results and verification records
```

The [architecture/code reading route](docs/architecture.md) explains responsibilities and integration boundaries. [Setup review](evidence/SYS-03/README.md#sys-03-independent-review), [face-loader evidence](evidence/FAC-01/README.md) and [liveness data evidence](evidence/LIV-03/README.md) retain their original execution attribution.

## Submission contents

Submit the source code with the bundled inference files, module guides, notebooks and selected evidence. Generated profile/attendance databases, raw datasets, personal recordings, caches, environments and full training checkpoints remain local or separately acquired. The [course coverage](docs/requirements.md#separate-submission-deliverables) lists the group report, demonstration video and individual deliverables required in addition to source code. This README is not the page-limited group report.
