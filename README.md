# Facial Attendance

Facial recognition with emotion and liveness for COS30082.

COS30082 group project combining face verification, anti-spoofing and emotion recognition. The first required application is webcam attendance with employee registration. The feature modules can later support workplace sentiment analysis or mental-health monitoring workflows.

This integration branch includes a webcam GUI with YuNet detection and the supplied trained Face, Emotion and AENet-Full checkpoints. The local workspace saves named recognition profiles and shows unverified matches. Attendance saving requires a separate verified workflow. Controlled registration uses generated test images and an isolated memory gallery.

## Webcam GUI: Standard and Developer modes

Place or link the three supplied checkpoints in `artifacts/models/` as `face-recognition-resnet18.pt`, `emotion-recognition-resnet18-v1.pt`, and `liveness-aenet-full-epoch18.pt`. The local workspace loads the liveness checkpoint automatically. Place or link the YuNet ONNX file under `artifacts/sys05-yunet/`. Use `--liveness-checkpoint PATH` or `--yunet-model PATH` to select another location.

```sh
cd /Users/elvyx/Downloads/fa-sys03-mac/artifacts/ui03-ui04-emotion-integration
PYTHONPATH=src artifacts/ui04-runtime/bin/python -m facial_attendance.ui.webcam_app \
  --local-workspace --camera 0 --detector yunet \
  --face-checkpoint artifacts/models/face-recognition-resnet18.pt \
  --emotion-checkpoint artifacts/models/emotion-recognition-resnet18-v1.pt \
  --opencv-threads 1 --diagnostics
```

**Standard Mode is the default**, including with `--diagnostics`. The toolbar contains camera connection status, **Restart Camera**, and **Developer Mode**. The camera remains the main view, with a scrollable analysis panel on the right. Face Recognition shows **Unverified** for an identity match. Anti-Spoofing shows the last completed scan. Emotion Detection shows the supplied ResNet-18 model's Sad, Happy or Neutral facial-expression label. The earlier EmotionCNN remains available through an explicit `--emotion-checkpoint` path. Attendance and liveness configuration are shown below the results.

The supplied Emotion checkpoint has SHA-256 `0dcecbdbc7195bfb8cfd4529e0aa287d9b40c43da323311ad3803d687fcbfd24`. The integration runtime requires `torch` and `torchvision`.

The webcam preview also shows an Identity-only label above the face outline, repositioned at edges, in both modes: Unknown, Checking..., or a current matched name marked UNVERIFIED. See [identity overlay behavior and retest](docs/ui-03-identity-overlay.md).

Select **Developer Mode** in the header to open technical diagnostics. A separate resizable window has scrollable **Runtime and presentation** and **Raw current SYS-02** tabs. **Open / close diagnostics** is visible exclusively in Developer Mode. Select **Standard Mode** to hide both that button and the diagnostics window; returning to Developer Mode reopens the same window. Switching modes preserves the running camera, models, workers, gallery, controller and typed registration input.

Developer diagnostics has **Pause / Resume refresh**, **Copy displayed JSON**, and **Export privacy-safe snapshot** controls. Pause freezes only the diagnostic tabs; the camera, inference and authorization continue. The LIVE/PAUSED indicator includes each displayed snapshot's UTC timestamp. **Capture latest completed inference (historical)** freezes a separate diagnostic copy rather than letting a checking state replace it. Resume returns to the current raw tab. Copy uses the exact displayed JSON; export removes names/IDs, contexts, images, embeddings, gallery records, arbitrary messages and local paths. See [diagnostic controls and verification](docs/ui-03-diagnostic-controls.md).

The primary preview can briefly retain historical display text while current inference is checking. Such text never authorizes saving. Technical age limits, continuity reasons, selected checkpoint metadata, raw confidence and timing measurements belong in Developer Mode. `--diagnostics` enables aggregate console timing counters; it does not select the initial mode. The diagnostic snapshot is not an attendance or enrollment record.

For owner-operated recognition measurements without trained liveness, use the separate [Recognition Evaluation / SANDBOX](docs/ui-03-recognition-sandbox.md). It permits consented test reference capture into its own memory-only gallery and records distinct evaluation scores. It cannot enroll into the normal gallery or authorize attendance; the normal GUI gates above remain unchanged.

See [dual-mode delivery and physical retest](docs/ui-03-dual-mode.md), [Emotion continuity policy](docs/ui-03-emotion-continuity.md), [Face integration contract](docs/ui-03-face-inference-registration-contract.md), and [trained adapter contract](docs/ui-04-trained-adapter-contract.md). Runtime setup and checkpoint requirements are in [integration delivery](docs/ui-03-integration-delivery.md). These distinguish generated-input checks, owner observations and pending physical/reviewer verification.

## Local workspace

In **Developer Mode**, select **Register new person** and ask the new person to face the webcam. The form opens when the app detects one unregistered face. While waiting, select **Cancel registration request** to cancel. Enter a display name, confirm consent to keep a face template on this device, and capture five fresh reference photos using the pose prompts. Losing the face or detecting multiple faces resets the attempt. Wait for a completed `live` three-second scan in the current attempt to enable **Capture reference**. Each selected photo must individually pass liveness before it counts toward the five required by **Save profile**. An `uncertain` selected photo requires a fresh capture. A `spoof` selected photo discards the attempt and requires **Retry**. The matcher checks for duplicate faces before saving. Saved profiles appear in webcam recognition after reopening the app. **Clear local profiles** is also in Developer Mode and requests confirmation before deleting all entries.

The workspace keeps names, test IDs and averaged recognition templates in the ignored `artifacts/local-recognition/profiles.sqlite3` database. **Liveness** uses AENet-Full epoch 18 on CPU. Every three seconds, it averages the spoof scores from valid single-face frames in the scan. A mean at or above the checkpoint threshold (0.464838…) gives `spoof`; a lower mean gives `live`. A scan with no valid frame gives `uncertain`. A local profile is optional. The main panel shows the last completed scan. Developer Mode shows the current frame score, the last scan mean and its valid-frame count. Attendance saving is disabled in this workspace.

## Local attendance

Add `--attendance` to the Local workspace command to show the manual Attendance panel. Start or end a local session from this panel. Sessions have unique IDs and last until ended or until the app closes. The verification status and confirmation control come from the dedicated attendance service.

Recording is currently blocked pending an approved, versioned liveness decision for the same capture and face as recognition. The three-second AENet-Full preview continues to run. Its scan result carries no attendance authorization.

The attendance writer uses `artifacts/attendance/attendance.sqlite3` by default, with one record per person per session and timestamps in UTC. For isolated testing, use both `--local-profile-db PATH` and `--attendance-db PATH` with separate test files. A count of recent saved records appears after a successful write. The database is created on the first authorized write.

## Three-model webcam trial

Add both options below to run the historical comparison with the original trial checkpoints.

```sh
--liveness-trial-checkpoint '/Users/elvyx/Downloads/liveness-inference.pt' \
--liveness-trial-checkpoint '/Users/elvyx/Downloads/liveness-inference (1).pt'
```

The trial uses the detected BGR crop, converts it to RGB and applies the documented LIV-03 bilinear 224×224 transform with pixel values scaled to [0, 1]. It calculates the softmax score for class 1 (spoof) from the checkpoint's two outputs and compares it with that checkpoint's recorded threshold. The original LIV-04 inference code and quality policy are required to confirm equivalence. Treat a score below threshold as an experimental observation. Genuine registration and attendance remain blocked, and the SYS-02 gate continues to use simulated liveness.

To use memory-only test profiles with the historical comparison, remove `--local-workspace`. Select **Create test profile…**, enter a display name and confirm consent. Capture at least three distinct reference photos of the same person, then select **Save test profile**. The trial assigns a test ID and matches later webcam frames against its own in-memory gallery. **Clear test profiles** removes those entries, and closing the trial discards them. When supplying a read-only gallery with `--gallery`, matching uses that gallery.

Compare a live face, a phone-screen photo and a printed photo, leaving the frame between cases. In Developer Mode, use **Runtime and presentation → Pause refresh** while a score is visible to inspect or export both scores and thresholds. The export excludes photos, embeddings, names and gallery data.

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
│   ├── recognition/               # Data/pair loaders, trained CPU embedding and matching
│   ├── liveness/                  # Reserved for the trained anti-spoofing module
│   ├── emotion/                   # ResNet-18 and legacy EmotionCNN CPU inference
│   └── ui/                        # Shared webcam pipeline, modes and guarded registration
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

[Requirements coverage](docs/requirements.md) maps the brief to the implementation areas. [Environment/data evidence](reports/README.md) links the earlier configured-data, CPU/IPython and strict CUDA runs; their source hashes and original 46-test scaffold coverage are historical. Current integration verification is described in [dual-mode delivery](docs/ui-03-dual-mode.md). [Data evidence](docs/data-loading.md#observed-evidence-and-acceptance) separates acquisition and real dataset checks from fixtures. Independent reviewer execution remains pending. Checkpoints, environments, biometric data and newly generated diagnostic/test logs stay local and outside Git.

