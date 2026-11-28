# Facial Attendance

COS30082 desktop webcam application for face recognition, anti-spoofing, local registration, and facial expression preview. It includes a separate Attendance test mode for manual and automatic recording to local SQLite.

**Demo status.** Five-sample registration and local recording have been exercised on Apple Silicon macOS. Attendance writes are limited to a separate test workspace. A live clip of a different person produced a false identity match and a test attendance record, so the demo does not enable real attendance. See [verification results and remaining checks](docs/attendance-demo.md#verification).

## Run the demo

Use a Python **3.13.x** environment with Tk and the project dependencies. See [runtime setup](docs/setup.md) for macOS, Windows, and Linux. Run the commands from the repository root.

Place the team-supplied files at these local paths.

| Artifact | Path |
| --- | --- |
| Selected metric-learning Face checkpoint | `artifacts/models/face-recognition-resnet18.pt` |
| Emotion ResNet-18 checkpoint | `artifacts/models/emotion-recognition-resnet18-v1.pt` |
| AENet-Full epoch 18 checkpoint | `artifacts/models/liveness-aenet-full-epoch18.pt` |
| YuNet detector | `artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx` |

With your environment activated, launch the isolated **Attendance test · Auto record** demo on macOS:

```sh
PYTHONPATH=src python -m facial_attendance.ui.webcam_app \
  --local-workspace --attendance --attendance-verification-test \
  --attendance-auto-retry-test --attendance-auto-record-test \
  --camera 0 --detector yunet \
  --face-checkpoint artifacts/models/face-recognition-resnet18.pt \
  --emotion-checkpoint artifacts/models/emotion-recognition-resnet18-v1.pt \
  --local-profile-db artifacts/attendance-verification-test/demo-box/profiles.sqlite3 \
  --attendance-db artifacts/attendance-verification-test/demo-box/attendance.sqlite3 \
  --opencv-threads 1
```

The local webcam workspace uses the tight YuNet box for Face Recognition. Register again in the new demo workspace to generate compatible profiles. The two files are created on the first successful registration and attendance write respectively. Reuse the same box profile path when reopening the demo. For manual recording, omit `--attendance-auto-record-test`. For preview and local profiles with recording locked, keep `--attendance` and omit the three attendance test flags and both database overrides. See the [demo guide](docs/attendance-demo.md) for the Windows command, storage, and checks.

## Walk through the app

1. **Register.** Open **Developer Mode**, select **Register new person**, and show one unregistered face. Enter a display name, confirm consent, and follow the five pose prompts. A live three-second preview enables capture. Each selected photo passes its own liveness check before it counts. Select **Save profile** after five accepted references.
2. **Show the result.** Switch to **Standard Mode**. The overlay centers the name within the width of the face outline. **Face Recognition** shows **Preview match** when a profile matches. **Anti-Spoofing** shows the last completed three-second scan. **Emotion Detection** shows the model's facial expression estimate.
3. **Record attendance.** Select **Start session** and show one enrolled face. Manual mode enables **Record attendance** after current verification. Auto record waits for at least three continuous seconds of verified matching and writes once per person per session. A new session permits another record.
4. **Review.** **Developer Mode** opens **Review details** with the match score and threshold, latest liveness sample, last scan, and attendance mode. **Technical details** contains the diagnostic JSON. **Pause updates** freezes the review view, and **Export snapshot** exports an allowlisted diagnostic snapshot.

Registration and attendance are separate operations. Attendance combines face recognition and liveness from the same capture and needs at least five consecutive live samples spanning one second, with gaps of at most 0.5 seconds. The displayed three-second preview and Emotion results carry no recording authority. The optional automatic retry starts one new attempt with fresh frames after spoof. See [the full workflow](docs/attendance-demo.md#how-verification-works).

## Development and delivery

- [Demo and verification guide](docs/attendance-demo.md) covers settings, manual checks, SQLite and release limits.
- [Runtime setup](docs/setup.md) covers dependency installation and Windows, macOS, Linux, and Colab environments.
- [Architecture](docs/architecture.md) and [requirements coverage](docs/requirements.md) describe the current integration and outstanding checks.
- [Data loading](docs/data-loading.md) documents the assigned dataset and local paths. [Verification evidence](reports/README.md) indexes historical infrastructure and dataset runs.
- [Recognition Evaluation sandbox](docs/ui-03-recognition-sandbox.md) and the [earlier integration notes](docs/ui-03-integration-delivery.md) document the separate evaluation tools and earlier implementation.

Run the regression suite from the repository root with the compatible environment active:

```sh
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 python -m unittest discover -s tests -v
```

Some tests require the supplied local checkpoints or a native Tk desktop session. See [the full local regression command](docs/attendance-demo.md#run-the-regression-suite). GitHub Actions installs the lightweight test dependencies for the source job and the pinned CPU dependencies for the Windows and macOS jobs. Local checkpoints, biometric profiles, attendance databases, personal recordings, and diagnostic exports stay under ignored `artifacts/`.

## Repository layout

- `src/facial_attendance/recognition/`, `liveness/`, `emotion/` contain the model and evaluation modules.
- `src/facial_attendance/ui/` contains the webcam, registration, and review interface.
- `src/facial_attendance/attendance/` contains the verification policy, session service, and SQLite writer.
- `scripts/`, `tests/`, `configs/`, and `docs/` contain entry points, regression coverage, configuration, and guides.
