# Webcam and Attendance demo

This guide covers the current Apple Silicon macOS demo for review. Both attendance modes write to an explicitly selected test database. Live face recognition can still match the wrong person, so the production writer remains disabled.

## Launch

Use Python 3.13.x and a working Tk installation. Install or reuse an environment from [runtime setup](setup.md), then add the team-supplied [model files](../README.md#run-the-demo). Run the [macOS launch command](../README.md#run-the-demo) from the repository root.

For a Windows review, use the equivalent PowerShell command below and the Windows CPU environment from the setup guide. The current version has not been verified end to end with a physical Windows webcam.

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m facial_attendance.ui.webcam_app `
  --local-workspace --attendance --attendance-verification-test `
  --attendance-auto-retry-test --attendance-auto-record-test `
  --camera 0 --detector yunet `
  --face-checkpoint artifacts/models/face-recognition-resnet18.pt `
  --emotion-checkpoint artifacts/models/emotion-recognition-resnet18-v1.pt `
  --local-profile-db artifacts/attendance-verification-test/demo/profiles.sqlite3 `
  --attendance-db artifacts/attendance-verification-test/demo/attendance.sqlite3 `
  --opencv-threads 1
```

Select the camera with `--camera INDEX`. For a different checkpoint location, pass `--liveness-checkpoint PATH` or `--yunet-model PATH` explicitly. The demo uses the selected AENet-Full epoch 18 release and YuNet. An existing compatible Python environment can replace the example interpreter.

| Flag | Effect |
| --- | --- |
| `--attendance` | Shows session controls. Recording stays disabled with this flag alone. |
| `--attendance-verification-test` | Enables attendance checks with two separate database paths under `artifacts/attendance-verification-test/`. The window title shows **Attendance test**. |
| `--attendance-auto-retry-test` | Starts one fresh attempt after spoof. A second spoof requires **Retry verification**. A new session or an explicit retry restores the allowance. |
| `--attendance-auto-record-test` | Saves after at least three seconds of continuous verification. The title shows **Auto record** and there is no separate confirmation button. |
| `--diagnostics` | Prints aggregate performance metrics to the console. **Developer Mode** controls the review window. |

## Reviewer walkthrough

| Check | What to do | What to look for |
| --- | --- | --- |
| Register a consenting person | Open **Developer Mode**, dismiss **Review details** if it covers the main window, and select **Register new person**. Show exactly one unregistered face, enter a name, and confirm consent. Follow the five poses and select **Save profile**. | **Registration saved** and the name appears on a subsequent preview match. Each retained reference passed its own liveness check. |
| Show preview and review | Switch to **Standard Mode**. Return to **Developer Mode** and open **Review details**. | The main view shows the name, last scan, and expression. The review window shows score, threshold, latest liveness sample, scan time and count, and attendance mode. **Technical details** opens the optional diagnostic views. |
| Record an enrolled person | Select **Start session**. Keep one enrolled person in frame. In manual mode, use **Record attendance** after it is enabled. | **Attendance recorded** and a new entry in **Recorded this session**. Auto record typically needs a little over three seconds of continuous verification. |
| Check duplicate prevention | Keep the same person in frame or verify them again during the same session. Then end the session, start a new one, and repeat. | The same session stays at one record. The new session gets one new record. |
| Check rejection | Use a new test session for an unknown person, an authorized phone-screen replay, or an authorized printed photo. Leave the frame before changing inputs. | No record is expected for these cases. Treat a record for the wrong person as a failed check even if the app reports success. |
| Check interruption | Leave the frame, add another face, or change the person while the test is waiting to record. | The app discards the previous verification. The next successful record requires a fresh attempt. |

**Pause updates** only freezes diagnostics. **Export snapshot** produces an allowlisted JSON report. **Copy JSON** in Technical details copies the current technical tab and can include identity metadata. Use the export for sharing review evidence.

## How verification works

- **Preview.** AENet-Full averages the scores from valid single-face samples over about three seconds. The checkpoint threshold is `0.46483826637268066`. A mean greater than or equal to the threshold yields `spoof`; a lower mean yields `live`. A scan without valid samples yields `uncertain`. The preview is informational for Attendance.
- **Naming a face.** The preview and Attendance name the best-matching profile only if its cosine score reaches the FAC threshold `0.4619967254593038` and beats every other profile by the margin in `configs/recognition.toml` (0.15). Otherwise the result is unknown, even when the best score is above the threshold. With one profile the score must reach the threshold plus the margin, about 0.612. The registration duplicate check stays on the threshold alone, so a face close to two profiles is refused as a duplicate.
- **Registration.** A completed live preview from the current attempt enables **Capture reference**. The five requested poses cover center, slight left and right, up and down. Only individually live captures count. An uncertain capture asks for a new photo; spoof discards the attempt and requires **Retry**. The gallery stores one averaged template per profile and rejects duplicate matches before saving.
- **Attendance.** The candidate reuses Trang's temporal gate and the [LIV-02 same-capture contract](https://github.com/Quynh-Trang12/facial-attendance/blob/25eafa26091be2f9dcd6b17aeffb75d801448435/docs/liveness-gating.md#registration-and-attendance-flow). It requires at least five consecutive live samples over at least one second, with gaps no longer than 0.5 seconds. The accepted recognition result must come from the exact gate-passing capture, under the current attempt, action, capture and face IDs. A camera failure, missing or multiple faces, changed identity, unknown result or stale completion cannot carry over an authorization. Emotion has no role in the write decision.
- **Options.** Automatic retry ends the spoof attempt and starts another with fresh samples. Auto record adds a minimum three-second live span measured from the first valid sample of the same continuous attempt. It uses the latest verified capture and goes through the same service pre-write checks. These are isolated test policies. The recording mode does not change the liveness model, recognition threshold, or preview scan.

## Where records are stored

The launch examples keep profiles in `artifacts/attendance-verification-test/demo/profiles.sqlite3` and attendance in `artifacts/attendance-verification-test/demo/attendance.sqlite3`. Each is a different database. Use the paths supplied on the command line when opening another test workspace. With the preview-only launch, the default profile database is `artifacts/local-recognition/profiles.sqlite3`.

The `attendance` table stores `attendance_id`, `person_id`, `person_name`, `session_id`, `recorded_at` (UTC), and `status` (`present`). SQLite enforces one row per `(session_id, person_id)`. The app displays success after the transaction completes. The two databases, diagnostic exports, recordings, and checkpoints stay out of Git.

To check counts without printing names, open the SQLite file after the first successful record and run:

```sql
SELECT session_id, COUNT(*) AS records
FROM attendance
GROUP BY session_id;
```

## Verification

| Evidence | Observed result |
| --- | --- |
| Physical webcam registration baseline, 25 November 2026 | A profile was enrolled with five reference captures and matched on webcam. The source baseline preserves 129 files from base `798f2e7` plus the then-current working changes. The stored replay check rejected spoof enrollment. |
| macOS physical Attendance test | The local auto-record test databases contain stored rows and pass SQLite integrity checks. This confirms local writes; the records alone do not validate every identity match or physical spoof case. |
| Offline replay with the actual Face and AENet-Full checkpoints | 17 distinct clips. All eight screen clips produced zero attendance records. Two clips of the enrolled participant wrote one row each in separate test sessions after about 3.0–3.3 seconds. One other live participant was falsely matched and generated one test record. |
| Regression suite on Apple Silicon macOS | 504 passed and one optional legacy-checkpoint test skipped. Coverage includes stale and mismatched results, duplicate recording, SQLite failure, registration, changing modes, stable sidebar scrolling, and diagnostic export. |
| Windows and independent reviewer checks | Physical Windows webcam verification and independent reviewer acceptance are pending. |

The observed wrong-person match is the blocker for real attendance. The replay set represents a small set of prerecorded cases; it does not establish a general false-accept or spoof-rejection rate. The next release check uses separate enrollment and test captures, including the previously misidentified person, several lighting conditions, and physical phone and print trials. Keep all test data consented and local.

Tested macOS model versions: Face metric ResNet-18 `cb38ea47ceff`, Emotion ResNet-18 epoch 10, and AENet-Full epoch 18 `f16fbfaf36046f75`. Face recognition uses the selected FAC cosine threshold `0.4619967254593038`. The native webcam run used Python 3.13.17, PyTorch 2.13.0, torchvision 0.28.0, and OpenCV 4.13.0.92 in an existing local environment. The repository pins define the shared setup; the exact pinned OpenCV version was not used for this webcam run.

### Run the regression suite

From the repository root in the prepared macOS Python environment:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
  FA_NATIVE_ATTENDANCE_TESTS=1 \
  FA_FACE_CHECKPOINT="$PWD/artifacts/models/face-recognition-resnet18.pt" \
  FA_LIVENESS_RELEASE="$PWD/artifacts/models/liveness-aenet-full-epoch18.pt" \
  FA_LIVENESS_YUNET="$PWD/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx" \
  python -m unittest discover -s tests -v
```

The supplied ResNet-18 Emotion integration tests look for `artifacts/emotion-resnet18-v1/best.pt` directly. Put the same checkpoint there if your model is set up only at the launch path. Set `FA_EMOTION_CHECKPOINT` to the separate legacy Emotion checkpoint for its compatibility tests. The optional historical liveness comparison uses `FA_TRIAL_FULL` and `FA_TRIAL_BASELINE` for its two legacy checkpoint paths. Report test skips alongside the total.

## Historical checkpoint comparison

For the earlier two-checkpoint liveness comparison, pass `--liveness-trial-checkpoint PATH` twice with the original experiment files. Keep the Face and Emotion checkpoint options and omit the Attendance test flags. Developer Mode's **Technical details → Model details** shows the comparison scores and thresholds. The experimental scores do not authorize registration or attendance. Without `--local-workspace`, test enrollment in this comparison stays in memory for the running app.
