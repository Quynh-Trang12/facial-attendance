# Architecture

The COS30082 application runs in one Python desktop process. Feature modules own their models and preprocessing. The GUI loads the selected checkpoints before starting the webcam. Registration and attendance use different operations and different local SQLite files.

| Area | Responsibility |
| --- | --- |
| `recognition/` | Dataset and evaluation helpers, normalized embeddings from the selected metric-learning Face model, cosine matching, and gallery checks. |
| `liveness/` | AENet-Full inference on the original YuNet face crop and the consecutive-live gate. |
| `emotion/` | The ResNet-18 facial expression model, with a legacy EmotionCNN adapter for compatibility. |
| `ui/` | Webcam lifecycle, bounded detector and model workers, guarded presentation, local registration, and review diagnostics. |
| `attendance/` | Same-capture verification in the isolated test workspace, session handling, and transactional attendance persistence. |
| `config.py`, `environment.py`, `runtime_checks.py` | Shared paths, version and device checks, and infrastructure evidence. |

## From webcam to record

1. The webcam delivers source frames. YuNet detects faces on sampled frames while the camera preview continues. The controller rejects stale work and requires a single valid face for verification.
2. The recognition adapter crops its own square region, converts BGR to RGB, and uses the selected Face transform. Liveness and Emotion use the original detector box from the same source frame. Emotion is informational.
3. The GUI presents recognition as a preview match and shows the last three-second liveness scan. The overlay and review window read presentation state without granting authority.
4. Registration binds each reference to its current capture and attempt. It requires five accepted live samples and checks for duplicate identities before storing an averaged template.
5. The Attendance test coordinator consumes the current liveness and recognition results, binds same-frame evidence to an attendance attempt, and reuses the temporal gate. The Attendance Service rechecks the session, current capture, liveness, identity and gallery before writing. Manual and automatic recording share that writer.

The source separates the three-second liveness preview, the verification gate, and the additional delay in automatic recording. Technical views can display historical results independently. They do not feed registration or attendance.

## Storage and review boundaries

- The local profile store holds display names and averaged biometric templates. The attendance store holds event rows and enforces a unique person-and-session pair. There is no remote database or web service.
- Attendance writes require isolated test paths. The group keeps real recording disabled because webcam replay testing reproduced a wrong-person match. See [the demo workflow, local storage and evidence](attendance-demo.md).
- Diagnostics exports use an allowlist. Keep profiles, SQLite attendance files, checkpoints, recordings and private evidence in ignored `artifacts/`. Store shareable verification summaries in `reports/` or the relevant guide.
- `configs/project.toml` resolves shared paths from the repository root. [Runtime setup](setup.md) and [data loading](data-loading.md) document the supported environments and assigned dataset.
