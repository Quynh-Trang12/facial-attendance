# Embedding and enrollment schema

This schema fixes what the face module takes in, what it stores for each registered person and how a webcam face is matched (UI-01.1). It is the face-module part of the SYS-02 interface contract. Data roles, pair rules and score direction come from the [face data protocol](https://github.com/Quynh-Trang12/facial-attendance/blob/89d3eec48edfddb0d4f5bec7de4521d649dacf47/docs/face-data-protocol.md); this document does not repeat them. Values marked **pending** are parameters whose final number is set by the task named in the same row. Code reads `S`, `D` and `τ` from the checkpoint and never hard-codes them. Training settings and selection rules are in [experiment settings](https://github.com/Quynh-Trang12/facial-attendance/blob/71f3b56fd7fdc6e90106ad6fdeae8e8b23749237/docs/experiment-settings.md).

## Face input

Every image that reaches the embedding model, whether from the competition data or the webcam, goes through the same steps:

| Step | Rule | Final value set by |
| --- | --- | --- |
| Crop | One square face crop. Training images are already face crops and are used whole. A webcam frame is cropped around the detector box, using a square with side `m × max(box width, box height)` around the box centre, clipped to the frame. | `m` **pending**, default 1.2: UI-03 compares webcam crops with training samples |
| Size | Resize the crop to `S × S` pixels with bilinear interpolation. `S = 64`, the size of every image in the [full-data audit](https://github.com/Quynh-Trang12/facial-attendance/blob/f92eb649c3aaaaf7cea16e79f9403cb2dd915ba0/reports/data/colab-audit-2026-11-05.json). The loader's 112-pixel resize is for inspection only. | Fixed |
| Colour order | RGB. The loader decodes with Pillow in RGB. OpenCV webcam frames are BGR and must be converted before cropping. | Fixed |
| Scaling | `uint8` to `float32`, divided by 255, giving `[0, 1]` | Fixed |
| Normalization | Per channel, `(x − mean) / std` with mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`. Both approaches use these values, so the UI has one preprocessing path whether a backbone is partially fine-tuned from ImageNet weights or trained from scratch. | Fixed; stored in the checkpoint |
| Layout | Tensor `[N, 3, S, S]`, `float32`, on the model's device | Fixed |

## Embedding and gallery record

The model returns one vector per face. The face module L2-normalizes it, so stored and live embeddings always have length 1 and the dot product is the cosine similarity. For the classification approach, the embedding is the layer before the classification head, not the class logits.

The gallery file has a header with `schema_version`, `model_id`, `model_version`, `embedding_dim` and `people`. Each registered person has one record in `people`, holding one averaged template:

| Field | Type | Unit or range | Final value set by |
| --- | --- | --- | --- |
| `person_id` | string | Unique, non-empty, generated at registration (for example `emp-0001`); never derived from the name | UI-01.4 |
| `name` | string | Display name, UTF-8 (Vietnamese diacritics allowed), 1–64 characters | Entered at registration (UI-01.4) |
| `embeddings` | list of `float32[D]` | Exactly one entry after enrollment, the averaged template; L2 norm `1 ± 1e-3`; each value in `[−1, 1]` | Computed at enrollment |
| `created_at` | string | ISO 8601 UTC timestamp, for example `2026-11-20T03:15:00Z` | Set at enrollment |
| `n_images` | integer | Number of accepted frames averaged into the template, 3–10 | Set at enrollment |

`D = 512`, the embedding length of the model head in the [experiment settings](https://github.com/Quynh-Trang12/facial-attendance/blob/71f3b56fd7fdc6e90106ad6fdeae8e8b23749237/docs/experiment-settings.md#training-settings). The gallery reads it from the checkpoint and rejects any record whose length differs.

Two header fields identify the model, so the gallery can be checked against a checkpoint without loading the weights:

| Field | Value | Set by |
| --- | --- | --- |
| `model_id` | The training run's `run_id` | Checkpoint chosen by FAC-05 |
| `model_version` | First 12 hex characters of the SHA-256 of the `state_dict` tensors, in sorted name order | Checkpoint chosen by FAC-05 |

`model_version` hashes the weights, not the checkpoint file. The file also holds `meta`, which stores this value and `τ`, so hashing the file would change the value whenever `meta` is written.

The gallery holds biometric data. It lives under the ignored `artifacts/` directory, next to attendance records, and is never committed. Enrollment frames and face crops are not stored once their embeddings are computed. UI-01.4 chooses the file format. Whatever it chooses must store `float32` values exactly (no rounding to short decimal text) and keep all five record fields.

## Decision rule

For a live embedding `e` and enrolled templates `g_1 … g_K`:

1. Compute `s_k = e · g_k` for every template in the gallery (cosine similarity, higher means more likely the same person, as in the protocol).
2. Take the best match `k* = argmax s_k`.
3. If `s_k* ≥ τ`, the match is accepted (`accepted = true`). Otherwise the result is "unknown" and registration is offered. An empty gallery always gives "unknown".

`τ` is a single threshold for the deployed checkpoint. FAC-05 chooses it on the seeded `val_data` pair list defined in the experiment settings, never on the official trials. It is stored in the checkpoint metadata, so a threshold from one model is never applied to another. Its operating point, a false-accept rate of at most 1% on those pairs, is defined in the [experiment settings](https://github.com/Quynh-Trang12/facial-attendance/blob/71f3b56fd7fdc6e90106ad6fdeae8e8b23749237/docs/experiment-settings.md#selection-rules).

An accepted match is provisional. Recognition may run in parallel with liveness, but the controller shows the identity as verified, or records attendance, only after liveness returns `live` for the same capture and face and the attendance checks pass. Recognition alone never authorizes attendance ([LIV-02](https://github.com/Quynh-Trang12/facial-attendance/blob/25eafa26091be2f9dcd6b17aeffb75d801448435/docs/liveness-gating.md), SYS-02 §4.2).

## Enrollment

Registration (UI-01.4) builds one averaged template per person:

- **Frames.** Capture 5 frames, with 3 accepted as the minimum and 10 as the maximum. Each frame must contain exactly one detected face. The face box must be at least `S` pixels on its shorter side, so the crop is not upscaled.
- **Pose and lighting.** The person faces the camera with a neutral expression, without sunglasses or a mask, under even front lighting. Across the five frames, ask for small head turns (straight, slightly left, right, up, down; about 15° at most), so the template covers the poses seen during attendance.
- **Liveness.** Each frame is checked by liveness on its own. Only a frame whose result is `live` becomes a retained sample whose embedding may enter the template. What spoof, uncertain and failed results do is in the [liveness gating](#liveness-gating-ui-013) section.
- **Template.** `g = normalize(mean(normalize(e_1), …, normalize(e_n)))`. Normalizing each frame first stops any one frame from dominating, and normalizing again keeps the template on the unit sphere.
- **Consistency guard.** Each accepted frame must score at least `τ` against the template. If one does not, the frames may show different people or a bad capture, so the attempt ends without storing the template and registration starts again as a new attempt.
- **Duplicate guard.** Before storing, match `g` against the gallery. If the best score is at least `τ`, reject the registration and show the existing identity's name. That person is already enrolled, or the two people cannot be told apart at this threshold.

## Versioning

A gallery is valid only for the model that built it. Templates from different checkpoints live in different embedding spaces, and the threshold belongs to one checkpoint. On start-up the face module compares the gallery header's `model_id`, `model_version` and `D` with the loaded checkpoint. If they differ, it refuses to match and asks for re-enrollment. Since enrollment images are not stored, re-enrollment is the only way to migrate after a model change. Retraining therefore happens before people register for the demonstration, not after.

## Liveness gating (UI-01.3)

Reviewed on 10 Nov 2026 against [LIV-02 v1](https://github.com/Quynh-Trang12/facial-attendance/blob/25eafa26091be2f9dcd6b17aeffb75d801448435/docs/liveness-gating.md) and [SYS-02 v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d) §4.2 and §5.3. LIV-02 returns `live`, `spoof` or `uncertain`; the threshold `t_spoof` and its quality rules come from LIV-04. This section fixes what each result does to enrollment. Gating of attendance is defined in LIV-02 and is not repeated here.

### Effect of one liveness result on registration

| Result for one enrollment sample | Effect |
| --- | --- |
| `live`, from `source = model`, with the same `attempt_id`, `capture_id` and `face_id` as the crop | The sample is retained, and its embedding may enter the template |
| `uncertain` | The sample is not retained. A fresh capture is needed; the attempt continues |
| Missing, invalid or failed result | The sample is not retained, the UI shows the LIV-02 error state, and nothing is saved |
| Result for another attempt or capture, or arriving after a reset | Discarded without changing the display or the retained samples |
| `spoof` | The attempt is aborted. Every retained sample and its embedding is discarded, and a retry starts a new attempt |
| `source = stub` | Demonstration only; never stored |

A live result applies to its own capture only. It never carries over to another frame.

### Before the template is saved

1. At least 3 retained samples, each live on its own (5 targeted, 10 at most, as in [Enrollment](#enrollment)).
2. The template, the consistency guard and the duplicate guard use retained samples only.
3. The registration details are valid and the user confirms.
4. The final save check runs immediately before writing (LIV-02 step 7): the attempt is still active and not aborted, every sample is still live, and the event has not been saved already.
5. The identity and its template are saved together. Success is shown only after the save succeeds, so a failed save leaves no partial identity.

### Embedding and model versions

The gallery header's `model_id`, `model_version` and `embedding_dim` describe the recognition checkpoint only. On load the gallery refuses a file whose values differ from the loaded checkpoint (`FaceGallery.load` in [`gallery.py`](https://github.com/Quynh-Trang12/facial-attendance/blob/82a74f8c1ec8f8b769b541c3e0527dd28ca85c3e/src/facial_attendance/ui/gallery.py), gallery branch). The liveness result also has a `model_version` field, but it names the liveness checkpoint and is never written to the gallery header.

### Findings

| # | Before this review | Now | Source |
| --- | --- | --- | --- |
| 1 | A rejected frame was dropped and enrollment went on, so 4 live frames and 1 spoof still gave a template | A spoof aborts the attempt and discards every retained sample | LIV-02 step 4a; SYS-02 §5.3 |
| 2 | A frame was embedded and matched only after it passed liveness | Recognition may run in parallel; its result stays provisional until liveness is `live` for the same capture | LIV-02 state changes; SYS-02 §4.2; UI-02 §2 |
| 3 | A match at or above `τ` showed the identity and recorded attendance | The face module returns `accepted`; display and saving wait for liveness and the attendance checks | SYS-02 §4.2; LIV-02 rules for saving records |
| 4 | Open question on consecutive live frames, the threshold and the spoof message | Closed: each sample is checked on its own, `t_spoof` comes from LIV-04, the messages are in LIV-02 | LIV-02 score and display sections |
| 5 | No sample identifiers, user confirmation or final save check | Added above | LIV-02 steps 6R–8 |

### Left for later tasks

- `FaceGallery.register` cannot check where a template came from, so the enrollment step built in UI-03 must enforce the rules above, as the UI-01.4 gallery notes state.
- The UI-01.5 walkthrough (Notion v0.2) does not yet include user confirmation or the final save check; this is recorded in the UI-01.5 review.
- Webcam tests of these rules belong to UI-04.6, using LIV-02 test cases 4, 10 and 12.

## Face-module values for SYS-02

These are the face-module entries for the shared interface contract:

| Item | Value |
| --- | --- |
| Input crop | Square around the detector box, side `m × max(w, h)`, `m` default 1.2 (pending UI-03) |
| Input size | `64 × 64` |
| Colour order | RGB (convert from OpenCV BGR) |
| Scaling and normalization | `/255`, then ImageNet mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)` |
| Tensor | `float32 [N, 3, S, S]` |
| Checkpoint format | PyTorch `.pt` file holding `state_dict` plus a `meta` dictionary: `model_id`, `model_version`, `architecture`, `approach` (`classification` or `metric`), `embedding_dim` (`D`), `input_size` (`S`), `mean`, `std`, `threshold` (`τ`) and `threshold_source` |
| Embedding output | `float32[N, 512]`, L2-normalized by the face module |
| Match output | `person_id`, `name`, `score` (best cosine similarity), `accepted` (`score ≥ τ`); `person_id` and `name` are empty when not accepted |
| Failure behaviour | No face, more than one face or a face box smaller than `S` produce no embedding and no match. A spoof, uncertain, missing or failed liveness result never lets a match be shown as verified or saved. The UI shows the reason instead of an identity. |
