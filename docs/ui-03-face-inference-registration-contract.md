# UI-03.1 — Face inference and registration contract

Prepared 18 November 2026 on `feature/ui03-ui04-emotion-integration`, based on verified SYS-05 `798f2e7b9484a682e50c70b0b4689cd2fde316dc`. The trained Face preview and controlled generated-input registration mechanics are now implemented; genuine live enrollment/attendance remain blocked by missing trained liveness. See [current implementation and verification](ui-03-integration-delivery.md). Owner/reviewer agreement is pending.

## Sources and selected decision

Read against [SYS-02 Final v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d), [UI-02](https://app.notion.com/p/412863094ebe80c09ffce13f894f16ae), and the current acceptance criteria, Requirements rows 79–87, on 18 November. The older SYS-02 gallery warning describes the previous prototype: the latest inspected UI-01 implementation now supports schema 0.2. The compatible gallery implementation and its 25 tests are now reused verbatim in this branch, without merging unrelated changes.

| Authoritative implementation | Inspected revision |
| --- | --- |
| [FAC-04 network and checkpoint loader](https://github.com/Quynh-Trang12/facial-attendance/blob/5700e02ce42fbe65a316dc578998b47e996926d2/src/facial_attendance/recognition/model.py) | `5700e02ce42fbe65a316dc578998b47e996926d2` |
| [FAC-05 selected decision](https://github.com/Quynh-Trang12/facial-attendance/blob/6212d86305f425dc4a4d3ce251966661e6d58bef/evidence/FAC-05/selection.json) | `6212d86305f425dc4a4d3ce251966661e6d58bef`, selection blob `620d5c1c8df61a451d24a8ba79fbb45cb318d9dc` |
| [UI-01 gallery 0.2](https://github.com/Quynh-Trang12/facial-attendance/blob/82a74f8c1ec8f8b769b541c3e0527dd28ca85c3e/src/facial_attendance/ui/gallery.py) | `82a74f8c1ec8f8b769b541c3e0527dd28ca85c3e` |

Selected approach: **metric**, model ID `20261114-125242-metric-74a8657-s42`, weights version `cb38ea47ceff`. Compare cosine similarity and accept when **score >= 0.4619967254593038**. FAC-05 selected the lowest validation cosine with FAR <= 1%; this is a published validation decision, not a new webcam measurement. The source and values are pinned in `configs/ui03-recognition-selection.json`. Do not recalibrate this threshold in the GUI.

## Request and preprocessing

`FaceRequest(context, bbox, frame)` carries immutable context fields `contract_version=sys02-v1`, `attempt_id`, `action=registration|attendance`, `capture_id`, `face_id`. The exact associated source is BGR `uint8[H,W,3]`. The original detector box contains integer, half-open coordinates with `0 <= x1 < x2 <= W`, `0 <= y1 < y2 <= H`. No-face, multiple-face and invalid boxes fail before inference. `face_id` identifies a selected region, not a person or tracked identity.

The recognition adapter owns a separate copy of its input. Proposed source crop follows SYS-02: square centered on the detector box, side `1.2 * max(width,height)`; round outer bounds outward, clip to source boundaries, reject zero area. Clipping can make an edge crop rectangular; do not invent padding or change the shared detector box. This default margin and edge behavior require validation with the selected face checkpoint before UI-03.2 acceptance. Store the crop policy/version with the adapter and gallery-generation configuration.

Convert BGR to RGB once; apply the FAC-04 deterministic evaluation transform: bilinear resize to 64×64, float32 /255, channel normalization mean `(0.485,0.456,0.406)`, std `(0.229,0.224,0.225)`. Model tensor is `[N,3,64,64]`. Never reuse Emotion's grayscale tensor or guessed liveness preprocessing.

FAC-04's `embed()` actually emits **raw** `[N,512]` embeddings before the identity head. The adapter must reject nonfinite/zero-norm outputs and L2-normalize them to satisfy the SYS-02/gallery interface. Classification logits are not embeddings. Gallery vectors are finite 512D unit vectors (norm 1 ± 1e-3); scoring is cosine between compatible embeddings.

## Loading and scoring

The adapter loads the selected face `best.pt` on CPU with `weights_only=True`, construct `resnet18-3x3-stem-no-maxpool` with the checkpoint's identity-head shape, load strictly, call `eval()`/`inference_mode()`, and recompute FAC-04's weights version. That algorithm hashes sorted tensor name, dtype and contiguous bytes, then takes 12 hex digits; it differs from the Emotion full digest including shape.

Reject wrong model ID/version/approach, architecture, 64-pixel input, 512D output, mean/std or invalid tensors. A training checkpoint can legitimately record `threshold=None`: bind the separately verified FAC-05 selection to the matching verified weights without editing the checkpoint. A conflicting embedded threshold is an error. The supplied Face file SHA-256 is bc4502836f306c886323c444f11d6ddd7fe02515ba40a467d22f486fbfdcbc4f; its weights version and finite deterministic output are verified independently of that file hash.

Require gallery schema/model ID/version/dimension compatibility before comparing. Select maximum finite cosine; use deterministic person-ID order for exact ties. An empty compatible gallery yields Unknown (`person_id=null`, `name=null`, `score=null`, `accepted=false`); incompatible or invalid galleries yield an operational error, not Unknown. A valid below-threshold comparison yields Unknown with its best score. An accepted comparison returns canonical `person_id`, registered `name`, score, `accepted=true`, model ID/version. Do not independently invent an `identity_id`.

## Identity display and enrollment flow

| Current condition | Planned UI | Genuine operation |
| --- | --- | --- |
| Same-capture trained live + accepted match | Registered name and ID | Further attendance checks required |
| Same-capture trained live + valid Unknown | `Unknown — Register a new person` | No attendance; explicit new registration attempt |
| Pending/invalid/model unavailable | Processing/Unavailable; clear old identity | Block |
| Spoof/uncertain/stub/no face/multiple faces | Warning/retry/single-face guidance | Block |

Registration is an explicit new action/attempt, never an automatic consequence of Unknown. Collect person ID and display name: nonempty trimmed Unicode text, name 1–64 characters; duplicate IDs casefold-match and cannot overwrite an existing person. Duplicate names are allowed in the inspected gallery; ID remains canonical. Validate an already-known person/duplicate template against the selected matching policy before enrollment; do not create a second identity silently.

Target 5 accepted captures, minimum 3 and maximum 10. Each retained sample needs its own current valid trained liveness, source association, face quality and compatible normalized embedding; count validation alone does not prove those checks. The UI-01 schema supplies the consistency rule: every retained vector must meet the selected threshold against the averaged normalized template; the duplicate guard rejects a template already matching a gallery identity at that threshold. Face size must be at least 64 pixels on its shorter side. Pose/lighting and crop-domain acceptance still require physical validation. Retry, spoof, camera failure, incompatible model or attempt cancellation discards uncommitted samples; uncertain samples are never retained.

Average individually normalized accepted vectors, reject a zero/nonfinite mean, then L2-normalize the average. Gallery schema 0.2 stores one vector in `embeddings` plus `person_id`, `name`, `created_at`, `n_images`; root fields are `schema_version`, `model_id`, `model_version`, `embedding_dim`, `people`. The current gallery implementation rejects legacy incompatible schemas rather than silently migrating.

Planned sequence: recheck active context and eligibility → validate template/identity → stage gallery change → explicit storage success → report enrollment success → start a fresh attendance attempt → recognize against updated gallery. Confirm success only after the chosen storage operation succeeds; roll back on failure. Cross-restart persistence is a documented implementation choice, not a separate acceptance prerequisite. The webcam performs no gallery writes. Controlled generated-fixture tests exercise in-memory registration and temporary-gallery save/reload only.

## Errors, correlation and verification

Every completion echoes the original full context. Never relabel old embeddings with new capture IDs. A newer sample, Retry, loss/change of face, multiple faces, camera failure, model/gallery replacement or action change invalidates current results. Liveness and recognition must match the same capture and face before any future authorization.

Return typed recognition errors: `invalid_input`, `version_mismatch`, `model_unavailable`, `gallery_incompatible`/`storage_error` at their owning boundaries, or `inference_error`; never substitute a demo identity after failure. Recognition/liveness errors block genuine operations. Emotion errors affect its display only.

`recognition/contracts.py` and 8 tests check selected metadata binding, threshold equality/adjacent values, invalid scores, normalized dimension, ID/name/sample bounds and gated display states. Those original contract tests do not themselves load a model; the new trained-integration tests load the actual Face checkpoint, run matching and isolated enrollment/reload, without authorizing a real person. Emotion remains a separate checkpoint at `/Users/elvyx/Downloads/best(Dung).pt`.

Owner acceptance: verify the selected source values; review the crop/edge proposal and raw-to-normalized boundary; agree known/unknown UI and explicit enrollment; agree error and per-sample liveness rules. The matching face checkpoint is now supplied; physically validate crop/domain behavior and review known/unknown, controlled enrollment/subsequent recognition and duplicates. Genuine enrollment requires trained liveness before UI-03.2 can be fully accepted. Independent agreement remains pending.
