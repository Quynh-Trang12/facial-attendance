# UI-04.1 — Trained adapter and GUI contract

> Historical adapter contract and verification notes. See the [current demo guide](attendance-demo.md) for trained liveness, registration and isolated attendance.

Prepared 18 November 2026 against [SYS-02 Final v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d), [UI-02](https://app.notion.com/p/412863094ebe80c09ffce13f894f16ae) and Requirements rows 85–87. Design prepared for owner/model-owner agreement; no independent approval or full UI-04.2 completion claimed.

Current extension: the selected Face checkpoint is now integrated alongside Emotion. Its displayed matches are trained but UNVERIFIED, because liveness remains simulated; genuine registration/attendance are blocked. Controlled registration embeds only internally generated fixtures into a separate memory gallery. See [current delivery and tests](ui-03-integration-delivery.md).

## Adapter boundary

All adapters take `FaceRequest(context,bbox,frame)` and return `Completion(original_context,result)` or a typed `OperationError(module,code,message,blocks_persistence)`. Context is exactly `contract_version`, `attempt_id`, `action`, `capture_id`, `face_id`. The source is the immutable original BGR uint8 frame with valid source-coordinate half-open box. `face_crop` is a view of that source; adapters make copies before conversion or normalization. Only local in-memory pixels are passed; frame data is not serialized into debug JSON.

Each module owns its crop, tensor, preprocessing version, model loading and output validation. A module's preprocessed tensor is never the shared source. Load once on a bounded adapter worker using CPU-compatible, weights-only deserialization and explicit architecture construction. Verify file identity, weights and metadata before inference. No model is silently downloaded, no inference adapter writes attendance/gallery data, and no module can authorize itself.

## Output fields and provenance

| Module | Required result semantics |
| --- | --- |
| Recognition (trained preview implemented) | Canonical `person_id`, `name`, finite cosine `score` or null, `accepted=true|false`, selected `model_id`, `model_version`, `source=model`, checkpoint hash, preprocessing version; compatible gallery. Name/ID remain UNVERIFIED. See UI-03 contract. |
| Liveness (planned trained adapter) | `status=live|spoof|uncertain`, machine-readable `reason`, finite `p_spoof` in [0,1] or null on failure, `source=model`, `model_version`, `preprocessing_version`, `decision_policy_version`; record checkpoint identity in adapter provenance. |
| Emotion (implemented) | Explicit `class_id=0|1|2`, matching `label=Negative|Neutral|Positive`, finite predicted-class softmax confidence; `source=model`, `model_id`, full weights `model_version`, `checkpoint_sha256`, `preprocessing_version`, `selected_epoch`. |

Emotion provenance extends the application envelope additively; existing `Emotion(id,label,confidence)` remains `source=stub` with empty model metadata. Context remains `sys02-v1`. Invalid model provenance clears Emotion with a display-only error. A stub cannot carry fabricated trained-model metadata.

The selected liveness design is RGB binary classification, 0=live, 1=spoof; higher `p_spoof` indicates more spoof evidence. Numeric input size, normalization, crop/context, threshold and quality policy must come from the actual selected trained checkpoint/config and LIV validation. These are **not supplied for this integration**, so no numeric values are guessed. A low spoof score alone is insufficient: report live only after the versioned quality/confidence policy passes, otherwise uncertain. The controller must consume the adapter's validated decision, not independently re-threshold it.

Emotion copies the detector-region crop, BGR→RGB→Pillow grayscale, bilinear 48×48, float32 /255, `[1,1,48,48]`; no augmentation or mean/std. Version: `pillow-bgr-rgb-gray48-bilinear-div255-v1`. Architecture is EmotionCNN [32,64,128], dropout 0.3, selected epoch 9; output logits `[1,3]`, softmax then argmax. Confidence is not accuracy or a diagnosis. Recognition uses its own RGB/ImageNet preprocessing; liveness remains checkpoint-specific.

## Scheduling and call order

1. Capture maintains only the latest preview frame; Tk never waits for detector/model work.
2. Existing SYS-05 scheduling throttles analysis (minimum 300 ms), with one detector worker and one adapter worker, each bounded. Frame/context tokens and age limits reject superseded/expired work. A slow initial Torch load may lose its first result; it never blocks preview rendering or creates extra workers.
3. Validate exactly one original-frame face, clear previous model fields, create the current request.
4. Run liveness first. Spoof aborts remaining work and requires Retry. Uncertain never runs recognition; Emotion may still run for display. Missing/invalid/error liveness blocks recognition and genuine operations.
5. Only a valid current live decision reaches recognition. The present mixed harness permits its existing **simulated** live path to demonstrate simulated identity, with no authorization.
6. Run Emotion for display on eligible non-spoof requests. Emotion errors clear Emotion without changing independently valid identity/liveness decisions.
7. Tk accepts only context-matching, timely completions and updates current labels/optional debug JSON. The existing GUI scheduling, rendering and diagnostics are retained.

The existing function `evaluate_adapters` delegates through `run_stubs`, whose historical name describes its authorization scope. The mixed implementation reuses that bounded dispatcher; it is not a production trained-liveness authorization route. Current `PrototypeController` deliberately rejects `Liveness(source=model)`, and `Envelope.can_save` is always false. This guard is preserved. Integrating trained liveness and genuine persistence requires a separately tested production controller/policy, not changing a label or relaxing the stub guard.

## Display, resets and authorization

| State | Current/planned display | Genuine save |
| --- | --- | --- |
| Current mixed result | MIXED DEMO; trained Face matches UNVERIFIED when enabled, Liveness simulated, Emotion trained CNN when enabled | Always false |
| Trained Emotion current | Architecture, model ID, weights prefix, epoch and checkpoint-hash prefix; full metadata in debug JSON | Emotion grants no permission |
| Processing/new sample | Clear old results, show Processing | Block |
| No face | Clear identity/liveness/Emotion; no-face guidance | Block |
| Multiple faces | Clear all current results; require one face | Block |
| Spoof | Warning, stop request; discard uncommitted enrollment samples in future production flow | Block |
| Uncertain | Retry/fresh-frame guidance; no verified identity | Block |
| Emotion model/hash/inference failure | Emotion Unavailable + typed error; preserve independently valid other fields | No additional Emotion gate |
| Camera/liveness/recognition failure | Clear invalid current fields and show error/recovery guidance | Block |

All previous identity/liveness/Emotion fields clear on a fresh analysis, face loss, multiple faces, camera failure or Retry. The bounded outline hold is visual only, not identity tracking; it cannot keep old model fields current. Face IDs are not persistent identities.

Late callbacks retain their original full context and cannot overwrite a new capture or retry. On camera restart the generation changes. A valid completion for an old generation/expired frame is discarded before controller application. Clean close invalidates state and stops the existing workers; an uninterruptible native call may require its bounded shutdown wait, and no unlimited thread/queue fallback is added.

Typed Emotion failures are `model_unavailable` for missing file/runtime, `version_mismatch` for checkpoint/hash/metadata mismatch, `invalid_input` for bad source/crop, and `inference_error` for invalid logits/model failures. All use `module=emotion`, `blocks_persistence=false`. Context mismatch discards a completion rather than fabricating a prediction/error for a new face.

## Agreement and remaining acceptance

UI-04.1 design acceptance requires Vỹ and chị Dung to agree the fields, actual Emotion preprocessing/provenance, future liveness configuration, call order, spoof/uncertain behavior and displayed states. This document and tests prepare that review; they do not record agreement on anyone's behalf.

UI-04.2 preserves actual trained Emotion output alongside the trained Face preview. Trained liveness integration, real spoof rejection, genuine registration/attendance, physical current-version webcam acceptance and independent integrated review remain pending. The optional controlled enrollment worker is bounded to one job/result slot; it prepares drafts without mutation, and Tk commits only a current registration attempt into the isolated test gallery. The camera path exposes no enrollment writer. See `ui-03-integration-delivery.md` for current setup, results and checklist. No Final Owner Evidence is produced for incomplete criteria.
