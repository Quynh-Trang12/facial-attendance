# Experiment settings and model selection

This document fixes how models are configured, selected and traced to the results we report and demonstrate (SYS-04). It applies to all four models. Data roles, pair rules and score direction come from the [face data protocol](face-data-protocol.md), and the checkpoint and gallery fields come from the [embedding schema](embedding-schema.md); this document does not repeat them.

No face model has been trained yet, so every number in the training tables is an unmeasured starting value. A run may change one if it records the new value and the reason in its manifest. A change to a selection rule needs a revision of this document before results are compared.

## Training settings

Both face approaches use the same backbone, input and embedding size, so the comparison measures the approach and not the architecture.

| Setting | Classification (FAC-03) | Metric learning (FAC-04) |
| --- | --- | --- |
| Backbone | ResNet-18 with a 3×3 stride-1 stem and no max-pool, for 64-pixel input; trained from scratch, no frozen layers | Same |
| Input | `64 × 64` RGB, normalized as in the [embedding schema](embedding-schema.md#face-input) | Same |
| Embedding | Global pooling, then a linear layer to `D = 512` | Same |
| Head and loss | 4,000-way softmax with cross-entropy; the embedding is the layer before this head | Triplet loss, margin 0.2, on L2-normalized embeddings |
| Sampling | Shuffled images | P×K batches, `P = 64` identities and `K = 4` images each, with semi-hard online mining |
| Optimizer | SGD, learning rate 0.1, momentum 0.9, weight decay 5e-4, cosine decay | SGD, learning rate 0.05, momentum 0.9, weight decay 5e-4, cosine decay |
| Batch size and length | 256, up to 30 epochs | 256, up to 30 epochs; one epoch is as many batches as one pass over the training images |
| Augmentation | Horizontal flip; 4-pixel pad and random crop | Same |
| Seed | 42 | 42 |

If triplet training collapses from scratch, a warm start from the FAC-03 backbone is allowed as a separate run with its own `run_id`. Each run saves a resumable checkpoint every epoch, because Colab sessions can end mid-run. A third approach is not part of this baseline.

This table has the shape the report needs for its training configuration section. Each owner fills in the row for their model, and the values must match that model's final manifest:

| Model | Owner | Optimizer | Learning rate | Batch size | Epochs | Loss |
| --- | --- | --- | --- | --- | --- | --- |
| Classification CNN | Quế An | SGD | 0.1 | 256 | Up to 30 | Cross-entropy |
| Metric-learning CNN | Quế An | SGD | 0.05 | 256 | Up to 30 | Triplet, margin 0.2 |
| Anti-spoofing | Trang | To be filled in by the owner | | | | |
| Emotion detection | chị Dung | To be filled in by the owner | | | | |

## Data

Partition roles are defined in the [protocol](face-data-protocol.md#partition-roles). SYS-04 adds two rules for training runs:

- **Filtered data.** How overlapping images are removed is still an open decision in [face data leakage](face-data-leakage.md). Whatever is accepted, a run records the hash of the exclusion list it used, so results from different data can't be mixed by accident.
- **Sampler check.** The P×K sampler must confirm that every training identity still has at least `K` images after filtering. The smallest identity has 15 images before filtering.

## Selection rules

These rules are fixed before any comparison. Selection never uses the official trials.

| Decision | Evidence | Data |
| --- | --- | --- |
| Best epoch, both approaches | Highest cosine AUC | Validation pairs |
| Approach and final model (FAC-05) | Higher validation-pair AUC of the two final models | Validation pairs |
| Threshold `τ` | Lowest score whose false-accept rate is at most 1% | Validation pairs |
| Reported verification result | ROC and AUC on all 8,805 trials, scored once per final model | `verification_pairs_val.txt` |
| Reported classification accuracy, during training | Top-1; monitored but not used for selection | `val_data` |
| Reported classification accuracy, final | Top-1, once per final classification model | `test_data` |

**Validation pairs.** The list is built once from `val_data` with seed 42 and reused unchanged for every checkpoint of both approaches. It holds one same-identity pair for each identity that still has two images, and an equal number of different-identity pairs drawn uniformly. Before filtering, that is 4,000 of each.

**Liveness and emotion.** Each module selects its checkpoint and threshold on validation data and reports on separate held-out data. The [LIV-01 protocol](https://github.com/Quynh-Trang12/facial-attendance/blob/docs/liv-01-anti-spoofing-protocol/docs/liveness-protocol.md) (under review) defines this for liveness. chị Dung records the same for emotion under EMO-01.

## Cosine and Euclidean comparison

On L2-normalized vectors, `‖a − b‖² = 2 − 2·cos(a, b)`. The two measures then rank every pair identically and give the same AUC, so comparing them shows nothing.

- **Cosine similarity** uses the model's raw output; scale does not affect it.
- **Euclidean distance** is computed on the same raw output before normalization, and negated as in the [protocol](face-data-protocol.md#score-direction).

This gives four official AUC values, one per approach and measure. Cosine stays the deployed measure unless FAC-05 records a reason to change it. SYS-06 implements both.

## Traceability

Every training run commits a manifest at `reports/runs/<run_id>/manifest.json`, where `run_id` is `YYYYMMDD-<approach>-<short commit>-s<seed>`.

| Group | Fields |
| --- | --- |
| Identity | `run_id`, `module`, `approach`, `git_commit`, `git_dirty`, `seed` |
| Data | `dataset_manifest_sha256`, `exclusions_sha256`, `pairs_file_sha256`, or the module's equivalents |
| Settings | Full configuration, any change from this document with its reason, framework versions, GPU model |
| Selection | `selected_epoch`, selection metric and value, `threshold`, `threshold_source` |
| Output | `model_id`, `model_version`, checkpoint storage location |

- A reported result needs `git_dirty` to be false, or the report must say why it isn't.
- Checkpoints stay out of git, in `artifacts/checkpoints` and a shared Drive folder.
- Every number and plot in the report names its `run_id`. Pair scores from SYS-06 are saved beside the manifest, so ROC and AUC can be recomputed without the model.
- The GUI reads `model_id` and `model_version` from the checkpoint, and the gallery stores them, so the demonstrated model links back to its manifest.

## What must be disclosed

- **Closed-set selection.** Validation pairs contain only identities the model trained on, so `τ` may be too permissive for people it has never seen. SYS-07.6 checks it on the webcam. The AUC on those pairs is a selection measure, not an estimate of real-world performance.
- **`val_data` and metric learning.** The brief says `val_data` is not needed for metric learning. This project uses it for selection anyway, because the official trials must stay out of selection, and the report must say so.
- **Unfiltered data.** As published, `val_data` shares 195 images and `test_data` 167 with `verification_data`, as the [leakage note](face-data-leakage.md) records, so selection data is not fully separate from the official trials. A result from a run without an exclusion list must say so and can't be called independent.
- **Reuse of official trials.** If any run does use them for a selection step, the report names that step and does not call the result independent.
- **Clean-trial AUC.** An AUC on the trials with no shared image may be reported as extra analysis only. It is not evidence that overlap is gone.
- **What a hash can't show.** Zero hash overlap does not rule out the same person appearing across partitions, as the [leakage note](face-data-leakage.md) explains.
