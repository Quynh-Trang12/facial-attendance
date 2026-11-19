# Face embedding models

This page covers the classification-based face embedding (FAC-03): how it is built and trained, how a run is recorded, and how to use the checkpoint. Settings and selection rules come from the experiment settings (SYS-04); input, checkpoint and gallery fields come from the embedding schema (UI-01.1). Neither is repeated here.

## Network

`src/facial_attendance/recognition/model.py`, architecture name `resnet18-3x3-stem-no-maxpool`.

| Part | Definition |
| --- | --- |
| Backbone | torchvision ResNet-18 with `weights=None`. The stem is a 3×3 stride-1 convolution and the max-pool is removed, so a 64×64 face keeps 64×64 maps in the first block. |
| Embedding | Global average pooling (512 values), then `Linear(512, 512)`. `embed()` returns this raw vector, before any normalization. |
| Head | `Linear(512, 4000)`, one output per training identity, trained with softmax cross-entropy. Only `forward()` uses it. |
| Trainable layers | All. Trained from scratch; nothing is frozen and no pre-trained weights are loaded. |

The embedding is the layer before the head, so verification never needs the head and a person outside the 4,000 training identities still gets an embedding.

## Training

```bash
python scripts/train_classification.py --download --output-root /content/drive/MyDrive/fac-03-runs
python scripts/train_classification.py --download --resume /content/drive/MyDrive/fac-03-runs/<run_id>
```

The team notebook `fac-03-train-classification.ipynb` runs these on a Colab GPU and resumes automatically after a disconnect. Settings are read from [`configs/face-classification.toml`](../configs/face-classification.toml). Training uses `load_face_layout` semantics: the exclusion list from FAC-02 is applied before any dataset is built.

| Step | Rule |
| --- | --- |
| Each epoch | Shuffle with seed `42 + epoch`, train with mixed precision on CUDA, then one pass over `val_data` gives top-1 accuracy and the embeddings for the validation pairs |
| Selection | The epoch with the highest cosine AUC on the validation pairs is saved as `best.pt`; a tie keeps the earlier epoch. Top-1 on `val_data` is monitored only. |
| End of run | `best.pt` is scored once on `test_data` (top-1) and again on `val_data` |
| Resume | `last.pt` holds model, optimizer, scheduler and scaler state after every epoch. A resume refuses any changed setting except `num_workers` and `eval_batch_size`. |

**Validation pairs.** `validation_pairs.py` builds the list once per run from the filtered `val_data`, seed 42: one same-identity pair for each identity with two images, and as many distinct different-identity pairs drawn uniformly. The run saves it as `val-pairs.txt` in the official `left right label` format, with paths relative to the dataset root, so FAC-04 and the shared evaluator can reuse it unchanged. The official trials are never used for selection.

## Run directory

| File | Content |
| --- | --- |
| `config.json`, `changes.json` | The settings, and any value that differs from the config file with its reason |
| `val-pairs.txt` | The selection pairs |
| `history.json` | Per epoch: learning rate, train loss and top-1, `val_data` top-1 and loss, validation-pair cosine and Euclidean AUC, seconds |
| `manifest.json` | The SYS-04 traceability fields, the architecture, trainable layers, loss, data hashes (exclusion list, used samples, identity mapping, both pair lists), environment, every session's git commit, the selection and the final accuracy |
| `last.pt` | Resume state; not deployable |
| `best.pt` | The deployable checkpoint: `state_dict` and `meta` with the embedding-schema keys plus `num_classes`. `threshold` stays empty; FAC-05 writes it into a copy, `deploy.pt` (see [Deployed model](#deployed-model)). |

`run_id` is `YYYYMMDD-HHMMSS-classification-<commit>-s<seed>` in UTC, and it is also the checkpoint's `model_id`. `model_version` is the first 12 hex digits of SHA-256 over every `state_dict` tensor's name, dtype and bytes, in name order. Checkpoints stay out of git.

## Using a checkpoint

```bash
# Two images: cosine similarity (higher = same) and Euclidean distance (lower = same)
python scripts/score_face_pair.py --checkpoint best.pt left.jpg right.jpg

# Inputs for the shared evaluator, written to a new directory
python scripts/export_face_embeddings.py --download --checkpoint best.pt --val-pairs <run>/val-pairs.txt --output <run>/exports
```

The export writes `official-embeddings.npz` and `val-pairs-embeddings.npz`, each with `paths` (the exact strings of the trial file) and `raw_embeddings` (float32, before L2 normalization), and `val_data-predictions.csv` / `test_data-predictions.csv` (`id,true,predicted`) with `face_labels.json`. These match the evaluator's verification and classification inputs, so FAC-05 computes ROC/AUC without loading the model. A checkpoint without an identity head, such as the metric-learning model, gets only the two NPZ files; `export-summary.json` then records `"classification": null`. Scoring applies no threshold.

In code, `load_checkpoint(path)` returns the model in eval mode and its `meta`, and refuses weights whose hash differs from `meta["model_version"]`. `eval_transform(meta["input_size"])` is the matching preprocessing.

## Deployed model

The face comparison (FAC-05) selected the metric-learning model and its cosine threshold on the validation pairs. The GUI uses this one checkpoint; why it was chosen and how well it does are in the [FAC-05 evidence](../evidence/FAC-05/README.md#fac-055--selected-model-and-limitations).

| Item | Value |
| --- | --- |
| Approach | Metric learning (triplet loss), FAC-04 |
| `model_id` | `20261114-125242-metric-74a8657-s42` |
| `model_version` | `cb38ea47ceff` |
| File | `MyDrive/fac-04-runs/20261114-125242-metric-74a8657-s42/deploy.pt`: `best.pt` with the threshold written into `meta`, same weights |
| Input | One RGB face crop, `eval_transform(64)`: resize to 64×64, scale to [0, 1], ImageNet mean and std |
| Embedding | `model.embed(x)`, `float32[512]`. The metric model's raw vectors are short (median pair distance 0.21), so L2-normalize before storing or comparing, as the embedding schema does. |
| Score | Cosine similarity, `s = ê · ĝ` on L2-normalized vectors; higher means more likely the same person |
| Decision | Same person when `s ≥ τ`, otherwise unknown |
| `τ` | `meta["threshold"]` = **0.4619967254593038**: the lowest validation-pair cosine with a false-accept rate of at most 1 % |
| `meta["threshold_source"]` | The rule, the validation-pair hash and the SHA-256 of `selection.json` |

Make `deploy.pt` from the selected `best.pt`; the source file is not changed:

```bash
python scripts/set_face_threshold.py --checkpoint <run>/best.pt \
    --selection evidence/FAC-05/selection.json --output <run>/deploy.pt
```

It refuses a checkpoint whose `model_id`, `model_version` or approach differs from the selection, and checks that the written file reloads with the same `model_version` and `τ`. Because `model_version` hashes only the weights, a gallery built with `best.pt` stays valid with `deploy.pt`.

Read `τ` from the loaded checkpoint's `meta`, never from a constant in GUI code, and refuse a gallery whose `model_id` or `model_version` differs from it. Another checkpoint needs its own `τ` from the same rule and a new enrollment.

## Limits

- The validation pairs contain identities the model trained on, so their AUC selects an epoch; it does not estimate performance on new people.
- Augmentation randomness after a resume depends on worker seeding, so a resumed run is not bit-identical to an uninterrupted one. Data order per epoch is.
