# FAC-02 — Face data audit, cleaning, loaders and sampler

Task requirements, assignments and review records: [Project Tracker](../../README.md#project-tracker), task `FAC-02`. Partition roles come from the FAC-01 face data protocol; interfaces are in the [loader contract](../../docs/data-loading.md#loader-and-sampling-contract).

**Decision.** Cleaning follows the team's 9-step plan; step 5 uses option C, and step 4 drops the pixel-hash and near-duplicate searches.

## Rule (step 5, option C)

An image found in several partitions stays in the first of `verification_data`, `test_data`, `val_data`, `train_data`; the other copies are dropped. An identity keeps one copy of an image. Nothing is refilled, no image moves between partitions, and the official trials, identity mapping and targets stay unchanged. The rule produces [`configs/face-exclusions.tsv`](../../configs/face-exclusions.tsv): each row gives the dropped path, the reason, its SHA-256 and the kept copy. `load_face_layout` applies it by default and refuses a row that does not match the data.

## Results on the full data

Colab CPU, 12 Nov 2026, code commit `85da9f1`, unfiltered manifest `97ceaaa8…` (the same data as the 8 Nov run).

| Check | Before filtering | After filtering |
| --- | --- | --- |
| Shared images: train/val, train/test, val/test | 12, 6, 1 | 0, 0, 0 |
| Shared images: train, val, test with verification | 8,396, 195, 167 | 0, 0, 0 |
| Duplicate copies inside one identity | 169 (in 127 groups) | 0 |
| Images: train, val, test, verification | 380,638, 8,000, 8,000, 69,097 | 372,057, 7,804, 7,832, 69,097 |
| Rows listed, images the loader dropped | 8,945 | 8,945 (8,581 train, 196 val, 168 test) |
| Undecodable images; image size and mode | 0; all 64×64 RGB | same |
| Official trials and identity mapping | 8,805 trials | hashes unchanged |

Coverage after filtering (step 6):

| Split | Identities with 0 / 1 / 2 / 3+ images | Identities able to form a positive pair |
| --- | --- | --- |
| `train_data` | 0 / 0 / 0 / 4,000 (min 13, median 89) | 4,000 |
| `val_data` | 2 / 192 / 3,806 / 0 | 3,806 |
| `test_data` | 2 / 164 / 3,834 / 0 | 3,834 |

Sampling (steps 8–9): 100,000 seeded training pairs and one full epoch of 1,453 P×K batches (64 identities × 4 images, seed 42) drew no held-out, listed or identical-positive image, and every batch had exactly 64 identities × 4 distinct files. A real batch decoded to `[256, 3, 64, 64]`. All 99 unit tests passed on the same runtime.

Reports: [unfiltered audit](colab-audit-raw-2026-11-12.json), [filtered audit](colab-audit-filtered-2026-11-12.json), [loader check](colab-loader-filtered-2026-11-12.json), [unit tests](colab-unit-tests-2026-11-12.log). The 465,735-row inventory CSV (step 2) is kept outside the repository.

## Preprocessing (step 7)

Decode with Pillow, convert to RGB, keep the native 64×64 size, scale to `[0, 1]`, then normalize per channel with mean `(0.485, 0.456, 0.406)` and std `(0.229, 0.224, 0.225)`; both approaches use the same input. No face detection or alignment is added, because the images are already face crops. Random augmentation applies only to training, after filtering.

## Limits

- Only byte-identical copies are found. Pixel-hash and near-duplicate searches (step 4) were left out, so re-encoded or slightly edited copies of one photo may remain across partitions.
- Hash overlap of 0 is not identity separation: the unfiltered audit names at least 3,580 of 4,000 training identities in `verification_data`, and 1,002 of 8,805 trials involve one of them. The verification AUC must not be reported as a result on unseen people.
- `val_data` and `test_data` become uneven: 2 identities have no image left in each, so classification accuracy is measured on 7,804 and 7,832 images.

## Re-run

Run the team's Colab notebook `fac-02-clean-data.ipynb` (CPU runtime, `KAGGLE_API_TOKEN` secret), or locally:

```bash
python scripts/audit_face_data.py --download --raw --write-exclusions new.tsv --inventory evidence/runs/FAC-02/inventory.csv --output evidence/runs/FAC-02/raw.json
python scripts/audit_face_data.py --download --exclusions new.tsv --output evidence/runs/FAC-02/filtered.json
python scripts/check_face_data.py --download --exclusions new.tsv --image-size 64 --output evidence/runs/FAC-02/loader.json
```

Expect the counts above; `--write-exclusions` never overwrites a list, so compare `new.tsv` with the committed one.
