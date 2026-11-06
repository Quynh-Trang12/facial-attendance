# Face data leakage: finding and decision needed

| | |
| --- | --- |
| Found by | **FAC-01.3** (Review identity labels and leakage risks), full-data audit on Colab, 5 Nov 2026 |
| Decided and fixed in | **FAC-01.4** (Revise the split and pair-loader design) |
| Affects | FAC-01.2 loader, FAC-05/SYS-04 verification evaluation |
| Status | **Open: option C is proposed and waiting for a decision.** Nothing in the loader, audit or protocol has changed on this branch. The option C code below is a proposal: a local draft exists, tested on generated fixtures only. It has not been pushed or run on the full data. |

## What was checked

The [audit report](../reports/data/colab-audit-2026-11-05.json) hashed and decoded all 465,735 files of the assigned competition (commit `651c25a`). The [loader re-run](../reports/data/colab-loader-review-2026-11-05.json) on the same data passed, and its counts, identity mapping and trials-file hash match the earlier local run, so the data itself is the same.

**Clean:** every image decodes (all 64×64 RGB), no trial pair has conflicting labels, no trial repeats or compares a file with itself, and no image appears under two identities.

## The problem

The same image bytes appear in more than one partition. That fails the audit rule in the [face data protocol](face-data-protocol.md#data-checks).

| Partitions sharing identical images | Shared images |
| --- | --- |
| `train_data` and `verification_data` | 8,396 |
| `val_data` and `verification_data` | 195 |
| `test_data` and `verification_data` | 167 |
| `train_data` and `val_data` | 12 |
| `train_data` and `test_data` | 6 |

The official trials use only 9,355 of the 69,097 verification images, so most shared images are never scored. The [trial overlap count](../reports/data/colab-trial-overlap-2026-11-05.json) measures what does matter:

| Official trials (`verification_pairs_val.txt`) | Count |
| --- | --- |
| All trials | 8,805 |
| Trials using an image that is also in `train_data` | **929** (527 same, 402 different) |
| Trials using an image that is also in any classification split | 1,002 (11.4%) |
| Clean trials, no shared image | 7,803 |

**Why it matters.** The verification AUC is the required result. If a model trains on all of `train_data`, about one trial in nine compares a face the model has already seen, so the reported AUC is likely higher than the model's real performance. The 12 + 6 train/val/test duplicates have the same effect, on a much smaller scale, on validation selection and test accuracy. The 195 + 167 val/test copies in `verification_data` also mean those partitions are not independent of the verification result.

The overlap comes from the competition data as published, not from how the project split it. Any run that loads the data unchanged inherits it.

There are also 169 duplicate copies inside single identities (a warning, not leakage). Sampled positive pairs can then compare two identical files.

The audit has not yet compared `val_data` with `test_data`, so any overlap between those two is still unknown.

The competition data cannot be changed. The fix has to be in how we use it.

## Options

### Rejected options

**A. Keep the training data and report two AUCs (rejected).** Every model would be scored on all 8,805 trials and on the 7,803 clean trials. Rejected because nothing is corrected: the overlap stays in training and is only measured and disclosed, FAC-01.4 asks for a correction, and results built on data known to leak put the marks at serious risk.

**B. Remove shared images from `train_data` only (rejected, previously recommended).** This was the earlier recommendation because it removes the largest overlap with a small loader change. Rejected because it does not reach zero overlap:

- 195 `val_data` and 167 `test_data` images would still be shared with `verification_data`.
- `val_data` and `test_data` would still not be compared with each other.
- Its proof relied partly on two AUCs being close. An AUC can suggest a problem, but it can't show that overlap is gone.

### C. Filter every partition until hash overlap is 0 (proposed)

Each distinct image is kept in one partition only, and the loader loads that filtered data instead of the data as published.

| Part | Rule |
| --- | --- |
| Which copy is kept | First match in the order `verification_data`, `test_data`, `val_data`, `train_data`. Verification comes first because the required AUC uses every official trial. Val and test come before train because they hold only two images per identity. |
| Duplicates inside one identity | One copy is kept, so a sampled positive pair can't compare identical images. |
| Where the rule lives | `configs/face-exclusions.tsv`, a committed list of paths and reasons generated from the full-data hashes. No image files are copied, changed or committed. |
| Loader | `FaceLayout.discover` applies the list by default. Unfiltered data needs an explicit `exclusions=None`. A list that doesn't match the dataset raises an error. |
| What stays the same | All 8,805 official trials, and the identity mapping and class targets. |

**Expected effect.** These figures are estimates until the full-data run. The real counts may be lower, because one shared image can have several copies.

| Partition | Now | Expected after filtering |
| --- | --- | --- |
| `train_data` | 380,638 | About 372,000 (about 2% fewer) |
| `val_data` | 8,000 | About 7,800 |
| `test_data` | 8,000 | About 7,800 |
| `verification_data` | 69,097 | Unchanged |

### Evidence required for option C

Option C is accepted only when a full-data audit of what the loader actually loads shows all of the following:

1. Hash overlap is 0 for all six pairs among train, val, test and verification.
2. The number of images the loader dropped equals the number listed.
3. No image in a seeded sample of 100,000 training pairs shares a hash with held-out data or appears in the list, and no positive pair compares identical images.
4. No duplicate copies remain inside an identity.
5. The loader check passes on the filtered data, and no training identity has lost all its images.

The AUC on all trials and on the 7,803 previously clean trials may be reported as extra analysis. Neither number counts as evidence that overlap is gone.

### Limits of option C

- **Hash overlap of 0 does not mean no shared people.** A hash finds only byte-identical files. The same person in a different photo, or the same photo re-encoded, isn't detected. Verification images have no identity labels, so this can't be measured, and the report must say so.
- **Val and test become uneven.** Some identities will have one image or none, so closed-set accuracy is measured on slightly fewer images.

## Decision needed

- [ ] Accept option C.
- [ ] Accept the change to `data_loader.py`, so that the filtered data becomes the default.
- [ ] Choose how `val_data` and `test_data` handle removed images:
  - **Drop only (recommended).** This is simpler and moves no image between partitions, but leaves uneven counts per identity.
  - **Refill from `train_data`.** This keeps two images per identity, but needs more code and changes what each split contains.
- [ ] Name the FAC-01.4 reviewer.

## Next steps after the decision

1. Run the audit on the unfiltered data on Colab, which also generates the list and gives the first `val_data`/`test_data` comparison.
2. Commit the list, then audit the filtered data and run the loader check on Colab.
3. Replace the estimates above with the measured counts, and record the rule in [face-data-protocol.md](face-data-protocol.md) and [data-loading.md](data-loading.md).
4. Set the face input size `S = 64` in the [embedding schema](embedding-schema.md), from the audit's image-size histogram.
5. Update FAC-01.3, FAC-01.4 and FAC-01.5 evidence in the WBS sheet, and ask the reviewer to check the corrected loader and the filtered audit.
