# Face data leakage: finding, options and decision

| | |
| --- | --- |
| Found by | **FAC-01.3** (Review identity labels and leakage risks), full-data audit on Colab, 5 Nov 2026 |
| Decided and fixed in | **FAC-01.4** (Revise the split and pair-loader design) |
| Affects | FAC-01.2 loader, FAC-05/SYS-04 verification evaluation |
| Status | **Decided on 8 Nov 2026: option C, with E and G, by the FAC-01.4 owner.** The code is written and passes its tests on generated fixtures. It has not run on the full data, so the counts in the option tables are still estimates. Reviewer confirmation is pending. See [Decision](#decision) and [Implementation report](#implementation-report). |

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

### Shared files versus shared people

The question behind the audit is whether the verification AUC shows how well a model handles faces it has never learned. The brief describes verification as open-set, where the model may meet people it has not seen. Overlap happens at three levels:

| Level | Measured? |
| --- | --- |
| 1. The same file (byte-identical) | Yes. These are the counts above. A hash finds them. |
| 2. The same photo, re-encoded, cropped or slightly edited | No. A hash misses it; a perceptual hash or embedding comparison would be needed. |
| 3. The same person in a different photo | Not fully, because verification images have no identity labels. A lower bound can be measured: each byte-identical match names the training identity that a verification image belongs to. |

The 20 train/verification examples in the audit report are ordered by hash, so close to random, and they belong to 20 different training identities. This suggests that the shared images are spread over many people, but 20 examples are not a measurement.

Removing a file does not remove the person. After any hash-based filter, the model still trains on that person's other photos (at least 15 per identity, 95 on average). A hash overlap of 0 therefore shows that no file is shared; it does not show that the AUC measures performance on unseen people. How much the AUC changes after filtering has not been measured.

Shared identities between `train_data` and `val_data`/`test_data` are by design: classification is closed-set, so these partitions hold the same 4,000 people. That is not leakage and is not removed.

## Options

### Rejected options

**A. Keep the training data and report two AUCs (rejected).** Every model would be scored on all 8,805 trials and on the 7,803 clean trials. Rejected because nothing is corrected: the overlap stays in training and is only measured and disclosed, FAC-01.4 asks for a correction, and results built on data known to leak were judged to put the marks at serious risk. The brief itself says nothing about duplicates or filtering, so that risk has not been confirmed with the tutor (see G).

**B. Remove shared images from `train_data` only (rejected, previously recommended).** This was the earlier recommendation because it removes the largest overlap with a small loader change. Rejected because it does not reach zero overlap:

- 195 `val_data` and 167 `test_data` images would still be shared with `verification_data`.
- `val_data` and `test_data` would still not be compared with each other.
- Its proof relied partly on two AUCs being close. An AUC can suggest a problem, but it can't show that overlap is gone.

### C. Filter every partition until hash overlap is 0 (chosen)

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

- **Hash overlap of 0 does not mean no shared people.** A hash finds only byte-identical files. The same person in a different photo, or the same photo re-encoded, isn't detected. Verification images have no identity labels, so only a lower bound of shared people can be measured (see E), and the report must say so.
- **Val and test become uneven.** Some identities will have one image or none, so closed-set accuracy is measured on slightly fewer images.
- **Most removed training images are never scored.** Only 667 of the 9,355 scored verification images are shared with `train_data`, so roughly 92% of the 8,396 removed training images don't touch the official trials. The cost is still small (about 2%), and one rule for the whole folder is simpler.

### Other options

**D. Filter only against images the official trials use (not chosen).** The same rule as C, but overlap is measured against the 9,355 verification images the trials use, not all 69,097. The other 59,742 are never loaded.

| Partition | Now | Expected after filtering |
| --- | --- | --- |
| `train_data` | 380,638 | About 379,800 |
| `val_data` | 8,000 | About 7,970 |
| `test_data` | 8,000 | About 7,970 |

- Gain: `val_data` and `test_data` stay almost as supplied (about 27 and 26 images removed instead of 195 and 167), and almost all training data is kept.
- Cost: an audit of the whole `verification_data` folder still finds shared files, so zero overlap holds only for "what the loader actually loads". Both sides would have to accept that scope. It is one more rule to explain.

**F. Hold out whole identities for an evaluation on unseen people (not added).** Some identities are removed from `train_data` before training, and seeded verification pairs are built from their images. Because the model never trains on those people, this evaluation has no identity overlap at all, which is the only way to get a guaranteed level-3 result from this data. The official AUC is still reported on `verification_pairs_val.txt`, as the brief requires.

| Identities held out | Training images left (about 95 per identity) | Classes left | `val_data`/`test_data` left (their 2 + 2 images go too) |
| --- | --- | --- | --- |
| 200 | About 361,600 | 3,800 | 7,600 each |
| 400 | About 342,600 | 3,600 | 7,200 each |

- Gain: a real open-set number, and selection data on unseen people. It also addresses two disclosed limits in the [SYS-04 experiment settings](https://github.com/Quynh-Trang12/facial-attendance/blob/71f3b56fd7fdc6e90106ad6fdeae8e8b23749237/docs/experiment-settings.md) (pull request 6, not merged): the threshold is chosen on closed-set pairs, and the brief says `val_data` is not needed for metric learning.
- Cost: fewer classes, a different identity mapping, more code, and SYS-04 must be revised. It does not make hash overlap 0 on its own, so it would be added to C or D, not replace them.

Removing every training identity that appears in verification would also be possible in principle, but only the identities found through byte-identical files are known, so it can't reach zero. It could also cost a large part of the training set. Each removed identity takes about 95 training images and 2 + 2 `val_data`/`test_data` images with it:

| Identities removed | Training images left | `val_data`/`test_data` left |
| --- | --- | --- |
| 600 | About 323,500 | 6,800 each |
| 1,000 | About 285,500 | 6,000 each |
| 2,000 | About 190,000 | 4,000 each |
| 3,500 | About 47,500 | 1,000 each |

How many identities would be affected is unknown. If they were spread evenly, the 667 scored shared images could touch up to about 600 identities, and all 8,396 shared images up to about 3,500. Clustering would make both numbers lower. E measures this.

### Additions that fit any option

**E. Measure shared identities (added).** The next full-data audit also records, from the byte-identical matches: how many training identities the shared images belong to (for all shared images and for the scored ones), and how many official trials involve a known training identity. This only reads data and doesn't change the loader. It gives the lower bound for level 3 and the numbers the report needs.

**G. Ask the tutor (to be asked; no work waits on the answer).** One question, which doesn't need to block the work: the supplied data shares images between `train_data` and `verification_data`; is filtering it and reporting what was done expected, or is disclosure enough, and may images be removed from `val_data` and `test_data`? The answer settles the marking risk behind rejecting A. No record of this having been asked was found in the team's files.

## Decision

Decided on 8 Nov 2026 by the FAC-01.4 owner. The reviewer checks it against the [evidence required for option C](#evidence-required-for-option-c).

| Question | Decision | Why, and what it costs |
| --- | --- | --- |
| What the fix must achieve | No shared files (level 1). Level 3 is measured and disclosed, not removed. | Only a held-out set of people could remove level 3 (option F), and that changes the classes and the SYS-04 settings. The report must not call the AUC a result on unseen people. |
| Which filter | **C**, whole folders | It reaches zero overlap on every partition, which D does not, and it is one rule. D would keep more images, but C costs only about 2% of `train_data` (estimate). |
| `val_data` and `test_data` | **Drop only**, no refill | No image moves between partitions. The cost is uneven counts per identity, and accuracy is measured on slightly fewer than the 8,000 supplied images, which the report must state. |
| Loader default | The filtered data becomes the default in `data_loader.py` | The loader must really load the clean data. Unfiltered data needs an explicit `exclusions=None`. |
| E | Added | It gives the level 3 lower bound the report needs, and it only reads data. |
| F | Not added | The cost above is too high for the remaining schedule. |
| G | To be asked by the FAC-01.4 owner | Nothing waits for the answer. If the tutor forbids filtering `val_data` or `test_data`, the list can be regenerated, because it is one file. |

## Implementation report

Status of the work on branch `fac-01/face-data-protocol-audit`.

| File | Change |
| --- | --- |
| `src/facial_attendance/recognition/data_loader.py` | `FaceLayout.discover` reads the exclusion list by default; `ClassificationFaces` drops the listed images and records `excluded_count`. A listed path that is not a classification image raises an error, and so does a training identity left with no images. |
| `src/facial_attendance/recognition/download.py` | `download_face_data` passes the list through. |
| `src/facial_attendance/recognition/audit.py` | Adds `val_data`/`test_data` to the checked pairs, generates the list, hash-checks 100,000 sampled training pairs, and records the shared-identity lower bound (E) under `known_identities` on unfiltered runs. |
| `scripts/audit_face_data.py` | Adds `--raw`, `--write-exclusions` and `--exclusions`. |
| `scripts/check_face_data.py` | Reports how many images the list dropped per split. |
| `tests/test_face_audit.py`, `tests/test_face_data.py` | Cover the keep order, zero overlap after filtering, a mismatched list, sampler cleanliness and E. |

**Checked so far.** The complete unit suite passes (79 tests), on generated fixtures only.

**Not checked yet.** Each row below needs the full-data Colab run.

| Evidence | Result |
| --- | --- |
| 1. Hash overlap is 0 for all six partition pairs, including the first `val_data`/`test_data` comparison | Pending |
| 2. Images dropped by the loader equal images listed | Pending |
| 3. 100,000 sampled training pairs are clean | Pending |
| 4. No duplicate copies remain inside an identity | Pending |
| 5. The loader check passes and no training identity lost all its images | Pending |
| E. Identities named by shared images; official trials involving a named identity | Pending |
| Measured counts per partition, replacing the estimates above | Pending |

## Next steps

1. Run the audit on the unfiltered data on Colab, which generates the list, then audit the filtered data and run the loader check in the same session.
2. Commit the list and the reports, and fill in the table above.
3. Set the face input size `S = 64` in the [embedding schema](embedding-schema.md), from the audit's image-size histogram.
4. Update FAC-01.3, FAC-01.4 and FAC-01.5 evidence in the sheet, and ask the reviewer to check the corrected loader and the filtered audit.
5. Ask the tutor the question in G.
