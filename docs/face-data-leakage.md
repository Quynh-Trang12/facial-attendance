# Face data leakage: finding and decision needed

| | |
| --- | --- |
| Found by | **FAC-01.3** (Review identity labels and leakage risks), full-data audit on Colab, 5 Nov 2026 |
| Decided and fixed in | **FAC-01.4** (Revise the split and pair-loader design), owner Quế An |
| Affects | FAC-01.2 loader (Trang), FAC-05/SYS-04 verification evaluation |
| Status | **Open: waiting for a team decision.** Nothing has been changed in the loader or the protocol yet. |

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

**Why it matters.** The verification AUC is the required result. If a model trains on all of `train_data`, about one trial in nine compares a face the model has already seen, so the reported AUC is likely higher than the model's real performance. The 12 + 6 train/val/test duplicates have the same effect, on a much smaller scale, on validation selection and test accuracy.

There are also 169 duplicate copies inside single identities (a warning, not leakage). Sampled positive pairs can then compare two identical files.

The competition data cannot be changed. The fix has to be in how we use it.

## Options

| | A. Keep training data, report both AUCs | B. Remove shared images from training (recommended) |
| --- | --- | --- |
| What changes | Nothing in training. Every final model is scored on all 8,805 trials and also on the 7,803 clean trials. | The images in `train_data` that are byte-identical to any `verification_data`, `val_data` or `test_data` image are excluded from training through a committed exclusion list. Identity mapping stays the same. |
| Leakage | **Still present** in training. It is only measured and disclosed. | **Removed** for the official trials and for val/test. |
| Cost | A clean-trial list and one paragraph in the protocol. | An exclusion list, a small loader change reviewed by Trang, tests, and one more Colab audit to prove no overlap remains. Roughly 8,400 of 380,638 training images (about 2%) are removed; the smallest identity has 15 images now, and its count after removal still has to be checked. |
| Fits FAC-01.4? | Weak. FAC-01.4 asks to **correct** the findings, not only record them. | Yes. It corrects the finding and gives a revised protocol to check. |

With option B, both AUCs are still reported. If the leakage is gone, the two numbers should be close, which is the evidence for the fix.

## Decision needed before the next step

Who: Quế An (FAC-01.4 owner) with Trang (owner of the FAC-01.2 loader). A reviewer for FAC-01.4 also has to be named, because chị Dung is unavailable.

- [ ] Choose option A or B.
- [ ] If B: Trang agrees to an exclusion-list change in `data_loader.py`.
- [ ] Decide whether the 169 within-identity duplicates are also removed, or left as a known warning.
- [ ] Name the FAC-01.4 reviewer.

## Next steps after the decision

1. Record the decision and the revised rule in [face-data-protocol.md](face-data-protocol.md).
2. Option B only: generate the exclusion list from the full data, add it to the loader with tests, and rerun the audit on Colab. The train/val, train/test and train/verification failures must reach 0.
3. Both options: commit a fixed list of the 7,803 clean trials, so every final model also reports a clean-trial AUC.
4. Set the face input size `S = 64` in the [embedding schema](embedding-schema.md), from the audit's image-size histogram.
5. Update FAC-01.3, FAC-01.4 and FAC-01.5 evidence in the WBS sheet, and ask the reviewer to check the corrected loader (FAC-01.4) and the rerun audit.
