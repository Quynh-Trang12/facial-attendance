# LIV-03 — Anti-spoofing audit and loader

Document version: **LIV-03-v9**, revised **28 November 2026**. Task owner: **Trang**. Independent reviewer: **chị Dung**. LIV-03 is defined in the [Project Tracker](../README.md#project-tracker), where its updates and records are maintained.

LIV-03 inspects or audits the frozen post-review selection of 560,722 CelebA-Spoof images without changing labels, splits or exclusions. Outputs include retained-row counts, crop previews and source-integrity receipts. Historical preparation evidence remains attributed. Complete crop semantics and source-video independence remain unverified.

## Start here

| Goal | Route | Inputs |
| --- | --- | --- |
| Inspect counts and previews | [Inspect the current selection](#audit-the-current-frozen-selection) | Source ZIP and saved evidence |
| Verify every retained image and box | [Run audit](#audit-the-current-frozen-selection) | Frozen metadata and the original extracted dataset, or the complete archive on Colab |
| Understand exclusions | [Crop review](#source-crop-review-and-filtered-cache-view) | Saved decisions and previews |
| Reconstruct earlier preparation experiments | [Historical reproduction](#reproduce-the-results) | Original dataset and the inputs identified for that historical route |

Use the [project environment](setup.md). Model inference and training are described in the [liveness guide](liveness-training-and-evaluation.md); they do not require replaying historical selection experiments.

## Source-crop review and filtered cache view

Review checked 353 likely problem cases against their original images. Confirmed exclusions remove wrong-subject crops or crops with too little usable face content; they do not relabel images or move them between splits. A valid print/screen attack remains useful training data even when the deployment detector cannot locate a face.

| Split | Original rows | Excluded | Retained |
| --- | ---: | ---: | ---: |
| Train | 444,357 | 243 | 444,114 |
| Validation | 49,504 | 25 | 49,479 |
| Test | 67,170 | 41 | 67,129 |

The [review ledger](../evidence/liveness/crop-review/filtered-view-v4-20261126/reviewed.csv) records all 353 decisions; the [quarantine list](../evidence/liveness/crop-review/filtered-view-v4-20261126/quarantine.csv) identifies exclusions and their reasons. A second pass resolved 66 initially unclear cases: 53 excluded and 13 retained. All declared review cases are resolved; unreviewed crop semantics remain outside that claim.

[ReviewedCacheView](../src/facial_attendance/liveness/datasets/reviewed_cache.py) applies the [frozen indices](../evidence/liveness/crop-review/filtered-view-v4-20261126/cache-view.json) identically to images, labels and teacher maps. It preserves original cache rows and rejects unresolved reviewed cases in normal use. The selected model predates these exclusions, so filtering does not undo its training exposure. [Sampled cache readback](../evidence/liveness/crop-review/filtered-view-v4-20261126/cache-readback.json) checks six actual rows; synthetic tests check complete index alignment, not every cached pixel.

## Audit the current frozen selection

Inspect is the default. The [data notebook](../notebooks/liveness-data-preparation-and-audit.ipynb) displays saved counts and 33 preview crops covering all eleven source types in each split. Its [inspection summary](../evidence/liveness/crop-review/selection-inspection-20261127/summary.json) binds those previews to the selection; this is not a full source scan.

Audit uses [audit_liveness_selection.py](../scripts/audit/audit_liveness_selection.py). It verifies input hashes, original row order, disjoint image assignments, train/validation subject separation and the exact complement of the exclusion list, then hash-checks and decodes every retained image and box. It writes a new run; checkpoints, source images and frozen decisions stay unchanged.

<details>
<summary>Local execution</summary>

From the repository root with the project environment active:

```sh
python scripts/run_liveness_notebook.py --notebook notebooks/liveness-data-preparation-and-audit.ipynb --action inspect
python scripts/run_liveness_notebook.py --notebook notebooks/liveness-data-preparation-and-audit.ipynb --action audit
```

Run inspect without the dataset. Run audit only when the complete source dataset exists at the configured location. Local audit performs no extraction or installation and creates a unique output run under the existing [evidence directory](../evidence/), using the runs/LIV-03/local layout. A failure retains its partial report.

</details>

<details>
<summary>Google Colab execution</summary>

Open the data notebook, set PROJECT_FOLDER and ARCHIVE_FOLDER to the shared folder IDs/URLs, choose ACTION = "audit", then Run all on a fresh Python 3.13 CPU runtime. Allow at least 85 GiB free VM storage plus variable Drive-cache overhead and several hours for the full scan.

Colab verifies the release, installs its isolated CPU environment, runs the selection tests, extracts the 74-part archive and restores the frozen annotations/indices. It does not reselect validation subjects or rerun old exclusion policies. Commenters read the shared project and write new logs, summaries and previews under their private MyDrive/facial-attendance-outputs, separated by project ID. The revised native Colab route has not been executed.

</details>

The 27 November local full audit verified all 560,722 retained images and boxes. Its original summary and checksum are retained in the [verification record](../evidence/liveness/package-refactor/verification.json) under historical_full_source_audit. This establishes source integrity for that run; complete label validity, cache-pixel parity, model effectiveness and source-video independence remain unverified.

## Evidence and source access

- [Current review summary](../evidence/liveness/crop-review/filtered-view-v4-20261126/summary.json): Current counts, decisions and source-type breakdowns
- [LIV-03 evidence index](../evidence/LIV-03/README.md): Original execution records and meaningful review findings
- [12 November baseline summary](../evidence/LIV-03/colab/colab-cpu-2026-11-12/audit-summary.json): Historical 561,575-image scan and three checked batches
- [13 November exact-stage summary](../evidence/LIV-03/local/quality-2026-11-13/summary.json): Exact-duplicate findings and stricter crop guards
- [14 November search summary](../evidence/LIV-03/local/quality-2026-11-14/summary.json): Complete radius-6 search and bounded content inspection
- [Historical experimental summary](../evidence/LIV-03/local/experimental-2026-11-14/summary.json): Narrower 339,925-image candidate-policy subset

[Original CelebA-Spoof acquisition](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-downloads) is subject to its research-only agreement. Raw images and full caches are external assets. [Full baseline outputs](https://drive.google.com/drive/folders/11rMGigAD9uu9WZT5WeqnN-agSvDOxwYd), [execution records](https://drive.google.com/drive/folders/1j2gu6F0AhgIFpbY57FXMUCiaKob8B7te) and the [original source snapshot](https://drive.google.com/drive/folders/1wp-WH2Rgaow7_gi9OMOt3xmXD1-rd1zB) retain their original permissions and attribution.

<a id="original-and-project-split-assignments"></a>
<a id="data-selection-and-split-integrity"></a>

## Historical print/screen selection and split integrity

An inventory identifies image paths, labels and integrity hashes. Original split assignment comes from the downloaded train/test annotations; project split assignment adds a validation subset from original train. The CSV fields are `original_split` and `split`. Quarantine is an exclusion record, not another evaluation split.

The downloaded archive contains 494,405 original training images from 8,192 subjects and 67,170 test images from 1,004 subjects. It supplies no validation assignment. Project validation uses the first StratifiedGroupKFold split: ten folds, shuffle enabled, seed 42, stratified by source type and grouped by subject. Its 821 subject IDs are fixed before model evaluation.

Historical LIV-03 selected types 0, 1, 2, 3, 7, 8 and 9 for live/print/screen loader work. Current AENet training uses all eleven types with the frozen crop-review indices above. These populations must not be mixed in reported metrics.

## Historical print/screen labels and preprocessing

Published JSON labels override folder names. Live is 0; spoof is 1. Supplied boxes are scaled from the dataset's 224-coordinate convention to image pixels, clipped and resized for the selected reader. The earlier strict loader and the later source-client reproduction reader have different crop contracts; each receipt records its recipe. Historical diagnostic batches contain eight RGB images, shaped 8 × 3 × 224 × 224, with aligned int64 labels.

## Scope and limitations

Known subject overlap, file/pixel duplicates and supplied relationship IDs can be checked. Unavailable source-video/session IDs prevent establishing capture independence; illumination classes and filenames are not substitutes. Complete visual correctness was not established by the automated scans. Findings remain attached to the producing implementation rather than relabelled as results of the current code.

<a id="reproduce-the-results"></a>

## Reproduce the historical baseline results

This route reconstructs the earlier print/screen preparation. It is separate from current frozen-selection audit and AENet training.

### Storage plan

| Storage component | Approximate space |
| --- | ---: |
| All 74 compressed parts | 71.7 GiB |
| Extracted files, including estimated filesystem overhead | 76.4 GiB |
| Extraction reserve | 2 GiB |
| One historical full audit’s outputs | 0.13 GiB |
| **Combined estimated footprint on one volume** | **150.3 GiB** |
| **Minimum free space before download, all on one volume** | **160 GiB** |
| **Minimum free space on extraction volume, archives stored elsewhere** | **85 GiB** |

The free-space allowances include headroom above the estimated footprint. They exclude the environment and variable Drive caching. If archives are on another volume, that volume must also accommodate their 71.7 GiB. The 74 parts form one seekable ZIP; the extractor reads them directly without a merged ZIP, verifies CRCs and refuses an existing extraction root.

<a id="set-review-paths"></a>
<a id="fresh-reproduction"></a>
<a id="local-execution"></a>

<details>
<summary>Local execution — historical preparation</summary>

Use the existing Python 3.13 environment. Download all original parts into a separate archive folder: parts 001–073 are 1,048,576,000 bytes each; part 074 is 453,369,048 bytes. The commands below use repository-relative paths and a new reviewer output directory. The archive folder may be on another mounted volume.

```sh
python scripts/preparation/prepare_liveness_data.py --parts-dir "../CelebA-Spoof-zips" --output-dir data/raw/liveness/celeba-spoof-full --full
python scripts/preparation/prepare_liveness_data.py --output-dir data/raw/liveness/celeba-spoof-full --configure-validation 10
python scripts/audit/audit_liveness_inventory.py --data-root data/raw/liveness/celeba-spoof-full --data-description data/raw/liveness/celeba-spoof-full/data-description.json --output-dir evidence/runs/LIV-03/local/reviewer-01/initial
```

For the recorded archive, the initial audit exits 1 after finding 37 train/validation identical-file groups. Continue only when the report confirms that finding with no unrelated blocking error. Resolve its affected training subjects and rerun:

```sh
python scripts/preparation/prepare_liveness_data.py --output-dir data/raw/liveness/celeba-spoof-full --resolve-audit evidence/runs/LIV-03/local/reviewer-01/initial/audit.json
python scripts/audit/audit_liveness_inventory.py --data-root data/raw/liveness/celeba-spoof-full --data-description data/raw/liveness/celeba-spoof-full/reviewed-data-description.json --output-dir evidence/runs/LIV-03/local/reviewer-01/final
```

The six affected training subjects are 5940, 7141, 7143, 7145, 7312 and 8807. Excluding their 328 images produces 544 quarantined source images in total, including earlier original train/test subject conflicts; originals remain intact. An already prepared dataset can start at the final audit.

<a id="data-layout"></a>

The extracted root directly contains Data, metas, the original README and sibling bounding-box files. Preparation adds validation-subjects.json, project annotation files and data descriptions. Point `--data-root` at this root, not Data or an individual split.

Verify validation-subject identity independently of JSON whitespace:

```python
import hashlib
import json
from pathlib import Path

root = Path("data/raw/liveness/celeba-spoof-full")
ids = sorted(
    json.loads((root / "validation-subjects.json").read_text())["validation_subjects"]
)
print(
    len(ids),
    hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode("ascii")).hexdigest(),
)
```

Expected: 821 subjects and SHA-256 ce7f612210a5ef3e1f085012bfb979fc75ef53f15324b89d2c8d9400188b552f. A mismatch requires investigating source annotations and dependencies, not selecting a different fold.

<a id="5-final-audit-and-loader-check"></a>

Under current full-verification semantics, final exit 2 means the pipeline completed with incomplete quality evidence; exit 1 means failure, and exit 0 means accepted evidence. The original Colab PASS used earlier acceptance semantics and remains historical.

</details>

<a id="google-colab-review"></a>
<a id="google-colab-execution"></a>

<details>
<summary>Google Colab execution — historical preparation</summary>

Run the same five preparation/audit commands in a separate compatible Colab CPU environment after mounting the archive folder and supplying project code. Define the paths for that environment and save new outputs to a writable destination. The current data notebook instead restores frozen metadata; it does not rebuild this historical baseline. [Original Colab evidence](../evidence/LIV-03/README.md#full-data-colab-execution--12-november-2026-utc) identifies the executed source/runtime.

</details>

<a id="inspect-and-compare-the-results"></a>

### Inspect historical outputs

Read the final summary for counts, requirement results, runtime and hashes; audit.json retains detailed per-image findings and inventory.csv retains excluded rows as well as selected ones. Each batch has a tensor and PNG preview.

The recorded baseline decoded 561,575 original rows and quarantined 544 source images. Its selected population was:

| Split | Selected images | Selected subjects |
| --- | ---: | ---: |
| Train | 320,660 | 7,362 |
| Validation | 35,646 | 821 |
| Test | 49,646 | 1,002 |

Blocking image errors, overlaps and missing families were each empty. There were 2,022 disagreements between folder names and JSON labels; JSON labels take precedence. Batch alignment and saved readback passed for all three batches. Source-video and session independence remain unverified.

The historical 561,574 valid-face count precedes stricter guards; the later fingerprint scan recorded 561,573 valid inputs and two errors in already excluded attacks. These counts identify different checks, not repaired source files.

<a id="quality-audit-and-evidence-gate"></a>
<a id="complete-search-and-recorded-content-review--14-november-2026"></a>

## Experimental readiness and candidate handling

<details>
<summary>Historical candidate policy and quality reproduction</summary>

Exact deduplication retained 403,362 images. The complete radius-6 DCT-hash search recorded 564,712 whole-image pairs and 1,053,782 crop pairs; views may overlap. Candidates are not duplicate verdicts. Inspection covered 39 natural pairs: 27 duplicates, three distinct and nine related presentations. A separate 25-image box/label inspection did not certify unseen rows or independent reviewer approval.

Direct-pair-quarantine-v2 retained 339,925 images: train 277,903, validation 31,211 and test 30,811. Additional exclusions were 27 confirmed near duplicates, 20 identical crops and 63,390 conservatively quarantined candidates. Representatives prefer test, validation, train, then lexical path order. Reviewed distinct pairs are exempt; crop-only similarity is diagnostic. The policy rechecks remaining direct constraints instead of merging transitive perceptual components. It is not the current frozen training selection.

The full quality assessment remains FAIL because two of 28 controlled crop variants were missed; natural-pair recall, complete visual review and capture relationships remain unverified. Experimental readiness uses a narrower, separately bound receipt. It does not turn missing full-verification evidence into PASS.

For a new historical quality run, use the inventory from your own baseline scan. Allow about 1 GiB for fingerprints/reports in addition to source data. The [quality CLI](../scripts/audit/review_liveness_data.py) exposes exact, near, near-complete, provenance, visual, preview, assess and experimental commands. Inspect its arguments before choosing stages:

```sh
python scripts/audit/review_liveness_data.py --help
python scripts/audit/review_liveness_data.py exact --data-root data/raw/liveness/celeba-spoof-full --inventory data/raw/liveness/celeba-spoof-full/audits/current-schema-2026-11-12/inventory.csv --data-description data/raw/liveness/celeba-spoof-full/reviewed-data-description.json --output-dir evidence/runs/LIV-03/local/quality-review-01
```

Use the same data/inventory/description/output arguments for near, provenance and visual. Near-complete additionally requires a new `--search-output`. Supply recorded pair/visual decisions through `--near-reviews` and `--visual-review`; repeat `--supporting-evidence` for attachments. Assess consumes the review CSVs; experimental writes a separate receipt using `--candidate-dir` and `--experiment-output`. Without the recorded exemptions, regenerated counts can differ. Previewing a sample does not approve unseen candidates.

The same CLI can run in a separate Colab workflow with mounted inputs and private outputs. The revised historical quality/experimental Colab route remains unexecuted. Preserve actual exit codes: completed discovery with unresolved evidence commonly returns 2; the recorded full assessment returned 1. Do not suppress a different failure to match these reference codes.

</details>

## Implementation reference

- **Frozen selection and full retained-row audit:** [audit_liveness_selection.py](../scripts/audit/audit_liveness_selection.py), [reviewed_cache.py](../src/facial_attendance/liveness/datasets/reviewed_cache.py)
- **Source crop scanning and review decisions:** [crop_content_scan.py](../src/facial_attendance/liveness/audit/crop_content_scan.py)
- **Historical audit, split checks and inventory:** [source_integrity_audit.py](../src/facial_attendance/liveness/audit/source_integrity_audit.py), [split_rules.py](../src/facial_attendance/liveness/datasets/split_rules.py)
- **Exact/near duplicate policy and inspection:** [duplicate_scan.py](../src/facial_attendance/liveness/audit/duplicate_scan.py), [duplicate_review.py](../src/facial_attendance/liveness/audit/duplicate_review.py), [candidate_policy.py](../src/facial_attendance/liveness/audit/candidate_policy.py)

### Dataset API

For the historical public loader, supply a matching accepted audit or experimental-readiness receipt; a standalone inventory is insufficient. The current full-source training adapter and reviewed view are described above.

```python
from pathlib import Path
from torch.utils.data import DataLoader
from facial_attendance.liveness.datasets.selected_dataset import LivenessImages
from facial_attendance.recognition.data_loader import image_transform

root = Path("data/raw/liveness/celeba-spoof-full")
output = Path("evidence/runs/LIV-03/local/experiment-review-01")
dataset = LivenessImages(
    root,
    root / "audits/current-schema-2026-11-12/inventory.csv",
    "train",
    transform=image_transform(224),
    audit_file=output / "experimental-readiness.json",
)
images, labels = next(iter(DataLoader(dataset, batch_size=8)))
```

The loader verifies bound metadata and image/box hashes. It returns cropped PIL RGB images and integer labels without a transform, or 8 × 3 × 224 × 224 float32 batches with eight int64 targets in the example. It does not add augmentation or learned normalization.

### Fixture tests

From the repository root with the project environment active, run `python -m unittest tests.liveness.test_liveness_data -v` with [the source path configured](setup.md#run-unit-tests). These fixtures need no downloaded images or GPU. Real-data integrity, crop semantics and model accuracy require their separately attributed evidence.
