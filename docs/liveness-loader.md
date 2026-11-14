# LIV-03 — Anti-spoofing audit and loader

Document version: **LIV-03-v4**, revised **14 November 2026**. Task owner: **Trang**. Independent reviewer: **chị Dung**. LIV-03 is defined in the [Project Tracker](../README.md#project-tracker), where its updates and records are maintained.

LIV-03 audits 561,575 CelebA-Spoof source images, face crops and fixed splits to prepare a reproducible training subset. Commands produce an inventory, candidate decisions, exclusions, checked batches and an experimental-readiness receipt. Missing capture provenance and incomplete visual review remain documented limits. Experimental readiness does not establish model effectiveness.

## Requirements and recorded results

The project brief requires a trained anti-spoofing model and integration with face recognition. It does not require exhaustive manual adjudication of perceptual candidates, complete capture-session metadata or visual confirmation of every dataset annotation before training. The public loader accepts a bound experimental-readiness receipt after the checks below pass. Full visual/provenance verification remains a separate assessment; its historical FAIL or INCOMPLETE results are preserved.

The historical Colab execution passed annotation mapping, preprocessing and batch checks under its recorded source hashes. It did not establish complete visual correctness, within-split deduplication, near-duplicate independence or video/session independence. The full-verification route returns INCOMPLETE when its stronger evidence requirements are missing; historical PASS fields remain unchanged.

| LIV-03 requirement | Recorded result | Evidence |
| --- | --- | --- |
| Load live/spoof images with correct labels/preprocessing and retain documented partitions | Historical automated checks: 561,575 decoded; original test retained; validation fixed by subject. Complete visual correctness unverified | [Generated summary](../evidence/LIV-03/colab/colab-cpu-2026-11-12/audit-summary.json), selection and preprocessing below |
| Prevent frames from known common videos leaking across training/evaluation | Controlled known-ID rejection tests PASS; real-data independence UNVERIFIED due to missing source-video IDs | [Colab test log](../evidence/LIV-03/colab/colab-cpu-2026-11-12/colab-cpu-tests-2026-11-12.log), [scope and limitations](#scope-and-limitations) |
| Provide aligned sample batches and traceable source/split records | PASS: three batches of 8 images, with alignment and saved readback verified; full inventory retained | [Full outputs and previews](https://drive.google.com/drive/folders/11rMGigAD9uu9WZT5WeqnN-agSvDOxwYd), [artifact checks](../evidence/LIV-03/colab/colab-cpu-2026-11-12/colab-artifact-checks-2026-11-12.log) |

## Experimental readiness and candidate handling

The [14 November experimental summary](../evidence/LIV-03/local/experimental-2026-11-14/summary.json) records a loadable subset of 339,925 images under direct-pair-quarantine-v2. Readiness requires source-consistent annotation mapping, valid selected inputs, subject and supplied-media/session split checks, live/print/screen coverage in every split, exact deduplication, completion of the declared candidate policy, and checked tensor/target batches. The receipt binds the inventory, source annotations, exclusions, candidate graph, reviews and relevant preprocessing/selection implementation. The loader rechecks image and box hashes on access. Adding an unrelated model module does not invalidate the receipt.

Published JSON annotations remain the label source. An attributed review that rejects a label or box excludes that image. Empty review rows, unavailable capture IDs and measured pHash misses remain informational limitations; they are not converted into PASS claims or training prohibitions. Known split overlap, altered bound inputs/evidence, missing attack families and unresolved policy constraints still prevent loading.

| Split | After exact deduplication | Experimental subset | Additional exclusions |
| --- | ---: | ---: | ---: |
| Train | 318,256 | 277,903 | 40,353 |
| Validation | 35,465 | 31,211 | 4,254 |
| Test | 49,641 | 30,811 | 18,830 |
| Total | 403,362 | 339,925 | 63,437 |

The additional exclusions comprise 27 confirmed near duplicates, 20 identical face crops and 63,390 conservatively quarantined candidates. Quarantine is not a duplicate verdict: some excluded images may be distinct. Original files and original split assignments are preserved, but this filtered test subset is not the unmodified benchmark. Report its metrics separately.

The finite candidate policy handles the recorded 1,618,494 candidate occurrences across both views as follows:

- Confirmed duplicates use deterministic representatives, preferring test, then validation, then train, followed by lexical path order. Contradictory review decisions are rejected. Exact face-crop equality is deduplicated; opposite binary labels on identical crops are quarantined.
- Unexempted whole-image candidates across splits create direct exclusion constraints. Within a split, unreviewed whole-image candidates of the same source type also create constraints. A priority-ordered selection quarantines an image if a directly related, higher-priority image was retained. It does not merge transitive perceptual components. Reviewed cross-split source relations create constraints even without whole-image hash support.
- Reviewed distinct pairs are exempt. Reviewed related presentations and different annotated source types remain valid contrasts within the same split. Crop-only pHash similarity is diagnostic in every split: without whole-image support, exact crop equality or an attributed review, it does not justify excluding an image or claiming a duplicate.
- A second traversal checks that no actionable candidate pair remains in the retained subset. The recorded run passed this check; a separate signature-aggregate verification also passed. Neither establishes recall outside radius 6 or independence of unavailable capture records.

This policy closes candidate handling through explicit disposition and quarantine, without requiring a human verdict on every pair. Human inspection remains useful for reinstating excluded images. The generated review queue records one reason/reference per excluded image; the anchor table groups direct references and prioritizes cross-split cases. These anchors are review aids, not inferred source-media/session IDs. Previewing several members never approves unseen members. Add explicit distinct or related-source decisions to the attributed review JSON and regenerate into a new output directory to reconsider exclusions; other direct constraints may still exclude an image.

<details>
<summary>Local experimental execution</summary>

Use the existing project environment from [setup](setup.md). The paths below identify the recorded local inputs; substitute your own baseline inventory, exact-stage directory and complete candidate-graph directory for independent reproduction. Reviews are optional: omit their arguments when unavailable, and record that no exemptions were supplied. Outputs reuse existing fingerprints and the compact graph without copying raw data.

```sh
data="data/raw/liveness/celeba-spoof-full"
inventory="$data/audits/current-schema-2026-11-12/inventory.csv"
scan="evidence/runs/LIV-03/local/quality-2026-11-13"
candidates="evidence/runs/LIV-03/local/quality-2026-11-14"
experiment="evidence/runs/LIV-03/local/experiment-review-01"
python scripts/review_liveness_data.py experimental --data-root "$data" --inventory "$inventory" --data-description "$data/reviewed-data-description.json" --output-dir "$scan" --candidate-dir "$candidates" --near-reviews "$candidates/natural-pair-reviews.json" --visual-review "$candidates/visual-review-inspected.csv" --experiment-output "$experiment"
```

Exit 0 means experimental readiness passed. The output includes **experimental-readiness.json**, **selection.csv**, **candidate-policy.json**, **review-priority.csv**, **review-anchors.csv**, and three tensor/preview pairs. Use a fresh output directory for each revision. Repeatable `--supporting-evidence` arguments bind additional review attachments.

To inspect four examples from the highest-priority direct anchor, run:

```sh
python scripts/review_liveness_data.py experimental-preview --data-root "$data" --inventory "$inventory" --data-description "$data/reviewed-data-description.json" --output-dir "$experiment" --anchor-index 0 --preview-offset 0 --preview-count 4 --preview-output "$experiment/quarantine-anchor-0.png"
```

Increase the offset to inspect more members or the anchor index to inspect another group. A JSON sidecar records exactly which examples were displayed; no verdict is generated.

</details>

<details>
<summary>Google Colab experimental execution</summary>

After the notebook baseline and exact stage, use the existing isolated interpreter and the same peer quality-route variables defined below. Upload current source files; the historical Colab source snapshot predates experimental readiness.

```python
candidate_dir = review / "candidates-complete"
stage("complete-candidates", quality_script, "near-complete", *quality_arguments,
      "--search-output", candidate_dir, expected=2)
experiment = review / "experiment-01"
stage("experimental", quality_script, "experimental", *quality_arguments,
      "--candidate-dir", candidate_dir, "--experiment-output", experiment, expected=0)
```

Supply the same attributed pair and visual review files with `--near-reviews` and `--visual-review` to reproduce the same exemptions. Without them the policy has no manual exemptions and counts may differ. The new experimental route has been executed locally on Linux; Colab and native Windows/macOS execution remain unrun.

</details>

Pass the receipt through the public API's `audit_file` argument, as shown in the [Dataset API](#dataset-api). For the recorded run, the public API loaded all three splits and verified 24 real tensor/target samples and saved readback. No model training or effectiveness measurement is claimed.

## Evidence and source access

| Resource | What to inspect |
| --- | --- |
| [Selected audit summary](../evidence/LIV-03/colab/colab-cpu-2026-11-12/audit-summary.json) | Counts, requirement results, preprocessing, source/runtime hashes and batch checks |
| [Full Colab outputs](https://drive.google.com/drive/folders/11rMGigAD9uu9WZT5WeqnN-agSvDOxwYd) | **audit.json**, **inventory.csv**, three tensor batches and their PNG previews |
| [Complete execution record](https://drive.google.com/drive/folders/1j2gu6F0AhgIFpbY57FXMUCiaKob8B7te) | Initial/final reports, preparation records, source hashes and logs |
| [Historical Colab source](https://drive.google.com/drive/folders/1wp-WH2Rgaow7_gi9OMOt3xmXD1-rd1zB) | 31 files used for the recorded 12 November execution; this snapshot predates the quality workflow |
| [Execution evidence index](../evidence/LIV-03/README.md) | Runtime, timings, verification scope and earlier Linux evidence |

Drive access depends on the folder's existing permissions and the dataset agreement. Public repository users can use the source in their clone and obtain CelebA-Spoof from the author's download below; the repository does not distribute raw data. A saved report establishes its recorded execution, not a new run on another device.

<a id="original-and-project-split-assignments"></a>

## Data selection and split integrity

An **inventory** is a CSV containing one row per source image, as defined in LIV-01. A **data description** is a JSON file recording the archive, source URL, received date, annotation paths, selection rules and fixed validation subject file.

An **original split assignment** is the train/test assignment in the downloaded annotations. A **project split assignment** is the train/validation/test assignment used by the loader after the agreed validation selection and exclusions. The CSV uses `original_split` for original split assignments and `split` for project split assignments. The data description uses `split_annotations` for annotation paths and `split_evidence` for their documented source. **Quarantine** records excluded images for review; it is not a training or evaluation split.

The source is [CelebA-Spoof](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-downloads), subject to the author's research-only agreement and redistribution restrictions. Raw images, crops, tensors and per-image inventories stay out of Git. Retain the archive on its existing local drive or Drive storage; no merged ZIP copy is required.

The downloaded dataset's [original annotation folder](#data-layout) annotations contain **494,405 train images from 8,192 subjects** and **67,170 test images from 1,004 subjects**. These observed dataset counts differ from the paper's full-dataset counts. Its **README** and supplied metadata provide train/test, not an original validation split.

The downloaded dataset differs from the earlier LIV-01-v2 assumption of original train/validation/test. Project validation comes only from original train and uses the first fold of StratifiedGroupKFold (10 folds, shuffle enabled, seed 42). Stratification balances source-type proportions, and grouping keeps each subject’s images together. Split preparation uses the full source annotations for types 0,1,2,3,7,8,9. The exact 821 validation subject IDs are fixed in **validation-subjects.json**; later extraction and audit reuse that file, rather than drawing another split.

The original annotations share subject IDs **5028, 7332 and 9735**. Their **216 source-train images** are recorded in quarantine (set aside from training and evaluation); all original test split assignments remain intact. No image is moved from original test into train or validation. Original files and labels are preserved for review.

The first full image audit additionally found **37 byte-identical image groups across train/validation**, involving six training subject IDs: **5940, 7141, 7143, 7145, 7312, 8807**. All **328 images** of those training subjects are quarantined, including 288 live/print/screen images that otherwise meet the selection rules. This preserves the fixed validation and original test split assignments and removes entire affected training identities, rather than only hiding duplicate filenames. The final audit independently rechecks that each additional quarantined subject has byte-identical image evidence in evaluation.

Final selected counts after all integrity exclusions are:

| Project split | Images | Subjects |
| --- | ---: | ---: |
| train | 320,660 | 7,362 |
| validation | 35,646 | 821 |
| test | 49,646 | 1,002 |

<details>
<summary>Annotation counts before duplicate-subject exclusions</summary>

Before integrity exclusions, the selected live/print/screen scope has:

| Project split | Images | Subjects | Live / spoof |
| --- | ---: | ---: | --- |
| train | 320,948 | 7,368 | 45.54% / 54.46% |
| validation | 35,646 | 821 | 45.56% / 54.44% |
| test | 49,646 | 1,002 | Original test distribution, unchanged |

These annotation-derived counts precede the six-subject duplicate exclusion. The [final expected results](#inspect-and-compare-the-results) include that exclusion and all image checks.

</details>

All **561,575 source images** remain in the inventory, including 544 quarantined training records and the predeclared outside-scope attacks.

## Labels and preprocessing

The [author's schema](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#annotation-detail) identifies source type at attribute 40 and live/spoof at attribute 43. Use the original JSON labels: 0=live and 1=spoof. The source has 2,022 training images in a live directory whose JSON says spoof type 7; directory names must not override those labels. Such disagreements are reported explicitly.

| Source type | Family | Binary target | Treatment |
| --- | --- | --- | --- |
| 0 | live | 0 | Include |
| 1,2,3 | print | 1 | Include |
| 7,8,9 | screen | 1 | Include |
| 4,5,6,10 | outside_scope | 1 | Inventory and decode; exclude from the project classifier |
| Unknown or conflicting JSON labels | invalid | — | Block acceptance |

Use identical preprocessing across labels and splits: the supplied single RetinaFace box, RGB conversion, bilinear resize to 224×224, float32 CHW scaled by 1/255 to [0,1]. The loader applies no augmentation or learned normalization. The model and webcam adapter must use consistent preprocessing when implemented under LIV-04 and SYS-02.

The original **&lt;image-stem&gt;_BB.txt** stores the x/y coordinates, width, height and confidence in a 224-coordinate system. Coordinates are scaled to the original dimensions and clipped to image bounds. Boxes losing over half their area outside the image are rejected; smaller clipping remains visible in quality review records. Nonfinite, nonpositive or empty boxes are reported rather than silently falling back to a full image. No detector runs during loading. A valid supplied box does not independently prove that exactly one face is visible; visual examples support label review, not a census of face count.

<a id="common-video-checks-and-evidence-limits"></a>

## Scope and limitations

This dataset supplies still images and 44 image attributes, but no documented original-video or session mapping. Unknown relationships are left blank. The real-data report therefore marks common-media independence **UNVERIFIED**. Filenames, lighting and attack type do not establish original-video relationships.

When documented relationships are available, **data-description.json** can name a CSV with exactly `relative_path,source_media_id,session_id`. All frames/crops derived from the same known original media must share its globally consistent ID. The audit rejects a known ID across train, validation and test splits even when image hashes or subject IDs differ. The regression suite checks that rejection and its valid same-split counterpart.

Exact-file SHA-256 does not detect recompressed or near-duplicate images. Sample visual review does not establish every annotation's correctness, motion-based replay performance, model accuracy or webcam behavior. Dataset acquisition and decoding evidence are distinct from the controlled regression fixtures. No Windows/macOS execution or model training is implied by the Linux loader run.

<a id="reviewer-starting-point"></a>

## Reproduce the results

Use these instructions to verify the recorded results or prepare the loader on another device. Choose the route matching your available inputs, then expand **Local execution** or **Google Colab execution** below. A section link may leave its collapse closed; click its title to reveal the steps. Keep the full dataset and agreed split rules unchanged.

| Starting point | Route | Needed before running |
| --- | --- | --- |
| Archive downloaded to local/external storage | [Local execution](#local-execution) | Project environment, all 74 parts, space calculated below and writable extraction/output destinations |
| Archive in Drive | [Google Colab execution](#google-colab-execution) | Drive access, Python 3.13 CPU runtime, about 85 GiB free VM disk plus variable Drive cache overhead; at least 1 GiB Drive output space |
| Complete prepared dataset | [Final audit](#5-final-audit-and-loader-check) | All source images/boxes, reviewed data description and referenced annotation/findings files; new output directory |
| Code only | [Fixture tests](#fixture-tests) | Project environment; no dataset download or GPU; does not replace a full-data audit |

The loader requires extracted images and box files. Neither a Drive URL nor a saved report alone is loadable training data. Training, application startup and webcam use are outside this task; use the repository's [setup guide](setup.md) for the wider project.

### Source, environment and locations

Use a checkout containing [prepare_liveness_data.py](../scripts/prepare_liveness_data.py), [check_liveness_data.py](../scripts/check_liveness_data.py) and the [liveness modules](../src/facial_attendance/liveness). The shared Drive source is also available above; it is a source copy, not a Git checkout. For a local run, select the [repository root](../README.md) as the working directory and follow [environment setup steps 1–2](setup.md#1-open-a-terminal-and-select-the-environment). Use Python 3.13 and the pinned project dependencies. CPU is sufficient; do not choose a GPU solely for these decoding/hash checks.

Download every archive part from the author's links under [download and extract](#1-download-and-extract), subject to the dataset agreement. The [local layout](#set-review-paths) shows where archives, extracted data and outputs belong. Those locations can be on different volumes; an external dataset does not need copying into the repo. Check free space on the actual destination volume before download or extraction.

### Storage plan

The figures below separate archive size, extracted storage and working space. Here, **storage partition** refers to the filesystem or volume whose free space is used; for example, Windows C:\\, a macOS volume mounted at /Volumes/MLData, or a Linux filesystem mounted at / or /mnt/data. Two folders on the same filesystem share its available space. **Google Drive** refers to the cloud service. Values are rounded; the filesystem estimate assumes 4 KiB allocation blocks.

| Storage item | Approximate space | How to use this figure |
| --- | ---: | --- |
| All 74 compressed archive parts | 71.7 GiB | Space on the storage partition holding the download |
| Uncompressed file contents | 73.1 GiB | Payload only; this excludes filesystem overhead |
| Extracted files including filesystem overhead | 76.4 GiB | Replaces the payload figure above; do not add the two |
| Extractor reserve | 2 GiB | Added to the extracted allocation for working/output space |
| Minimum free space checked before extraction | 78.4 GiB | Extracted allocation plus reserve, when archives are on a different storage partition |
| Archive, extracted allocation and reserve on one storage partition | 150.1 GiB | Combined requirement before downloading onto that partition |
| One full audit's generated outputs | About 125 MiB | Inventory, report and sample batches; fresh reproduction runs two audits |

Prepare **at least 160 GiB free before downloading** if the archive and extracted dataset share one storage partition, or **at least 85 GiB free on the partition used for extraction** if the archive is stored elsewhere. These planning allowances provide headroom above the estimates; they exclude environment installation, other datasets and variable Google Drive cache usage. Check the actual destination partition: filesystem allocation can differ, and the extractor calculates its minimum requirement from archive metadata before extraction. Auditing an already prepared dataset needs output space, not another extracted copy.

If the laptop has insufficient space, use [Colab](#google-colab-execution) or put the archive and/or extracted dataset on an existing external or mounted storage partition. Change only the path variables below; the repository and review outputs can remain on the laptop. Copying the dataset into the repository is unnecessary. Preparation needs write access to the extracted root; auditing an already prepared root can read it while writing outputs elsewhere.

The 74 parts form **one seekable ZIP**, not 74 independent archives. Keep all parts available until extraction and CRC checks finish. The extractor reads them directly without creating another merged ZIP. It refuses an existing output directory and has no resume mode. Do not rerun extraction into a partial directory; preserve the failure output and choose a new destination with sufficient space. This guide does not require deleting archives or source data.

If no suitable storage is available, run the fixture tests and record the full-data verification limit. Keep the documented full-data selection and split rules unchanged.

<a id="local-execution"></a>

<details>
<summary>Local execution</summary>

#### Local prerequisites

Follow the path initialization and fresh/prepared sequence below in one terminal with the project environment active. Use your file manager to create only the dataset parent and output parent directories before extraction; the dataset root itself must not already exist.

#### Set review paths

The default layout is:

```text
Project/
├── CelebA-Spoof-zips/
│   ├── CelebA_Spoof.zip.001
│   ├── ...
│   └── CelebA_Spoof.zip.074
└── facial-attendance/                 working directory
    ├── scripts/
    ├── src/
    ├── configs/project.toml
    ├── data/raw/liveness/
    │   └── celeba-spoof-full/         created by extraction
    └── evidence/runs/LIV-03/
        └── local/reviewer-01/        this run's outputs
```

Choose one initialization block. Substitute your actual mounted paths before running it. Paths containing spaces are supported because every command quotes the variables.

<details>
<summary>Linux/macOS path variables</summary>

**Linux/macOS — bash/zsh**

```bash
parts="../CelebA-Spoof-zips"
data="data/raw/liveness/celeba-spoof-full"
review="evidence/runs/LIV-03/local/reviewer-01"
```

For an external Linux drive, replace the first two assignments, for example:

```bash
parts="/mnt/ml-data/CelebA-Spoof-zips"
data="/mnt/ml-data/celeba-spoof-full"
```

On macOS, use the corresponding mounted paths from Finder. For example:

```bash
parts="/Volumes/MLData/CelebA-Spoof-zips"
data="/Volumes/MLData/celeba-spoof-full"
```

Use your actual mounted volume name; these examples do not mount a drive.

</details>

<details>
<summary>Windows PowerShell path variables</summary>

**Windows — PowerShell**

```powershell
$parts = "../CelebA-Spoof-zips"
$data = "data/raw/liveness/celeba-spoof-full"
$review = "evidence/runs/LIV-03/local/reviewer-01"
```

For an external drive, replace the first two assignments, using your actual drive letter:

```powershell
$parts = "E:/MLData/CelebA-Spoof-zips"
$data = "E:/MLData/celeba-spoof-full"
```

</details>

Forward slashes work in the Python commands.

Create the parent and review folders using your file manager. Leave the dataset root absent for fresh extraction. Check the destination drive's free space in its properties against the storage table above. Use a new review name for each run; existing results are not overwritten.

For shorter commands, set script paths once in the same terminal:

```bash
prepare="scripts/prepare_liveness_data.py"
audit="scripts/check_liveness_data.py"
```

In PowerShell, prefix the two variable names with `$`:

```powershell
$prepare = "scripts/prepare_liveness_data.py"
$audit = "scripts/check_liveness_data.py"
```

The path variables control this terminal session. They do not edit [configs/project.toml](../configs/project.toml). Every audit command below passes `--data-root`, so an external dataset does not silently fall back to the repository default.


#### Fresh reproduction

#### 1. Download and extract

Download **CelebA_Spoof.zip.001** through **CelebA_Spoof.zip.074** from the [author's download page](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-downloads) or its [linked Google Drive folder](https://drive.google.com/drive/folders/1OW_1bawO79pRqdVEVmBzp8HSxdSwln_Z?usp=sharing), subject to the author's dataset agreement. Put the 74 files directly in the archive directory selected above, not in another nested folder. Preserve filenames; browser suffixes such as **(1)** are not accepted. Parts 001–073 must each contain 1,048,576,000 bytes; part 074 must contain 453,369,048 bytes.

```sh
python "$prepare" --parts-dir "$parts" --output-dir "$data" --full
```

The command checks part sizes and destination space before extracting, then checks every file's ZIP CRC. Success writes **extraction.json** with **1,123,165 extracted files**. The archive's top-level **CelebA_Spoof** wrapper is removed by the extractor, so the chosen root directly contains the following:

#### Data layout

```text
celeba-spoof-full/                    data
├── README
├── extraction.json
├── Data/
│   ├── train/<subject>/<live-or-spoof>/<image>
│   └── test/<subject>/<live-or-spoof>/<image>
│       each image has a sibling <image-stem>_BB.txt
└── metas/
    ├── intra_test/
    │   ├── train_label.json
    │   └── test_label.json
    └── ...                          preserve other original metadata
```

Do not point `--data-root` at **Data**, **train**, an individual subject, or a parent that still contains the wrapper folder. All paths inside the data description are relative to the extracted root.

#### 2. Generate validation and project annotations

```sh
python "$prepare" --output-dir "$data" --configure-validation 10
```

This uses the original train/test annotations and the pinned scikit-learn 1.9.1 implementation: first fold of StratifiedGroupKFold, 10 folds, shuffle enabled, seed 42, stratified by source type and grouped by subject. It sets aside the three subjects shared by original train/test, preserves original test assignments and generates **validation-subjects.json**, the four **project_*.json** annotation files and **data-description.json** in the extracted root. No file from an owner-specific subset directory is required.

For the recorded archive, expect **821 validation subjects**. Check their identity independently of JSON whitespace or OS line endings:

Open **validation-subjects.json** to inspect the subject list. For an independent fingerprint check, run `python`, then enter this readable Python block (replace the root with your chosen path):

```python
from pathlib import Path
import hashlib
import json

root = Path("data/raw/liveness/celeba-spoof-full")
subjects = json.loads((root / "validation-subjects.json").read_text())
ids = sorted(subjects["validation_subjects"])
canonical = json.dumps(ids, separators=(",", ":")).encode("ascii")
print(len(ids))
print(hashlib.sha256(canonical).hexdigest())
```

Use `exit()` to return to your terminal before the next shell command.

Expected count: **821**. Expected canonical subject-list SHA-256:

```text
ce7f612210a5ef3e1f085012bfb979fc75ef53f15324b89d2c8d9400188b552f
```

If this differs, compare source annotation hashes and installed dependency versions before continuing. Do not choose a different fold to force matching counts. This fingerprint hashes only the sorted subject IDs; the audit separately records the actual file checksum.

#### 3. Run the initial full audit

```sh
python "$audit" --data-root "$data" --data-description "$data/data-description.json" --output-dir "$review/initial"
```

Progress is printed every 10,000 images. Fresh reproduction performs two full image scans: this investigation and the final audit after exclusions. An already prepared dataset needs only the final scan.

For this archive, this first run is expected to return **exit code 1** because it finds **37 identical-file groups across train/validation**. Its report and **rejected-inventory.csv** are needed by the next step. Continue only after confirming the expected SHA-256 overlaps, no missing attack families and no blocking errors in selected images. An unrelated failure must be investigated, not passed to duplicate resolution.

The one empty type-4 box remains an explicitly excluded error. The **summary.json** file in the initial audit folder records failure; it is not an accepted final result. Windows PowerShell exposes the exit code through `$LASTEXITCODE`; bash/zsh uses `$?` immediately after the command.

#### 4. Resolve the recorded duplicate training subjects

```sh
python "$prepare" --output-dir "$data" --resolve-audit "$review/initial/audit.json"
```

The command derives affected training subjects from the rejected inventory. For this archive they are **5940, 7141, 7143, 7145, 7312 and 8807**. It records **quality-findings.json**, writes new training/quarantine annotations and **reviewed-data-description.json**, and reuses unchanged validation/test annotations. It excludes all 328 images of these training subjects; source files are preserved. Total quarantine becomes **544 images** including the original train/test subject conflicts.

#### 5. Final audit and loader check

```sh
python "$audit" --data-root "$data" --data-description "$data/reviewed-data-description.json" --output-dir "$review/final"
```

This performs another complete image audit, validates a candidate inventory, loads three diagnostic sample batches and checks saved tensor readback. Use it directly when a complete dataset already has the current **reviewed-data-description.json**, referenced project annotations, **validation-subjects.json** and **quality-findings.json**. Set the root to that dataset and use a fresh review output directory; extraction and split preparation are unnecessary in that case.

Expect **exit code 2**, **result=INCOMPLETE**, **pipeline_completed=true** and **accepted=false** when automated checks finish but required quality evidence remains unavailable. Exit code 1 means execution or an automated integrity check failed. Exit code 0 is reserved for accepted evidence. The historical Colab report used the earlier acceptance semantics.

</details>

<a id="google-colab-review"></a>
<a id="google-colab-execution"></a>

<details>
<summary>Google Colab execution</summary>

#### Open and run the notebook

Start from the shared [facial-attendance folder on Drive](https://drive.google.com/drive/folders/1wp-WH2Rgaow7_gi9OMOt3xmXD1-rd1zB?usp=sharing). It contains the source files and **colab-liveness.ipynb**; no ZIP, manual source upload or manually added cells are needed.

1. In Drive, add shortcuts named **facial-attendance** and **CelebA-Spoof-zips** directly under **My Drive**. Point the first to the shared project and the second to the author's complete 74-part archive folder you are authorized to use. Shortcuts avoid exposing or configuring private parent paths.
2. In the shared project folder, open [colab-liveness.ipynb](https://drive.google.com/file/d/1Tp-3M5zgA1m_cuJMdv_2bQ3EH-YcHHhT/view) with **Google Colaboratory**. The same [notebook source](../notebooks/colab-liveness.ipynb) is included in this repository.
3. Choose **Runtime → Run all** and approve Google's Drive access prompt. The notebook selects CPU packages and the CPU environment check automatically; leave the default runtime/device settings unchanged. Do not copy cells from the general setup guide.
4. Follow progress in the existing cells. At completion, inspect the displayed batch previews and open the generated run folder beneath the Drive project's evidence/runs/LIV-03/colab directory.

The notebook copies only source/config/test files onto the VM, creates an isolated project environment, runs tests, extracts all archive entries, generates validation, performs the initial audit, resolves proven training duplicates and runs the final audit/loader checks. Each run receives a unique name. Logs, preparation records, reports, inventory and tensor/preview pairs are saved to Drive; the original archive remains unchanged.

**Before Run all:** the project must contain the shared source/config/test folders, the archive must contain all 74 parts, and the account must be able to read both folders and write results inside the project folder. If the shared project is read-only, use a writable copy of its source folders and notebook, with your project shortcut pointing to that copy. Allow at least **85 GiB free VM disk plus variable Drive cache overhead**, **1 GiB Drive output space**, and approximately **three hours** on hardware comparable to the recorded 2-core CPU VM. The notebook checks available VM space and Python 3.13 compatibility before processing; insufficient resources stop execution without changing the dataset or splits.

This notebook runs the implemented setup and LIV-03 data workflow. It does not launch registration, trained face models or a webcam UI, which are not implemented in this repository. The [recorded full-data Colab execution](../evidence/LIV-03/README.md#full-data-colab-execution--12-november-2026-utc) used CLI entry points; the notebook's syntax and runner were checked, but the notebook itself has not been executed end-to-end. Its outputs must be generated by its own run.

<details>
<summary>Outputs, interruptions and runtime compatibility</summary>

<a id="1-prepare-the-environment-and-drive-folders"></a>
<a id="6-run-the-final-audit-and-inspect-the-batches"></a>

The notebook saves initial and final outputs beneath each unique run's outputs folder and generated annotation/configuration JSON beneath its preparation folder. Compare the final **summary.json** with [the expected results](#inspect-and-compare-the-results). A final PASS still leaves video independence UNVERIFIED due to missing source-video IDs.

If a cell fails, inspect its saved stage log. Do not continue duplicate resolution after an unrelated initial-audit error. Extraction and image scans have no resume mode; preserve logs and use a fresh runtime for another complete attempt. Verify saved files are visible in Drive before disconnecting and deleting the runtime.

The recorded run used Python 3.13.18, 12.67 GiB RAM and 107.72 GiB total VM disk, with 24.27 GiB free after verification. Google does not guarantee identical allocation in later sessions. If the default Python version changes or VM space is insufficient, use a compatible runtime or the local route above; no device configuration is required for the current notebook workflow. Google documents [runtime versions](https://research.google.com/colaboratory/runtime-version-faq.html), [resource limits and Drive caching](https://research.google.com/colaboratory/faq.html), and [Drive shortcuts](https://support.google.com/drive/answer/9700156?hl=en&co=GENIE.Platform%3DDesktop).

</details>

</details>

### Output routing

Without `--output-dir`, the audit CLI creates a unique UTC run directory under the repository's ignored evidence/runs/LIV-03/local directory. Use `--execution-route colab` on Colab; the notebook sets this automatically and saves beneath the corresponding colab route in Drive. An explicit `--output-dir` remains the exact destination, so the local commands group initial/final stages under one local run. `--summary-output` still selects an explicit summary file; place it under the same route/run to keep evidence together. Selected reports follow the same local/colab/run-name layout in the [evidence index](../evidence/LIV-03/README.md).

### Inspect and compare the results

The review directory contains:

```text
reviewer-01/
├── initial/
│   ├── audit.json
│   ├── summary.json
│   └── rejected-inventory.csv
└── final/
    ├── audit.json
    ├── summary.json
    ├── inventory.csv
    ├── train.pt / validation.pt / test.pt
    └── train.png / validation.png / test.png
```

Read **summary.json** in the final audit folder for counts, requirements, runtime and hashes. **audit.json** additionally retains per-image batch records and all folder-label disagreements. **inventory.csv** contains every original image, including exclusions; it is not just the classifier's selected rows. The summary is generated by the command, not assembled manually.

| Final check | Field in the summary | Expected result for the recorded source |
| --- | --- | --- |
| Original rows / decoded originals | `inventory_rows` / `image_integrity.raw_images_decoded` | 561,575 / 561,575 |
| Selected train / validation / test | `counts.<split>.selected` | 320,660 / 35,646 / 49,646 |
| Selected subjects | `counts.<split>.subjects` | 7,362 / 821 / 1,002 |
| Quarantined source images | `quarantined_rows` | 544 |
| Blocking errors / overlaps / missing families | `blocking_image_errors`, `overlaps`, `missing_families` | Empty lists |
| Valid face inputs / excluded invalid box | `image_integrity.valid_face_inputs` / `excluded_image_errors` | 561,574 / one type-4 box error |
| Folder/JSON label disagreements | `folder_label_disagreement_count` | 2,022; JSON labels take precedence |
| Each saved batch | `batches.<split>.shape`, `image_dtype`, `target_dtype`, `range` | 8 × 3 × 224 × 224 float32 images; eight int64 targets; pixels in [0,1] |
| Alignment / saved readback | `batches.<split>.alignment`, `saved_batch_readback` | PASS for train, validation and test |
| Original-video/session independence | `requirements.known_common_media` | UNVERIFIED: relationship IDs are unavailable |

Here, the **split placeholder** means train, validation or test, and the batch fields are inside each split's batch object. In the full **audit.json**, folder disagreements are recorded as a path list rather than the summary's count.

<details>
<summary>Inspect saved tensors alongside the PNG previews</summary>

Inspect the three PNG contact sheets in tensor order, then inspect a saved batch:

Run `python`, then enter:

```python
from pathlib import Path
import torch

review = Path("evidence/runs/LIV-03/local/reviewer-01")
batch = torch.load(review / "final/train.pt", weights_only=True)
print(batch["images"].shape, batch["images"].dtype)
print(batch["targets"].dtype)
print(list(zip(batch["relative_paths"], batch["targets"].tolist())))
```

Set `review` to your actual output directory. Use `exit()` to return to the terminal.

Repeat for **validation.pt** and **test.pt**. Match each path/target to the corresponding batch row in **audit.json** and to the preview. Sampling covers every selected source type before filling the eight-image batch; this diagnostic does not alter split distributions.

</details>

Compare counts and the original annotation hashes recorded under `validation_from_train.original_annotation_sha256`:

| Original annotation | Recorded SHA-256 |
| --- | --- |
| **train_label.json** | fe96b85f575b98d7c231cbf24eb249014b57e95a9f7c1ebd3624034c30d7a30a |
| **test_label.json** | 5024238665680fd6df3a1b4b1cf00b75cc60386e6c6978822df86cfe794bd14d |

For comparison with an owner run, use its automatically generated summary from [LIV-03 evidence](../evidence/LIV-03/README.md). Compare selection rules, canonical validation subjects, relative image paths/labels, source annotation hashes, exclusions, counts and requirement results. Timestamps, absolute roots, received dates and source-code hashes for different revisions can differ. Newly generated JSON file hashes can also differ with OS line endings; binary tensor/preview hashes are records of the producing run, not substitutes for numerical image/target checks across platforms. Investigate unexplained changes instead of copying old hashes into new evidence.

A complete owner-run claim requires generated evidence for the stated source revision. Fixture logs and reports from earlier revisions keep their own execution scope. Record independent findings and the review outcome under LIV-03 in the [Project Tracker](../README.md#project-tracker).

<details>
<summary>Fixture tests without downloaded images</summary>

### Fixture tests

These checks need the project environment but no downloaded CelebA-Spoof images or GPU.

**Linux/macOS**

```bash
PYTHONPATH=src python -m unittest discover -s tests -p test_liveness_data.py -v
```

**Windows PowerShell**

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -p test_liveness_data.py -v
```

Expect exit code 0 and **OK**; inspect skipped tests separately. The tensor/CLI integration test requires Torch and Torchvision. A skipped integration check does not verify actual tensor loading. Use [the full suite](setup.md#run-unit-tests) for broader configuration/runtime checks; the discovered test count can change with the revision.

</details>

<details>
<summary>Troubleshooting and platform execution limits</summary>

| Symptom | Check/action |
| --- | --- |
| Not enough extraction space | Check available space on the volume selected by the data root; use a larger mounted destination and a new root. Keep all 74 parts available. |
| Missing part or unexpected part size | Check exact filenames, the archive directory and the sizes above; finish/retry the affected download before extraction. |
| Existing directory/file refused | Use a new run name/output path. If the dataset is already prepared, run the final audit rather than repeating preparation. |
| Missing **Data**/**metas** or unsafe path | Point to the extracted root shown in the layout; keep annotation paths relative to that root. |
| Validation fingerprint/count differs | Check original annotation hashes, dependency versions and the first-fold parameters; preserve the mismatch for review. |
| Initial audit fails for another reason | Inspect **audit.json**, overlaps and blocking errors before running resolution. |
| Memory pressure during decoding | Try `--workers 1`; ordered inventory/selection rules stay the same. Annotations and inventory still need memory. |
| Final report says PASS but a requirement is UNVERIFIED | Automated checks passed only within their stated scope. Missing original-video relationships remain an evidence limitation. |

Full extraction/audit execution is recorded on owner Linux and Colab CPU using the project Python 3.13 environment. The same CLI accepts Windows/macOS paths, but complete LIV-03 execution on those systems has not been established by the Linux run. The supported setup platforms and their separate runtime evidence are in [setup](setup.md#platform-verification). This audit does not establish model accuracy, training or webcam behavior.

</details>

## Quality audit and evidence gate

The baseline scan above produces diagnostic evidence. The [quality entry point](../scripts/review_liveness_data.py) supports the following full-verification stages. Experimental readiness uses their bound fingerprints and candidate graph, then applies the separate criteria above; it does not require every stronger verification claim to pass.

| Step | Work and evidence | Acceptance boundary |
| --- | --- | --- |
| 1 | Requirement results in the audit report | Exit 0 means accepted; 1 means failure; 2 means incomplete evidence |
| 2 | Exact file/pixel/crop fingerprints, duplicate groups and exclusions | All source rows scanned; label conflicts retained for review |
| 3 | Near-duplicate candidates on whole images and face crops; controlled transformation measurements | Candidate adjudication and observed search limitations remain explicit |
| 4 | Source metadata inspection and relationship review records | Capture-media/session identities require evidence; unknown remains unknown |
| 5 | Per-image visual review records and bounded previews | Full visual verification needs every retained row; experimental readiness excludes known rejections and records incomplete coverage |
| 6 | Quality assessment bound to the inventory, source annotations and review records | Full verification and experimental readiness use separate receipts; altered bound evidence is rejected |
| 7 | Tests, real-data checks and independent review | Fixture success does not establish full-data visual truth or reviewer confirmation |

The [13 November local summary](../evidence/LIV-03/local/quality-2026-11-13/summary.json) records 2,590 additional exact-duplicate exclusions and zero remaining file/pixel duplicate groups in the cleaned subset. Its fingerprint scan records 561,573 valid face inputs and two errors in already excluded attacks; the historical 561,574 count predates the stricter box guard. The overall assessment fails because controlled crop transformations were missed; source relationships and complete visual correctness remain unverified.

Exact deduplication keeps a deterministic representative, preferring test over validation over train, then lexical path order. Conflicting labels/source types are excluded pending adjudication. The original inventory and original split assignments remain unchanged; the exclusion table defines a cleaned subset. Report results for that subset separately from the original benchmark. Crop equality alone does not trigger exclusion: one source face can appear in distinct genuine and attack presentations.

Near-duplicate discovery uses 64-bit DCT hashes and Hamming radius 6 on whole images and supplied face crops. The current default searches every pair in this hash radius, using four 16-bit blocks and full-distance verification, without a neighbor limit or output cap. A pair within radius 6 must have at least one block differing by at most one bit. The optional diagnostic `--max-pairs` cap leaves the search incomplete and prevents acceptance. The 13 November evidence used approximate LSH and remains attributed to that implementation. Controlled JPEG, resize and crop trials measure their own recovery; natural-pair recall and false-positive rate require separately adjudicated examples. Same identity or similar appearance does not prove duplication or common capture media.

### Complete search and recorded content review — 14 November 2026

The [selected summary](../evidence/LIV-03/local/quality-2026-11-14/summary.json) records a complete radius-6 search on the 403,362 images retained after exact deduplication, reusing the bound 13 November fingerprints: 564,712 whole-image candidate pairs and 1,053,782 face-crop candidate pairs. The two views can contain the same image pair. Neither view was truncated or neighbor-limited. These are candidates, not confirmed duplicates or proof of visual independence outside the chosen hash radius.

Content inspection covered 39 natural pairs: 27 were recorded as duplicates, three as distinct and nine as related portrait/presentation content requiring further adjudication. This deliberately selected diagnostic set does not estimate population precision or recall. One confirmed duplicate joins source training subject 9225 and source test subject 7082, despite their different IDs. The duplicate review excludes the training copy while preserving the test copy. Other reviewed cross-split duplicates also demonstrate that subject IDs alone cannot establish content independence. Duplicate components use confirmed edges only; transitive perceptual candidates must not be merged automatically. The unreviewed candidate graph contains a component with 103,605 images.

The resulting exclusion table removes 27 additional duplicate images, leaving train 318,230, validation 35,464 and test 49,641, or 403,335 images total. Source files, the full inventory and original split assignments are preserved. A separate visual inspection checked 25 selected originals, supplied boxes, crops and resized inputs. All 25 boxes visibly selected the intended face; visible attack evidence supported 13 binary spoof labels. Remaining labels and all uninspected rows remain unconfirmed. These are assistant inspection records, not independent human reviewer confirmation.

Inspection of all ten packaged metadata JSON schemas still found no per-image source-media or capture-session mapping. That full-verification assessment remains FAIL: two of 28 controlled crop variants are missed even with the supplied ideal coordinate shifts. Natural-pair coverage, complete visual review and capture relationships remain unverified. It does not authorize training on its own; the experimental route above generates a separate receipt under its narrower criteria.

The [source paper](https://arxiv.org/html/2007.12342#S1.T1) describes session counts through illumination/environment categories. These categories do not identify individual capture sessions. The provenance stage records the packaged metadata inspected; it never creates capture IDs from filenames or condition labels.

<details>
<summary>Local quality execution</summary>

Use the existing project environment from [setup](setup.md), with the repository as working directory. For already prepared data, set paths and choose a new quality run directory:

```sh
data="data/raw/liveness/celeba-spoof-full"
inventory="$data/audits/current-schema-2026-11-12/inventory.csv"
description="$data/reviewed-data-description.json"
quality="evidence/runs/LIV-03/local/quality-review-01"
python scripts/review_liveness_data.py exact --data-root "$data" --inventory "$inventory" --data-description "$description" --output-dir "$quality"
python scripts/review_liveness_data.py near --data-root "$data" --inventory "$inventory" --data-description "$description" --output-dir "$quality"
python scripts/review_liveness_data.py provenance --data-root "$data" --inventory "$inventory" --data-description "$description" --output-dir "$quality"
python scripts/review_liveness_data.py visual --data-root "$data" --inventory "$inventory" --data-description "$description" --output-dir "$quality"
python scripts/review_liveness_data.py preview --data-root "$data" --inventory "$inventory" --data-description "$description" --output-dir "$quality" --preview-offset 0
python scripts/review_liveness_data.py assess --data-root "$data" --inventory "$inventory" --data-description "$description" --output-dir "$quality"
```

The default near stage writes the complete review CSV. To inspect the same fixed-radius candidates in a compact SQLite graph without expanding dense signature groups, run the following after the exact stage, using another new output directory:

```sh
python scripts/review_liveness_data.py near-complete --data-root "$data" --inventory "$inventory" --data-description "$description" --output-dir "$quality" --search-output "${quality}-complete"
```

The compact search returns exit 2 because discovery alone is unverified review evidence. It does not substitute for the review CSV consumed by assessment. After recording decisions, use repeatable `--supporting-evidence /path/to/attachment` arguments on the assess command to bind the reviewed previews and supporting records as well as the review tables. Existing input records and generated queues should be preserved separately from edited review copies.

Use the inventory generated by your own baseline scan when the historical local path is unavailable. Fingerprints and reports need about 1 GiB of free space in addition to existing source data. Existing outputs are refused. Exit 2 from a completed stage means unresolved evidence, not execution failure. Inspect that result before proceeding to the next stage; independent evidence collection can continue while a requirement is unresolved.

</details>

<details>
<summary>Google Colab quality execution</summary>

Supply the current source/config/test files before this route; the linked historical Drive snapshot predates the new quality modules. The [shared notebook](../notebooks/colab-liveness.ipynb) runs the baseline scan and reports incomplete acceptance. It does not perform the new complete quality workflow automatically. After its baseline scan, use its existing isolated interpreter, data root and output paths in a notebook code cell:

```python
quality_script = repo / "scripts/review_liveness_data.py"
quality = review / "quality"
quality_inventory = review / "final/inventory.csv"
quality_arguments = ["--data-root", data, "--inventory", quality_inventory,
                     "--data-description", data / "reviewed-data-description.json",
                     "--output-dir", quality]
expected_codes = {"exact": 0, "near": 2, "provenance": 2, "visual": 2, "assess": 1}
for phase, expected_code in expected_codes.items():
    stage("quality-" + phase, quality_script, phase, *quality_arguments,
          expected=expected_code)
```

These expected codes describe the full-verification reference results: assessment fails because controlled crop variants were missed. Inspect each actual result rather than changing expected codes to suppress a finding. The current default near stage is uncapped; an explicit diagnostic cap is recorded as incomplete coverage. The new Colab quality route has not been executed in the recorded local run. Keep complete outputs with the producing runtime; changing paths requires a new local assessment rather than editing historical hashes.

</details>

Review **near-candidates.csv** with verdict distinct, duplicate or uncertain, plus reviewer and evidence. Candidate identity columns are bound to the search; review columns can be filled without changing them. Confirmed duplicates with matching labels/source types create additional deterministic exclusions during assessment; contradictory decisions remain unresolved. Review **visual-review.csv** using confirmed, incorrect or uncertain for both label and box. Review **relationship-review.csv** with confirmed IDs or evidence-backed not_applicable; an empty ID remains unknown unless its inapplicability is established. The assessment records counts and hashes without authenticating the reviewer or independently establishing ground truth.

For full verification, use a new assessment name with `--review-output` to retain earlier assessments, then pass the resulting quality report to the baseline audit through `--quality-report`. The public loader accepts either a fully accepted audit or an experimental-readiness receipt supplied through `audit_file`; a standalone inventory is not sufficient. It verifies the applicable inventory, source, review and implementation hashes, applies bound exclusions and checks image/box hashes on access. Diagnostic batches alone do not authorize training. Original inventory counts and cleaned-subset counts are reported separately.

## Implementation reference

### Code responsibilities

| Component | Responsibility |
| --- | --- |
| [data_contract.py](../src/facial_attendance/liveness/data_contract.py) | Source label mapping and the per-image record contract |
| [dataset_files.py](../src/facial_attendance/liveness/dataset_files.py) | File hashes, duplicate-safe JSON reading and dataset path boundaries |
| [face_crop.py](../src/facial_attendance/liveness/face_crop.py) | Bounding-box handling, RGB decoding and face cropping |
| [split_rules.py](../src/facial_attendance/liveness/split_rules.py) | Shared subject assignment rules and overlap/family checks |
| [image_inventory.py](../src/facial_attendance/liveness/image_inventory.py) | Annotation inventory construction and CSV validation/read/write |
| [dataset_audit.py](../src/facial_attendance/liveness/dataset_audit.py) | Full image integrity checks and aggregation of audit results |
| [data_loader.py](../src/facial_attendance/liveness/data_loader.py) | Accepted image/target loading with evidence, image and box checks |
| [image_fingerprints.py](../src/facial_attendance/liveness/image_fingerprints.py) | Exact RGB fingerprints and perceptual candidate search |
| [near_search.py](../src/facial_attendance/liveness/near_search.py) | Complete fixed-radius candidate graph with source and implementation bindings |
| [candidate_policy.py](../src/facial_attendance/liveness/candidate_policy.py) | Direct candidate constraints, explicit review exemptions and conservative exclusions |
| [experimental_readiness.py](../src/facial_attendance/liveness/experimental_readiness.py) | Experimental subset, readiness checks and a bound training receipt |
| [sample_batches.py](../src/facial_attendance/liveness/sample_batches.py) | Shared tensor/target alignment, representative batches and saved readback |
| [quality_audit.py](../src/facial_attendance/liveness/quality_audit.py) | Full source-bound fingerprint scan and exact-duplicate exclusions |
| [quality_review.py](../src/facial_attendance/liveness/quality_review.py) | Candidate, provenance and visual review evidence; acceptance assessment |
| [audit_report.py](../src/facial_attendance/liveness/audit_report.py) | Runtime metadata, report models and JSON/summary output |

The loader uses inventory validation and shared image/file functions; it does not import the audit workflow or reporting. Preparation and inventory construction share subject constraints and split assignment logic. Inventory construction reads annotations without decoding images; the audit performs decoding and hashing before the inventory is used for loading.

<details>
<summary>Dataset API and tensor contract</summary>

### Dataset API

After experimental readiness passes, run this block with the existing project environment active and the [src directory](../src) on `PYTHONPATH`. Replace the input/output paths for your own run. Fully accepted legacy audits remain supported through the default audit beside their inventory.

```python
from pathlib import Path
from torch.utils.data import DataLoader
from facial_attendance.liveness.data_loader import LivenessImages
from facial_attendance.recognition.data_loader import image_transform

root = Path("data/raw/liveness/celeba-spoof-full")
inventory = root / "audits/current-schema-2026-11-12/inventory.csv"
experiment = Path("evidence/runs/LIV-03/local/experiment-review-01")
dataset = LivenessImages(root, inventory, "train", transform=image_transform(224),
                         audit_file=experiment / "experimental-readiness.json")
images, targets = next(iter(DataLoader(dataset, batch_size=8)))
print(images.shape, images.dtype, targets.shape, targets.dtype)
print(dataset.samples[0].relative_path)
```

Expected: image shape 8 × 3 × 224 × 224, float32; target shape 8, int64. Without a transform the dataset returns cropped PIL RGB images and integer labels. The loader verifies the applicable receipt, source/evidence bindings and original image/box hashes, applies no augmentation or learned normalization, and excludes outside-scope rows and the bound selection. Failed readiness and incomplete full-audit receipts are rejected; diagnostic limitations alone do not invalidate experimental readiness.

</details>
