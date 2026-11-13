# LIV-03 — Anti-spoofing audit and loader

Document version: **LIV-03-v1**, created **12 November 2026**. Task owner: **Trang**. Independent reviewer: **chị Dung**. LIV-03 is defined in the [Project Tracker](../README.md#project-tracker), where its updates and records are maintained.

LIV-03 audits all 561,575 CelebA-Spoof images and loads live/print/screen face crops as 224×224 RGB tensors (0=live, 1=spoof). It preserves original test assignments, derives validation from training subjects, checks subject/file leakage and records exclusions. Reproducible commands generate the source/split inventory, audit report and aligned sample batches. Video independence remains unverified due to unavailable source-video IDs.

## Requirements and recorded results

The recorded full-data Colab execution passed the label/preprocessing/split checks and batch checks. Original-video independence remains unverified; PASS in the generated report covers only the available automated checks.

| LIV-03 requirement | Recorded result | Evidence |
| --- | --- | --- |
| Load live/spoof images with correct labels/preprocessing and retain documented partitions | PASS: 561,575 source images decoded; original test assignments preserved; validation fixed by subject | [Generated summary](../evidence/LIV-03/colab/colab-cpu-2026-11-12/audit-summary.json), selection and preprocessing below |
| Prevent frames from known common videos leaking across training/evaluation | Controlled known-ID rejection tests PASS; real-data independence UNVERIFIED due to missing source-video IDs | [Colab test log](../evidence/LIV-03/colab/colab-cpu-2026-11-12/colab-cpu-tests-2026-11-12.log), [scope and limitations](#scope-and-limitations) |
| Provide aligned sample batches and traceable source/split records | PASS: three batches of 8 images, with alignment and saved readback verified; full inventory retained | [Full outputs and previews](https://drive.google.com/drive/folders/11rMGigAD9uu9WZT5WeqnN-agSvDOxwYd), [artifact checks](../evidence/LIV-03/colab/colab-cpu-2026-11-12/colab-artifact-checks-2026-11-12.log) |

## Evidence and source access

| Resource | What to inspect |
| --- | --- |
| [Selected audit summary](../evidence/LIV-03/colab/colab-cpu-2026-11-12/audit-summary.json) | Counts, requirement results, preprocessing, source/runtime hashes and batch checks |
| [Full Colab outputs](https://drive.google.com/drive/folders/11rMGigAD9uu9WZT5WeqnN-agSvDOxwYd) | **audit.json**, **inventory.csv**, three tensor batches and their PNG previews |
| [Complete execution record](https://drive.google.com/drive/folders/1j2gu6F0AhgIFpbY57FXMUCiaKob8B7te) | Initial/final reports, preparation records, source hashes and logs |
| [Project source on Drive](https://drive.google.com/drive/folders/1wp-WH2Rgaow7_gi9OMOt3xmXD1-rd1zB) | 31 individually accessible source/config/test files, including all eight liveness modules; folder structure matches the repository |
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

The original **&lt;image-stem&gt;_BB.txt** stores the x/y coordinates, width, height and confidence in a 224-coordinate system. Coordinates are scaled to the original dimensions and clipped to image bounds. Nonfinite, nonpositive or empty boxes are reported rather than silently falling back to a full image. No detector runs during loading. A valid supplied box does not independently prove that exactly one face is visible; visual examples support label review, not a census of face count.

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

This performs another complete image audit, validates the accepted inventory, loads all three sample batches and checks saved tensor readback. Use it directly when a complete dataset already has the current **reviewed-data-description.json**, referenced project annotations, **validation-subjects.json** and **quality-findings.json**. Set the root to that dataset and use a fresh review output directory; extraction and split preparation are unnecessary in that case.

Expect **exit code 0**, **result=PASS** and **accepted=true** for the available automated checks. The common-video requirement remains UNVERIFIED for the downloaded images; **all_task_requirements_verified** remains false.

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
| [data_loader.py](../src/facial_attendance/liveness/data_loader.py) | Selected image/target loading with image and box checksum verification |
| [audit_report.py](../src/facial_attendance/liveness/audit_report.py) | Runtime metadata, report models and JSON/summary output |

The loader uses inventory validation and shared image/file functions; it does not import the audit workflow or reporting. Preparation and inventory construction share subject constraints and split assignment logic. Inventory construction reads annotations without decoding images; the audit performs decoding and hashing before the inventory is used for loading.

<details>
<summary>Dataset API and tensor contract</summary>

### Dataset API

After a successful final audit, run this block in a Python session with the project environment active and the [src directory](../src) on `PYTHONPATH`. Replace both paths for your machine. This optional example shows the API; the final audit already exercises batch loading automatically, including on Colab.

```python
from pathlib import Path
from torch.utils.data import DataLoader
from facial_attendance.liveness.data_loader import LivenessImages
from facial_attendance.recognition.data_loader import image_transform

root = Path("data/raw/liveness/celeba-spoof-full")
inventory = Path("evidence/runs/LIV-03/local/reviewer-01/final/inventory.csv")
dataset = LivenessImages(root, inventory, "train", transform=image_transform(224))
images, targets = next(iter(DataLoader(dataset, batch_size=8)))
print(images.shape, images.dtype, targets.shape, targets.dtype)
print(dataset.samples[0].relative_path)
```

Expected: image shape 8 × 3 × 224 × 224, float32; target shape 8, int64. Without a transform the dataset returns cropped PIL RGB images and integer labels. The loader verifies the inventory and original image/box hashes, applies no augmentation or learned normalization, and excludes quarantine/outside-scope rows.

</details>
