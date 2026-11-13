# LIV-03 execution evidence

The [Project Tracker](../../README.md#project-tracker), Requirements tab, LIV-03 maintains task status and reviewer confirmation. The [loader guide](../../docs/liveness-loader.md) defines the data description, inventory, labels, fixed subject splits, preprocessing and commands.

## Evidence layout

```text
evidence/LIV-03/
├── README.md
├── local/<run-name>/
└── colab/<run-name>/
```

Routine generated outputs use the same route/run structure beneath the ignored evidence/runs/LIV-03 directory. Historical timestamps, paths and hashes inside saved JSON/log files remain unchanged when files are relocated.

## Independent reproduction

Use the [reproduction guide](../../docs/liveness-loader.md#reproduce-the-results) to choose fixture checks, a full audit of prepared data, or fresh extraction and preparation. It defines storage allowances, portable paths, the initial expected duplicate failure, resolution and the final audit. These steps generate their own summaries and preserve the owner records.

The final review command writes **summary.json**, **audit.json**, **inventory.csv**, three tensor batches and three previews inside the final audit directory. The summary is created automatically from the report. Compare [the expected results](../../docs/liveness-loader.md#inspect-and-compare-the-results), original annotation hashes, validation subject fingerprint, source hashes and evidence limits. A report from an earlier code revision or a fixture test log does not establish a new full-data execution.

New audits default to the ignored runs directory, separated by local/colab route and a unique UTC run name. Explicit output directories remain supported for initial/final stages. Promote selected reports into the matching route/run directory here, preserving their bytes; existing output filenames are refused.

<a id="full-data-colab-execution--12-november-2026-utc"></a>

## Colab CPU audit — selected evidence

Recorded at **12 November 2026, 16:12:05 UTC**.

The [selected summary](colab/colab-cpu-2026-11-12/audit-summary.json) was generated automatically by the final audit CLI on Colab CPU and copied byte-for-byte into the Colab run folder. Its audit SHA-256 is **b9698a92dfd5eb32c9c50475dd744d9d4a000ec0735e0c60c1da77830bad5e51**. Source-code hashes identify the executed workspace files; this source snapshot was uploaded without Git metadata.

The [facial-attendance/evidence/runs/LIV-03/colab/colab-cpu-2026-11-12 folder](https://drive.google.com/drive/folders/1j2gu6F0AhgIFpbY57FXMUCiaKob8B7te) retains initial/final audit reports and inventories, all three tensor/preview pairs, preparation JSON, source-file hashes, installation/test logs and generated stage timings. The [final output folder](https://drive.google.com/drive/folders/11rMGigAD9uu9WZT5WeqnN-agSvDOxwYd) contains the full 561,575-row inventory and traceable sample batches. Access remains subject to the dataset agreement and folder permissions.

| Check | Recorded result |
| --- | --- |
| Runtime | Python 3.13.18; CPU, 2 cores; 12.67 GiB RAM; 107.72 GiB VM disk |
| Storage | 105.79 GiB free before extraction; 24.27 GiB free after verification; archive read from Drive without a full VM copy |
| Extraction | 1,123,165 files, all ZIP CRCs passed |
| Validation | 821 subjects; canonical fingerprint matches the fixed selection |
| Initial audit | All 561,575 images checked; 37 train/validation byte-duplicate groups; exit 1 as expected |
| Resolution | Six additional training subjects quarantined; 544 total quarantined images |
| Final audit | PASS; 561,575 raw images decoded; zero blocking errors and zero reported split overlaps |
| Selected images | Train 320,660; validation 35,646; test 49,646 |
| Batches | Train/validation/test each 8 × 3 × 224 × 224, float32 images and int64 targets; alignment and saved readback PASS |
| Common-video independence | UNVERIFIED; no source-video IDs supplied; all-task-requirements flag remains false |

The initial and final scans took **3,936.835** and **4,042.242 seconds**, respectively. The single invalid face input is the already excluded type-4 box; its JPEG decoded successfully. The final inventory and all three tensor-file SHA-256 values match the earlier Linux full-data run. The [artifact checks](colab/colab-cpu-2026-11-12/colab-artifact-checks-2026-11-12.log) also verified that the final reports, inventory, tensors and previews copied to Drive match the VM originals and that all 31 uploaded source files retained their hashes.

The [Colab liveness tests](colab/colab-cpu-2026-11-12/colab-cpu-tests-2026-11-12.log) record **23 passing tests**, including controlled common-media leakage rejection and its valid counterpart. These tests supplement the full source-image execution. The completed runtime was released after the outputs were preserved and verified; raw images can be reconstructed from the original Drive archive.

<a id="earlier-linux-data-audit--12-november-2026"></a>

## Local Linux audit — historical evidence

Recorded at **12 November 2026, 01:19:56 UTC**.

The [historical audit report](local/full-data-2026-11-12/audit.json) is retained byte-for-byte under the local route. Its original data-root paths identify the producing run; source images, inventories and batches remain outside Git. It records all 561,575 images decoded, zero blocking image errors, zero subject-ID or identical-file overlap, and selected counts of train 320,660, validation 35,646 and test 49,646. The single invalid box is a previously excluded type-4 mask. This run predates the automatic summary export and saved-batch readback added to the command; its implementation hashes identify the code it executed.

## Source and protocol findings

- The local **CelebA_Spoof.zip.001–.074** parts total 76,999,417,048 bytes. The archive contains 78,458,144,960 payload bytes.
- Original intra-test annotations identify 494,405 training images and 67,170 test images. No original validation annotation is supplied in this dataset.
- Subject IDs 5028, 7332 and 9735 occur in both original splits. Their 216 original training records are recorded in quarantine; original test split assignments are preserved.
- The first full scan found 37 byte-identical groups across train/validation, involving training subjects 5940, 7141, 7143, 7145, 7312 and 8807. All 328 source images of those six subjects are recorded in quarantine. Validation/test split assignments and the fixed subject assignment remain unchanged; the full inventory still contains every original image.
- One source type-4 mask has an empty box file (**004046_BB.txt**, for source image 004046 under subject 3329 in the original training spoof directory). Type 4 was excluded by the original classifier protocol. Its source bytes and failure record are preserved; the reviewed audit checks raw pixel integrity separately and does not use a full-image training fallback.
- Rich JSON annotations identify 2,022 original training images under live folders as PC spoof. Use attributes 40 and 43 for source type and binary label; paths remain unchanged.
- Validation uses the first fold of StratifiedGroupKFold (10 folds, shuffle enabled, seed 42), stratified by source type and grouped by subject ID. Its 821 subject IDs were fixed before the full-dataset run. This is project validation derived from original train, not an original validation split.

## Software verification

[Output-routing verification](local/output-routing-2026-11-13/tests.log) recorded **70 passing tests** in the existing local Python 3.13 environment. It checks local/Colab default paths, explicit output overrides and invalid routes, alongside the full regression suite. This code change did not rerun the full source-image audit; the selected Colab report retains its original implementation hashes.


[Reviewer workflow checks](local/reviewer-workflow-tests-2026-11-12/reviewer-workflow-tests-2026-11-12.log) verify the documented CLI sequence on generated images with spaces in storage paths: validation generation, expected duplicate rejection, resolution, final audit and loader API. Fresh preparation from the complete original annotations independently reproduced all 821 validation subjects and the documented canonical fingerprint. Exact bash snippets were also executed against the existing prepared dataset for path/space checks, validation fingerprint, saved-batch inspection and one loader batch. These checks did not repeat full extraction or the full source-image audit; Windows/macOS execution was not performed.

[Data-component checks](local/data-modules-tests-2026-11-12/data-modules-tests-2026-11-12.log) record 69 passing tests for the separated loader, inventory, image, split, audit and reporting components. Coverage includes annotation-only inventory construction, loader import isolation, shared subject validation during preparation, and the existing label, image-integrity, leakage and evidence checks. Current components also read all 561,575 saved inventory rows and reproduced all 24 saved sample images/targets/paths with unchanged inventory and batch hashes. The full image audit was not repeated.

[Reporting checks](local/reporting-tests-2026-11-12/reporting-tests-2026-11-12.log) record 67 passing tests after separating report construction and JSON output from the CLI. The tests cover automatic summaries, saved-batch readback, source hashes including the reporting module, unknown-media limits and failure after an accepted inventory. This verification uses controlled fixtures; the full dataset audit was not repeated.

[Software checks](local/summary-export-tests-2026-11-12/summary-export-tests-2026-11-12.log) record 66 passing tests in the existing Linux Python 3.13 project environment. Fixtures exercise audit/loader output, automatic summary generation, preserved result fields, identical repeated exports, overwrite refusal, saved tensor readback and failure reporting. These tests do not represent another full-dataset audit.

## Evidence limits

The dataset supplies still images without documented original-video or capture-session IDs. Their absence does not prove video independence. Exact-file SHA-256 does not establish near-duplicate or recompressed-image independence. Sample image review does not establish every label or face count. Model accuracy, motion-based replay detection, GPU training and webcam inference are outside this loader audit.

## Local logs

Historical execution logs remain in ignored local run storage. The selected Colab summary and logs linked above are supplied with the repository; full inventories and batches are retained in the linked Drive run.
