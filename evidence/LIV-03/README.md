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

## Experimental readiness — 14 November 2026

The [experimental summary](local/experimental-2026-11-14/summary.json) records PASS for experimental readiness under direct-pair-quarantine-v2. The public loader accepts its bound receipt without requiring unavailable capture IDs, exhaustive visual confirmation or perfect pHash transformation recovery. Historical full-verification results below remain unchanged. Readiness still checks annotation/source consistency, selected input integrity, subject and supplied-media/session splits, attack-family coverage, exact duplicates, the declared candidate policy and representative tensor batches.

The experimental subset contains train 277,903, validation 31,211 and test 30,811, or 339,925 images. Beyond exact deduplication, it excludes 27 confirmed near duplicates, 20 identical face crops and 63,390 conservatively quarantined candidates. Quarantined images are not claimed to be duplicates. Whole-image candidate constraints and reviewed source relations use direct pairs rather than transitive perceptual components. Crop-only pHash similarity is diagnostic regardless of split. All source files and original split assignments are preserved; metrics on this filtered test subset must be reported separately from the original benchmark.

The complete candidate set has a recorded disposition; no actionable pair remains under the declared scope. Independent signature-aggregate counting verified that result and zero remaining file/pixel/crop duplicate groups. It also records 45,711 retained cross-split crop-similarity occurrences as diagnostic evidence, without asserting that they are duplicates or independent captures. Review queues and direct-anchor previews support optional reinstatement review rather than requiring human adjudication of every perceptual pair.

All 89 configured tests passed. The public dataset API loaded all three splits, and 24 real tensor/target samples passed alignment and saved readback. Regression checks retain rejection of known leakage, rejected annotations, changed images, altered source/selection evidence and failed readiness checks. The 26/28 controlled crop result and unavailable capture provenance remain explicit limitations. This is Linux local evidence; model training, model effectiveness, new Colab/native Windows/macOS execution and independent human review are not recorded. See the [experimental guide](../../docs/liveness-loader.md#experimental-readiness-and-candidate-handling) for commands and scope.

The matching ignored local run is experimental-2026-11-14-v2. It contains the receipt, exclusions, policy, review queues, previews, public loader checks, independent counting and test log. Earlier execution directories retain their own implementation snapshots and results.

## Complete search and content review — 14 November 2026

The [selected summary](local/quality-2026-11-14/summary.json) records complete radius-6 search in both pHash views, reusing the bound full-scan fingerprints. It found 564,712 whole-image and 1,053,782 face-crop candidate pairs without truncation or a neighbor cap. These per-view counts can overlap. The compact graph, complete review CSV, implementation snapshots and review attachments remain in the matching ignored local run directory.

Inspection of 39 natural pairs recorded 27 duplicates, three distinct pairs and nine unresolved related-source pairs. Twenty-two confirmed duplicate pairs cross subject IDs and splits; this demonstrates that subject-disjoint assignments alone do not establish content independence. Confirmed duplicate exclusions leave train 318,230, validation 35,464 and test 49,641. Source files, original split assignments and historical evidence are preserved. A separate inspection confirmed 25 face boxes and supported 13 binary spoof labels among 25 deliberately selected images. These are assistant visual review records, not independent human review or a population accuracy estimate.

All 81 configured tests passed. Three new batches, eight images each, passed source/target alignment and saved readback on the revised subset. Its bound fingerprint index has zero remaining file/pixel duplicate groups. Image and box SHA-256 checks matched the inventory for all 1,775 images used by the scored/reviewed examples. The public loader rejected incomplete evidence. The final assessment remains FAIL because crop controls still miss two of 28 variants, capture relationships are unknown and complete visual coverage is unavailable. No training or new Colab/native Windows/macOS quality execution is recorded. See the [quality guide](../../docs/liveness-loader.md#complete-search-and-recorded-content-review--14-november-2026) for the execution boundary and commands.

## Local quality audit — 13 November 2026

The [selected summary](local/quality-2026-11-13/summary.json) records a full 561,575-image fingerprint scan, exact-duplicate exclusions, near-candidate search, source metadata inspection, visual review queues, quality assessment and three real diagnostic batches. Raw data, the original inventory and historical reports are preserved. The fingerprint source files are retained with hashes captured during the scan; later loader/acceptance revisions have separate tests and real batch checks.

Exact deduplication excludes 2,590 additional images, leaving train 318,256, validation 35,465 and test 49,641 in a cleaned subset. Verification finds zero remaining file/pixel duplicate groups and zero selected image/box errors. These exclusions do not replace the original benchmark assignments.

The overall assessment is FAIL: whole-image fingerprints recovered 28/28 variants for each tested JPEG setting and resize, but only 6/28 crop variants. A supplementary control combining whole-image and face-crop matches with known box-coordinate shifts recovered 26/28 crop variants; two still failed. This ideal-box control does not establish natural-pair recall or real detector consistency. Both views hit the candidate budget, retaining one million candidate rows, and 14,357 queries saturated their neighbor limit. Capture-media/session IDs remain unavailable for all 403,362 selected images. The visual queue contains 7,205 clipped boxes and 2,021 folder/JSON disagreements; no complete visual confirmation or independent reviewer confirmation is recorded.

All 77 configured unit tests passed. Three real cleaned-subset diagnostic batches passed image/target alignment and saved readback; the public loader rejected incomplete historical evidence. These checks establish software behavior, not full visual truth, model performance or completion of LIV-03. The new Colab workflow and native Windows/macOS quality execution were not run.

Complete local outputs remain in the ignored runs directory under the matching local quality run: fingerprints, exclusions, candidate/review tables, previews, source snapshot, assessment, batches and logs. The [quality guide](../../docs/liveness-loader.md#quality-audit-and-evidence-gate) defines reproduction and the evidence gate.

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
