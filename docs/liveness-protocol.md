# LIV-01 — Anti-spoofing data and attack protocol

Version **LIV-01-v2**, revised **5 November 2026** (initial output: 5 November). Owner: Trang. Independent reviewer: chị Dung. This is the inspectable design handover for LIV-01.1–LIV-01.5, with a [machine-readable dataset/split specification](../configs/liveness-protocol.json). It contains no measured model results or invented reviewer confirmation. The 4 November draft checkpoint has passed; this version must be recorded with its actual date.

## Scope and requirement boundary

The [project brief](reference/project-brief.pdf), sections 3 and 6, requires a trained anti-spoofing model integrated with face recognition and real-time liveness feedback. The group WBS additionally requires live/print/replay data, splits by subject/session/video where supported, and webcam tests with people, devices and lighting outside training conditions.

Use one **RGB CNN for binary live/spoof classification**, trained from scratch or partially fine-tuned. Run it on the webcam face crop and display the liveness result. The selected application behavior is to block attendance on spoof and show a warning; attendance requires both a live result and verified identity. This blocking policy is a project design choice, not an explicit rule in the brief. A missing/invalid face produces no attendance decision.

Motion analysis and multi-frame score aggregation are outside this baseline. There is no prescribed recording duration, frame-sampling rate, minimum valid-frame count or fixed test matrix. Model implementation and execution belong to later tasks; this handover defines the data and attack protocol.

## Source selection and access

Select **CelebA-Spoof** for the RGB appearance baseline. The authors provide [downloads and the Dataset Agreement](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-downloads). The agreement permits non-commercial research and limits copying/distribution to the stated internal organizational use. Keep images and derived dataset copies out of the shared repository; members should use the author's source. Commercial deployment requires a separate rights assessment. Public download availability does not imply an unrestricted license.

The [paper, §3.3](https://arxiv.org/html/2007.12342) reports 625,537 images, 10,177 subjects and subject-disjoint train/validation/test partitions in an 8:1:1 ratio. These describe the full release, including attack types excluded below; they are not local observations or counts for our subset. Still-image data supports appearance learning; it cannot establish temporal replay detection. Held-out webcam video replay is therefore a mandatory physical attack test.

On 5 November, the author's linked Drive folder could be listed through the connected Drive tool: archive parts `.001`–`.074`, approximately 77 GB compressed. No image archive was downloaded or decoded. Download quota, extraction size, metadata integrity and subject membership remain unverified. Selecting this source does not require training on the entire release. If a smaller working subset is used, document its selection and counts, retain subject-disjoint partitions and all selected attack families, and report results as subset results.

Replay-Attack was considered because it supplies video and official subject partitions. It was not selected for this handover: [the publisher's Zenodo record](https://zenodo.org/records/4593128) requires an EULA, non-commercial research use, an institutional email and a signatory holding a permanent position. The team has not supplied access approval. Do not replace this access process with an unofficial mirror. The assigned face-recognition competition has no established print/replay annotations and must not be relabelled as liveness data.

## Labels and input/output contract — LIV-01.1

Train a **binary** live/spoof classifier; retain attack subtype for evaluation. Project label `0 = live`, `1 = spoof`; prediction score `p_spoof` increases toward spoof. This is the classifier output, distinct from the face-verification similarity score; it is not assumed to be a calibrated probability. This is a proposed liveness convention for SYS-02 review, not an already agreed system-wide interface.

| Source spoof-type code | Project family | Binary label | Treatment |
| --- | --- | --- | --- |
| 0: Live | live | 0 | Include |
| 1: Photo, 2: Poster, 3: A4 | print | 1 | Include; preserve original code |
| 7: PC, 8: Pad, 9: Phone | screen | 1 | Include; preserve display category |
| 4, 5, 6: cut/mask variants; 10: 3D mask | outside selected scope | — | Exclude with reason; never relabel live |
| Missing/unknown code | invalid | — | Quarantine and stop manifest acceptance |

Code names and annotation positions come from the [author's annotation schema](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#annotation-detail): index 40 is spoof type; index 43 is the supplied binary label. Verify the binary encoding against the release before converting it; do not assume every source uses the project encoding. Screen images are not evidence that a moving video was presented. Webcam records distinguish `screen_photo` from `video_replay` explicitly.

Input at the data-design boundary is an original image or clip plus source identity/partition/attack metadata. Output is an inventory row with the columns in the JSON specification, plus selected/excluded/quarantined counts. Record session or parent-media identifiers only where the source supports them. Preserve native media and relative paths. The output inventory belongs under ignored `data/raw/liveness/`; put only aggregate audit results under `reports/`.

Face crop, detector, RGB conversion, input dimensions and normalization must be settled with SYS-02 before training. Use identical preprocessing for live and spoof, fit learned transforms only on train, and quarantine decode/no-face/multiple-face failures. Keep failure counts by partition and family so preprocessing cannot silently remove difficult attacks.

## Split specification and leakage gates — LIV-01.2

| Partition | Assignment | Permitted use | Local count |
| --- | --- | --- | --- |
| train | Official training subjects | Fit model and learned preprocessing | Not audited |
| validation | Official validation subjects | Choose checkpoint and threshold | Not audited |
| test | Official test subjects | Final locked evaluation | Not audited |
| webcam_test | New local participants, never used in training/selection | Physical print/screen/replay tests | Not run |

The JSON is a **split design manifest**, not a per-file observed inventory. Exact subject IDs, selected counts and file hashes require release metadata. Preserve official membership when selecting the live/print/screen subset; do not recreate the published ratio by randomly splitting images. If the supplied release has only train/test metadata, resolve the release's validation protocol before proceeding; do not borrow test for tuning or invent an official validation list.

Before LIV-02 consumes data, generate the inventory and enforce these gates:

1. Record source/release, acquisition date and the annotation version used. Validate every selected path and annotation. Inspect actual examples from each source code, including excluded masks, against labels.
2. Verify subject sets are pairwise disjoint across train/validation/test. All live images and attacks derived from one subject inherit that subject's partition. Keep session/video/parent-media groups together wherever those identifiers exist. Do not split extracted frames or augmented crops independently.
3. Hash original media and reject exact duplicates across partitions, including renamed copies. Keep known source-image derivatives in the same partition; byte hashes alone cannot identify related captures. Quarantine conflicts and resolve membership rather than silently moving official test samples.
4. Report selected/excluded/quarantined counts by split, subject and family. Require live, print and screen in every selected partition; report imbalance. Training-only sampling/weights may address imbalance later; do not rebalance test to improve a result.
5. Freeze the accepted inventory hash before training. Keep test and webcam_test inaccessible to model/threshold selection. Record all exclusions and any deviation from the official protocol. Missing subject or split information blocks acceptance.

These gates are specified, not executed on CelebA-Spoof in this version. A per-file manifest, leakage pass or class counts must not be claimed from the publisher totals.

## Physical webcam protocol

Test these cases with the same trained model, preprocessing and decision threshold:

| Case | Presentation to the capture camera | Expected behavior |
| --- | --- | --- |
| Live | Person physically present | Display live; allow attendance if identity is verified |
| Print | Printed face photograph | Display spoof and block attendance |
| Screen photo | Still face photograph on a display | Display spoof and block attendance |
| Video replay | Prerecorded face video playing on a display | Display spoof and block attendance |

Use consenting local participants who were not used for training or model selection. Include capture/presentation devices and lighting conditions outside the training setup where those details can be established. Record the actual people, devices and lighting used; disclose missing source metadata rather than claiming verified device/lighting independence. Keep framing and face size comparable across live and attack cases. The tests may share conditions; a full cross-product of people, devices, lights and repeated recordings is not required.

For each attempt, record the true case, participant ID, device/lighting condition, displayed prediction and whether attendance was allowed or blocked. Record the model/preprocessing version and threshold for the test session. Recording every attempt as a video is not required. Keep personal media local and obtain permission for any photos/videos used in attacks.

Choose the checkpoint and threshold using validation data, then keep them unchanged for final dataset tests and webcam attempts. Classify an image as spoof when `p_spoof >= threshold`. No averaging across frames or clip-level calibration is required. If test results motivate a change, identify it and separate development attempts from the subsequent final assessment.

Report live rejection and successful spoof detection separately, including correct/total attempts for print, screen photo and video replay. Note the conditions where the model fails. These are tests of an image-based classifier against physical attacks, not a claim that the model analyzes motion. Display liveness status in the live system; use the same attendance-blocking behavior in the demo.

## Review, corrections and handover — LIV-01.3–LIV-01.5

| Check | Expected result for independent review |
| --- | --- |
| Source/license | Author source and permitted use identified; no unrestricted/commercial claim |
| Labels | Unknown labels rejected; mask types excluded; screen photos not called verified video replay |
| Subject leakage | A deliberately repeated subject across splits fails acceptance; a valid disjoint counterpart passes |
| Duplicate leakage | Same bytes under another filename across splits fails; genuinely distinct disjoint records pass |
| Group leakage | Frames/crops inherit parent split; missing group metadata disclosed |
| Threshold leakage | Changing threshold after viewing test is a protocol deviation requiring a new holdout |
| Input failure | Missing or invalid face input produces no attendance decision |
| Scope/evidence | Publisher totals, planned test cases and observed measurements remain distinct |

Owner design checks on 5 November corrected three risks before handover: avoid a gated source without team access; preserve `screen` rather than inventing temporal replay labels; distinguish a split specification from a downloaded-data manifest. This is an owner self-review, not chị Dung's review or a response to findings she has not made.

The initial v1 design checks passed JSON parsing, label consistency, test-matrix arithmetic and local links; they did not execute the dataset or classifier. Revision v2 removes the fixed capture matrix and frame-aggregation requirements because the brief/WBS do not require them. JSON/Markdown consistency, label mapping, the four webcam cases, removal of fixed capture settings, local links and `git diff --check` passed for this revision using the existing Python 3.13 environment. No model or webcam execution is claimed.

For independent review, chị Dung reads this document and JSON, checks the sources and mapping, and records: reviewed version/hash, date, issue/location, expected behavior, decision, correction and retest result. Design review can proceed by inspection of these files without training or webcam execution. Once data is available, the data audit checks actual inventory records against the stated rules. Do not check reviewer confirmation until her evidence exists. No message has been sent on the owner's behalf.

| Task | Prepared here | Remaining acceptance evidence |
| --- | --- | --- |
| LIV-01.1 | Source choice, labels, contract, risks and test cases | Reviewer agreement |
| LIV-01.2 | Inspectable dataset/split specification and leakage gates | Reviewer inspection; acquired-data inventory when available |
| LIV-01.3 | Source/license assessment and review cases | chị Dung's independent review |
| LIV-01.4 | Owner design corrections documented | Resolve actual reviewer findings and obtain her retest |
| LIV-01.5 | Versioned handover and walkthrough instructions | Independent walkthrough/confirmation |

LIV-01 remains **ready for review, not accepted complete**. The [WBS and Acceptance Criteria](https://docs.google.com/spreadsheets/d/14iyTbZmOgO3WBDBv0jlLX9f2WhTDOMOeW1GbgnflECw/edit) require named human review; this handover cannot replace it. No tracker status or confirmation cells have been changed.
