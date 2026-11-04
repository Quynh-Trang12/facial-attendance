# LIV-01 — Anti-spoofing data and attack protocol

Version **LIV-01-v1**, prepared **5 November 2026**. Owner: Trang. Independent reviewer: chị Dung. This is the inspectable design handover for LIV-01.1–LIV-01.5, with a [machine-readable dataset/split specification](../configs/liveness-protocol.json). It contains no measured model results or invented reviewer confirmation. The 4 November draft checkpoint has passed; this version must be recorded with its actual date.

## Source selection and access

Select **CelebA-Spoof** for the RGB appearance baseline. The authors provide [downloads and the Dataset Agreement](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-downloads). The agreement permits non-commercial research and limits copying/distribution to the stated internal organizational use. Keep images and derived dataset copies out of the shared repository; members should use the author's source. Commercial deployment requires a separate rights assessment. Public download availability does not imply an unrestricted license.

The [paper, §3.3](https://arxiv.org/html/2007.12342) reports 625,537 images, 10,177 subjects and subject-disjoint train/validation/test partitions in an 8:1:1 ratio. These describe the full release, including attack types excluded below; they are not local observations or counts for our subset. Still-image data supports appearance learning; it cannot establish temporal replay detection. Held-out webcam video replay is therefore a mandatory physical attack test.

On 5 November, the author's linked Drive folder could be listed through the connected Drive tool: archive parts `.001`–`.074`, approximately 77 GB compressed. No image archive was downloaded or decoded. Download quota, extraction size, metadata integrity and subject membership remain unverified. No dataset-specific environment changes are needed for this design task.

Replay-Attack was considered because it supplies video and official subject partitions. It was not selected for this handover: [the publisher's Zenodo record](https://zenodo.org/records/4593128) requires an EULA, non-commercial research use, an institutional email and a signatory holding a permanent position. The team has not supplied access approval. Do not replace this access process with an unofficial mirror. The assigned face-recognition competition has no established print/replay annotations and must not be relabelled as liveness data.

## Labels and input/output contract — LIV-01.1

Train a **binary** live/spoof classifier; retain attack subtype for evaluation. Project label `0 = live`, `1 = spoof`; output score `p_spoof` increases toward spoof. This is a proposed liveness convention for SYS-02 review, not an already agreed system-wide interface.

| Source spoof-type code | Project family | Binary label | Treatment |
| --- | --- | --- | --- |
| 0: Live | live | 0 | Include |
| 1: Photo, 2: Poster, 3: A4 | print | 1 | Include; preserve original code |
| 7: PC, 8: Pad, 9: Phone | screen | 1 | Include; preserve display category |
| 4, 5, 6: cut/mask variants; 10: 3D mask | outside selected scope | — | Exclude with reason; never relabel live |
| Missing/unknown code | invalid | — | Quarantine and stop manifest acceptance |

Code names and annotation positions come from the [author's annotation schema](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#annotation-detail): index 40 is spoof type; index 43 is the supplied binary label. Verify the binary encoding against the release before converting it; do not assume every source uses the project encoding. Screen images are not evidence that a moving video was presented. Webcam records distinguish `screen_photo` from `video_replay` explicitly.

Input at the data-design boundary is an original image or clip plus source identity/partition/attack metadata. Output is an inventory row with the columns in the JSON specification, plus selected/excluded/quarantined counts. Preserve native media and relative paths. Unknown session or device fields remain explicitly `unknown`; they are not fabricated IDs. `source_media_id` links derivatives to the original where available. The output inventory belongs under ignored `data/raw/liveness/`; put only aggregate audit results under `reports/`.

Face crop, detector, RGB conversion, input dimensions and normalization must be settled with SYS-02 before training. Use identical preprocessing for live and spoof, fit learned transforms only on train, and quarantine decode/no-face/multiple-face failures. Keep failure counts by partition and family so preprocessing cannot silently remove difficult attacks.

## Split specification and leakage gates — LIV-01.2

| Partition | Assignment | Permitted use | Local count |
| --- | --- | --- | --- |
| train | Official training subjects | Fit model and learned preprocessing | Not audited |
| validation | Official validation subjects | Choose checkpoint and threshold | Not audited |
| test | Official test subjects | Final locked evaluation | Not audited |
| webcam_test | New local participants, never used in training/selection | Physical attack/domain-shift assessment | Planned 96 clips; none collected |

The JSON is a **split design manifest**, not a per-file observed inventory. Exact subject IDs, selected counts and file hashes require release metadata. Preserve official membership when selecting the live/print/screen subset; do not recreate the published ratio by randomly splitting images. If the supplied release has only train/test metadata, resolve the release's validation protocol before proceeding; do not borrow test for tuning or invent an official validation list.

Before LIV-02 consumes data, generate the inventory and enforce these gates:

1. Save release identity, archive/annotation SHA-256 hashes, access source and acquisition date. Validate every selected path and annotation. Inspect actual examples from each source code, including excluded masks, against labels.
2. Verify subject sets are pairwise disjoint across train/validation/test. All live images and attacks derived from one subject inherit that subject's partition. Keep session/video/parent-media groups together wherever those identifiers exist. Do not split extracted frames or augmented crops independently.
3. Hash original media and reject exact duplicates across partitions, including renamed copies. Inspect near-duplicate/source-image candidates as well; byte hashes cannot establish absence of related captures. Quarantine conflicts and resolve membership rather than silently moving official test samples.
4. Report selected/excluded/quarantined counts by split, subject and family. Require live, print and screen in every selected partition; report imbalance. Training-only sampling/weights may address imbalance later; do not rebalance test to improve a result.
5. Freeze the accepted inventory hash before training. Keep test and webcam_test inaccessible to model/threshold selection. Record all exclusions and any deviation from the official protocol. Missing subject or split information blocks acceptance.

These gates are specified, not executed on CelebA-Spoof in this version. A per-file manifest, leakage pass or class counts must not be claimed from the publisher totals.

## Physical webcam protocol

Use **3 consenting local participants × 2 capture cameras × 2 lighting conditions × 4 presentations × 2 takes = 96 ten-second clips**. Reserve every participant and clip for final assessment. Obtain agreement for recording and use of each participant's photograph/video; store pseudonymous IDs and keep consent records separate from shared evidence.

Choose two available capture devices with different models; record model, resolution, frame rate and auto-exposure. Compare them with the dataset's device documentation. Until their absence is verified, describe them as new local devices, not proven unseen training devices. Record diffuse frontal light and dim/backlit light, room/background and approximate face distance (0.5–1 m). Local scenes are outside the source collection; shared broad lighting categories do not by themselves prove an unseen illumination distribution.

| Case | Presentation to the capture camera | Expected behavior |
| --- | --- | --- |
| Live | Participant present, natural head/eye movement | Accept liveness if quality permits |
| Print | Participant's A4 face photo, both rigid and hand-held across takes | Reject spoof |
| Screen photo | Same person's still photo displayed on phone/tablet | Reject spoof |
| Video replay | Prerecorded face video, visibly moving/blinking, played on a display | Reject spoof despite facial motion |

Use two different display models across the two screen takes and record brightness/model. Record print size, paper finish and printer. Keep faces similarly framed across cases; test at least one tightly framed attack without obvious paper/screen borders. Randomize case order before capture. Store `subject_id`, session/take, source photo/video ID, camera, presentation device, lighting, true family, media hash, model/preprocessing/threshold versions, score and outcome. Never include these final clips in training; any later reuse requires a new untouched test cohort.

For the planned image model, sample 5 frames/s and average valid `p_spoof` scores per clip. Require at least 25 valid single-face samples; otherwise return a quality/error outcome and do not create attendance. Freeze this rule and the validation-selected threshold before test access. Include separate no-face, multiple-face, corrupt-media and camera-disconnect checks: each must return an explicit error with no attendance. These checks are additional to the 96 presentation clips.

Classify a valid score as spoof when `p_spoof >= threshold`. The threshold is selected from validation images and transferred unchanged to mean clip scores. This transfer is a predeclared baseline assumption, not clip-level calibration; report its webcam performance separately and do not tune aggregation or threshold on final clips.

At evaluation, report attack acceptance rate separately for print, screen photo and video replay, and live rejection rate, with numerator/denominator and quality-error counts. Report per-camera/lighting results and clip-level uncertainty; adjacent frames are not independent trials. Select the threshold on validation by minimum balanced live/spoof error (ties choose the lower threshold, which rejects more spoof). Keep that threshold for final test/webcam results. These are design decisions, not achieved performance targets. A small local test cannot certify workplace security or resistance to masks/deepfakes.

## Review, corrections and handover — LIV-01.3–LIV-01.5

| Check | Expected result for independent review |
| --- | --- |
| Source/license | Author source and permitted use identified; no unrestricted/commercial claim |
| Labels | Unknown labels rejected; mask types excluded; screen photos not called verified video replay |
| Subject leakage | A deliberately repeated subject across splits fails acceptance; a valid disjoint counterpart passes |
| Duplicate leakage | Same bytes under another filename across splits fails; genuinely distinct disjoint records pass |
| Group leakage | Frames/crops inherit parent split; missing group metadata disclosed |
| Threshold leakage | Changing threshold after viewing test is a protocol deviation requiring a new holdout |
| Runtime/data failure | Invalid media or insufficient face evidence produces error/no attendance |
| Scope/evidence | Publisher totals, planned clips and observed measurements remain distinct |

Owner design checks on 5 November corrected three risks before handover: avoid a gated source without team access; preserve `screen` rather than inventing temporal replay labels; distinguish a split specification from a downloaded-data manifest. This is an owner self-review, not chị Dung's review or a response to findings she has not made.

Local verification used the existing Python 3.13.16 environment: JSON parsing, selected/excluded label consistency, binary mapping, absent observed-data claims, pending-review state, the 96-clip/50-sample arithmetic and local Markdown links all passed. `git diff --check` passed. No training, GPU execution, image decoding, webcam capture or independent review was performed; these design checks do not validate a dataset or classifier.

For independent review, chị Dung reads this document and JSON, checks the sources and mapping, and records: reviewed version/hash, date, issue/location, expected behavior, decision, correction and retest result. She should walk through one valid inventory and each failure case above once actual metadata is available. Do not check reviewer confirmation until her evidence exists. No message has been sent on the owner's behalf.

| Task | Prepared here | Remaining acceptance evidence |
| --- | --- | --- |
| LIV-01.1 | Source choice, labels, contract, risks and test cases | Reviewer agreement |
| LIV-01.2 | Inspectable dataset/split specification and leakage gates | Reviewer inspection; acquired-data inventory when available |
| LIV-01.3 | Source/license assessment and review cases | chị Dung's independent review |
| LIV-01.4 | Owner design corrections documented | Resolve actual reviewer findings and obtain her retest |
| LIV-01.5 | Versioned handover and walkthrough instructions | Independent walkthrough/confirmation |

LIV-01 remains **ready for review, not accepted complete**. SYS-03 independent reproduction is also still pending in existing evidence. The [WBS and Acceptance Criteria](https://docs.google.com/spreadsheets/d/14iyTbZmOgO3WBDBv0jlLX9f2WhTDOMOeW1GbgnflECw/edit) require named human review; this handover cannot replace it. No tracker status or confirmation cells have been changed.
