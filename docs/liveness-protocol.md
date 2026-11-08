# LIV-01 — Anti-spoofing data and attack protocol

Protocol version: **LIV-01-v2**, revised **8 November 2026**. Owner: **Trang**. Independent reviewer: **chị Dung**. Status: **Pending review**.

This protocol defines the data selection, labels, split rules and physical attack tests for LIV-01.1–LIV-01.5. The accompanying [JSON specification](../configs/liveness-protocol.json) stores the label mapping, split rules and planned test cases. The per-image inventory will be generated during LIV-03 data auditing.

## Scope

Sections 3 and 6 of the [project brief](reference/project-brief.pdf) require a trained anti-spoofing model integrated with face recognition and real-time liveness feedback.

The baseline uses live, print and screen images. Train, validation and test contain different people: all images showing one person, including spoof images made from that person’s face, stay in one split. Crops, resized images and frames extracted from the same original media stay with that original media. Webcam evaluation covers live faces, prints, screen photographs and video replays.

Use one **RGB CNN for binary live/spoof classification**, trained from scratch or partially fine-tuned. Run it on the webcam face crop and display the liveness result. Block attendance on a spoof result and show a warning. Attendance requires both a live result and verified identity. A missing/invalid face produces no attendance decision.

The model classifies one face image at a time. It does not analyze movement or combine predictions from a sequence of frames. Data auditing and loader implementation belong to LIV-03; model training and evaluation belong to LIV-04.

## Source selection and access

Select **CelebA-Spoof** for the RGB appearance baseline. The authors provide [downloads and the Dataset Agreement](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#dataset-downloads). The agreement permits non-commercial research and prohibits further copying, publication or distribution, except for internal copies at a single site within the same organization. Keep original images and processed copies, such as resized images and face crops, out of the shared repository; members should use the author's source. Commercial deployment requires a separate rights assessment. Public download availability does not imply an unrestricted license.

The [paper, §3.3](https://arxiv.org/html/2007.12342) reports 625,537 images, 10,177 subjects and subject-disjoint train/validation/test partitions in an 8:1:1 ratio. These are the authors’ counts for the complete published dataset, including attack types excluded below. The team’s selected images and subjects must be counted separately after download and filtering.

CelebA-Spoof provides still images. An image classifier can learn visual differences between a live face and a face photographed from paper or a screen. During a video-replay test, it classifies an image captured by the webcam while the attack video plays on a display. This tests detection of the replay presentation from its appearance; it does not test motion analysis or establish replay-detection accuracy before execution.

Access check on 5 November 2026: the author's linked Drive folder listed archive parts .001–.074, approximately 77 GB compressed. Image download and decoding were not tested; download quota, extraction size, annotation integrity and the supplied subject/split assignments remain unverified. A smaller training subset is permitted if its selection and counts are documented, each selected image keeps its official split assignment and all selected attack families remain represented, and results are reported as subset results.

Replay-Attack was considered because it supplies video and official subject partitions. It was not selected: [the publisher's Zenodo record](https://zenodo.org/records/4593128) requires an EULA, non-commercial research use, an institutional email and a signatory holding a permanent position. Access approval is not recorded. Acquisition requires the publisher’s approval process. The assigned face-recognition competition has no established print/replay annotations and must not be relabelled as liveness data.

## Labels and input/output contract — LIV-01.1

Train a **binary** live/spoof classifier; retain attack subtype for evaluation. Project label 0 = live, 1 = spoof; prediction score p_spoof increases toward spoof. This is the classifier output, distinct from the face-verification similarity score; it is not assumed to be a calibrated probability. LIV-01 defines the dataset labels and score direction. The shared interface specification in SYS-02 records how the liveness adapter exposes its result to other modules; LIV-02 defines registration/attendance behavior.

| Source spoof-type code | Project family | Binary label | Treatment |
| --- | --- | --- | --- |
| 0: Live | live | 0 | Include |
| 1: Photo, 2: Poster, 3: A4 | print | 1 | Include; preserve original code |
| 7: PC, 8: Pad, 9: Phone | screen | 1 | Include; preserve display category |
| 4, 5, 6: cut/mask variants; 10: 3D mask | outside selected scope | — | Exclude with reason; never relabel live |
| Missing/unknown code | invalid | — | Quarantine and stop manifest acceptance |

Code names and annotation positions come from the [author's annotation schema](https://github.com/ZhangYuanhan-AI/CelebA-Spoof#annotation-detail): index 40 is spoof type; index 43 is the supplied binary label. Check the downloaded annotation files before converting their binary labels to the project mapping. A screen-image label does not establish that a moving video was displayed. Webcam records therefore distinguish screen_photo from video_replay.

### Per-image inventory

During LIV-03, create a CSV inventory with one row per original image. Use the columns listed in the JSON specification: relative_path, subject_id, split, source_type, binary_label, attack_family and sha256. CSV stores the image records; JSON stores the rules used to create and check those records.

Use subject IDs, attack labels and split assignments from the dataset’s annotation files. Where available, also record:

- **Recording-session ID:** identifies the capture session in which the image was produced. Record it only if the dataset documentation or supplied metadata identifies that session.
- **Source-media ID:** identifies the original image or video from which a sample was created. A frame extracted from a video keeps that video’s ID; a crop keeps its original image’s ID. For samples generated by the team, record this relationship during processing.

Do not infer a recording session merely because images share lighting or indoor/outdoor labels. If the source dataset does not provide a relationship, leave it unknown and report the limitation.

Keep original media, processed image copies and the CSV inventory under ignored data/raw/liveness/. Save summary counts and audit findings under reports/, for example counts of selected, excluded and unreadable images by split and attack family.

Before training, document the selected detector, face-crop method, RGB conversion, input dimensions and normalization. Keep the model’s preprocessing and the shared adapter interface consistent. Use identical preprocessing for live and spoof, fit learned transforms only on train, and set aside unreadable images and images with no detected face or multiple detected faces for investigation. Keep failure counts by partition and family so preprocessing cannot silently remove difficult attacks.

## Split rules and overlap checks — LIV-01.2

| Partition | Assignment | Permitted use | Local count |
| --- | --- | --- | --- |
| train | Official training subjects | Fit model and learned preprocessing | Not audited |
| validation | Official validation subjects | Choose checkpoint and threshold | Not audited |
| test | Official test subjects | Final locked evaluation | Not audited |
| webcam_test | New local participants, never used in training/selection | Physical print/screen/replay tests | Not run |

The paper describes official train, validation and test partitions. When downloading the dataset, locate the annotation files that identify each image’s person, label and split, and check that they match the published partition scheme. Missing or ambiguous split information must be resolved before training. Do not use test images to select a model or threshold.

When selecting the live/print/screen subset, keep each selected image in its original split. For example, an image assigned to official test remains in test after filtering. Do not randomly redistribute selected images to reproduce the published 8:1:1 ratio.

During LIV-03, generate the CSV inventory and perform these checks:

1. **Record the downloaded data.** Save the source URL, the dataset version or archive names, the date the team downloaded or received the files, and the annotation files used.
2. **Check labels against images.** Open sample images for every source attack-type code, such as 1 = Photo and 9 = Phone. Compare their visible content with the annotation and project mapping. Include excluded mask codes to check that the filter removes the intended images. Record mismatches; inspecting examples does not verify every label.
3. **Check people across splits.** Compare the subject-ID sets for train versus validation, train versus test, and validation versus test. No person may occur in both sets in any pair. All live and spoof images representing one person stay in that person’s assigned split.
4. **Keep related samples together.** Frames extracted from one video stay in that video’s split. Face crops, resized copies and training augmentations such as flips or brightness changes stay in their original image’s split. Use documented recording-session and source-media relationships where available. Lighting labels such as normal, strong, backlit or dark, and environment labels such as indoor/outdoor, describe conditions; they do not establish that two images came from the same recording.
5. **Check identical files and related images.** Calculate a SHA-256 hash from each original file’s bytes. Identical files have the same hash even when renamed, so reject copies appearing across splits. Resizing or recompressing an image can change its hash; also use known source-media relationships to keep those related copies together. Investigate conflicts without silently moving official test images into train.
6. **Count the selected data.** Report selected, excluded and unresolved image counts by split, person and attack family. Include live, print and screen in each selected partition and report imbalance. Apply any sampling or class weights only during training; preserve the evaluation distribution.
7. **Fix the inventory before training.** Save the accepted CSV and its SHA-256 hash so later runs can identify the exact inventory used. Model and threshold selection use validation, not test or webcam-test results. Record exclusions and any deviation from the official split rules. Resolve missing person/split assignments before accepting the inventory.

Use the dataset annotations to identify the person, label and split for each image. Open the downloaded images to check for unreadable or damaged files, and calculate a SHA-256 hash for each file to find identical copies, even when their filenames differ. After recording which images are kept in the inventory, count those records by split and label to report the number of images selected for the project.

CelebA-Spoof inventory generation and overlap checks have not been executed. The team’s selected image counts and split assignments remain unaudited.

## Physical webcam protocol

Test these cases with the same trained model, preprocessing and decision threshold:

| Case | Presentation to the capture camera | Expected behavior |
| --- | --- | --- |
| Live | Person physically present | Display live; allow attendance if identity is verified |
| Print | Printed face photograph | Display spoof and block attendance |
| Screen photo | Still face photograph on a display | Display spoof and block attendance |
| Video replay | Prerecorded face video playing on a display | Display spoof and block attendance |

Use consenting local participants who were not used for training or model selection. Keep framing and face size comparable across live and attack cases.

Where practical, include camera/display devices or lighting conditions different from those documented in training, to test behavior under changed conditions. Record the actual devices and lighting used. If training-device information is unavailable, do not claim that the test devices are unseen.

Cover the four presentation types in the table. Testing every combination of person, device and lighting is not required; record the combinations actually used.

### One webcam test attempt

1. Present the live face, print, screen photograph or playing video to the webcam.
2. Trigger a check that captures one image for evaluation. Do not select an image after viewing its prediction. If no valid face is obtained, record the capture failure separately.
3. Apply the same preprocessing, trained model and threshold used for the other cases. The adapter classifies the captured face as spoof when p_spoof is at or above the threshold; the application displays the result and applies the attendance rule.
4. Record the true presentation type, participant ID, devices, lighting, predicted status and whether attendance was allowed or blocked. Record model version, preprocessing version and threshold for the test session. Keep personal media local and obtain permission for photographs/videos used in attacks.

Each attempt uses one captured image. Scores from multiple frames are not averaged, and no separate threshold is fitted for an entire video clip. The video-replay case has a moving video as the attack presentation, but the classifier still evaluates one camera image per attempt.

Select the checkpoint and threshold using validation data, then keep them fixed for final dataset and webcam tests. If findings lead to changes, record the changed settings and distinguish development attempts from the subsequent final assessment.

Report correct/total predictions separately for live, print, screen-photo and video-replay attempts. Also report capture failures and whether the application saved or blocked attendance. Identify conditions associated with errors. Use the same attendance-blocking behavior in the demonstration.

## Independent review — LIV-01.3–LIV-01.5

Start with this protocol and the [JSON specification](../configs/liveness-protocol.json). Review the following:

- **Source and licensing:** the selected source supports the proposed labels and its terms permit the planned research use.
- **Label mapping:** selected source codes map consistently to live, print and screen; masks are excluded and unknown codes are rejected. Screen photographs and video replays remain distinct webcam test cases.
- **People and related images:** check that the split rules reject a person or identical file shared across partitions, permit separate people with unrelated images, and keep generated frames/crops in the split of their original media. LIV-01 reviews these rules; LIV-03 executes the checks on the downloaded data.
- **Threshold selection:** validation selects the checkpoint and threshold; final dataset and webcam tests do not influence selection.
- **Input failures:** missing or invalid faces produce no attendance decision.
- **Reported results:** verify that the publisher’s dataset size is attributed to the paper, the team’s subset counts come from its inventory, and webcam outcomes are reported only after execution.

Record the reviewed version, review date, findings, agreed corrections and review outcome in the LIV-01 review evidence on the WBS Google Sheet. LIV-01.4 covers corrections and reviewer rechecks; LIV-01.5 covers the final protocol walkthrough. Acquired-data inventory and leakage results are assessed during LIV-03.

Independent review and reviewer rechecks are pending. The [WBS](https://docs.google.com/spreadsheets/d/1ItpF11JBlpcTnNiADI-Qm-cDmexBuLRzuHBW5uDChLw/edit?usp=sharing) records task acceptance.
