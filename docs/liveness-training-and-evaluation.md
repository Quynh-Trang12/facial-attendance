# Anti-spoofing training and evaluation

Document version: **LIV-04-v17**, revised **28 November 2026**. Task owner: **Trang**. Independent reviewer: **Vỹ**. LIV-04 is defined in the [Project Tracker](../README.md#project-tracker), where its updates and records are maintained.

LIV-04 compares four ResNet-18 liveness configurations and supplies a selected checkpoint, saved results and runnable inference/reproduction routes. Inputs are CelebA-Spoof images, fixed splits and bound caches. The model predates crop review. Native revised Colab/Windows/macOS execution and full attendance integration remain unverified.

## Results and limits

The selected aenet-full checkpoint is from epoch 18. Its retained-validation ACER is 0.1674%, but public-test ACER is 6.176%: it accepts 12.222% of spoof images and rejects 0.131% of live images. The 25 recorded camera trials were correctly classified, but this bounded sample does not establish population accuracy or capture-session independence.

| Configuration | Selected epoch | Epoch budget | Original validation ACER | Retained validation ACER |
| --- | ---: | ---: | ---: | ---: |
| binary-classification | 13 | 15 | 0.2829% | 0.2739% |
| binary-geometric | 11 | 15 | 0.2662% | 0.2618% |
| binary-semantic | 21 | 25 | 0.1962% | 0.1933% |
| aenet-full | 18 | 25 | 0.1734% | 0.1674% |

All four configurations first trained for 15 epochs; binary-semantic and aenet-full then continued to 25. The final table therefore compares unequal training budgets. Validation selects weights and threshold; public test has prior experimental exposure and is not an untouched independent benchmark.

## After crop review

[Crop review](liveness-data-preparation-and-audit.md#source-crop-review-and-filtered-cache-view) excluded 243 train, 25 validation and 41 test images. Current retained counts are 444,114, 49,479 and 67,129 respectively. The checkpoints retain their pre-review training exposure. Recalculation removes the same confirmed defects from saved predictions for every model, without changing weights or thresholds; lower filtered error rates are not a model improvement.

The [artifact index](../evidence/liveness/submission-artifacts.json) links the model, selections, predictions and camera report, with download destinations and checksums. The [current test report](../evidence/liveness/crop-review/filtered-view-v4-20261126/filtered-test-evaluation.json) covers 67,129 retained rows; the [original report](../evidence/liveness/liveness-continuation/evaluation/aenet-full/6715d53461e6d123/public-test/evaluation.json) preserves its 67,170-row attribution. The camera report contains nine live, eight screen and eight print recordings. Personal recordings are not shipped in the source ZIP.

## Portable inference and saved inspection

Start from the repository root with the [CPU environment](setup.md#1-open-a-terminal-and-select-the-environment) active. Saved inspection needs no raw dataset, weights or training caches:

```sh
python scripts/run_liveness_notebook.py --action inspect
python scripts/run_liveness_notebook.py --notebook notebooks/liveness-data-preparation-and-audit.ipynb --action inspect
```

Both commands execute saved-evidence rendering and retain tables/figures in the notebook. They stop on a failed binding and preserve its output. The training notebook contains 24 figures and the data notebook three previews. Optional raw crop verification is controlled by VERIFY_SOURCE_CROPS in the training notebook and requires original images.

To try your own input:

1. Complete the OS-specific environment setup above.
2. Use the bundled 44.8 MB [checkpoint](../artifacts/models/liveness/liveness-inference.pt). The [artifact index](../evidence/liveness/submission-artifacts.json) supplies a download source if it is missing. Default inference verifies the bundled file checksum; an explicit run directory selects its archived validation winner.
3. For full images or recordings, verify the bundled face detector with the preparation CLI below; it downloads only when the asset is missing.
4. Use the appropriate image or folder command below. The checkpoint's validation threshold remains fixed.

```sh
python scripts/preparation/prepare_liveness_detector.py
python scripts/run_liveness_notebook.py --action inference --device cpu --input "my-photo.jpg"
python scripts/run_liveness_notebook.py --action inference --device cpu --input "face.png" --face-crop
python scripts/run_liveness_notebook.py --action inference --device cpu --input "my-clips" --cache "evidence/liveness/local-inference/my-clips"
```

The face-crop command expects an existing PIL-compatible RGB face image and does not require the detector. The full-image command detects and crops one face. An image/frame with zero or multiple faces returns uncertain and cannot permit attendance.

For MP4 inference, pass a folder containing files directly, not one MP4 path. Names follow Name-YYYYMMDD_HHMMSS-kind-lighting.mp4, for example Reviewer-20261128_120000-live-bright.mp4; kind is live, print or screen and lighting is bright or dim. These labels describe presentation conditions, not predicted outcomes. Invalid labels and duplicate IDs/content stop the batch. Image inference prints JSON; folder inference saves **latest.json** and per-recording outputs below the chosen cache directory.

<details>
<summary>Optional independent checkpoint checksum check</summary>

Run this Python code from the repository root after downloading the file:

```python
import hashlib
import json
from pathlib import Path

asset = json.loads(Path("evidence/liveness/submission-artifacts.json").read_text())[
    "selected_model"
]
checkpoint = Path(asset["local_path"])
assert checkpoint.stat().st_size == asset["bytes"]
with checkpoint.open("rb") as stream:
    assert hashlib.file_digest(stream, "sha256").hexdigest() == asset["sha256"]
print("Selected checkpoint verified")
```

</details>

Native checks cover Linux CPU inference and one selected-checkpoint CUDA face-crop inference run on an RTX 3050. Windows x64 and macOS Apple Silicon have CI routes; their current native runs remain unverified. These file/frame checks do not exercise the attendance application’s webcam flow. [Latest verification](../evidence/liveness/package-refactor/verification.json) identifies the exact tested inputs and revision fingerprints.

## Model and training procedure

AENet follows Zhang et al., [CelebA-Spoof](https://arxiv.org/abs/2007.12342v3), with a fine-tuned ImageNet-initialized ResNet-18 backbone. Binary-classification learns live/spoof only; binary-semantic adds 40 face attributes, source/spoof type and illumination; binary-geometric adds pseudo-depth/reflection; aenet-full combines both groups. Only the binary head is needed for deployment.

- **Images and labels:** RGB source-client crops, bilinear 224 × 224 resize, values in [0,1], no channel standardization; live 0, spoof 1
- **Original training rows:** 444,357
- **Project validation rows:** 49,504, derived from original training data
- **Original test rows:** 67,170
- **Budget:** Four 15-epoch runs; binary-semantic and aenet-full continue from epoch 15 to 25
- **Optimizer:** SGD; learning rate 0.005, momentum 0.9, weight decay 0.0001
- **Matched conditions:** Seed 42, batch 256 from epoch one, FP32, GPU augmentation and identical input and model-selection rules
- **Augmentation:** Training-only horizontal flip; ColorJitter brightness 0.3, contrast 0.3, saturation 0.2, hue 0.05
- **Loader workers:** Default 12 capped by available CPUs; the run configuration records the actual count
- **Auxiliary weights:** Face attributes 1; spoof type 0.1; illumination 0.01; depth and reflection MSE 0.1 each

PRNet supplies relative depth for live images; the reflection teacher supplies three RGB reflection grids for spoof images. Spoof depth and live reflection targets are zero under this training convention. Each original training row has four 14 × 14 grids. These pseudo-targets are model estimates, not measured physical depth/reflection. Reviewed images and maps use identical original row indices.

At each epoch, validation chooses the threshold where APCER and BPCER are closest (the EER operating point). Threshold ties prefer lower ACER, then lower threshold. Checkpoint selection minimizes validation ACER, then binary loss, then epoch. APCER is spoof accepted as live divided by all spoof inputs; BPCER is live rejected as spoof divided by all live inputs; ACER averages them. Test/camera results do not select weights or threshold. Source-video independence remains unverified due to unavailable source-video IDs.

## Execution and resume

| Notebook ACTION | Work | Colab runtime |
| --- | --- | --- |
| inspect | Render saved results | CPU |
| prepare | Stage and checksum existing caches | CPU |
| train | Verify inputs and start/resume the declared experiment | GPU |
| inference | Score camera recordings | GPU; local CLI also supports CPU |
| evaluate | Score public-test images with the selected model | GPU |

The default is inspect. TARGET_EPOCHS = 25 means stop at epoch 25, not train 25 additional epochs. Shared inputs remain readable to Commenters; new results go to their private MyDrive/facial-attendance-outputs, separated by project ID. Existing private runs retain source/data/optimizer/RNG/cursor checks. The notebook rejects historical output names for a new filtered-data training run; choose a new TRAINING_OUTPUT_NAME when membership changes.

<details>
<summary>Google Colab execution — training and evaluation</summary>

Open the [training notebook](../notebooks/liveness-training-and-evaluation.ipynb), set PROJECT_FOLDER to the shared folder URL/ID and follow its setup cells. Mounting resolves the project by ID, including shared-folder shortcuts, before verifying source checksums.

For training, use prepare on CPU to verify persistent face arrays, row-correction pixels and geometric targets, then select a GPU runtime and train. Allow about 69 GiB for train/validation base arrays, plus correction pixels, the 1.3 GiB geometric array and the environment. Runtime storage is measured before staging; an existing SSD copy/partial is reused. Rclone transfers are checksum-verified, with mounted Drive as the documented fallback. RAM disk is selected only when capacity and training memory headroom are sufficient. A runtime switch discards VM RAM/SSD but preserves Drive inputs.

For evaluate, the notebook stages only the roughly 10.1 GB public-test cache, verifies its frozen input bindings and writes new reports/predictions to the selected output. It does not rebuild training targets. Completed results are reused only with matching checkpoint/input signatures. Inspect later on CPU to read the saved results. The revised native Colab route remains unexecuted.

</details>

<details>
<summary>Local execution — inspection, inference and advanced CUDA entry points</summary>

Use the portable commands above for normal local inspection/inference. The [comparison runner](../scripts/run_liveness_comparison.py) is POSIX orchestration, and the [training CLI](../scripts/training/train_liveness_aenet.py) requires CUDA. Existing compatible CUDA 12.6 builds are retained; Colab uses CUDA 13.0 with the same public dependency versions. An unavailable requested CUDA device fails without CPU fallback. CPU/CUDA prediction caches are separate.

For an already prepared Linux GPU environment, [public-test orchestration](../scripts/run_liveness_public_test.py) accepts a separate runtime directory and writable result location:

```sh
python scripts/run_liveness_public_test.py evaluate --source-root . --project-root /path/with-capacity/liveness-evaluation --run-root evidence/liveness/liveness-continuation --output-root /path/to/writable/marker-results
```

Inputs must include the published test cache and frozen training outputs. This command performs evaluation, not training. Use each entry point's `--help` for overrides; relative dataset/camera settings resolve from the project root through [project.toml](../configs/project.toml).

</details>

Each complete epoch archive contains model heads, optimizer/RNG state and the completed-update cursor. **last-training.pt** resumes optimization; **liveness-inference.pt** contains binary deployment weights and the selected threshold. Source/data/recipe changes outside explicit compatibility declarations stop resume. [export_liveness_checkpoint.py](../scripts/release/export_liveness_checkpoint.py) exports a verified archived epoch without fitting a new threshold.

## Recalculate saved results

Use the existing environment with the source path configured as in [setup](setup.md#run-unit-tests). These commands validate original predictions and retained indices, then recalculate metrics without model inference. Choose new output names:

```sh
python scripts/evaluation/evaluate_liveness_predictions.py --split validation --project . --cache-view evidence/liveness/crop-review/filtered-view-v4-20261126 --output evidence/liveness/validation-recalculated.json
python scripts/evaluation/evaluate_liveness_predictions.py --split test --project . --cache-view evidence/liveness/crop-review/filtered-view-v4-20261126 --output evidence/liveness/test-recalculated.json
```

<a id="evaluation-without-training"></a>
<a id="final-evidence-and-model-distribution"></a>
<a id="cuda-inference-on-local-and-colab-runtimes"></a>

## Evidence and integration

The [artifact index](../evidence/liveness/submission-artifacts.json) is the entry point for model/cache downloads, checksums, historical selections, score receipts and current validation/test/camera outputs. The source ZIP includes code, notebooks and selected evidence; large weights, datasets, caches and personal recordings remain separate assets.

The [adapter](../src/facial_attendance/liveness/inference/liveness_adapter.py) accepts a PIL RGB face crop or a uint8 HWC BGR frame. It returns live/spoof/uncertain, spoof score, frozen threshold, model identity and `allow_attendance`. Full-frame boxes are source-pixel x, y, width, height. A tracked stream can require five consecutive live observations spanning at least one second with no gap over 0.5 seconds. This temporal gate is distinct from a clip's median sampled-frame score. Identity verification, registration and recording attendance remain the application's responsibility.

The [capture CLI](../scripts/capture_liveness_trial.py) writes flat MP4s and metadata under the selected input folder unless overridden. Capture timestamps identify recordings, not independent sessions. Model comparisons and physical trials retain their producing source/input/device attribution. [Architecture and code reading](architecture.md#interview-reading-route) identifies the implementation boundaries.

## Notebook visual analysis

Sections 14–16 place learning curves, validation/test metrics and paired error galleries beside their interpretation. The [selected-example inventory](../evidence/liveness/visual-analysis-filtered/selected-examples-input.json) and [saved report](../evidence/liveness/visual-analysis-filtered/selected-examples.json) bind nine retained validation examples to model and teacher outputs. Inspection reads embedded crops/maps by default; raw source verification is optional.

The validation-error plot compares misclassified-score tails for all four models; the public-test plot separates 26 live errors and 5,774 spoof errors. Compare axis values, not panel heights. Auxiliary predictions are pseudo-label diagnostics, not physical ground truth or a new independent test. Original 26 November GPU replay checked 40 image/configuration decisions across ten examples, with maximum spoof-score difference 0.001280; the current report retains nine examples after crop review. A background-face crop of only 23 × 37 source pixels was excluded through that review rather than described as an ordinary model failure.

<details>
<summary>Optional regeneration of auxiliary reports and the public-test cache</summary>

To regenerate the selected examples in a prepared CUDA environment, supply the four verified training checkpoints and original source/teacher assets, choose a new output parent and run:

```sh
python scripts/audit/inspect_liveness_examples.py --project . --inventory evidence/liveness/visual-analysis-filtered/selected-examples-input.json --output /path/to/new-output/selected-examples.json
```

The optional `--checkpoint-root` points to a directory with binary-classification.pt, binary-semantic.pt, binary-geometric.pt and aenet-full.pt. On Colab, set RUN_AUXILIARY_ANALYSIS = True and run inference setup before section 16; the default False reads the saved report. New reports preserve their own runtime attribution and do not change model weights.

The test-cache builder streams one NPY to stdout and records original image/box hashes and sampled tensor parity:

```sh
python scripts/run_liveness_public_test.py build-test-cache --data-root data/raw/liveness/celeba-spoof-full --receipt /path/to/new-output/decoded-faces.json > /path/with-capacity/decoded-faces.npy
```

Allow roughly 10.1 GB for the array. Publish its receipt alongside the completed file; interrupted output is not a valid cache. Rows retain original annotation order, uint8 RGB pixels and the recorded crop recipe.

</details>
