# Dataset storage

Raw data, caches, weights, credentials and personal recordings stay outside Git. Use the [release inventory](../configs/liveness-release.json) and [artifact index](../evidence/liveness/submission-artifacts.json) for supplied liveness assets, destinations and checksums. Paths below are relative to the repository root; storage can be on another mounted volume.

## Face-recognition storage

```text
data/.kagglehub/competitions/11-785-fall-20-homework-2-part-2/
├── classification_data/{train_data,val_data,test_data}/<identity>/
├── verification_data/
└── verification_pairs_val.txt
```

- **Use the downloaded default:** Keep paths.face_data in [project.toml](../configs/project.toml)
- **Read an existing extracted dataset:** Pass `--data-root` pointing at the root shown above
- **Select download/cache storage:** Pass `--cache-dir`; `--download` uses KaggleHub's returned dataset location
- **Save another configuration:** Pass `--config`; relative values resolve from the repository root

The loader does not move or rename existing data. `--data-root` selects local files and cannot be combined with `--download`. For Windows TOML values, use forward slashes, for example `face_data = "C:/Datasets/face-data"`. The [data-loading guide](../docs/data-loading.md) owns download/token instructions, commands, API contracts and observed counts.

## Anti-spoofing data (LIV-03)

[Current frozen-selection audit](../docs/liveness-data-preparation-and-audit.md#audit-the-current-frozen-selection) verifies the post-review data. [Historical reproduction](../docs/liveness-data-preparation-and-audit.md#reproduce-the-results) reconstructs earlier extraction/split/quality experiments; it is not a prerequisite for loading the supplied current selection.

| Input | Location under this data directory | Required for |
| --- | --- | --- |
| Extracted CelebA-Spoof | **raw/liveness/celeba-spoof-full** | Full source audit or input/cache generation |
| Prepared arrays and receipts | **liveness/face-cache/faces**, by train/validation/test | Colab training or public-test inference |
| Geometric targets and receipt | **liveness/face-cache/geometry** | Training with geometric supervision |
| Lossless transfer shards | **liveness/face-cache/transport** | Optional face-array materialization |
| Current recordings | **liveness/camera-trials** or a caller-selected folder | Camera inference only |

The extracted root directly contains Data, metas, the original README and image box files. Preparation adds validation-subjects.json, explicit annotations and data descriptions. The loader reads extracted files; it does not stream source images from archive parts. For a fresh archive extraction, allow 160 GiB when archive and output share a volume, or 85 GiB for extraction when archives are stored elsewhere. The [storage plan](../docs/liveness-data-preparation-and-audit.md#storage-plan) explains the reserve.

Training uses the persistent arrays and exact frozen row indices; it does not rerun historical candidate-policy selection. A local inference-only checkout needs the selected checkpoint and optional detector, not these training caches. Shared Drive inputs remain readable to Commenters; new Colab outputs use their private MyDrive. See the [training guide](../docs/liveness-training-and-evaluation.md#execution-and-resume).

## Liveness working directory

```text
facial-attendance/
├── data/
│   ├── raw/liveness/celeba-spoof-full/       # Source images and split metadata
│   └── liveness/
│       ├── camera-trials/                   # Flat MP4s; optional metadata/ subfolder
│       ├── face-cache/{faces,geometry,transport}/
│       └── source-crop-corrections/          # Bound correction inputs
├── artifacts/{pretrained,detectors,checkpoints}/
└── evidence/liveness/                       # Selected reports and new run outputs
```

Camera inference reads MP4s directly from the input folder and ignores subfolders. The capture CLI writes operator sidecars to its metadata subfolder by default. The source ZIP requires no team recording; use your own inputs with the [filename/command instructions](../docs/liveness-training-and-evaluation.md#portable-inference-and-saved-inspection). Original recording paths and historical metadata remain attributed to their original execution records, not required sidecars for new inference.
