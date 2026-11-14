# Dataset storage

Choose the dataset task:

- **FAC-01 face recognition:** [storage options](#face-recognition-storage).
- **LIV-03 anti-spoofing:** [CelebA-Spoof storage and reproduction](#anti-spoofing-data-liv-03).

## Face-recognition storage

The following download/cache flags apply to FAC-01.

The configured default matches the dataset actually downloaded through KaggleHub:

```text
data/.kagglehub/competitions/11-785-fall-20-homework-2-part-2/
├── classification_data/
│   ├── train_data/<identity>/
│   ├── val_data/<identity>/
│   └── test_data/<identity>/
├── verification_data/
└── verification_pairs_val.txt
```

`paths.face_data` and `paths.kaggle_cache` in `configs/project.toml` describe these locations relative to the repository root. The CLI reads the same config in local/API mode, so no machine-specific absolute path is embedded in code. The downloaded data stays ignored by version control.

### Choose the storage option

| Situation | Setting |
| --- | --- |
| Use the default downloaded dataset | Keep `configs/project.toml` unchanged |
| Use a dataset extracted elsewhere | Follow [Use an existing dataset](#use-an-existing-dataset) below |
| Change the KaggleHub download/cache location | Pass `--cache-dir` |
| Use a separate configuration file | Pass `--config` |

The configurable dataset root supports existing folder locations without renaming or copying the dataset. `--data-root` selects existing extracted files; `--cache-dir` controls download/cache storage.

### Use an existing dataset

The commands below use the environment selected in [setup](../docs/setup.md). Without activation, replace `python` with `.venv/bin/python` on Linux/macOS or `.\.venv\Scripts\python.exe` in Windows PowerShell (or the executable of an environment stored elsewhere).

Choose the folder containing `classification_data/`, `verification_data/` and `verification_pairs_val.txt`, rather than an individual train/validation folder.

**For one run:** pass that folder to `--data-root`. For example, if your dataset is at `C:/Datasets/face-data`:

```bash
python scripts/check_face_data.py --data-root "C:/Datasets/face-data" --output evidence/runs/FAC-01/external-review.json
```

Replace the example path with your actual dataset folder. Use a new output filename for each run.

<details>
<summary>Optional: remember this dataset location for future runs</summary>

1. Open [configs/project.toml](../configs/project.toml).
2. Under `[paths]`, replace the existing `face_data = ...` line with your dataset folder. For the example above:

```toml
face_data = "C:/Datasets/face-data"
```

3. Save the file. Run the check without `--download` or `--data-root`:

```bash
python scripts/check_face_data.py --output evidence/runs/FAC-01/configured-review.json
```

**`face_data`** is a configuration key: it tells the loader where to read existing images. It does not create, move or download the dataset. **Absolute paths** point to a location on your machine; **relative paths** are resolved from the repository root. Use forward slashes in Windows TOML paths, as shown above.

This setting affects local mode. `--download` uses KaggleHub's returned download/cache location instead. A command-line `--data-root` overrides the saved location for that run.

</details>

Use [the data-loading guide](../docs/data-loading.md) for actual counts, API/local commands, dataset interfaces, sampling rules and limits. The assigned source is the [11-785 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). Decoding and DataLoader collation work on CPU; the training caller later chooses where to move tensors.

The anti-spoofing dataset has its own [storage and reproduction instructions](#anti-spoofing-data-liv-03). Emotion storage remains reserved in the project configuration. Keep credentials, raw face images, per-image inventories and personal attendance records out of Git.

### Local files, API and working directory

- **Local mode** (no `--download`) reads existing files from `face_data` or `--data-root` and does not invoke the Kaggle download API. Those files can be in the Kaggle cache, a manually extracted folder or a mounted disk.
- **API mode** (`--download`) invokes KaggleHub. It reuses a valid cached copy instead of downloading the same archive every time. **Cache** is storage, not a third loading mode.
- The earlier `/tmp` test changed only the terminal working directory. The dataset remained under this repository's `data/` tree. `/tmp` is not a default dataset location.

## Anti-spoofing data (LIV-03)

Use the [anti-spoofing data guide](../docs/liveness-loader.md#reproduce-the-results) for the complete sequence. The repository supplies code and documentation, not the original CelebA-Spoof images or owner-local preparation files. Download the 74 parts from the author's source and generate project metadata with the preparation command.

| Item | Default location, relative to the repository root | Purpose |
| --- | --- | --- |
| Original archive parts | A **CelebA-Spoof-zips** folder beside **facial-attendance** | All 74 exact filenames are needed together for extraction |
| Extracted dataset | The **celeba-spoof-full** directory under **data/raw/liveness** | Must directly contain **Data**, **metas**, original **README** and image box files |
| Generated preparation files | Inside the extracted root | **validation-subjects.json**, project annotation JSON, **data-description.json**, then reviewed annotations/findings |
| Reviewer outputs | A new **reviewer-01** directory following the [review-output layout](../docs/liveness-loader.md#inspect-and-compare-the-results) | Separate initial/final reports, inventories, summaries and sample batches |

The extracted root and archive directory can be on different mounted drives. The [path initialization blocks](../docs/liveness-loader.md#set-review-paths) give Linux/macOS and Windows examples; quoted variables keep spaces in paths intact. Pass the selected root explicitly through `--data-root` and point `--data-description` at its corresponding JSON. The repository default in [configs/project.toml](../configs/project.toml) does not need to change.

Plan about **160 GiB before download** if compressed and extracted data share a drive, or **85 GiB free for extraction** when the 71.71 GiB archive is elsewhere. The [storage calculation](../docs/liveness-loader.md#storage-plan) explains the measured sizes, allocation estimate and filesystem limits. A full audit adds about 125 MiB. Full verification covers every source image; fixture tests have a separate software-only scope.

Follow [fresh reproduction](../docs/liveness-loader.md#fresh-reproduction) to generate metadata from the original annotations, investigate the initial duplicate findings and produce the reviewed data description. With a complete prepared root, go directly to [the final audit and loader check](../docs/liveness-loader.md#5-final-audit-and-loader-check), then prepare an [experimental-readiness receipt](../docs/liveness-loader.md#experimental-readiness-and-candidate-handling) for training. The loader requires extracted files; it does not stream images from ZIP parts or Google Drive links. Preparation and extraction refuse existing outputs, so select new destinations for a fresh reproduction rather than mixing old and new files.

The [anti-spoofing Colab workflow](../docs/liveness-loader.md#google-colab-execution) was executed on a standard CPU VM with 107.72 GiB total disk. It reads archive parts from Drive through canonical filename aliases and extracts images onto the VM, avoiding a full local archive copy. Results persist beneath the existing project folder in Drive, following the [Colab output layout](../docs/liveness-loader.md#6-run-the-final-audit-and-inspect-the-batches). The guide checks free space and explains cache overhead, runtime limits and recovery.

The historical Colab baseline uses the notebook from the [shared Drive project](https://drive.google.com/drive/folders/1wp-WH2Rgaow7_gi9OMOt3xmXD1-rd1zB?usp=sharing) after preparing the [two folder shortcuts](../docs/liveness-loader.md#google-colab-execution). It handles baseline setup, extraction, preparation and batch checks. Experimental readiness requires current source files and the additional [Colab experimental commands](../docs/liveness-loader.md#experimental-readiness-and-candidate-handling); that new route has not been executed on Colab.
