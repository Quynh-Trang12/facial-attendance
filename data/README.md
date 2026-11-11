# Dataset storage

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

## Choose the storage option

| Situation | Setting |
| --- | --- |
| Use the default downloaded dataset | Keep `configs/project.toml` unchanged |
| Use a dataset extracted elsewhere | Follow [Use an existing dataset](#use-an-existing-dataset) below |
| Change the KaggleHub download/cache location | Pass `--cache-dir` |
| Use a separate configuration file | Pass `--config` |

The configurable dataset root supports existing folder locations without renaming or copying the dataset. `--data-root` selects existing extracted files; `--cache-dir` controls download/cache storage.

## Use an existing dataset

The commands below use the environment selected in [setup](../docs/setup.md). Without activation, replace `python` with `.venv/bin/python` on Linux/macOS or `.\.venv\Scripts\python.exe` in Windows PowerShell (or the executable of an environment stored elsewhere).

Choose the folder containing `classification_data/`, `verification_data/` and `verification_pairs_val.txt`, rather than an individual train/validation folder.

**For one run:** pass that folder to `--data-root`. For example, if your dataset is at `C:/Datasets/face-data`:

```bash
python scripts/check_face_data.py --data-root "C:/Datasets/face-data" --output evidence/runs/FAC-01/external-review.json
```

Replace the example path with your actual dataset folder. Use a new output filename for each run.

<details>
<summary>Optional: remember this dataset location for future runs</summary>

1. Open [`configs/project.toml`](../configs/project.toml).
2. Under `[paths]`, replace the existing `face_data = ...` line with your dataset folder. For the example above:

```toml
face_data = "C:/Datasets/face-data"
```

3. Save the file. Run the check without `--download` or `--data-root`:

```bash
python scripts/check_face_data.py --output evidence/runs/FAC-01/configured-review.json
```

`face_data` is a configuration key: it tells the loader where to read existing images. It does not create, move or download the dataset. Absolute paths point to a location on your machine; relative paths are resolved from the repository root. Use forward slashes in Windows TOML paths, as shown above.

This setting affects local mode. `--download` uses KaggleHub's returned download/cache location instead. A command-line `--data-root` overrides the saved location for that run.

</details>

Use [the data-loading guide](../docs/data-loading.md) for actual counts, API/local commands, dataset interfaces, sampling rules and limits. The assigned source is the [11-785 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). Decoding and DataLoader collation work on CPU; the training caller later chooses where to move tensors.

`data/raw/liveness/` and `data/raw/emotion/` are reserved config paths, not already downloaded datasets. Their source/license, labels and split protocols remain separate tasks. Credentials, raw face images and personal attendance records are local data rather than repository deliverables.

## Local files, API and working directory

- Local mode (no `--download`) reads existing files from `face_data` or `--data-root` and does not invoke the Kaggle download API. Those files can be in the Kaggle cache, a manually extracted folder or a mounted disk.
- API mode (`--download`) invokes KaggleHub. It reuses a valid cached copy instead of downloading the same archive every time. Cache is storage, not a third loading mode.
- The earlier `/tmp` test changed only the terminal working directory. The dataset remained under this repository's `data/` tree. `/tmp` is not a default dataset location.
