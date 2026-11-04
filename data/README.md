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

If your files are elsewhere (for example `/content/drive/MyDrive/face-data` on Colab), pass `--data-root` or edit `face_data`; do not rename or copy the dataset just to fit this tree. `--cache-dir` overrides only where KaggleHub downloads/cache data, while `--data-root` selects existing extracted files. A custom config can be supplied with `--config`.

Use [the data-loading guide](../docs/data-loading.md) for actual counts, API/local commands, dataset interfaces, sampling rules and limits. The assigned source is the [11-785 competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). Decoding and DataLoader collation work on CPU; the training caller later chooses where to move tensors.

`data/raw/liveness/` and `data/raw/emotion/` are reserved config paths, not already downloaded datasets. Their source/license, labels and split protocols remain separate tasks. Never commit credentials, raw face images or personal attendance records.

## Local files, API and working directory

- Local mode (no `--download`) reads existing files from `face_data` or `--data-root` and does not invoke the Kaggle download API. Those files can be in the Kaggle cache, a manually extracted folder or a mounted disk.
- API mode (`--download`) invokes KaggleHub. It reuses a valid cached copy instead of downloading the same archive every time. Cache is storage, not a third loading mode.
- The earlier `/tmp` test changed only the terminal working directory. The dataset remained under this repository's `data/` tree. `/tmp` is not a default dataset location.
