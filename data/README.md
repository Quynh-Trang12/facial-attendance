# Local datasets

Download the assigned [face verification competition data](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation) using your own Kaggle account. Accept any competition access requirements yourself; never commit credentials or raw face images.

Place the extracted data under `raw/face/` (or configure another location in `configs/project.toml`). The course brief expects:

```text
raw/face/
├── classification_data/
│   ├── train_data/<identity>/
│   ├── val_data/<identity>/
│   └── test_data/<identity>/
├── verification_data/
└── verification_pairs_val.txt
```

The loader now validates this layout and reads classification images and verification pairs. See [data loading](../docs/data-loading.md) for API/local commands, interfaces, sampling and evidence. Authenticated acquisition has succeeded, and the actual extracted competition layout matches the loader. The recorded checks cover split metadata, all listed verification paths and sample decoding; they are not a full duplicate/leakage audit. Both embedding approaches train on the specified training split. Each verification trial contains two image paths and a same/different label. The brief requires ROC/AUC on `verification_pairs_val.txt`.

`raw/liveness/` and `raw/emotion/` are reserved paths. Select and document their sources, labels, licenses and splits in the corresponding data tasks. Do not substitute another face-recognition dataset for the assigned one. Data loaders should live beside their model modules when implemented; notebooks must not become the application runtime.
