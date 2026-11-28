# Emotion data loader

The loader reads the prepared three-class dataset and its manifest. It does not create new
splits or change labels. Configuration is in `configs/emotion.toml`.

| Item | Baseline |
|---|---|
| Model IDs | Negative=0, Neutral=1, Positive=2 |
| Dataset | `data/processed/emotion_3class/` |
| Image tensor | Float32, `[1, 48, 48]`, grayscale |
| Batch | Float32 images `[B, 1, 48, 48]`; Int64 targets `[B]` |
| Scaling | Divide pixel values by 255; range `[0,1]` |
| Augmentation | None, on any partition |
| Training order | Seeded shuffle, seed 42 |
| Validation/test order | Fixed manifest order; no shuffle |
| Batch size / workers | 32 / 0 by default; suitable for Windows CPU checks |

This is the implemented baseline preprocessing for training. The GUI owner still needs to
confirm face-crop handling and use the same transform when the trained adapter is integrated.
Do not add dataset mean/std normalization or augmentation only at inference.

## Run

Use the project Python 3.13 environment and dependencies. From the repository root:

```powershell
.\.venv\Scripts\python.exe scripts/check_emotion_data.py
```

On other systems, use the corresponding environment interpreter. `--config` selects another
emotion config, `--data-root` selects existing processed data, and `--output` selects a new
evidence filename. Relative arguments resolve from the repository root, even if the command
is launched from another working directory. No dataset download is performed.

The command checks real batches and one example of each target from every partition,
recording shapes, values, labels, paths, counts, hashes and package versions under
`reports/emotion/loader/`. It also checks deterministic preprocessing and reproducible
training order. Existing evidence files are never overwritten.

For a new environment, follow `docs/setup.md` and the selected CPU/CUDA profile. A notebook
kernel also needs `ipykernel`; this is notebook tooling, not part of the model itself.
Use `notebooks/emotion_data_loader.ipynb` for visible batches and image/target examples.
Face previews are local; clear those outputs before publishing the notebook.

## Use from training code

```python
from facial_attendance.emotion.data_loader import EmotionLayout, load_settings, make_loaders

settings = load_settings(repo_root / 'configs/emotion.toml', root=repo_root)
layout = EmotionLayout(settings.data_root)
loaders = make_loaders(layout, settings)
images, targets = next(iter(loaders['train']))
```

Decoding and collation happen on CPU. The training caller chooses when to move images and
targets to a GPU. `EmotionDataset` also returns PIL images and integer targets if no tensor
transform is supplied, matching the project's map-style dataset convention.

## Validation and limits

The loader checks the explicit mapping, manifest checksum, class counts, aggregation and
one-to-one path/content hashes. It rejects missing/extra files, invalid paths and duplicate
pixel hashes in the prepared manifest. Every loaded image is checked against its recorded
file hash before decoding.

The preparation stage audited pixels across the full dataset. The loader check decodes
sample batches and class examples; it does not repeat a full image audit or prove that
every annotation is correct. It does not measure model accuracy or test the GUI. Near
duplicates and identity overlap are outside this check. Independent review remains separate.
