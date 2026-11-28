# Revised FER2013 dataset

The original dataset is read from `paths.emotion_data` in `configs/project.toml`.
The revised dataset is written to `data/processed/emotion/`. Raw files stay unchanged.

## Reproduce the dataset

Use the project's Python 3.13 environment and dependencies. From the repository root:

```powershell
python scripts/prepare_emotion_data.py
```

On Windows, use `py -3.13` or the configured environment's interpreter if `python` points elsewhere.
The command refuses to overwrite an existing dataset. To reproduce into a different directory:

```powershell
python scripts/prepare_emotion_data.py --output data/processed/emotion-rebuild
```

From another working directory, invoke the script by its full path. Defaults and relative CLI
paths resolve from the repository root. `--data-root`, `--config`, and `--decisions` select
other local inputs. No Kaggle API call is made. Input images must be 48x48 grayscale.

## Manual review is part of the input

`reports/emotion/review_decisions.csv` records the owner's choices for 57 conflicting-label
groups. Six groups are excluded; 51 have a selected label. Each decision is keyed by a
SHA-256 hash of the decoded pixels plus image mode and size, using the EDA notebook's
definition. Display group numbers are only for convenience.

The review itself is subjective. Reapplying the saved choices to the same images is
reproducible. Another reviewer may disagree; any changed decisions should be saved in a
new CSV and used to create a new dataset version. This is not independent reviewer approval.

## Preparation rules

1. Decode and hash every source image. Require a recorded decision for every conflicting
   group, and reject stale decisions that do not match a current conflict group.
2. Exclude every member of the six manually excluded groups.
3. Apply the chosen label to every remaining member of each reviewed group, including test images.
4. Keep one image per identical decoded-pixel group. Prefer an existing test image when
   the group spans train and test; exclude its training copies. Otherwise choose the
   lexicographically first source path. Deduplication also removes repeated test copies.
5. Split remaining training images by class into approximately 80% train and 20% val.
   Rank each class by SHA256 of `seed|label|pixel_hash`, with seed 42; round each class's
   validation count to the nearest integer (halves rounded up). Keep every class in both subsets.
6. Copy images without re-encoding and verify every copy against its source file hash.
   Output filenames use pixel hashes. Publish the directory only after the build succeeds.

This guarantees that identical decoded pixels do not cross output splits. It does not check
identity overlap or near duplicates. The original folder-label mapping must still be overridden
when a training loader uses alphabetical order: model IDs are Angry=0, Disgust=1, Fear=2,
Happy=3, Sad=4, Surprise=5, Neutral=6.

## Outputs and evaluation

- `train/`, `val/`, `test/`: the revised image folders, each containing seven class folders.
- `manifest.csv`: one row per original image, including original label/split, selected label,
  output location or exclusion reason, hashes, and the review decision.
- `review_decisions.csv`: a copy of the exact review input.
- `label_mapping.json`: explicit model IDs.
- `summary.json`: measured counts, test-label changes, input/code hashes and build settings.

The test set has changed labels and fewer images. Call it a **revised FER2013 test set**.
Its scores are not directly comparable to the original benchmark. Keep model selection on
validation data. Report the actual number of retained test images whose labels changed,
as well as exclusions and class counts. This record describes dataset preparation, not model
accuracy or independent review.

Image data stays ignored by Git. The small decision CSV and summary can be versioned;
do not commit raw faces or notebook outputs containing face images. No Git publication is
performed by the preparation command.
