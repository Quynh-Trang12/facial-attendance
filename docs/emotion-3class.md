# Three-class FER2013

Model targets: Negative=0, Neutral=1, Positive=2.

| Target | Source emotions |
|---|---|
| Negative | Angry, Disgust, Fear, Sad |
| Neutral | Neutral |
| Positive | Happy |

Surprise is excluded. Source annotations remain in the manifest; training targets are aggregated.
Use `data/processed/emotion_3class/train`, `val` and `test`. Each contains three target folders.

## Reproduce

Use the project Python 3.13 environment. From the repository root:

```powershell
python scripts/prepare_emotion_sixclass.py --output data/processed/expression_selection
python scripts/prepare_emotion_3class.py --selection-root data/processed/expression_selection --output data/processed/emotion_3class_rebuild
```

The first command selects and deduplicates source images while preserving expression labels.
The second aggregates their training targets. Neither command trains a six-class model.
Both refuse to overwrite existing outputs. Use `py -3.13` if it selects your project interpreter.
The selection command reads `paths.emotion_data` from `configs/project.toml`; use `--data-root`
for another extracted location. Relative arguments resolve from the repository root.

If the verified selection already exists at `data/processed/emotion_6class/`, the aggregation
command can use its default selection path:

```powershell
python scripts/prepare_emotion_3class.py --output data/processed/emotion_3class_rebuild
```

## Preparation policy

Exclude Surprise first. Exclude identical-image groups with conflicting remaining source
labels, even if both labels would later map to Negative. This conservative selection rule
is applied before aggregation. Retain one copy of each other image, preferring an existing
test image over training copies. Validation is stratified by source emotion with seed 42
and approximately 20% of eligible training data. Preserve those partitions during aggregation.

No original expression annotation is manually changed. Images are copied without re-encoding;
all source and copied hashes are checked. Exact duplicate pixels cannot cross output splits.

`manifest.csv` records every source image, source emotion, target class, split and exclusion.
`emotion_mapping.json` and `label_mapping.json` define aggregation and model IDs. `summary.json`
records measured counts, settings, input/code hashes and limitations. The notebook at
`notebooks/emotion_eda.ipynb` contains source analysis and the final three-class distribution.

The groups are an experimental label definition, not a direct measurement of mood or mental
health. Original label errors may remain. No full visual, identity or near-duplicate review
is performed. Evaluate on these fixed partitions and report it as a three-class FER2013 task;
scores are not directly comparable to the original seven-class benchmark.
