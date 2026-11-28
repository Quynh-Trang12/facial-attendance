# Shared evaluation scripts

SYS-06.2 evaluates saved predictions or raw face embeddings. It does not train models or run image inference. The [evaluation protocol](evaluation-protocol.md) defines data roles, selection and limitations.

## Run

Use the project Python 3.13 environment from the repository root. In PowerShell:

```powershell
$python = '.venv/Scripts/python.exe'
& $python scripts/evaluate.py --help
$env:PYTHONPATH = 'src'
& $python -m unittest discover -s tests -p test_evaluation.py -v
```

All inputs below are deliberately synthetic. Their numbers check calculations and are not model performance. Each output directory must be new; the scripts refuse to overwrite one.

```powershell
& $python scripts/evaluate.py verification --trials tests/fixtures/evaluation/trials.txt --scores tests/fixtures/evaluation/classification_pairs.csv --approach classification --mode fixture --run-id demo-classification --checkpoint-id synthetic --output artifacts/evaluation/demo-classification
& $python scripts/evaluate.py verification --trials tests/fixtures/evaluation/trials.txt --scores tests/fixtures/evaluation/metric_pairs.csv --approach metric --mode fixture --run-id demo-metric --checkpoint-id synthetic --output artifacts/evaluation/demo-metric
& $python scripts/evaluate.py classification --predictions tests/fixtures/evaluation/emotion.csv --module emotion --split validation --run-id demo-emotion --checkpoint-id synthetic --output artifacts/evaluation/demo-emotion
& $python scripts/evaluate.py classification --predictions tests/fixtures/evaluation/face.csv --module face --label-map tests/fixtures/evaluation/face_labels.json --split validation --run-id demo-face --checkpoint-id synthetic --output artifacts/evaluation/demo-face
& $python scripts/evaluate.py liveness --predictions tests/fixtures/evaluation/liveness.csv --split webcam --policy-id synthetic-policy --run-id demo-liveness --checkpoint-id synthetic --output artifacts/evaluation/demo-liveness
```

Expected results:

| Synthetic example | Expected result |
| --- | --- |
| Classification-based pair scores | Cosine AUC=0.75; negative Euclidean AUC=0.75 |
| Metric-learning pair scores | Both AUCs=1.0 |
| Emotion predictions | Accuracy=2/3; macro F1=5/9; confusion matrix [[1,0,0],[1,0,0],[0,0,1]] |
| Face class predictions | Top-1 accuracy=2/3 |
| Liveness attempts | 6 total; 4 scored; spoof acceptance=1/3; live rejection=0; genuine-user blocking=2/3; 2 failures |

All 71 tests (63 existing published tests plus 8 evaluation tests) passed locally. Recorded executions are under [reports/evaluation/sys06-fixtures](../reports/evaluation/sys06-fixtures). Tests also cover tied AUC=0.5, inverted AUC=0, validation-only threshold selection, tied threshold boundaries, invalid inputs, trial order/count mismatch, raw embedding distances and output-overwrite refusal.

## Input contracts

CSV headers are required. Inputs must be nonempty, with complete rows. Labels must be integer IDs; scores must be finite. Per-image IDs must be unique. Keep detailed real predictions and embeddings under ignored artifacts/, not Git. Published examples and reports contain only synthetic identifiers.

### Face verification

- Trial text: one `left right label` row per pair, 1=same and 0=different. Score CSV columns: `left,right,label,cosine,euclidean_distance`. Paths, labels, row count and order must match the trial file exactly. Both classes must be present for ROC/AUC. Distances must be nonnegative; cosine must lie in [-1,1].
- Use `--mode official` for the required trial file: it enforces exactly 8,805 rows and permits no threshold selection. The caller must supply the official file; count validation cannot prove provenance. The input SHA-256 is saved so reviewers can check it.
- Use `--mode validation --select-threshold` only for the fixed validation pairs. This selects the lowest observed cosine threshold at FAR<=1%, handles tied scores together, and uses a finite reject-all threshold when necessary. It never tunes the distance threshold. Fixture mode deliberately cannot select a deployed threshold.
- Instead of `--scores`, use `--embeddings path/to/raw.npz`. The NPZ must contain string `paths[N]` and finite numeric `raw_embeddings[N,D]` with unique paths. Keys must match the trial file exactly, including relative prefixes. Pickle loading is disabled. Missing embeddings and zero-norm vectors fail. The script computes cosine and Euclidean distances on the same raw vectors, then negates distance for ROC.
- The exporter must supply embeddings before L2 normalization. The evaluator cannot recover raw norms from normalized embeddings or establish that a saved embedding came from a claimed checkpoint. Use the same official trials for both approaches; do not confuse synthetic score comparisons with measured model comparisons.

### Classification

CSV columns: `id,true,predicted`, using top-1 predicted class IDs from the caller. `--module emotion` fixes IDs 0=Negative, 1=Neutral, 2=Positive and includes absent classes in macro F1 with zero_division=0. For face classification, `--label-map` is the train-derived JSON object mapping integer IDs to identity names; unknown labels fail. Use `--split validation` or `--split test`. This evaluator does not choose emotion checkpoints or invent a classification head for metric learning.

### Liveness

CSV columns: `id,label,score,status,family`. Labels are live=0/spoof=1. Score is p_spoof in [0,1]. Status is live, spoof, uncertain or failure; only failure may have an empty score. Family is live for label 0; print, screen, screen_photo or video_replay for label 1.

Statuses must come from the actual versioned adapter policy, identified by `--policy-id`. The evaluator does not invent Trang's threshold or quality rules. ROC uses every available finite score, including scored uncertain/failure attempts; without both scored classes it reports null AUC with a reason. All attempts remain in operational denominators. True-label rows and status columns are explicitly ordered in the outcome table. Missing class-specific denominators produce null rates, not zero. Report uncertainty and failures separately; a blocked attack is not necessarily a correct spoof classification.

## Outputs and traceability

Each invocation writes metrics.json, saved numeric predictions/scores and, where defined, ROC points as CSV plus roc.png. The initial ROC threshold is positive infinity; JSON records it as null to remain valid JSON. The corresponding point is (0,0), meaning no samples accepted.

metrics.json records run/checkpoint/policy identifiers, input hashes, code commit, source hashes, dirty state, versions, metric counts and class/score direction. Record the actual preprocessing, dataset/split/exclusion manifests and checkpoint hash in the training/inference run manifest; link it through the run/checkpoint IDs. This command cannot verify image decoding, dataset split membership, semantic label correctness or model provenance from numeric predictions alone.

Real inference exports, full official-trial runs and independent reviewer execution remain separate integration/review work. The current evidence verifies metric calculations using synthetic inputs only.
