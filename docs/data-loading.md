# Face data loading

FAC-01.2 provides classification/pair datasets and baseline training-pair sampling for competition `11-785-fall-20-homework-2-part-2`. No other face dataset is substituted.

## Acquisition

The existing environment now uses **kagglehub 1.0.2**, pinned in the project after an explicitly authorized upgrade. It supports modern Kaggle API tokens. `recognition/download.py` keeps the reusable cache under the supported `KAGGLEHUB_CACHE` location, restores the previous variable afterwards and inspects the returned path without copying another set of images.

At [Kaggle API settings](https://www.kaggle.com/settings/api), choose **Generate New Token**. Store only the token text in `~/.kaggle/access_token` outside the repository, or set `KAGGLE_API_TOKEN` in the process that will download. The account owner supplies the token; never copy it into this repository. `kagglehub.login()` masks token input but stores credentials only in its current process, so a one-off login command does not configure another later process. Do not put a standalone token in `kaggle.json`, which is the legacy username/key JSON format. Accept competition rules in your account if required. No credentials belong in source or reports.

From the repo root, using the existing Python 3.13 environment:

```bash
~/.local/share/ml-frameworks/.venv/bin/python scripts/check_face_data.py \
  --download --output reports/data/kaggle-review.json
```

This invokes `kagglehub.competition_download("11-785-fall-20-homework-2-part-2")`. Cache files stay under ignored `data/.kagglehub/`; cached data may be reused. For offline use, pass the extracted dataset root returned in the report:

```bash
~/.local/share/ml-frameworks/.venv/bin/python scripts/check_face_data.py \
  --data-root /path/to/extracted/dataset \
  --output reports/data/local-review.json
```

Without either input flag, the script reads `paths.face_data` in `configs/project.toml`. Use a new output filename per run. Layout discovery accepts the course layout or one enclosing directory. Further nested archives or a different layout raise an explicit error; the authenticated run has verified the actual competition layout described below.

## Loader and sampling contract

Implementation: `src/facial_attendance/recognition/data_loader.py`.

| Interface | Behavior |
| --- | --- |
| `FaceLayout.discover(root)` | Require three classification splits, verification directory and trials file at one unambiguous root |
| `ClassificationFaces(...)` | Return `(RGB image/tensor, integer target)`; sort training identities deterministically and reuse that mapping for validation/test; reject unknown identities |
| `VerificationPairs(...)` | Return `(left image/tensor, right image/tensor, label)` in official file order; preserve `1=same`, `0=different` |
| `sample_training_pairs(train, count_per_label=N, seed=42)` | Draw N positives and N negatives from train only, using a local seeded generator |
| `PairFaces(records, transform=...)` | Load sampled pair records through the same pair-image interface |
| `image_transform(size)` | Inspection-only resize and RGB tensor conversion to `[3,size,size]` in `[0,1]` |

These map-style datasets work with PyTorch `DataLoader`. Without a transform they return PIL RGB images. The inspection CLI explicitly uses a configurable size, default 112, which does not settle final preprocessing under SYS-02.

For positives, choose uniformly among identities with at least two images, then choose two distinct images. For negatives, choose two different identities uniformly, then one image per identity. Sampling is with replacement across pairs; repeated pairs are possible. Singleton identities can supply negatives. This is a reproducible baseline, not exhaustive sampling or hard-negative mining. Official verification trials are never resampled.

Verification paths may be bare filenames or prefixed with `verification_data/`. Nonblank/noncomment lines require exactly two paths and label 0/1. Missing files, malformed labels/rows, escaping paths, unknown classification IDs and unreadable sampled images fail explicitly. Full duplicate/leakage audits and model evaluation remain separate tasks.

## Observed evidence and acceptance

- [Test log](../reports/data/loader-tests-kagglehub-1.0.2.log): **26 tests PASS**, including 19 loader/acquisition fixture tests and seven existing configuration tests. Mocked API tests establish the call contract, not real authentication.
- [CLI fixture result](../reports/data/fixture-kagglehub-1.0.2.json): **PASS** on temporary generated PNGs. Classification train batch `[4,3,112,112]`; validation/test and verification batches `[2,3,112,112]`; sampled-pair batch `[4,3,112,112]`. All seven loader checks pass. These are not Kaggle images.
- [Earlier acquisition attempt](../reports/data/kaggle-access-2026-11-04.json), under 0.3.13: **FAIL**, `UnauthenticatedError` before a network download. This historical failure was resolved by the modern-client upgrade and owner token configuration; it is not the current acceptance result.

The [authenticated Kaggle run](../reports/data/kaggle-authenticated-2026-11-04.json) and [download/inspection log](../reports/data/kaggle-authenticated-2026-11-04.log) completed with **PASS**, exit code 0, using KaggleHub 1.0.2 and the assigned competition. Observed counts:

| Data | Actual count |
| --- | --- |
| Training images | 380,638 across 4,000 identities |
| Validation images | 8,000 across the same 4,000 identities |
| Classification test images | 8,000 across the same 4,000 identities |
| Official verification trials | 8,805: 4,315 same-identity, 4,490 different-identity |

All seven loader checks passed: shared class mapping; finite classification batches from each split; all official trial paths and binary labels; finite verification batches; identity-correct training pairs; seeded reproducibility; and finite sampled-pair batches. Each image tensor batch inspected on the actual data had shape `[4,3,112,112]`. The sample manifest contains three positive and three negative training pairs.

The [local replay](../reports/data/local-verified-2026-11-04.json) also exited 0 with PASS, using the extracted cached files through `--data-root` rather than invoking the download function. Its split counts, complete trial manifest hash, identity mapping hash, sampled pairs and batch checks match the authenticated run.

No model was trained or scored. Only the inspected batches were decoded; this is not proof that every image is decodable or that the splits are free of duplicate/identity leakage. FAC-01.2 implementation and assigned-data verification are complete; Que An's independent protocol review remains separate.

Acceptance requires the authenticated command to exit 0 with `status=PASS`, actual split/identity counts, valid trial labels, finite classification/pair tensor batches and identity-correct sampled pairs. Reports include the trials-file hash, mapping hash, preview rows, seed and source hashes. Retain failures and use a new output file after fixes.

Primary references: [KaggleHub downloads/authentication](https://github.com/Kaggle/kagglehub) and [PyTorch datasets/DataLoader](https://docs.pytorch.org/docs/stable/data.html). The installed API signature, supplied brief and saved runs establish the version-specific behavior above.
