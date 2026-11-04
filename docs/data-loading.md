# Face data loading

FAC-01.2 provides classification/pair datasets and baseline training-pair sampling for competition `11-785-fall-20-homework-2-part-2`. No other face dataset is substituted.

## Download on your own machine

Use the assigned [Kaggle competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). `--download` calls `kagglehub.competition_download("11-785-fall-20-homework-2-part-2")`; it downloads on the first run and can reuse the cache later. It does not require manually uploading images to Colab or receiving another person's dataset folder.

### 1. Select the project interpreter

Open a terminal in the repository root, the directory containing `configs/`, `scripts/` and `requirements.txt`. Complete the CPU or CUDA installation in [setup](setup.md) if this is a new machine. Commands below use `python` to mean **that configured environment's interpreter**, not an arbitrary system Python. In Windows PowerShell, use `.\.venv\Scripts\python.exe`; on macOS use `.venv/bin/python` after completing its OS-specific setup. On the owner's machine use `~/.local/share/ml-frameworks/.venv/bin/python` in place of `python`; do not reinstall that environment.

```bash
python --version
python -m pip show kagglehub
```

Expected: Python 3.13.x and KaggleHub 1.0.2. The pinned client supports modern API tokens.

### 2. Configure your own Kaggle token once

Sign in to Kaggle, open the competition and accept its rules if requested. In [Kaggle API settings](https://www.kaggle.com/settings/api), select **Generate New Token**. Store the token text in `~/.kaggle/access_token`, outside the repository. If the folder/file does not exist, the following command creates it and prompts without echoing the token or putting it into shell history:

```bash
python -c "from pathlib import Path; from getpass import getpass; p=Path.home()/'.kaggle'/'access_token'; p.parent.mkdir(mode=0o700, parents=True, exist_ok=True); token=getpass('Kaggle API token: ').strip(); assert token, 'Empty token'; p.write_text(token); p.chmod(0o600)"
```

Run this only when adding/replacing your own token; it replaces that token file. On Linux/macOS the file permission limits access to your user. Do not put the new token text in `kaggle.json`, which expects legacy username/key JSON. `KAGGLE_API_TOKEN` is another supported authentication route. Do not print tokens in logs or commit them. A one-off `kagglehub.login()` in another process does not configure the later loader process.

### 3. Download and inspect

```bash
python scripts/check_face_data.py --download --output reports/data/kaggle-review.json
```

Expected: exit code 0 and JSON `status=PASS`. The script prints the dataset root, split counts, trial counts and batch checks. Download/extraction may take several minutes. Files remain under ignored `data/.kagglehub/`; the loader uses KaggleHub's returned root without copying a second dataset. Expected full-data counts are listed [below](#observed-evidence-and-acceptance).

For a repeat run, choose a new output filename, for example `kaggle-review-02.json`; reports are intentionally not overwritten. When data is already present, local mode skips the Kaggle API:

```bash
python scripts/check_face_data.py --output reports/data/local-review.json
```

This reads `paths.face_data` from `configs/project.toml`. To use a manually extracted directory or another disk, explicitly select its dataset root:

```bash
python scripts/check_face_data.py --data-root "/path/to/extracted/dataset" --output reports/data/external-review.json
```

That root must contain `classification_data/`, `verification_data/` and `verification_pairs_val.txt`, as shown in [the storage guide](../data/README.md). Discovery also accepts one enclosing directory. Further nesting or incomplete extraction raises an error. API mode reads `paths.kaggle_cache` from the same config; `--cache-dir` overrides that cache and `--config` selects another TOML file. `--data-root` selects existing files and cannot be combined with `--download`.

## Download on Google Colab

First run steps 1–4 in [the Colab setup guide](setup.md#google-colab-cpu-or-nvidia-gpu). Those steps define `python`, `reports`, `command_env` and the `run(...)` helper. In Colab's **Secrets** panel (key icon), add a secret named `KAGGLE_API_TOKEN`, paste your modern token as its value, and enable notebook access. Then execute this cell without printing the secret:

```python
from google.colab import userdata

# The project runs in a subprocess, so explicitly pass the notebook secret.
command_env['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
```

Then run the loader cell once in setup step 5. It uses the same competition API and repo-local cache as the local command. If the notebook secret cannot be read, check its spelling and notebook-access switch. No token needs to be embedded in a cell, source file or report. This secret handoff follows [KaggleHub's supported token mechanisms](https://github.com/Kaggle/kagglehub#authenticate); the current Colab acceptance run uploaded an image subset and did not transmit the owner's Kaggle token or exercise Colab authentication.

## If a step fails

| Symptom | Action |
| --- | --- |
| `User is not authenticated` | Check the selected interpreter/client version and token location, or the Colab secret handoff above |
| Permission/competition-rules error | Sign in to the same Kaggle account that owns the token and accept the competition rules; verify token permissions |
| Missing layout/files | Use the root of the extracted competition, not an archive or an individual split; finish extraction |
| Output file already exists | Choose a new report filename rather than deleting the previous evidence |
| Imports/package conflicts | Recheck the chosen environment/profile in setup; do not replace the owner's working GPU installation |

Fix the cause and rerun with a new output path. A failed report is evidence of a failed attempt, not an accepted loader run.

## Loader and sampling contract

Implementation: `src/facial_attendance/recognition/data_loader.py`. Identity mapping and image indexing are separate helpers. The inspection script separates classification decoding, official pair batches and training-pair sampling rather than combining them into one long function. Inline comments explain train-only mapping, lazy RGB decoding, preserved trial order, path boundaries and local RNG use.

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

### Colab checks with original competition images

The [CPU report](../reports/environment/colab-cpu-real-2026-11-04.json) and [Tesla T4 report](../reports/environment/colab-cuda-real-2026-11-04.json) both passed on the same original-image subset. The [input manifest](../reports/data/colab-real-input-2026-11-04.json) records 56 image hashes verified against the locally downloaded competition before upload and again on each runtime. Four identities contribute 12 train, 8 validation and 8 classification-test images; 28 distinct verification images support the first 16 unmodified official trials. No image bytes or credentials are included in these reports.

Each runtime passed all seven loader checks and 39 unit tests. A batch of four real training images also completed one tiny randomly initialized CNN optimizer update and finite inference on the selected device: CPU or `cuda:0`. This verifies decoded-image/model/device integration; it does not implement or assess the required embedding models. The subset run does not establish full-dataset Colab download, full image integrity or recognition accuracy. Full-layout and authenticated-download evidence below was obtained locally.

### Local full-dataset acquisition and inspection

The [current default-config run](../reports/data/config-default-audit-2026-11-04.json) passed on the real dataset without `--data-root`, invoked from `/tmp` to check working-directory independence. The following earlier runs retain their historical source hashes; their counts, mapping and sampled-pair outputs were compared against the refactor. The complete current suite has 46 passing tests.

- Historical pre-refactor [test log](../reports/data/loader-tests-kagglehub-1.0.2.log): **26 tests PASS**, including 19 loader/acquisition fixture tests and seven existing configuration tests. Mocked API tests establish the call contract, not real authentication.
- Historical pre-refactor [CLI fixture result](../reports/data/fixture-kagglehub-1.0.2.json): **PASS** on temporary generated PNGs. Classification train batch `[4,3,112,112]`; validation/test and verification batches `[2,3,112,112]`; sampled-pair batch `[4,3,112,112]`. All seven loader checks pass. These are not Kaggle images.
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
