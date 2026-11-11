# Face data loading

FAC-01.2 provides classification/pair datasets and baseline training-pair sampling for competition `11-785-fall-20-homework-2-part-2`. No other face dataset is substituted.

**Choose your route:** [Local machine](#download-on-your-own-machine) · [Colab](#download-on-google-colab) · [Troubleshooting](#if-a-step-fails).
**Reference:** [Loader contract](#loader-and-sampling-contract) · [Evidence](#observed-evidence-and-acceptance).

## Download on your own machine

The dataset comes from the assigned [Kaggle competition](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation). `--download` calls `kagglehub.competition_download("11-785-fall-20-homework-2-part-2")`; it downloads on the first run and can reuse the cache later. It does not require manually uploading images to Colab or receiving another person's dataset folder.

### 1. Select the project interpreter

**Before starting:** complete [setup](setup.md) and activate the project environment. Open a terminal at the repository root (`scripts/` and `requirements.txt` must be present).

Commands below use the selected environment's `python`. If activation is unavailable, replace `python` in every local command with `.venv/bin/python` on Linux/macOS or `.\.venv\Scripts\python.exe` in Windows PowerShell. For an environment outside `.venv`, use its executable instead.

**Check:**

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip show kagglehub
```

Expected: the environment selected during setup, Python 3.13.x and KaggleHub 1.0.2. The pinned client supports modern API tokens.

### 2. Save your Kaggle token — once

Already saved a working token? Continue to [step 3](#3-download-and-inspect).

1. **Accept the rules:** sign in and open the [competition page](https://www.kaggle.com/c/11-785-fall-20-homework-2-part-2/overview/evaluation).
2. **Generate a token:** open [Kaggle API settings](https://www.kaggle.com/settings/api) → **Generate New Token**.
3. **Copy the save command:** find **“Or save it to ~/.kaggle/access_token, where the client will read it automatically:”** and copy the command underneath.
4. **Save it locally:** run the copied command in **bash/zsh** on Linux/macOS. For **Windows PowerShell**, expand the instructions below instead.

**Next:** follow [step 3](#3-download-and-inspect) to download the dataset. Saving the token does not download it.

<details>
<summary>Windows PowerShell: save the token</summary>

The Kaggle command may use bash syntax. In your project environment, run this command, then paste **only the token text** at the prompt:

```powershell
python -c "from pathlib import Path; from getpass import getpass; p=Path.home()/'.kaggle'/'access_token'; p.parent.mkdir(mode=0o700, parents=True, exist_ok=True); token=getpass('Kaggle API token: ').strip(); assert token, 'Empty token'; p.write_text(token); p.chmod(0o600)"
```

Then continue to [step 3](#3-download-and-inspect).

</details>

The token is stored outside the repository in `~/.kaggle/access_token`, which [KaggleHub reads automatically](https://github.com/Kaggle/kagglehub#authenticate). Saving a new token replaces the existing file. The copied command contains a private token and may remain in shell history. Tokens are local credentials rather than repository content; legacy `kaggle.json` uses a different credential format.

### 3. Download and inspect

```bash
python scripts/check_face_data.py --download --output evidence/runs/FAC-01/kaggle-review.json
```

Expected: exit code 0 and JSON `status=PASS`. The script prints the dataset root, split counts, trial counts and batch checks. Download/extraction may take several minutes. Files remain under ignored `data/.kagglehub/`; the loader uses KaggleHub's returned root without copying a second dataset. Expected full-data counts are listed [below](#observed-evidence-and-acceptance).

**Rerunning:** use a new output filename, such as `kaggle-review-02.json`; existing reports are preserved.

<details>
<summary>Dataset already downloaded at the configured location</summary>

Local mode skips the Kaggle API:

```bash
python scripts/check_face_data.py --output evidence/runs/FAC-01/local-review.json
```

This reads `paths.face_data` from `configs/project.toml`.

</details>

<details>
<summary>Dataset extracted elsewhere or stored on another disk</summary>

Select its dataset root:

```bash
python scripts/check_face_data.py --data-root "/path/to/extracted/dataset" --output evidence/runs/FAC-01/external-review.json
```

That root must contain `classification_data/`, `verification_data/` and `verification_pairs_val.txt`, as shown in [the storage guide](../data/README.md). Discovery also accepts one enclosing directory. Further nesting or incomplete extraction raises an error. `--data-root` selects existing files and cannot be combined with `--download`.

</details>

<details>
<summary>Custom download cache or configuration</summary>

API mode reads `paths.kaggle_cache` from the same config; `--cache-dir` overrides that cache and `--config` selects another TOML file.

</details>

## Download on Google Colab

1. **Prepare Colab:** complete steps 1–4 in [the Colab setup guide](setup.md#google-colab-cpu-or-nvidia-gpu). They define `python`, `run_dir`, `command_env` and `run(...)`.
2. **Add the token:** in Colab's **Secrets** panel (key icon), create `KAGGLE_API_TOKEN`, paste the token as its value and enable notebook access.
3. **Pass it to the loader:** execute this cell without printing the secret:

```python
from google.colab import userdata

# The project runs in a subprocess, so explicitly pass the notebook secret.
command_env['KAGGLE_API_TOKEN'] = userdata.get('KAGGLE_API_TOKEN')
```

4. **Download:** return to setup step 5 and run the loader cell once. It uses the same competition API and repo-local cache as the local command.

**If the secret cannot be read:** check its spelling and notebook-access switch. No token needs to be embedded in a cell, source file or report.

<details>
<summary>Authentication reference and recorded test limits</summary>

This secret handoff follows [KaggleHub's supported token mechanisms](https://github.com/Kaggle/kagglehub#authenticate); the current Colab acceptance run uploaded an image subset and used existing dataset files rather than testing Colab token authentication.

</details>

## If a step fails

| Symptom | Action |
| --- | --- |
| `User is not authenticated` | Check the selected interpreter/client version and token location, or the Colab secret handoff above |
| Permission/competition-rules error | Sign in to the same Kaggle account that owns the token and accept the competition rules; verify token permissions |
| Missing layout/files | Use the root of the extracted competition, not an archive or an individual split; finish extraction |
| Output file already exists | Choose a new report filename rather than deleting the previous evidence |
| Imports/package conflicts | Recheck the selected environment and OS-specific dependency profile in setup |

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

For positive pairs, the sampler selects uniformly among identities with at least two images, then selects two distinct images. For negative pairs, it selects two different identities uniformly, then one image per identity. Sampling is with replacement across pairs; repeated pairs are possible. Singleton identities can supply negatives. This is a reproducible baseline, not exhaustive sampling or hard-negative mining. Official verification trials are never resampled.

Verification paths may be bare filenames or prefixed with `verification_data/`. Nonblank/noncomment lines require exactly two paths and label 0/1. Missing files, malformed labels/rows, escaping paths, unknown classification IDs and unreadable sampled images fail explicitly. Full duplicate/leakage audits and model evaluation remain separate tasks.

## Observed evidence and acceptance

### Colab checks with original competition images

The [CPU report](../evidence/SYS-03/colab-cpu-real-2026-11-04.json) and [Tesla T4 report](../evidence/SYS-03/colab-cuda-real-2026-11-04.json) both passed on the same original-image subset. The [input manifest](../evidence/FAC-01/colab-real-input-2026-11-04.json) records 56 image hashes verified against the locally downloaded competition before upload and again on each runtime. Four identities contribute 12 train, 8 validation and 8 classification-test images; 28 distinct verification images support the first 16 unmodified official trials. No image bytes or credentials are included in these reports.

Each runtime passed all seven loader checks and 39 unit tests. A batch of four real training images also completed one tiny randomly initialized CNN optimizer update and finite inference on the selected device: CPU or `cuda:0`. This verifies decoded-image/model/device integration; it does not implement or assess the required embedding models. The subset run does not establish full-dataset Colab download, full image integrity or recognition accuracy. Full-layout and authenticated-download evidence below was obtained locally.

### Local full-dataset acquisition and inspection

The [current default-config run](../evidence/FAC-01/config-default-audit-2026-11-04.json) passed on the real dataset without `--data-root`, invoked from `/tmp` to check working-directory independence. The following earlier runs retain their historical source hashes; their counts, mapping and sampled-pair outputs were compared against the refactor. The [unit-test command](setup.md#run-unit-tests) finds and runs the tests in the repository’s `tests/` folder.

The [authenticated Kaggle run](../evidence/FAC-01/kaggle-authenticated-2026-11-04.json) and [download/inspection log](../evidence/FAC-01/kaggle-authenticated-2026-11-04.log) completed with **PASS**, exit code 0, using KaggleHub 1.0.2 and the assigned competition. Observed counts:

| Data | Actual count |
| --- | --- |
| Training images | 380,638 across 4,000 identities |
| Validation images | 8,000 across the same 4,000 identities |
| Classification test images | 8,000 across the same 4,000 identities |
| Official verification trials | 8,805: 4,315 same-identity, 4,490 different-identity |

All seven loader checks passed: shared class mapping; finite classification batches from each split; all official trial paths and binary labels; finite verification batches; identity-correct training pairs; seeded reproducibility; and finite sampled-pair batches. Each image tensor batch inspected on the actual data had shape `[4,3,112,112]`. The sample manifest contains three positive and three negative training pairs.

No model was trained or scored. Only the inspected batches were decoded; this is not proof that every image is decodable or that the splits are free of duplicate/identity leakage. [Review notes and later leakage findings](../evidence/FAC-01/README.md#review-scope-and-findings) distinguish the loader checks from full-data/protocol evidence. Task progress is maintained in the [Project Tracker](../README.md#project-tracker).

The authenticated loader check records exit code 0 with `status=PASS`, actual split/identity counts, valid trial labels, finite classification/pair tensor batches and identity-correct sampled pairs. Reports include the trials-file hash, mapping hash, preview rows, seed and source hashes. Separate output files preserve failed attempts and subsequent retests.

Primary references: [KaggleHub downloads/authentication](https://github.com/Kaggle/kagglehub) and [PyTorch datasets/DataLoader](https://docs.pytorch.org/docs/stable/data.html). The installed API signature, supplied brief and saved runs establish the version-specific behavior above.
