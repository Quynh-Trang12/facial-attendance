# Setup

**Local machine:** follow steps 1–4 below, using only the branch for your machine.

**Colab:** skip steps 1–4 and follow the [Colab guide](#google-colab-cpu-or-nvidia-gpu).

## 1. Open a terminal and select the environment

Use the `facial-attendance` repository root as the working directory, containing `scripts/` and `requirements.txt`. Use Python **3.13.x**; the exact patch in `.python-version` is not required.

If you already have a compatible environment, activate it and skip creating `.venv`. Otherwise, use the appropriate branch:

**Linux x86_64 / macOS 14+ Apple Silicon — bash/zsh**

```bash
python3.13 -m venv .venv
source .venv/bin/activate
```

**Windows x64 — PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use `.\.venv\Scripts\python.exe` instead of `python` in the following steps.

Check the selected interpreter:

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

**Expected:** Python 3.13.x and an executable inside `.venv` (or the compatible environment you chose to reuse). This identifies the interpreter that the later `python` commands will use.

macOS requires native `arm64` Python. Intel Macs, Rosetta and Windows ARM64 are not supported by this dependency set. macOS currently uses CPU only; MPS is not supported.

## 2. Install dependencies into the environment from step 1

Skip this step if that environment already has the project dependencies. These are **terminal commands**, run from the same repository root as step 1. They target the existing `.venv` directly; activation is not needed for installation.

If step 1 reused an environment at another location, substitute its Python executable for the `.venv` executable below. The intended target is the same environment throughout setup.

**Expand only your platform below.**

<details>
<summary>Linux x86_64</summary>

First confirm pip's location:

```bash
.venv/bin/python -m pip --version
```

Expected: the pip location is inside this repository's `.venv`, with Python 3.13. A missing executable means step 1 has not prepared that environment at this location.

Choose **one** framework profile:

| Device | Command |
| --- | --- |
| NVIDIA GPU | `.venv/bin/python -m pip install -r requirements-cuda.txt` |
| CPU | `.venv/bin/python -m pip install -r requirements-cpu.txt` |

Then install the shared dependencies:

```bash
.venv/bin/python -m pip install -r requirements.txt
```

</details>

<details>
<summary>Windows x64 — PowerShell</summary>

First confirm pip's location:

```powershell
.\.venv\Scripts\python.exe -m pip --version
```

Expected: the pip location is inside this repository's `.venv`, with Python 3.13. This also works when PowerShell blocks activation.

Choose **one** framework profile:

| Device | Command |
| --- | --- |
| NVIDIA GPU | `.\.venv\Scripts\python.exe -m pip install -r requirements-cuda.txt` |
| CPU | `.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt` |

Then install the shared dependencies:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

</details>

<details>
<summary>macOS Apple Silicon</summary>

First confirm pip's location, then install from PyPI:

```bash
.venv/bin/python -m pip --version
.venv/bin/python -m pip install -r requirements.txt
```

Expected: the pip location is inside this repository's `.venv`, with Python 3.13. macOS uses the shared requirements directly, without a CPU/CUDA profile file.

</details>

**Next:** continue to step 3 using the environment selected in step 1. The NVIDIA profile requires a driver compatible with CUDA 12.6.

## 3. Check the device

Run one check using the same environment as steps 1–2. If you reused an environment elsewhere, substitute its Python executable below.

<details>
<summary>Linux x86_64 / macOS Apple Silicon — bash/zsh</summary>

**CPU, including macOS:**

```bash
.venv/bin/python scripts/check_environment.py --device cpu --operator team-check --output evidence/runs/SYS-03/team-cpu-check.json
```

**NVIDIA GPU — Linux only:**

```bash
.venv/bin/python scripts/check_environment.py --device cuda --operator team-check --output evidence/runs/SYS-03/team-cuda-check.json
```

</details>

<details>
<summary>Windows x64 — PowerShell</summary>

**CPU:**

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py --device cpu --operator team-check --output evidence/runs/SYS-03/team-cpu-check.json
```

**NVIDIA GPU:**

```powershell
.\.venv\Scripts\python.exe scripts/check_environment.py --device cuda --operator team-check --output evidence/runs/SYS-03/team-cuda-check.json
```

</details>

**Expected:** exit code 0, JSON `status: PASS` and the selected device. Each rerun needs a new output filename because existing results are preserved. The check runs a synthetic training step and inference pass; it does not measure model accuracy. Inaccessible CUDA produces an error rather than switching to CPU.

**If it fails:** the command, terminal output and generated JSON identify the failed check.

## Run unit tests

Run the suite after installing dependencies to check the setup, and after code or configuration changes to catch regressions before review.

The current tests check path/configuration handling, Python/device policy, dataset labels and trial ordering, image decoding and seeded pair sampling. For example, they can detect a change that maps validation identities incorrectly or alters official trial order. They use generated fixtures and mocked download calls, so the current suite needs no GPU, Kaggle token or competition download.

The command below uses `unittest` discovery to collect tests from `tests/`. As model and system-integration tests are added, the discovered test count and covered behavior can grow.

From the repository root, using the environment selected in step 1 (substitute its executable if it is outside `.venv`):

```bash
# Linux/macOS
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

```powershell
# Windows PowerShell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

**Expected:** exit code 0 and `Ran N tests` followed by `OK`, where `N` is the number discovered for the checked-out revision, not a fixed target. `FAILED` lists failures/errors to investigate; skipped tests are reported separately and do not establish that the skipped behavior works.

Passing the suite confirms only the behavior covered by its tests. The device check above exercises actual training/inference; dataset inspection and model-performance evaluation remain separate checks.

## 4. Download and check the dataset

Choose the dataset task:

- **FAC-01 face recognition:** follow [Download on your own machine](data-loading.md#download-on-your-own-machine).
- **LIV-03 anti-spoofing:** follow [the reviewer starting point](liveness-loader.md#reviewer-starting-point). Check its storage plan before downloading; it uses CelebA-Spoof rather than KaggleHub and needs no GPU for the audit/loader checks. The guide includes external-drive paths and commands that generate the required preparation files on a fresh device.

**For subsequent runs:** open a terminal at the repository root, activate the existing environment and run `python ...`. Environment creation and dependency installation are one-time setup steps.

Platform verification and previous execution results are in the [reference section](#reference).

## Google Colab: CPU or NVIDIA GPU

For LIV-03, complete environment steps 1–3 here, then follow the [Colab data, tests and audit workflow](liveness-loader.md#google-colab-review). That workflow includes its own liveness tests; skip the face-recognition data step for this task.

<details>
<summary>Using Colab? Expand and follow steps 1–5</summary>


**Order:** select runtime → prepare environment → install → check → load data.

Follow these cells in order in one notebook session. They run repository scripts in an isolated project environment using Colab's existing Python; they do not replace Colab's stock Torch/TensorFlow packages. The notebook itself continues to use the stock kernel, so run project commands through the `python` path defined below. Use the Python 3.13 interpreter supplied by Colab; another Python 3.13 patch release is acceptable. Matching the reference patch does not require a separate Python installation.

### 1. Put the repository in one directory and select the device

1. Upload/extract the repository to `/content/facial-attendance`.
2. Confirm `/content/facial-attendance/scripts/check_environment.py` exists.
3. Choose **Runtime → Change runtime type** → CPU or NVIDIA GPU.
4. Set `device` in the cell below to match your selection, then run it.

GPU availability depends on the account/runtime allocation. An unavailable requested GPU causes the check to fail.

```python
from pathlib import Path
from datetime import datetime, timezone
import os
import subprocess
import sys
import venv

repo = Path('/content/facial-attendance')
assert (repo / 'scripts/check_environment.py').is_file(), 'Extract the repo first'
assert sys.version_info[:2] == (3, 13), sys.version
# Match the Colab runtime you selected: 'cpu' or 'cuda'.
device = 'cpu'
if device == 'cuda':
    subprocess.run(['nvidia-smi'], check=True)
assert device in ('cpu', 'cuda')
run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
run_dir = repo / 'evidence/runs/SYS-03' / f'colab-{device}-{run_id}'
run_dir.mkdir(parents=True, exist_ok=False)
python = repo / '.venv/bin/python'
print('Kernel:', sys.version.split()[0], '| device:', device)
print('Project:', repo, '| evidence:', run_dir)
```

Keep a CPU environment and a CUDA environment in different Colab sessions. Each project environment uses one wheel profile.

### 2. Prepare a command runner and the isolated environment

Run this cell once. It creates the isolated environment and defines a runner that saves logs and stops on failure.

**Runner settings:** `capture_output=True` supports notebook output capture. `MPLBACKEND=Agg` prevents Colab's inline display backend from being inherited by noninteractive subprocesses without notebook plugins.

```python
command_env = {
    **os.environ,
    'MPLBACKEND': 'Agg',
    'PYTHONDONTWRITEBYTECODE': '1',
    'PYTHONPATH': os.pathsep.join([str(repo / 'src'), str(repo / 'tests')]),
}

def run(stage, command):
    result = subprocess.run([str(arg) for arg in command], cwd=repo,
                            env=command_env, capture_output=True, text=True)
    (run_dir / f'{stage}.log').write_text(result.stdout + result.stderr)
    print(stage, 'PASS' if result.returncode == 0 else 'FAIL')
    if result.returncode:
        print((result.stdout + result.stderr)[-3000:])
    result.check_returncode()

# Bootstrap pip through the existing kernel; keep installed packages in .venv.
venv.EnvBuilder(with_pip=False).create(repo / '.venv')
run('bootstrap-pip', [sys.executable, '-m', 'pip', '--python', python,
                      'install', 'pip'])
```

### 3. Install one wheel profile, then shared requirements

```python
profile = 'requirements-cpu.txt' if device == 'cpu' else 'requirements-cuda.txt'
run('install-framework', [python, '-m', 'pip', 'install', '-r', profile])
run('install-project', [python, '-m', 'pip', 'install', '-r', 'requirements.txt'])
```

The CUDA profile uses CUDA 12.6 wheels and needs a compatible NVIDIA driver. If installation or CUDA checks fail, inspect the stage log for the failing command. The archived [requirements.lock](../requirements.lock) describes a historical Linux/CUDA environment rather than the Colab dependency profile.

### 4. Run tests and the selected-device check

```python
run('unit-tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
run('environment', [python, 'scripts/check_environment.py', '--device', device,
                    '--operator', 'colab-review',
                    '--output', run_dir / 'environment.json'])
```

Expected: the discovered tests finish with `OK`; environment JSON has `status=PASS`, the requested device, finite gradients, changed weights, finite inference and a successful `pip check`. This step uses a tiny synthetic tensor batch to check infrastructure, not face-model accuracy.

### 5. Run the loader on real competition data

Configure your own Kaggle token using [the data-loading instructions](data-loading.md#download-on-google-colab), then run:

```python
run('loader', [python, 'scripts/check_face_data.py', '--download',
               '--output', run_dir / 'loader.json'])
```

This downloads/reuses the assigned competition and checks its actual data.

<details>
<summary>Use an existing uploaded, extracted or mounted dataset instead</summary>

Replace `--download` with `--data-root`, followed by the dataset root. For files already extracted at `repo / 'data/colab-sample'`, the alternative loader cell is:

```python
run('loader', [python, 'scripts/check_face_data.py',
               '--data-root', repo / 'data/colab-sample',
               '--output', run_dir / 'loader.json'])
```

Run this cell instead of the download cell above. That folder must have the [required layout](../data/README.md); the example subset is not shipped with the repository. Loader decoding/collation runs on CPU even when the model runs on GPU. The environment JSON's `face_dataset_present` describes the configured default path at the time it ran; an explicit `--data-root` can point elsewhere.

</details>

**Expected full-dataset counts:** 380,638 train images, 8,000 validation images, 8,000 classification test images, 4,000 identities, and 8,805 official trials. A deliberately uploaded subset will have smaller counts and must be labelled as such. `status=PASS` checks mappings, finite decoded batches, pair labels/paths and seeded training-pair rules; it does not decode every image or measure recognition accuracy.

Download the `run_dir` directory shown in step 1 before ending the session, then disconnect and delete the runtime in Colab. Runtime files are temporary; keep credentials and raw images out of Git.

</details>

## Reference

<details>
<summary>Platform verification</summary>

### Platform verification

Linux CPU/CUDA execution has recorded evidence. Windows 11 and Mac M1 CPU execution passed in the accepted independent review; see [review notes](../evidence/SYS-03/README.md#sys-03-independent-review). Historical dependency-resolution reports remain separate evidence. The macOS minimum follows the pinned [Torch wheels](https://pypi.org/project/torch/2.13.0/#files).

`requirements.lock` is an archived Linux/CUDA snapshot, not the team installation file.

</details>

<details>
<summary>Recorded remote evidence</summary>

### Recorded remote evidence

See [Google's runtime documentation](https://research.google.com/colaboratory/runtime-version-faq.html) for runtime changes and [official PyTorch wheel profiles](https://pytorch.org/get-started/previous-versions/) for installation background.

</details>

<details>
<summary>Dataset configuration</summary>

## Dataset configuration

`configs/project.toml` contains both `kaggle_cache` and `face_data`. The defaults now match the actual downloaded competition under `data/.kagglehub/competitions/11-785-fall-20-homework-2-part-2`. Both dataset CLI branches read this config; `--data-root`, `--cache-dir` and `--config` are explicit overrides for external disks or Colab mounts.

Dataset decoding/collation happens on CPU and does not require GPU access. In Colab, put the modern token in a private Kaggle token file or its supported environment/secret mechanism, then use [the data-loading commands](data-loading.md). Do not put tokens in notebook output or source control.

</details>

<details>
<summary>Execution evidence and review records</summary>

## Execution evidence and review records

- Current Python 3.13-only policy: [CPU runtime report](../evidence/SYS-03/cpu-python313-2026-11-04.json) and [46-test log](../evidence/SYS-03/python313-tests-2026-11-04.log).

CI discovers tests from `tests/` rather than checking a fixed test count. All CI jobs read `.python-version` and run **3.13.16 only**. The Linux check retains its existing `source-checks (3.13)` name. Windows x64 and macOS Apple Silicon CPU jobs cover the team's actual operating systems, not extra Python versions. Independent Windows/macOS CPU execution passed in the accepted review; GitHub CI execution remains unverified.

The workflow defines `source-checks (3.13)`, `OS compatibility (windows-latest)` and `OS compatibility (macos-15)`. The former `source-checks (3.12)` job is retired; repository ruleset configuration is separate from execution evidence.

[SYS-03 review evidence](../evidence/SYS-03/README.md#sys-03-independent-review) records the tested platforms and findings. [FAC-01 evidence](../evidence/FAC-01/README.md) documents loader checks and protocol findings. Task status and assignments are maintained in the [Project Tracker](../README.md#project-tracker).

Primary wheel reference: [official PyTorch CPU/CUDA installation commands](https://pytorch.org/get-started/previous-versions/).

</details>
