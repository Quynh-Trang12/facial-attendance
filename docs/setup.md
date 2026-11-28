# Setup

Use Python 3.13.x and the repository root as the working directory. Install once into a project environment, then follow the [module guide](../README.md#start-here). Inference and saved inspection use CPU; liveness training/evaluation reproduction uses the [Colab notebook](../notebooks/liveness-training-and-evaluation.ipynb).

## 1. Open a terminal and select the environment

Reuse an existing compatible environment, or create one using the platform block below. Supported targets are Linux x86_64, Windows x64 and macOS 14+ Apple Silicon with native arm64 Python. Intel/Rosetta Macs, Windows ARM64 and MPS are outside this dependency policy. Another 3.13 patch is accepted; CI reads its reference from [.python-version](../.python-version).

## 2. Install dependencies into the environment from step 1

Run only your platform's block. Each creates/activates the environment and installs the CPU profile plus [shared dependencies](../requirements.txt). If the environment already exists, skip creation/installation as appropriate.

<details>
<summary>Linux x86_64 — bash/zsh</summary>

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
```

</details>

<details>
<summary>Windows x64 — PowerShell</summary>

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
```

If activation is blocked, replace `python` with `.\.venv\Scripts\python.exe`; no policy change is needed.

</details>

<details>
<summary>macOS 14+ Apple Silicon — bash/zsh</summary>

```sh
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Native arm64 wheels come from PyPI; do not use the CUDA index.

</details>

For explicit local NVIDIA CUDA work on Linux/Windows, replace requirements-cpu.txt with [requirements-cuda.txt](../requirements-cuda.txt), then install shared requirements. Choose one wheel profile. [requirements.lock](../requirements.lock) is an attributed historical Linux/CUDA snapshot, not the installation recipe.

Confirm the interpreter and pip both belong to your chosen environment:

```sh
python -c "import sys; print(sys.executable); print(sys.version)"
python -m pip --version
```

## 3. Check the device

After activation, the same command works on all supported OSs:

```sh
python scripts/check_environment.py --device cpu --operator reviewer --output evidence/runs/SYS-03/reviewer-cpu.json
```

Expected: exit 0 and JSON status PASS. The report checks versions, numerical training/inference and image/metric APIs; dataset locations are reported separately. Use a new output filename on repeats. For an explicitly requested NVIDIA target, use `--device cuda`; inaccessible CUDA fails without CPU fallback. Resolve the reported failure before relying on that runtime.

## Run unit tests

Tests use generated fixtures, not a Kaggle token, GPU or downloaded dataset. Configure the source path and discover from the repository root:

```sh
# Linux/macOS
PYTHONPATH=src python -m unittest discover -s tests -t . -v
```

```powershell
# Windows
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -t . -v
```

Expected: OK; failures and skips describe their own scope. The cache-publication fixture checks real disk capacity, so leave more than 2 GiB free. Tests are grouped under [liveness](../tests/liveness/) and [recognition](../tests/recognition/), with shared configuration/runtime tests at the root. A passing suite does not establish dataset truth, accuracy or live webcam integration.

## 4. Download and check the dataset

- **Face recognition:** [Acquisition, token setup and loader checks](data-loading.md).
- **Liveness:** [Saved inspection and CPU inference](liveness-training-and-evaluation.md#portable-inference-and-saved-inspection), [full data audit](liveness-data-preparation-and-audit.md#audit-the-current-frozen-selection) or [Colab training](liveness-training-and-evaluation.md#execution-and-resume).

Liveness inference needs the selected checkpoint and optional detector; it does not require downloading training data. [Storage](../data/README.md) explains existing datasets, cache locations and path overrides.

## Google Colab: CPU or T4 GPU

<details>
<summary>Liveness</summary>

Use the [training/evaluation notebook](../notebooks/liveness-training-and-evaluation.ipynb) or [data audit notebook](../notebooks/liveness-data-preparation-and-audit.ipynb). Their [module guides](liveness-training-and-evaluation.md) own action/runtime selection, asset checks and outputs. Commenters read shared inputs and write results under their private MyDrive; no edit access to the shared project is required. Native revised Colab execution remains unverified.

</details>

<details>
<summary>Face recognition — isolated runtime and loader check</summary>

The following steps define `repo`, `python`, `run_dir`, `command_env` and `run(...)` used by the [Colab acquisition guide](data-loading.md#download-on-google-colab). They keep project packages separate from the notebook kernel. Match the selected device and retain logs; an unavailable GPU stops rather than switching devices.

### 1. Put the repository in one directory and select the device

1. Extract the repository under **/content** as **facial-attendance**.
2. Confirm the extracted project contains [check_environment.py](../scripts/check_environment.py) in its scripts directory.
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

repo = Path("/content/facial-attendance")
assert (repo / "scripts/check_environment.py").is_file(), "Extract the repo first"
assert sys.version_info[:2] == (3, 13), sys.version
# Match the Colab runtime you selected: 'cpu' or 'cuda'.
device = "cpu"
if device == "cuda":
    subprocess.run(["nvidia-smi"], check=True)
assert device in ("cpu", "cuda")
run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
run_dir = repo / "evidence/runs/SYS-03" / f"colab-{device}-{run_id}"
run_dir.mkdir(parents=True, exist_ok=False)
python = repo / ".venv/bin/python"
print("Kernel:", sys.version.split()[0], "| device:", device)
print("Project:", repo, "| evidence:", run_dir)
```

Keep a CPU environment and a CUDA environment in different Colab sessions. Each project environment uses one wheel profile.

### 2. Prepare a command runner and the isolated environment

Run this cell once. It creates the isolated environment and defines a runner that saves logs and stops on failure.

**Runner settings:** `capture_output=True` supports notebook output capture. `MPLBACKEND=Agg` prevents Colab's inline display backend from being inherited by noninteractive subprocesses without notebook plugins.

```python
command_env = {
    **os.environ,
    "MPLBACKEND": "Agg",
    "PYTHONDONTWRITEBYTECODE": "1",
    "PYTHONPATH": str(repo / "src"),
}


def run(stage, command):
    result = subprocess.run(
        [str(arg) for arg in command],
        cwd=repo,
        env=command_env,
        capture_output=True,
        text=True,
    )
    (run_dir / f"{stage}.log").write_text(result.stdout + result.stderr)
    print(stage, "PASS" if result.returncode == 0 else "FAIL")
    if result.returncode:
        print((result.stdout + result.stderr)[-3000:])
    result.check_returncode()


# Bootstrap pip through the existing kernel; keep installed packages in .venv.
venv.EnvBuilder(with_pip=False).create(repo / ".venv")
run(
    "bootstrap-pip", [sys.executable, "-m", "pip", "--python", python, "install", "pip"]
)
```

### 3. Install one wheel profile, then shared requirements

```python
profile = "requirements-cpu.txt" if device == "cpu" else "requirements-cuda.txt"
run("install-framework", [python, "-m", "pip", "install", "-r", profile])
run("install-project", [python, "-m", "pip", "install", "-r", "requirements.txt"])
```

The CUDA profile uses CUDA 13.0 wheels and needs a compatible NVIDIA driver. The existing Python 3.13 local runtime with PyTorch 2.13.0+cu126 is also accepted for inference; a valid local build is retained rather than replaced with Colab wheels. The [inference routes](liveness-training-and-evaluation.md#cuda-inference-on-local-and-colab-runtimes) identify the two runtimes and their cache separation. If installation or CUDA checks fail, inspect the stage log for the failing command. The archived [requirements.lock](../requirements.lock) describes a historical Linux/CUDA environment rather than the Colab dependency profile.

### 4. Run tests and the selected-device check

```python
run(
    "unit-tests", [python, "-m", "unittest", "discover", "-s", "tests", "-t", ".", "-v"]
)
run(
    "environment",
    [
        python,
        "scripts/check_environment.py",
        "--device",
        device,
        "--operator",
        "colab-review",
        "--output",
        run_dir / "environment.json",
    ],
)
```

Expected: the discovered tests finish successfully (OK); environment JSON has `status=PASS`, the requested device, finite gradients, changed weights, finite inference and a successful `pip check`. This step uses a tiny synthetic tensor batch to check infrastructure, not face-model accuracy.

### 5. Run the loader on real competition data

Configure your own Kaggle token using [the data-loading instructions](data-loading.md#download-on-google-colab), then run:

```python
run(
    "loader",
    [
        python,
        "scripts/check_face_data.py",
        "--download",
        "--output",
        run_dir / "loader.json",
    ],
)
```

This downloads/reuses the assigned competition and checks its actual data.

<details>
<summary>Use an existing uploaded, extracted or mounted dataset instead</summary>

Replace `--download` with `--data-root`, followed by the dataset root. For files already extracted at `repo / 'data/colab-sample'`, the alternative loader cell is:

```python
run(
    "loader",
    [
        python,
        "scripts/check_face_data.py",
        "--data-root",
        repo / "data/colab-sample",
        "--output",
        run_dir / "loader.json",
    ],
)
```

Run this cell instead of the download cell above. That folder must have the [required layout](../data/README.md); the example subset is not shipped with the repository. Loader decoding/collation runs on CPU even when the model runs on GPU. The environment JSON's `face_dataset_present` describes the configured default path at the time it ran; an explicit `--data-root` can point elsewhere.

</details>

**Expected full-dataset counts:** 380,638 train images, 8,000 validation images, 8,000 classification test images, 4,000 identities, and 8,805 official trials. A deliberately uploaded subset will have smaller counts and must be labelled as such. `status=PASS` checks mappings, finite decoded batches, pair labels/paths and seeded training-pair rules; it does not decode every image or measure recognition accuracy.

Download the output directory identified by `run_dir` in step 1 before ending the session, then disconnect and delete the runtime in Colab. Runtime files are temporary; keep credentials and raw images out of Git.

</details>

## Reference

<a id="platform-verification"></a>

### Platform verification

Current native checks cover Linux CPU behavior and one selected-checkpoint CUDA face-crop inference run. Earlier independent Windows 11/Mac M1 CPU reviews remain attributed in [SYS-03 evidence](../evidence/SYS-03/README.md#sys-03-independent-review); they do not certify this revision. CI retains source-checks (3.13), OS compatibility (windows-latest) and OS compatibility (macos-15). Each installs the relevant CPU dependencies, prepares the pinned detector and runs behavior tests, both notebook inspect routes and the CPU inference CLI. Linux also runs Ruff. Linux/Colab staging tests are explicitly excluded on other OSs; those exceptions do not exclude inspect/inference.

<a id="dataset-configuration"></a>

### Dataset configuration

[project.toml](../configs/project.toml) resolves relative locations from the repository root. CLI `--data-root`, `--cache-dir` and `--config` options select existing storage or another configuration; they do not imply moving data. Use [storage guidance](../data/README.md) and keep secrets out of source/output.

<a id="execution-evidence-and-review-records"></a>

### Execution evidence and review records

[SYS-03](../evidence/SYS-03/README.md), [FAC-01](../evidence/FAC-01/README.md) and the [current liveness verification](../evidence/liveness/package-refactor/verification.json) identify inputs, versions, devices and limits. Assignment/status/reviewer confirmation remain in the [Project Tracker](../README.md#project-tracker). [Colab runtime notes](https://research.google.com/colaboratory/runtime-version-faq.html) and [official PyTorch wheel profiles](https://pytorch.org/get-started/previous-versions/) are background references, not evidence of a run on this revision.
