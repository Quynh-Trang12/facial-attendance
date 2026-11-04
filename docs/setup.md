# Runtime setup and verification

The owner reference environment is Python 3.13.16 with PyTorch 2.13.0+cu126 on Linux/NVIDIA. It stays unchanged at `~/.local/share/ml-frameworks/.venv`. The team uses Python 3.13.x, with explicit CPU or CUDA execution. Other 3.13 patch releases, such as 3.13.18, are accepted; no patch-version test matrix is required. `.python-version` is the owner's development pin; it is not a command to replace another machine's interpreter.

## Choose the operating system and runtime first

Use Python **3.13.x** for the team's common setup. CI reads **3.13.16** from `.python-version`; do not downgrade a working 3.13 environment just to match that patch. An existing compatible environment can be reused. The owner's Linux GPU environment must not be reinstalled. These instructions define supported targets; wheel availability and dependency resolution are not the same as a successful native execution test.

| Target | Installation | Execution | Verification status |
| --- | --- | --- | --- |
| Linux x86_64 | One CPU/CUDA profile, then shared requirements | `cpu` or NVIDIA `cuda` | Executed locally; Colab evidence is separate |
| Windows x64 | One CPU/CUDA profile, then shared requirements | `cpu` or NVIDIA `cuda` | CPU dependency resolution passed; native CPU CI configured, pending execution |
| macOS 14+ on Apple Silicon | Shared requirements from PyPI; native arm64 Python | Explicit `cpu` | Dependency resolution passed; native CPU CI configured, pending execution |
| macOS Intel, or Apple Silicon using Rosetta/x86_64 Python | No supported install for the current pins | Unsupported locally | Torch 2.13 has no Intel macOS wheel; requires a separately selected/tested stack |
| Windows ARM64 / other architectures | Not covered by this setup | Unsupported target | Do not assume that availability of a Torch wheel covers every dependency |

The macOS minimum comes from the pinned [Torch release wheels](https://pypi.org/project/torch/2.13.0/#files). It is not a requirement to match the owner's OS or Python patch. On macOS, check **Apple menu → About This Mac** for the chip and OS version. The interpreter's architecture matters too: an Apple Silicon computer running x86_64 Python through Rosetta cannot use arm64 wheels.

`requirements.txt` pins the nine shared direct dependencies. `requirements-cpu.txt` selects official CPU wheels for Linux/Windows x64; `requirements-cuda.txt` selects CUDA 12.6 wheels for Linux/Windows NVIDIA. `requirements.lock` is only the owner's 58-package Linux/CUDA snapshot. Neither CUDA nor the strict owner snapshot is a macOS setup profile. MPS acceleration is outside the currently implemented/verified device modes.

### Windows: PowerShell

Open PowerShell in the repository root. If you already have a compatible environment, use its Python executable and skip environment creation. Otherwise, with Python 3.13 already installed:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-cpu.txt
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe scripts/check_environment.py --device cpu --operator windows-review --output reports/environment/windows-review.json
```

Directly invoking the executable avoids PowerShell activation/execution-policy issues. For an intentional NVIDIA run, use `requirements-cuda.txt` instead of the CPU profile, then explicitly use `--device cuda` and a new output filename. The GPU driver must support CUDA 12.6; Windows CPU compatibility does not establish CUDA compatibility on that machine. Do not run the owner's `.sh` launcher or copy `~/.local/share/ml-frameworks/...` paths onto Windows.

To run the source/loader tests in PowerShell:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

### macOS: Terminal on Apple Silicon, macOS 14+

Open Terminal in the repository root. Check the OS and Python architecture first:

```bash
sw_vers -productVersion
python3.13 -c "import platform; print(platform.machine())"
```

The architecture must be `arm64`. Use a native arm64 Python 3.13 installation; do not run an Intel interpreter under Rosetta for this dependency set. Reuse a compatible environment if present; otherwise:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python scripts/check_environment.py --device cpu --operator macos-review --output reports/environment/macos-review.json
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v
```

Install from PyPI on macOS; do not use CUDA wheels, `--device cuda`, or `--strict-owner`. CPU mode is deliberate, not an automatic fallback. An Intel Mac cannot install the pinned Torch stack natively. The previously verified Colab setup is an available alternative if the member chooses it; no runtime is switched silently and no untested older dependency stack is substituted.

### Linux or Colab

For a new Linux x86_64 environment, install one wheel profile, then shared requirements:

```bash
python -m pip install -r requirements-cpu.txt
python -m pip install -r requirements.txt
```

For NVIDIA, replace the first profile with `requirements-cuda.txt` and explicitly request `--device cuda`. On the owner's environment, install nothing. Colab has its own complete sequence below.

### Review failures without guessing the cause

Use a new output filename each time. The environment check records `os`, `architecture`, `macos_version`, Python and either the actual computation result or a failed-check traceback. It runs platform checks before importing the ML stack, so missing dependencies and unsupported OS/architecture can still produce evidence. If pip installation fails, retain its terminal output too; the environment script does not capture an earlier pip command.

Ask the reviewer to share the generated JSON and the failed command/output, rather than interpret the error themselves. A generic report of an OS mismatch does not identify whether the failing layer was a shell command, Python architecture, wheel availability, native library, or driver. Repeated installation is not the default remedy.

## Owner GPU check

Use the existing host launcher, which verifies TensorFlow/PyTorch coexistence before the project check:

```bash
~/.local/share/ml-frameworks/run_gpu.sh scripts/check_environment.py \
  --device cuda --strict-owner --operator owner-host \
  --output reports/environment/new-owner-check.json
```

`--strict-owner` also checks Python 3.13.16 and every exact owner-lock version. Without it, the check uses the supported Python range and shared direct package versions. Requested CUDA fails when inaccessible; it never silently becomes CPU.

## Explicit CPU check

A CUDA-built PyTorch can also execute CPU operations. The owner can test that path without installing or replacing anything:

```bash
~/.local/share/ml-frameworks/.venv/bin/python scripts/check_environment.py \
  --device cpu --operator cpu-check \
  --output reports/environment/new-cpu-check.json
```

Both devices run one optimizer step on four synthetic images, verify finite gradients and changed weights, then run finite `[4,2]` inference. This is a small infrastructure check, not dataset training or model accuracy. The JSON identifies the actual execution device separately from the installed CUDA build. Use a new output file each time.

## Google Colab: CPU or NVIDIA GPU

Follow these cells in order in one notebook session. They run repository scripts in a separate `.venv` using Colab's existing Python; they do not replace Colab's stock Torch/TensorFlow packages. The notebook itself continues to use the stock kernel, so run project commands through the `python` path defined below. Use the Python 3.13 interpreter supplied by Colab; a different patch release from the owner's 3.13.16 is acceptable. Do not install another Python just to match the patch.

### 1. Put the repository in one directory and select the device

Upload/extract the repository to `/content/facial-attendance`. It is ready when `/content/facial-attendance/scripts/check_environment.py` exists. In Colab choose **Runtime → Change runtime type** and select CPU or an NVIDIA GPU before proceeding. GPU availability depends on the account/runtime allocation; code does not silently switch to CPU.

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
reports = repo / 'reports/compatibility' / f'colab-{device}-{run_id}'
reports.mkdir(parents=True, exist_ok=False)
python = repo / '.venv/bin/python'
print('Kernel:', sys.version.split()[0], '| device:', device)
print('Project:', repo, '| reports:', reports)
```

Keep a CPU environment and a CUDA environment in different Colab sessions. Do not install both wheel profiles into the same `.venv`.

### 2. Prepare a command runner and the isolated environment

The runner saves each command's full output, prints a short stage result and raises on failure. `capture_output=True` also works under notebook output capture. `MPLBACKEND=Agg` avoids inheriting Colab's inline display backend into a subprocess that does not install notebook plugins; it affects these noninteractive checks only.

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
    (reports / f'{stage}.log').write_text(result.stdout + result.stderr)
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

The CUDA profile uses CUDA 12.6 wheels and needs a compatible NVIDIA driver. If installation or CUDA checks fail, inspect the stage log rather than changing devices or package versions silently. Do not use `--strict-owner` or install `requirements.lock` on Colab.

### 4. Run tests and the selected-device check

```python
run('unit-tests', [python, '-m', 'unittest', 'discover', '-s', 'tests', '-v'])
run('environment', [python, 'scripts/check_environment.py', '--device', device,
                    '--operator', 'colab-review',
                    '--output', reports / 'environment.json'])
```

Expected: 46 tests pass; environment JSON has `status=PASS`, the requested device, finite gradients, changed weights, finite inference and a successful `pip check`. This step uses a tiny synthetic tensor batch to check infrastructure, not face-model accuracy.

### 5. Run the loader on real competition data

Configure your own Kaggle token using [the data-loading instructions](data-loading.md#download-on-google-colab), then run:

```python
run('loader', [python, 'scripts/check_face_data.py', '--download',
               '--output', reports / 'loader.json'])
```

This downloads/reuses the assigned competition and checks its actual data. To use files you already uploaded, extracted or mounted, replace `--download` with `--data-root`, followed by their root path, for example `repo / 'data/colab-sample'`. That folder must have the [required layout](../data/README.md); the example subset is not shipped with the repository. Loader decoding/collation runs on CPU even when the model runs on GPU. The environment JSON's `face_dataset_present` describes the configured default path at the time it ran; an explicit `--data-root` can point elsewhere.

Expected full-dataset counts: 380,638 train images, 8,000 validation images, 8,000 classification test images, 4,000 identities, and 8,805 official trials. A deliberately uploaded subset will have smaller counts and must be labelled as such. `status=PASS` checks mappings, finite decoded batches, pair labels/paths and seeded training-pair rules; it does not decode every image or measure recognition accuracy.

Download the report directory before ending the session, then disconnect and delete the runtime in Colab. Runtime files are temporary; keep credentials and raw images out of Git.

### Recorded remote evidence

- [Earlier CPU fixture check](../reports/environment/colab-cpu-retest-2026-11-04.json): 39 tests and synthetic optimizer/inference passed on Python 3.13.18. Generated-image loader inputs were explicitly labelled as fixtures.
- [First CPU failure](../reports/environment/colab-cpu-first-failure-2026-11-04.json): inherited inline Matplotlib backend failed in the isolated subprocess. The runner above explicitly sets Agg to resolve it.
- Current real-image CPU/GPU results are indexed in [verification evidence](../reports/README.md). Image inputs are a byte-verified subset of the required competition; local acquisition/full-layout inspection is recorded separately. Webcam/GUI behavior, model training, ROC/AUC and full image/duplicate/leakage audits remain outside this infrastructure check.

See [Google's runtime documentation](https://research.google.com/colaboratory/runtime-version-faq.html) for runtime changes and [official PyTorch wheel profiles](https://pytorch.org/get-started/previous-versions/) for installation background.

## Dataset configuration

`configs/project.toml` contains both `kaggle_cache` and `face_data`. The defaults now match the actual downloaded competition under `data/.kagglehub/competitions/11-785-fall-20-homework-2-part-2`. Both dataset CLI branches read this config; `--data-root`, `--cache-dir` and `--config` are explicit overrides for external disks or Colab mounts.

Dataset decoding/collation happens on CPU and does not require GPU access. In Colab, put the modern token in a private Kaggle token file or its supported environment/secret mechanism, then use [the data-loading commands](data-loading.md). Do not put tokens in notebook output or source control.

## Current evidence and remaining review

- Current Python 3.13-only policy: [CPU runtime report](../reports/environment/cpu-python313-2026-11-04.json) and [46-test log](../reports/environment/python313-tests-2026-11-04.log).

- [Historical platform dependency resolution](../reports/environment/platform-resolution-2026-11-04.json): includes the now-retired Python 3.12 target alongside 3.13. The Python 3.13 Windows x64/macOS arm64 resolutions passed; this is not native execution and does not require members to repeat resolution tests.
- Earlier OS-policy execution: [Linux CPU](../reports/environment/cpu-os-policy-2026-11-04.json), [Linux CUDA](../reports/environment/cuda-os-policy-2026-11-04.json), and [46-test log](../reports/environment/os-policy-tests-2026-11-04.log).

- [Configured default dataset check](../reports/data/config-default-audit-2026-11-04.json): actual data loaded without `--data-root`, from a different working directory.
- [Fresh CPU-only installation](../reports/environment/cpu-wheel-clean-install-2026-11-04.json): project requirements installed into a separate temporary environment with actual CPU wheels.
- Colab real-image checks: [CPU](../reports/environment/colab-cpu-real-2026-11-04.json), Python 3.13.18 / CPU wheels; [Tesla T4 CUDA](../reports/environment/colab-cuda-real-2026-11-04.json), Python 3.13.16 / CUDA 12.6 wheels. Both checked the same byte-verified 56-image subset and a real-image optimizer/inference batch.
- Documented-cell execution: [CPU](../reports/environment/colab-cpu-guide-review-2026-11-04.json) and [CUDA](../reports/environment/colab-cuda-guide-review-2026-11-04.json), five cells executed in real IPython runtimes with the documented existing-data override. The API/Colab-secret branch was documented but not executed remotely.
- [CPU/IPython check](../reports/environment/cpu-notebook-audit-2026-11-04.json): optimizer/inference execution in a captured notebook context.
- [Strict owner CUDA check](../reports/environment/cuda-strict-audit-2026-11-04.json): host GPU preflight and current owner snapshot verified.
- Earlier environment/Keras and data runs remain historical evidence at their recorded source hashes. They are not silently relabelled as runs of current code.

The source/configuration suite contains 46 tests. All CI jobs read `.python-version` and run **3.13.16 only**. The Linux check retains its existing `source-checks (3.13)` name. Windows x64 and macOS Apple Silicon CPU jobs cover the team's actual operating systems, not extra Python versions. Native Windows/macOS execution is pending until the owner publishes the workflow and GitHub runs it.

If the branch ruleset still requires `source-checks (3.12)`, remove that obsolete requirement under Settings → Rules → Rulesets → main-pr-and-ci → required status checks. Keep `source-checks (3.13)`. After successful native runs, the Windows/macOS check names remain `OS compatibility (windows-latest)` and `OS compatibility (macos-15)`.

For SYS-03 review, Vy runs the documented check once on his own OS and chosen device, then records the result. No member is asked to install another Python patch, test every OS or rerun both CPU/CUDA profiles. CI handles the operating-system checks.

SYS-03 reviewer reproduction remains a human task for Vy. FAC-01.2 loader evidence is available for Que An's independent review. Keep findings and retests under new output filenames; do not replace original results or mark an unrun review as passed. All Git/GitHub publication is performed by the human owner.

Primary wheel reference: [official PyTorch CPU/CUDA installation commands](https://pytorch.org/get-started/previous-versions/).

## Colab CLI on this machine

The official [google-colab-cli](https://github.com/googlecolab/google-colab-cli) version 0.7.5 is installed user-wide at `~/.local/bin/colab` through `uv tool`, separate from ML libraries. The earlier [source-only transfer](../reports/environment/colab-source-only-input-2026-11-04.json) used generated fixtures. After separate explicit authorization, the same 56 original competition images and project files were uploaded to CPU and T4 runtimes under `/content/facial-attendance/`; the [real-image manifest](../reports/data/colab-real-input-2026-11-04.json) records individual image hashes and the archive hash. Inputs were stored in `data/colab-sample/`, the isolated runtime in `.venv/`, and results in `reports/compatibility/`. No Kaggle credentials, Git metadata or workspace instructions were uploaded. Both test runtimes were stopped after retrieving the evidence.
