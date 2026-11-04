# Development environment

## Runtime

The validated owner runtime is Python **3.13.16**, PyTorch **2.13.0+cu126**, torchvision **0.28.0+cu126**, Linux x86_64 and an NVIDIA RTX 3050 6GB Laptop GPU. TensorFlow 2.21.0 also exists in the shared lab environment; the application currently uses PyTorch and does not add TensorFlow as a project dependency. The full shared runtime's coexistence checks run through its existing launcher.

`requirements.txt` contains direct dependencies. `requirements.lock` contains their 58-package installed dependency closure for this Linux/Python/CUDA combination, including KaggleHub 1.0.2 and transitive CUDA libraries. Only KaggleHub was upgraded, with authorization; Python and the ML framework packages were preserved. It was derived from installed distribution metadata and checked against their active dependency requirements. It is a snapshot, not evidence of a successful clean install on another machine. Keep direct pins and the snapshot consistent when a deliberate change is reviewed; do not copy the entire unrelated lab environment into this project.

OpenCV supplies image/webcam support. Its GUI display functions require a desktop session and OS graphics libraries. Do not install another OpenCV variant in the same environment. The application GUI toolkit is still a SYS-02/UI decision; no unused UI framework has been added. CUDA training does not require an open webcam or desktop window.

## Existing owner environment

Reuse `~/.local/share/ml-frameworks/.venv`, kernel `ml-frameworks-gpu` (**Python 3.13 (TensorFlow + PyTorch GPU)**). Do not run the fresh-install procedure against this environment.

Read its README and use the existing host launcher from the repository root:

```bash
~/.local/share/ml-frameworks/run_gpu.sh scripts/check_environment.py \
  --operator "owner-host" \
  --output reports/environment/owner-check.json
```

The launcher tests both frameworks on the GPU in both import orders before running the project check. Restricted execution environments can hide NVIDIA devices; a restricted-process failure is not sufficient evidence of a broken host driver. Run the check from a host terminal with GPU access. The shared runtime has its own native compatibility hook for TensorFlow/PyTorch coexistence; this project does not recreate or modify it.

## Another developer's setup

First inspect and reuse a compatible Python 3.13 GPU environment if one already exists. Only a developer who needs a fresh environment should create one on their own machine. For the recorded **Linux x86_64 / Python 3.13.16 / NVIDIA CUDA 12.6** profile:

```bash
python3.13 --version
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock \
  --extra-index-url https://download.pytorch.org/whl/cu126
.venv/bin/python -m pip check
.venv/bin/python scripts/check_environment.py \
  --operator "Vy / independent execution" \
  --output reports/environment/vy-check.json
```

The interpreter patch must match `.python-version`. Access to the package indexes and an NVIDIA driver compatible with the wheel are required. The wheel supplies CUDA user-space dependencies; a separate local CUDA toolkit is not needed just to execute it. OpenCV may need OS graphics libraries such as `libGL.so.1`. Do not change the OS Python, GPU driver or owner's shared environment as a routine setup step.

Windows, macOS and different Python/CUDA profiles are not certified by this Linux snapshot. If a teammate uses another profile, record the actual differences and validate it as a separate review outcome before changing the agreed environment.

**If GPU access is unavailable:** stop the GPU workload and record the failing command and error. The proposed alternative is a teammate's compatible NVIDIA GPU host, subject to the user's/group's confirmation and the same check. Do not automatically switch runtime, launch paid cloud resources, install drivers or use CPU as a successful replacement. No alternate host has been independently approved/tested in this task.

## Check budget and behavior

This is an infrastructure check with one Adam update on four synthetic RGB tensors of shape `[4,3,32,32]`, then one inference pass. Expected project-check cost is seconds to about a minute after imports; the existing two-framework host preflight adds startup time. There is no architecture search, epoch training, dataset split, checkpoint selection or final accuracy comparison at this stage.

The script imports the pinned dependencies, checks the Linux dependency snapshot and storage configuration, requires model/data/labels on CUDA, and verifies finite loss, finite gradients and an actual parameter update. It then checks `eval()`/inference mode and finite GPU output with shape `[4,2]`, plus image conversion, a toy ROC/AUC API call and `pip check`.

Use a new output path per run. JSON contains timestamps, self-reported operator, source/config/dependency SHA-256 values and observed results. Existing evidence is never overwritten. FAIL exits nonzero and records a traceback when the script can start; launcher failures are retained in the host log. No result claims project AUC or webcam/GUI functionality.

## Evidence and review

The final source-checkout entry point was executed on **4 November 2026**. The [owner JSON result](../reports/environment/2026-11-04-owner/result.json) and [technical host log](../reports/environment/2026-11-04-owner/host.log) record **PASS**, exit code 0, all 49 locked distributions matched, finite loss `0.6945785284042358`, finite gradients and four updated parameter tensors. Inference returned finite `[4,2]` output on `cuda:0`, with gradients disabled. `pip check` found no broken requirements. The host preflight also passed both framework import orders. The seven source/configuration tests passed locally; the GitHub workflow is configured but has not run remotely.

This run verifies the owner host only; Vy's independent reproduction remains pending. Publication copies omit operator-identifying metadata. Original unredacted records are retained locally outside the repository publication folder; technical observations and source hashes are unchanged.

The saved SYS-03 run predates the FAC-01.2 additions. Its 49-package count and source hashes describe that earlier revision, not the current expanded repo. Current loader tests and acquisition evidence are documented in [data loading](data-loading.md); no new GPU result is inferred from a loader test.

A [post-upgrade compatibility run](../reports/environment/2026-11-04-kagglehub-upgrade/result.json) on 4 November passed the current 58-package project check and both host framework preflights. Standalone Keras 3.15.1 with TensorFlow 2.21.0 on Python 3.13.16 also completed a finite-loss GPU optimizer update (four parameter tensors changed) and finite `[4,2]` inference on `/GPU:0`. The exact [probe source](../reports/environment/2026-11-04-kagglehub-upgrade/probe.py) is preserved with its hash in the result. This establishes the tested configuration, not compatibility of every third-party model or future upgrade.

| Checkpoint | Owner deliverable | Remaining acceptance condition |
| --- | --- | --- |
| SYS-03.1 | Python pin, direct dependencies, Linux snapshot, runtime and fallback procedure | Vy confirms compatibility and group runtime agreement |
| SYS-03.2 | Executable CUDA training/inference check and owner logs/results | Vy reviews the observations |
| SYS-03.3 | Reproduction commands and evidence schema | Vy independently installs/runs and reports PASS or reproducible errors |
| SYS-03.4 | Error handling, retained diagnostics and a defined retest command | Resolve actual review findings; if none, record no fix needed after review |
| SYS-03.5 | Shared repo layout, setup commands and runtime limitations | Human publication and final reviewer walkthrough |

For independent review, Vy must run the check, retain `vy-check.json` and record the reviewed source/commit, commands, host/device, exit code and findings. Reading or copying the owner log is insufficient. Operator text is self-reported; the script cannot authenticate a reviewer or confirm the Sheets checkbox. If review exposes an error, retain the failing evidence, fix the cause and produce a new retest file; do not invent findings or mark an unrun retest as passed.

After the human owner publishes the repo, link actual GitHub file permalinks for setup/dependencies, the check and owner evidence in Sheets. Link Vy's actual run and review separately. Independent review is still pending; SYS-03 cannot be marked fully closed yet.
