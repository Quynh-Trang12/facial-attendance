# SYS-03 — Development environment

Task requirements, assignments and review records: [Project Tracker](../../README.md#project-tracker), Requirements and WBS tabs, task `SYS-03` and its subtasks.

## Selected evidence

| Criteria | Evidence | Scope |
| --- | --- | --- |
| SYS-03.1: versions/runtime | [Dependency pins](../../requirements.txt), [Python reference](../../.python-version), [setup](../../docs/setup.md) | Team Python 3.13.x; explicit CPU/CUDA selection |
| SYS-03.2: training and inference | [Linux CPU report](cpu-python313-2026-11-04.json), [46-test log](python313-tests-2026-11-04.log) | Recorded versions, loss, finite gradients, updated weights and finite `[4,2]` inference |
| SYS-03.2: NVIDIA execution | [Linux CUDA report](cuda-os-policy-2026-11-04.json) | Actual model/batch execution on `cuda:0`; archived snapshot checks reflect the implementation at that time |
| SYS-03.1–2: CPU installation | [Clean CPU-wheel report](cpu-wheel-clean-install-2026-11-04.json), [log](cpu-wheel-clean-install-2026-11-04.log) | Separate clean installation using CPU wheels |
| SYS-03.2: Colab compatibility | [CPU report](colab-cpu-real-2026-11-04.json) / [tests](colab-cpu-real-tests-2026-11-04.log); [T4 report](colab-cuda-real-2026-11-04.json) / [tests](colab-cuda-real-tests-2026-11-04.log) | Same byte-verified 56-image subset; 39 tests per runtime at the recorded revision |
| SYS-03.3–4: independent review/retest | [Project Tracker](../../README.md#project-tracker), SYS-03.3–SYS-03.4 review records | Windows 11 and Mac M1; Python 3.13.17; 46/46 tests and CPU training/inference passed |
| SYS-03.5: handover | [README](../../README.md), [setup guide](../../docs/setup.md), [test script](../../scripts/check_environment.py) | Shared setup and run instructions |

The Colab reports share the [input manifest](../FAC-01/colab-real-input-2026-11-04.json); it is stored once under FAC-01.

## SYS-03 independent review

Vy recorded successful independent installation and retesting on Windows 11 and Mac M1 with Python 3.13.17: 46/46 unit tests and CPU training/inference passed on both systems. Findings covered a Windows TOML path issue fixed before retesting and an installation path-length issue resolved by using a shorter folder path.

The Project Tracker does not identify the tested commit hash or review date. This reviewer record does not establish Windows CUDA execution or a GitHub CI run. The review record applies to the tested setup revision.

## Historical scope

All selected execution files are original 4 November 2026 results. Their embedded source hashes and version policies identify what ran; later source changes are outside those recorded runs. Additional troubleshooting history is stored outside the repository.
