# Task evidence

The [Project Tracker](../README.md#project-tracker) is the single source for task status, assignments and reviewer confirmation. Repository files document implementation and recorded evidence. This folder contains selected supporting artifacts, not the group report.

| Task | Evidence |
| --- | --- |
| SYS-03 | [Setup, training/inference and independent review](SYS-03/README.md) |
| FAC-01 | [Loader evidence and protocol findings](FAC-01/README.md) |
| FAC-02 | [Face data audit, cleaning rule and P×K sampler](FAC-02/README.md) |
| FAC-04 | [Triplet-loss face embedding, identity check and two-image demo](FAC-04/README.md) |
| FAC-05 | [Comparison of both face approaches: ROC/AUC, selection and threshold](FAC-05/README.md) |
| LIV-03 | [Current data audit, loader checks and limits](LIV-03/README.md) |
| Liveness training and evaluation | [Selected model and final evidence](liveness/submission-artifacts.json) |

Only tasks with supporting files have folders. Subtasks share the parent folder; shared artifacts are linked rather than copied. Other tasks remain indexed in the Project Tracker.

## Selection and new runs

Selected evidence directly supports a task criterion, identifies its source/input/device, and records the result. Distinct platforms or meaningful failure/fix/retest pairs can justify additional artifacts. Routine repeats do not expand the selected set automatically.

New command outputs go below this [evidence directory](./), using ignored runs/Task-ID subdirectories. A selected run can later be moved into the parent task folder and linked from its README and the Project Tracker. Old reports retain their original source hashes; acceptance does not imply that later changes were retested.

## Scope

Only selected task evidence is included in this directory. Earlier troubleshooting runs and historical notes stay in the ignored runs subdirectory of this [evidence directory](./) or in existing external archives. Paths embedded in execution artifacts describe their original run locations.

## Liveness package refactor evidence

The [package-layout verification](liveness/package-refactor/verification.json) records the source inventory tested for that refactor, its nested source snapshot, controlled resume check and numerical comparisons. Its CPU checks do not represent a new GPU training result or verification of every later source edit. Selected model reports, score receipts and reviewed crop evidence retain their original producing-code attribution because they explain the frozen checkpoint and selection.

The release inventory selects the liveness evidence required by the current notebooks, reproduction commands and these checks. Git ignores other liveness outputs by default, including raw source copies, resume working files, logs, superseded camera metadata and per-clip caches. Those files remain local or on Drive. A current source snapshot is identified by its package-relative inventory; duplicate implementation trees and optimizer tensors are not required in Git.

## Source verification — 29 November 2026

These checks apply to working source based on commit 22a9217 with the submission fixes, using the existing Linux x86_64 Python 3.13.16 runtime. A separate source copy contained all tracked files plus the new [bundled-inference regression](../tests/test_submission_inference.py), with no Git metadata, raw datasets, caches, local profiles, attendance databases or full training checkpoints. The selected inference weights and detector were included.

| Check | Observed result |
| --- | --- |
| Ruff 0.17.6 lint and format | PASS; no blanket rule exclusions. Source-check formatting also covers the notebooks. |
| Complete Linux behavior suite, including bundled Face/liveness opt-in paths | PASS on the final separate source copy: 925 discovered, 912 passed, 13 skipped. That run preceded removal of the four root-level compatibility files. |
| Shared environment check, explicit CPU device | PASS, including finite optimizer update/inference and dependency consistency. The installed CUDA-capable wheels were retained; this execution used CPU. |
| Bundled inference entry points | PASS: shared canonical liveness classes through compatibility imports; default liveness CLI and YuNet no-face handling; default portable Emotion and UI adapter agreement; trained Face/Emotion generated-input integration. |
| Real face input | PASS for inference on one supplied verification trial, producing finite 512-dimensional embeddings and similarity/distance scores. This is a smoke check, not a new AUC evaluation. |
| Source-copy liveness notebooks, inspect action | PASS for both notebooks after refreshing distribution checksums. Only saved-evidence inspection ran; no training, public-test inference or native Colab execution is claimed. |
| Syntax and local documentation links | PASS for Python/TOML/notebook code syntax and the README/documentation index links and anchors. |

The 13 skips comprise eight native Tk checks and five optional historical-checkpoint checks. Current bundled Emotion inference ran; the legacy Emotion and two-checkpoint comparison skips do not mean the deployed emotion or liveness model was skipped. Native GUI interaction could not be exercised because the available process could not connect to the graphical display. Windows/macOS CI has not been rerun on these local changes. The prior [CI run](https://github.com/Quynh-Trang12/facial-attendance/actions/runs/46167404777) tested commit 22a9217 before these fixes.

No model weights, gallery thresholds, dataset partitions or original experiment results were changed. Developer Mode intentionally owns registration. Historical webcam/replay outcomes retain their original revision/input attribution and do not establish a failure in the current application. Full training reproduction, current physical webcam accuracy and separate report/video deliverables require their own evidence.

### Follow-up: remove duplicate root modules

The four root-level liveness files for binary_model, face_detector, inference and temporal_liveness were removed, together with the compatibility-only test. UI, attendance and CLI import the canonical packages directly. No current source, script, test or notebook imports the removed module paths; historical source-transition records retain their original attribution.

Verification after removal: the complete Linux suite passed with 924 discovered tests, 911 passed and 13 skipped. Ruff lint/format, release fingerprints and saved liveness notebook inspection passed. Native GUI and other-OS limits above remain unchanged.
