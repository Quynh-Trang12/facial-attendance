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

New command outputs go under ignored `evidence/runs/<Task ID>/`. A selected run can later be moved into the parent task folder and linked from its README and the Project Tracker. Old reports retain their original source hashes; acceptance does not imply that later changes were retested.

## Scope

Only selected task evidence is included in this directory. Earlier troubleshooting runs and historical notes stay in the ignored runs subdirectory of this [evidence directory](./) or in existing external archives. Paths embedded in execution artifacts describe their original run locations.

## Current liveness package evidence

The [package-layout verification](liveness/package-refactor/verification.json) records the current source inventory, nested source snapshot, controlled resume check and numerical comparisons after the models/datasets/training/experiments refactor. Its CPU checks do not represent a new GPU training result. Selected model reports, score receipts and reviewed crop evidence retain their original producing-code attribution because they explain the frozen checkpoint and selection.

The release inventory selects the liveness evidence required by the current notebooks, reproduction commands and these checks. Git ignores other liveness outputs by default, including raw source copies, resume working files, logs, superseded camera metadata and per-clip caches. Those files remain local or on Drive. A current source snapshot is identified by its package-relative inventory; duplicate implementation trees and optimizer tensors are not required in Git.
