# Task evidence

The [Project Tracker](../README.md#project-tracker) is the single source for task status, assignments and reviewer confirmation. Repository files document implementation and recorded evidence. This folder contains selected supporting artifacts, not the group report.

| Task | Evidence |
| --- | --- |
| SYS-03 | [Setup, training/inference and independent review](SYS-03/README.md) |
| FAC-01 | [Loader evidence and protocol findings](FAC-01/README.md) |
| FAC-02 | [Face data audit, cleaning rule and P×K sampler](FAC-02/README.md) |

Only tasks with supporting files have folders. Subtasks share the parent folder; shared artifacts are linked rather than copied. Other tasks remain indexed in the Project Tracker.

## Selection and new runs

Selected evidence directly supports a task criterion, identifies its source/input/device, and records the result. Distinct platforms or meaningful failure/fix/retest pairs can justify additional artifacts. Routine repeats do not expand the selected set automatically.

New command outputs go under ignored `evidence/runs/<Task ID>/`. A selected run can later be moved into the parent task folder and linked from its README and the Project Tracker. Old reports retain their original source hashes; acceptance does not imply that later changes were retested.

## Scope

Only selected task evidence is included in this directory. Earlier troubleshooting runs and historical notes are stored outside the repository. Paths embedded in execution artifacts describe their original run locations.
