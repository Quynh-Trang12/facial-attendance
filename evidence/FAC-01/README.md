# FAC-01 — Face data protocol and loaders

Task requirements, assignments and review records: [Project Tracker](../../README.md#project-tracker), Requirements and WBS tabs, task `FAC-01` and its subtasks.

## Selected evidence for FAC-01.2

| Criterion | Evidence | Scope |
| --- | --- | --- |
| Split/pair interface and sampling design | [Loader contract](../../docs/data-loading.md#loader-and-sampling-contract), [implementation](../../src/facial_attendance/recognition/data_loader.py) | Train-only identity mapping, official trial order/labels and seeded identity-correct training pairs |
| Sample paths, identities and labels | [Configured-dataset report](config-default-audit-2026-11-04.json), [log](config-default-audit-2026-11-04.log) | Actual split counts, identity preview, trial preview and six sampled training pairs; invoked from another working directory |
| Assigned-data acquisition and inspection | [Authenticated download report](kaggle-authenticated-2026-11-04.json), [log](kaggle-authenticated-2026-11-04.log) | Real competition download and first-batch loader checks, not generated fixtures |
| Colab subset provenance | [Input manifest](colab-real-input-2026-11-04.json) | 56 original-image hashes, selected official trials and archive hash; execution reports are shared from [SYS-03](../SYS-03/README.md) |
| Independent review | [Project Tracker](../../README.md#project-tracker), FAC-01.2 review record | Review at commit `c4e4812`; 46/46 tests; no loader defect found in the reviewed contract |

## Review scope and findings

The FAC-01.2 review at commit `c4e4812` recorded 46/46 passing tests and no loader defect in the reviewed contract. It also identified that the real-data PASS reports predate that committed loader; full-data rerun, duplicate and leakage checks were tracked separately under FAC-01.3.

The FAC-01.4 evidence entry records 8,396 training images identical to verification images and 929/8,805 affected trials. This finding is separate from the earlier loader checks. Decisions, follow-up and reviewer confirmation are maintained in the Project Tracker.

The Project Tracker links a [leakage note](https://github.com/Quynh-Trang12/facial-attendance/blob/f92eb649c3aaaaf7cea16e79f9403cb2dd915ba0/docs/face-data-leakage.md) and [Colab audit](https://github.com/Quynh-Trang12/facial-attendance/blob/f92eb649c3aaaaf7cea16e79f9403cb2dd915ba0/reports/data/colab-audit-2026-11-05.json) at another revision. These external evidence files are stored at the linked revision, outside this checkout.

The selected local artifacts remain original 4 November 2026 observations. They validate acquisition and inspected batches; they do not establish full-image integrity, absence of leakage or model accuracy. Historical fixtures, acquisition failures and duplicate local replays are stored outside the repository.
