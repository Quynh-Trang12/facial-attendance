# SYS-05 execution evidence — 16 November 2026

Read [the owner self-review](../../../docs/sys-05-owner-self-review.md) first for
acceptance status, scope, limitations and outstanding reviewer/handover actions.
The files here are genuine outputs or explicitly attributed source records.
No webcam image, recording, real identity or attendance/enrollment record is stored.

| File | What it establishes |
| --- | --- |
| [full-tests-clean-runtime.json](full-tests-clean-runtime.json), [log](full-tests-clean-runtime.log) | Current working source: 131 passed, 0 failed, 0 skipped in a new venv with only three GUI dependencies. |
| [isolated-source-tests.json](isolated-source-tests.json), [log](isolated-source-tests.log) | Separate source snapshot, without .git or inherited working files: the same 131 tests passed; protected review JSON excluded. |
| [native-tk-controlled.json](native-tk-controlled.json) | Actual Tk widgets, generated camera source, 14 controlled cases PASS; no callback exceptions; all workers stopped. This is not a real-face accuracy test. |
| [isolated-native-tk.json](isolated-native-tk.json) | Native Tk startup/flow repeated successfully from the isolated source and fresh runtime, 14 cases PASS. |
| [controlled-demo.json](controlled-demo.json) | Four executed scenarios with generated distinct BGR channels, crop shape/source sharing, original identifiers, envelope, false save eligibility and stale rejection. Supplied regions are explicit developer fixtures. |
| [owner-physical-validation.json](owner-physical-validation.json) | Owner's eight macOS/YuNet physical PASS reports and exact movement observation, with runtime hashes captured in this session. Unreported hardware/timing metadata remain null; not reviewer evidence. |
| [runtime.json](runtime.json) | Actual fresh environment versions and detector/model licence hashes. Torch absent; no shared-environment replacement. |
| [benchmark.json](benchmark.json) | Executed synthetic CPU benchmark. No physical webcam FPS, native Tk throughput or glasses accuracy claim. |
| [remote-interface-audit.json](remote-interface-audit.json) | Current remote branch heads and source blob IDs read through GitHub; no team branch merge or other-owner modification. |
| [tracker-readonly-snapshot.json](tracker-readonly-snapshot.json) | Criteria and existing Owner Evidence from the specified Sheet ranges, read only. No Sheet update or reviewer confirmation is inferred. |
| `candidate-manifest.json` | Final review snapshot identifier, copied-file hashes and protected-file exclusion. It is local, uncommitted and not a reviewed/frozen release. |

Historical Windows/macOS results remain in the earlier documents and original
environment record. They were not overwritten or relabelled for this revision.
The protected `reports/environment/vy-mac-m1-review.json` is outside this evidence
directory and was not changed, staged, deleted or copied into the review package.

## Reproduce

Use [the handover guide](../../../docs/sys-05-handover.md) for platform commands.
The full test command remains `PYTHONPATH=src <python> -m unittest discover -s tests -v`.
The runtime and each report state the interpreter/source scope actually used.
Native Tk checks require desktop access. Physical glasses/lighting observations
require an owner/reviewer, not a generated test image. The demo machine/reviewer
must record failures or OWNER VALIDATION REQUIRED instead of inventing PASS.
