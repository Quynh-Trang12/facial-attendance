# SYS-05 owner final self-review — 16 November 2026

Owner: Vỹ. Independent reviewer: Trang. Repository:
`Quynh-Trang12/facial-attendance`. Branch: `feature/sys-05-webcam-gui-stubs`.

**Owner implementation, compatibility audit, regression verification and review
preparation are implemented. The owner reports the required macOS physical checks
passing. Independent review, the reviewer demonstration, actual handover delivery
and final release freeze remain unconfirmed.** This report does not close those
external acceptance actions or claim that the eleven tracker requirements are all
complete.

Candidate **SYS05-LOCAL-20261116-c00c004fc23a** identifies the UI runtime digest
`c00c004fc23ad4c4d1bec41c134163ebb30b62bb4b88b0ec3dcab720cf3d5268`.
Baseline remains `39ac3dfd3c841dd394f96f1ef43aa37ff46bdcac`; all redesign changes are
local and uncommitted. The runtime files hashed in the physical-validation record
have not been changed after that owner report. No stage, commit, push, merge or
publication has occurred.

## Completion labels and acceptance matrix

- **IMPLEMENTED** describes source/documents that exist.
- **AUTOMATED TEST PASS** describes the actual test or controlled native execution.
- **PHYSICAL TEST PASS** below is the owner's explicit macOS/YuNet report, not an
  independent reviewer result or quantified accuracy/FPS measurement.
- **OWNER VALIDATION REQUIRED** applies only to unexecuted platform/scene checks.
- **READY FOR INDEPENDENT REVIEW** means the identified local package can be reviewed;
  it does not mean review or a frozen release exists.
- **REVIEWER CONFIRMATION PENDING** preserves Trang's independent responsibility.

Sources read before the audit: [SYS-02 v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d),
[UI-02](https://app.notion.com/p/412863094ebe80c09ffce13f894f16ae),
[Requirements 39–49](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit#gid=2016101501&range=A39:S49),
and [WBS 41–51](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit#gid=827743122&range=A41:AM51).
No `PROJECT_CONTEXT.md` or applicable `AGENTS.md` was found. The Sheet was read
only. Exact criteria and prior owner text are preserved in
[`tracker-readonly-snapshot.json`](../reports/sys05/2026-11-16/tracker-readonly-snapshot.json).

| Task | Acceptance scope | Owner-side result / artifact | Remaining acceptance |
| --- | --- | --- | --- |
| SYS-05 | Runnable webcam/crop/GUI with replaceable, labelled stubs; show reviewer | IMPLEMENTED; AUTOMATED TEST PASS; PHYSICAL TEST PASS. Automatic preview, four fixtures and no genuine writes. | READY FOR INDEPENDENT REVIEW; REVIEWER CONFIRMATION PENDING; show revised build to Trang. |
| SYS-05.1 | Define flow/interfaces and agree sample outputs | IMPLEMENTED; AUTOMATED TEST PASS. Request/result dataclasses and runnable four-scenario walkthrough. | REVIEWER CONFIRMATION PENDING for sample-output agreement. |
| SYS-05.2 | Connect camera, stubs and inspectable GUI outputs | IMPLEMENTED; AUTOMATED TEST PASS; PHYSICAL TEST PASS on macOS/YuNet. No dropdown/freeze/evaluate requirement. | REVIEWER CONFIRMATION PENDING. |
| SYS-05.3 | Compare against SYS-02 and record reproducible findings | IMPLEMENTED; AUTOMATED TEST PASS. [Requirement-to-code-to-test traceability](sys-05-contract-traceability.md). | Owner-side audit complete; REVIEWER CONFIRMATION PENDING for Trang's separate run. |
| SYS-05.4 | Correct confirmed crop/camera/display findings | IMPLEMENTED; AUTOMATED TEST PASS; PHYSICAL TEST PASS for reported owner findings. Fix register below. | REVIEWER CONFIRMATION PENDING for affected-flow checks; slight outline hesitation retained. |
| SYS-05.5 | Rerun corrected flow and no-face/camera failures; assess replaceability | AUTOMATED TEST PASS for 131 tests and 14 native cases; PHYSICAL TEST PASS for owner clearing/recovery. | REVIEWER CONFIRMATION PENDING for independent retest and adapter assessment. |
| SYS-05.6 | Smoke intended demo runtime, document limitations | AUTOMATED TEST PASS in native macOS Tk; PHYSICAL TEST PASS on owner's macOS runtime. | Windows is OWNER VALIDATION REQUIRED if selected as demo/reviewer runtime; REVIEWER CONFIRMATION PENDING. |
| SYS-05.7 | Identify reviewed source/configuration and freeze appropriately | IMPLEMENTED. Runtime digest, required files, isolated dependencies and local snapshot manifest. | READY FOR INDEPENDENT REVIEW; final reviewed commit/freeze awaits review and owner authorization. |
| SYS-05.8 | Check identified candidate's files/setup/launch | IMPLEMENTED. Fresh minimal environment; required-file and model checks; isolated source check recorded below. | READY FOR INDEPENDENT REVIEW; Trang's fresh launch is REVIEWER CONFIRMATION PENDING. |
| SYS-05.9 | Demonstrate flow, explain simulation and answer integration questions | IMPLEMENTED; AUTOMATED TEST PASS. Four local interface demonstrations, native cases and English script/Q&A. | Demonstration to Trang has not occurred in this session; REVIEWER CONFIRMATION PENDING. |
| SYS-05.10 | Deliver runnable handover and actual receipt/execution evidence | IMPLEMENTED guide, manifest and copy-ready message. | Actual delivery/acknowledgement not performed; REVIEWER CONFIRMATION PENDING. No receipt invented. |

## Implementation and confirmed finding register

The default path opens camera 0 automatically, continuously previews the latest
frame, detects faces on a worker and automatically dispatches stubs for exactly
one valid source-frame region. Results are readable text, with secondary debug
JSON and permanent DEMO / SIMULATED OUTPUT labelling. Registration and injected
fake outcomes are developer-only. `can_save=false` and no attendance/gallery
calls prevent genuine persistence.

| Finding / reproduction | Root cause and correction | Regression / actual verification |
| --- | --- | --- |
| Owner reported lag in the earlier automatic GUI | Full-size conversion/detection, fixed post-read/post-render sleeps, repeated overlay/hidden JSON work were confirmed code paths. Time-budget pacing, resize-first rendering, bounded detector input, cached UI and one native OpenCV thread were implemented. Backend/Tk contribution was not independently quantified. | Existing scheduling/performance tests pass. Synthetic measurements below are scoped separately. Owner reports movement/preview responsiveness PASS. |
| Owner reported misses with glasses under Haar | Exact scene-specific cause not established. YuNet selected provisionally using verified CPU API/model provenance and lower measured synthetic cost; bounded Haar retained for comparison. | Owner now reports detection with and without glasses PASS on YuNet/macOS. This is not a measured general accuracy gain or a physical Haar/YuNet A/B result. |
| `receive(Completion(None, live))` after reset returned True | No active-context guard. Reject all untyped/orphan completions before result application. | `ContractAuditTests.test_orphan_and_untyped_completions_are_rejected_without_active_capture` PASS. |
| `begin_capture(frame, [None])` raised TypeError; missing dtype raised AttributeError | Shape/box validators assumed valid containers/metadata. Invalid metadata/regions now become typed `invalid_input`. | `test_malformed_frames_and_regions_produce_typed_invalid_input` PASS. |
| Emotion adapter returning None raised AttributeError outside adapter-call exception handling | Response type/module not validated. Invalid responses now produce module-specific errors; emotion remains display-only. | `test_malformed_adapter_response_is_typed_and_emotion_failure_is_display_only`; `test_wrong_result_type_cannot_masquerade_as_another_adapter` PASS. |
| Late uncertain liveness could retain an earlier provisional recognition object | Uncertain transition changed status without clearing recognition. Clear recognition on uncertain. | `test_uncertain_liveness_clears_a_provisional_identity` PASS. |
| Adapter-provided error severity could contradict its module | Controller previously accepted the severity flag as supplied. Controller now normalizes emotion-only errors to display-only and operation errors to blocking. | `test_error_severity_is_owned_by_controller` PASS; can_save remains false independently. |
| A ready old adapter result and new detector result in one Tk tick dropped the eligible new request | Dispatch preceded freeing the finished adapter slot. Drain the completed slot first, retaining original token/context checks. | `SchedulingAuditTests.test_finished_adapter_slot_is_freed_before_new_detection_dispatch` PASS. |
| Slow rendering could consume the result-age budget | Previously corrected age check uses acceptance time after rendering. | `SchedulingTests.test_expiry_is_checked_again_after_a_slow_tk_render` PASS. |
| Occasional box jitter/misses | Bounded display-only 70/30 geometry smoothing, max 450-ms historical outline, and immediate model-field clearing. No identity tracking or reused adapter evidence. | Stability/face-loss/multiple-face tests PASS. Owner notes slight hesitation during movement, acceptable in the reported run. |

No defect was invented for valid crop slicing, known fixture values, canonical
emotion mapping or stub save prohibition; their passing behavior was preserved.
All 120 previously passing tests remain; 11 focused audit regressions were added.

## Execution evidence and runtime

| Check | Actual result | Evidence |
| --- | --- | --- |
| Existing project command `PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -v` | 131 passed, 0 failed, 0 skipped | Executed in this session; fresh-runtime persisted rerun below provides the full log. |
| Fresh venv, only NumPy/Pillow/OpenCV | 131 passed, 0 failed, 0 skipped | [Summary](../reports/sys05/2026-11-16/full-tests-clean-runtime.json), [full log](../reports/sys05/2026-11-16/full-tests-clean-runtime.log). |
| Native Tk with generated frames / explicit controlled detections | 14 passed, 0 failed; no Tk callback errors; all worker threads stopped | [Native report](../reports/sys05/2026-11-16/native-tk-controlled.json). Root was withdrawn; physical visual usability is attributed separately to the owner. |
| Headless interface walkthrough | Four scenarios executed with source/crop/context/envelope, all saves false, stale callback rejected | [Controlled demonstration](../reports/sys05/2026-11-16/controlled-demo.json). Generated regions do not prove real detection. |
| Model/licence check | Published SHA-256 and file sizes matched | [Runtime/assets](../reports/sys05/2026-11-16/runtime.json). No model was silently downloaded during startup. |
| Current remote interfaces | Read-only inspection, no merges | [Branch heads and blob IDs](../reports/sys05/2026-11-16/remote-interface-audit.json). |
| Isolated source snapshot + fresh minimal runtime | 131 tests passed, 0 failed, 0 skipped; 14 native Tk cases passed again; protected JSON absent | [Source/test summary](../reports/sys05/2026-11-16/isolated-source-tests.json), [full log](../reports/sys05/2026-11-16/isolated-source-tests.log), [native report](../reports/sys05/2026-11-16/isolated-native-tk.json). |

Fresh verification runtime: macOS 27.2 arm64, Python 3.13.17, Tk/Tcl 8.6,
NumPy 2.2.6, Pillow 11.2.1, opencv-python 4.13.0.92. Torch is not installed in
that minimal venv. The existing project `.venv` and shared requirements were not
replaced. Initial sandbox DNS prevented pip access; an explicitly approved network
retry installed the three pins in the isolated environment successfully. That was
a setup-access failure, not a failed package compatibility test.

The full suite's native GUI checks are separate from the unittest count. Do not
add the 14 native cases to 131 and describe them as 145 unittests.

## Owner physical validation — attributed report

On 16 November the owner replied to the provided current YuNet launch command:

| Owner check | Result |
| --- | --- |
| Detection without glasses | PHYSICAL TEST PASS |
| Detection with glasses | PHYSICAL TEST PASS |
| Normal indoor lighting | PHYSICAL TEST PASS |
| Movement and preview responsiveness | PHYSICAL TEST PASS |
| Leaving the frame clears results | PHYSICAL TEST PASS |
| Re-entering restores results | PHYSICAL TEST PASS |
| Retry | PHYSICAL TEST PASS |
| Camera error and relaunch recovery | PHYSICAL TEST PASS |

Owner observation, retained verbatim: “di chuyển thì cái khung có hơi khựng nhưng
tạm ônr”. Interpretation: the box hesitates slightly during movement, but the
owner considers it acceptable in this run. The 300-ms analysis sampling is a
plausible contributor; it has not been isolated experimentally on the physical
scene. Continuous preview and sampled box updates are different rates. No new
FPS threshold, accuracy guarantee or zero-jitter claim is made.

[The attributed physical record](../reports/sys05/2026-11-16/owner-physical-validation.json)
contains the owner's eight reported outcomes and the current runtime hashes.
Camera model, actual capture resolution, exact test start time and measured
physical FPS were not supplied and remain unknown. No screenshot or video was
invented. Bright backlight, physical two-person detection, trained spoof attacks
and revised Windows execution were not reported as tested.

## Current measured performance, with scope

[The 16 November benchmark](../reports/sys05/2026-11-16/benchmark.json) used seeded
1920×1080 noise, one OpenCV native thread, two warmups and eight timing samples.

| Synthetic operation | Median / observed value |
| --- | --- |
| Prior full-source preview preparation | 15.488 ms |
| Current resize-first preview preparation | 2.771 ms |
| Full-HD Haar | 786.463 ms |
| Bounded 640-edge Haar | 76.148 ms |
| Bounded CPU YuNet | 13.150 ms |
| Reproduced old fixed-wait capture loop on a paced fake 30-Hz source | 13.78 published frames/s |
| Current capture loop on that paced fake source | 26.37 published frames/s |

These are not physical webcam FPS, photon-to-screen latency, native Tk throughput
or accuracy measurements. Actual owner responsiveness is qualitative PASS.
The optional five-second aggregate diagnostics measure real capture/GUI work when
the user runs them; local frame age excludes exposure and driver buffering.
The earlier 12 November measurements remain historical in the run guide.

## Release, limitations and handover

The local candidate is identified by runtime hash and a full copied-file manifest,
not by a new Git commit. The final snapshot is prepared under ignored
`artifacts/sys05-candidate-20261116/`, with model/licence copied only after explicit
hash verification. `scripts/prepare_sys05_candidate.py` reads HEAD and overlays
the SYS-05 allowlist; it never stages, commits, pushes or merges. The protected
environment JSON is expressly excluded.

[The handover guide](sys-05-handover.md) contains exact macOS/Windows commands,
model provenance, troubleshooting, adapter entry points, threading, original
context propagation, checkpoint responsibilities and an English walkthrough/Q&A.
UI-03/UI-04 own trained model adapters, preprocessing and policy validation,
gallery compatibility, attendance rules, final authorization and persistence.
SYS-05 does not implement those tasks to manufacture completion.

Remaining limits: slight sampled-box hesitation; no physical benchmark or broader
accuracy study; Windows native execution not run; native driver calls cannot be
forcibly interrupted safely; no trained recognition/liveness/emotion deployment;
no normal registration UI or genuine persistence. Synthetic success does not
substitute for trained anti-spoofing tests. These limits do not reopen owner checks
already explicitly reported PASS, but must travel with the candidate.

Remaining external actions: present this exact candidate to Trang; record her
sample-output agreement and independent findings; retest any findings; authorize
the eventual source commit/push; identify/freeze the reviewed version; actually
deliver the handover and record receipt. None of these events is claimed here.

## Final Owner Evidence — completed owner-side portions only

These entries are at most 60 English words each. They describe completed owner
verification, not reviewer approval or full tracker closure. They have not been
written to the Sheet.

**SYS-05.3**

Completed the owner-side SYS-02 v1 compatibility audit with source-to-test traceability. Verified BGR frames, half-open crops, correlated identifiers, typed outputs/errors, liveness gating, stale-result rejection and stub save prohibition. The 131-test suite and 14 native Tk cases passed. Trang’s independent compatibility review remains pending.

**SYS-05.4**

Corrected malformed-response handling, orphan completion acceptance and adapter-slot dispatch loss. Capture pacing, bounded detection and rendering optimizations remain verified. Regression tests passed; the owner reported all eight YuNet/macOS physical checks passing, including glasses and recovery. Slight box hesitation during movement remains documented. Reviewer retesting is pending.

**SYS-05.5**

Reran all 131 automated tests and 14 native Tk scenarios with no failures. Controlled tests verified no-face clearing, multiple-face blocking, camera/detector errors, retry, stale callbacks and shutdown. The owner separately confirmed physical face-loss, return and camera recovery on macOS. Trained adapters and Trang’s independent retest remain outside this verification.

**SYS-05.8**

Prepared and checked local candidate SYS05-LOCAL-20261116-c00c004fc23a. A fresh minimal environment and isolated source snapshot reproduced 131 passing tests and 14 native Tk cases. Required files, model checksum/licence and platform commands are documented. This is an uncommitted local candidate; Trang’s fresh launch and independent release confirmation remain pending.

**SYS-05.9** — no final entry yet. The local walkthrough was executed and the
English script/Q&A prepared, but demonstration to Trang and actual questions have
not occurred in this session.

**SYS-05.10** — no final entry yet. The runnable guide, snapshot and message are
prepared; actual transmission and recipient acknowledgement have not occurred.

## Existing Owner Evidence preserved; revision notes proposed separately

The exact prior Sheet evidence for SYS-05, SYS-05.1, SYS-05.2, SYS-05.6 and
SYS-05.7 remains in the read-only tracker snapshot and the Sheet. The historical
`39ac3df` document is preserved verbatim in
[the archive](sys-05-legacy-39ac3df.md). It still describes genuine earlier
Windows/macOS evidence. `reports/environment/vy-mac-m1-review.json` was never
modified, staged, deleted or copied into the candidate.

If the owner later updates the tracker, append a clearly dated redesign follow-up;
do not relabel historical 77-test/Haar/39ac3df evidence as current YuNet verification.
Proposed follow-up descriptions (each below 60 English words):

- **SYS-05:** The automatic YuNet redesign is identified as local candidate SYS05-LOCAL-20261116-c00c004fc23a. All 131 automated tests and 14 native Tk cases passed; the owner reported eight macOS physical checks passing. Model outputs remain simulated and non-persistent. Independent review, demonstration and actual handover remain pending.
- **SYS-05.1:** The original source/crop/context interfaces remain preserved. The revised run guide and headless walkthrough demonstrate four deterministic scenarios through the same replaceable adapter boundaries. Current identity, liveness and emotion outputs remain explicitly simulated. Reviewer agreement on sample outputs is still pending.
- **SYS-05.2:** The revised GUI automatically previews the webcam, detects one face and updates labelled simulated outputs without dropdowns, freezing or manual evaluation. Owner-reported YuNet/macOS physical checks passed. Debug JSON remains available; genuine attendance and enrollment saving remain disabled. Historical Windows results do not validate revised Windows execution.
- **SYS-05.6:** The owner reported YuNet/macOS passes for glasses, indoor lighting, responsiveness, face-loss/return, Retry and camera recovery. Controlled native Tk checks and 131 automated tests also passed. Slight box hesitation remains documented. Revised Windows and broader lighting/accuracy checks were not performed; independent review remains pending.
- **SYS-05.7:** The redesigned source is identified by local candidate SYS05-LOCAL-20261116-c00c004fc23a and its file manifest, with isolated runtime pins and verified model instructions. No new commit or final release freeze is claimed. Preserve 39ac3df as the historical baseline; freeze the reviewed version only after findings are resolved and the owner authorizes release actions.

## Owner follow-through checklist

- [x] Report current YuNet/macOS physical results honestly; eight checks PASS.
- [x] Retain the slight box-hesitation observation and untested scene/platform limits.
- [x] Preserve prior evidence and keep stub records disabled.
- [ ] Provide the identified local snapshot to Trang; record exact candidate received.
- [ ] Obtain independent review/sample-output agreement and retest actual findings.
- [ ] Present the walkthrough and record actual integration questions/answers.
- [ ] Deliver the handover, record recipient acknowledgement and observed execution.
- [ ] Authorize a source commit/push separately, then freeze the reviewed version.
- [ ] If Windows is the intended demo/reviewer runtime, run its documented setup,
  full tests, camera and recovery checks; label failures or unavailable checks honestly.
