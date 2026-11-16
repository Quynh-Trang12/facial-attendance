# SYS-05 owner-side SYS-02 / UI-02 traceability — 16 November 2026

This is implementation verification by the owner-side work session, not Trang's
independent review. The implementation remains a non-persistent demonstration.
All 131 tests passed in the existing runtime and a fresh minimal runtime; the
native Tk harness passed 14 controlled cases. Physical checks are separately
attributed in [the owner report](sys-05-owner-self-review.md).

Sources read again on 16 November:

- [SYS-02 Final v1.0](https://app.notion.com/p/40d863094ebe80d7a6f9c9ca4f7abc6d)
- [UI-02 scheduling and states](https://app.notion.com/p/412863094ebe80c09ffce13f894f16ae)
- [Requirements 39–49](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit#gid=2016101501&range=A39:S49)
- [WBS 41–51](https://docs.google.com/spreadsheets/d/1GkZogmafVGGMJrRnzRzTURIFnva_MPgCsVlqgpB9uR8/edit#gid=827743122&range=A41:AM51)

No Sheet or Notion content was changed. Relative implementation paths below are
under `src/facial_attendance/ui/`; tests are under `tests/`.

| SYS-02 / UI-02 requirement | Implementation | Automated test / native case | Observed result |
| --- | --- | --- | --- |
| §3.1 BGR uint8 `[H,W,3]`, nonempty source | `webcam_prototype.frame_size`; `live_preview.CameraFeed._run` | `ContractAuditTests.test_distinct_bgr_pixels_and_exact_half_open_crop_are_preserved`; `test_malformed_frames_and_regions_produce_typed_invalid_input` | AUTOMATED TEST PASS. Distinct `[11,22,33]` BGR channels survive; malformed metadata is typed invalid input. |
| §3.1 half-open source coordinates | `box_in_frame`; `face_detection.map_box`, `ScaledFaceDetector` | `ScaledDetectorTests.test_downscales_once_and_maps_outward_with_actual_non_integer_ratios`; `test_invalid_regions_are_rejected_before_scaling` | AUTOMATED TEST PASS. Integer starts floor, ends ceil using actual resized dimensions; invalid regions rejected. |
| §3.2 exact frame, box and crop association | `FaceRequest.face_crop`; `PrototypeController.begin_capture` | `WebcamPrototypeTests.test_crop_uses_half_open_original_frame_coordinates`; distinct-pixel audit test above | AUTOMATED TEST PASS. Crop shares read-only source memory, exact expected shape and channels; no model preprocessing applied globally. |
| §3.2 `contract_version`, attempt/action/capture/face IDs | `Context`; `begin_capture`; `receive` | `LiveControllerTests.test_wrong_capture_face_attempt_action_and_version_cannot_be_relabelled`; `ContractAuditTests.test_unknown_action_is_rejected` | AUTOMATED TEST PASS. Any superseded identifier is rejected; action limited to attendance/registration. |
| §4 recognition person/name/score/accepted/model metadata | `Recognition`; `StubAdapters.recognition`; `receive` | `WebcamPrototypeTests.test_live_scenarios_serialize_contract_and_never_allow_save`; `test_wrong_result_type_cannot_masquerade_as_another_adapter` | AUTOMATED TEST PASS. Match/Unknown fields and null stub model metadata serialize; incorrect adapter result types become typed errors. |
| §5 liveness status/reason/p_spoof/version/source | `Liveness`; `StubAdapters.liveness`; `receive` | `WebcamPrototypeTests.test_bad_liveness_is_an_operational_error`; `LiveControllerTests.test_all_scenarios_use_original_bgr_context_and_never_authorize_save` | AUTOMATED TEST PASS. Source remains stub; invalid scores/status/reason never count as live. |
| §5 liveness gating; spoof abort; uncertain hides identity | `run_stubs`; `receive`; `demo_identity` | `WebcamPrototypeTests.test_provisional_identity_clears_on_spoof_and_requires_retry`; `test_uncertain_skips_recognition_and_rejects_previous_capture`; `ContractAuditTests.test_uncertain_liveness_clears_a_provisional_identity` | AUTOMATED TEST PASS. Recognition runs only after live; spoof stops the sample; uncertain clears identity. |
| §6 explicit Negative=0 / Neutral=1 / Positive=2 | `EMOTION_LABELS`; `Emotion`; `receive` | `WebcamPrototypeTests.test_invalid_emotion_clears_display_only`; four-scenario demo; native `live_match/live_unknown/uncertain` | AUTOMATED TEST PASS. Mapping is explicit; invalid emotion is unavailable, never a guessed Neutral result. |
| §6 emotion cannot authorize records | `Envelope.can_save`; `_apply_error` | `ContractAuditTests.test_error_severity_is_owned_by_controller`; `test_malformed_adapter_response_is_typed_and_emotion_failure_is_display_only` | AUTOMATED TEST PASS. Emotion failure clears only emotion; other operational failures block the operation; no save path exists. |
| §7 canonical envelope / typed errors | `Envelope`, `OperationError`; `run_stubs`; `LiveController.completed` | `ContractAuditTests.test_malformed_current_batch_fails_closed`; `test_malformed_adapter_response_is_typed_and_emotion_failure_is_display_only` | AUTOMATED TEST PASS. Malformed predictions/batches become typed errors without a Tk callback exception. |
| §7 no orphan, stale or relabelled completions | `receive`; `LiveController.token`; `CameraFeed` generation | `ContractAuditTests.test_orphan_and_untyped_completions_are_rejected_without_active_capture`; `test_stale_adapter_response_is_discarded_without_relabelling_context`; native `stale_after_reset` | AUTOMATED TEST PASS. Original contexts remain attached; a completion without an active capture is refused. |
| §7 no-face / multiple faces clear current fields | `LiveController.begin_sample/detected`; `capture_error` | `LiveAppTests.test_next_sample_no_face_and_multiple_faces_clear_display`; native `face_loss/multiple_after_live` | AUTOMATED TEST PASS. All model fields clear; multiple faces never select an arbitrary face. |
| §8 camera failure / retry / shutdown invalidation | `CameraFeed`; `LiveWebcamApp.retry/close` | `WorkerTests.test_camera_failure_releases_on_owner_thread_and_retry_recovers`; `LiveAppTests.test_close_is_idempotent_and_late_tick_does_not_schedule`; native recovery/shutdown cases | AUTOMATED TEST PASS. Fourteen native cases report all workers stopped; owner separately reports physical recovery PASS. |
| §8/10 stub input never genuine persistence | frozen `Envelope.can_save=False`; no gallery/attendance calls in pipeline | `WebcamAppTests.test_all_scenarios_keep_save_disabled_in_both_modes`; all four demo payloads; native harness | AUTOMATED TEST PASS. All scenarios/actions remain non-persistent. |
| UI-02 background work, main-thread widgets, bounded scheduling | `CameraFeed`, `SingleWorker`; `LiveWebcamApp._tick` | `LiveAppTests.test_camera_preview_continues_and_work_slots_remain_bounded_while_detection_is_busy`; `SchedulingAuditTests.test_finished_adapter_slot_is_freed_before_new_detection_dispatch` | AUTOMATED TEST PASS. Latest capture selected only when slot is free; completed old slot drained before dispatch. |
| UI-02 throttle, expiry and delayed callbacks | `_next_detection`; `LiveController.expire`; post-render age check | `SchedulingTests.test_detector_throttles_and_chooses_latest_frame_after_busy_work`; `test_expiry_is_checked_again_after_a_slow_tk_render`; `LiveAppTests.test_slow_adapter_cannot_hide_face_loss_or_restore_old_identity` | AUTOMATED TEST PASS. No backlog, stale success or old authorization restoration. |
| UI-02 readable states and explicit demo status | `LiveWebcamApp.__init__/_render/_toggle_debug` | `LiveAppTests.test_default_constructor_contains_no_dropdowns_or_manual_evaluation_buttons`; `SchedulingTests.test_hidden_json_is_not_serialized_and_opening_debug_refreshes_it`; native harness | AUTOMATED TEST PASS. Native Tk reached expected states; owner reports physical workflow PASS. |
| Display stability must not become identity tracking | `BoxStability`; `LiveController.visual_regions` | `StabilityTests.test_one_miss_holds_only_unconfirmed_outline_and_clears_every_model_field`; `test_smoothing_only_changes_display_not_request_or_frame_association` | AUTOMATED TEST PASS. A dashed historical outline is bounded to 450 ms; adapter boxes remain current and unsmoothed. |

## Findings and scope boundary

The 16 November audit reproduced orphan completion acceptance, TypeError on a
malformed bbox, AttributeError on a malformed emotion response, and a finished
adapter slot causing avoidable dispatch loss. The corrective tests above now pass.
Earlier full-resolution work and pacing findings remain documented separately in
[the run guide](sys-05-phase-1.md). No additional defect was found in the valid
four-scenario field mapping, basic crop slicing, save prohibition or existing
context comparison; those were retained rather than rewritten.

The pipeline uses serial liveness-first dispatch, which is permitted by SYS-02.
No shared recognition normalization, liveness input size/threshold, model
architecture or genuine persistence eligibility is guessed. The current validator
deliberately refuses `source=model`; UI-03/UI-04 must introduce and verify a
separate trained integration. A passing stub contract is not trained model accuracy.

## Current team interfaces read without merging

Exact branch heads and blob IDs are recorded in
[`remote-interface-audit.json`](../reports/sys05/2026-11-16/remote-interface-audit.json).

| Current source | Reconciliation / integration responsibility |
| --- | --- |
| `main` `9439aea9813423f0aedd0e9930f10bea24fa184d` | Remote main advanced after this SYS-05 branch baseline. No unrelated SYS-03 changes were merged into the working tree. |
| `feature/ui-01-gallery-base` `82a74f8…` | Actual gallery schema 0.2 stores one averaged L2-normalized template with `n_images` 3–10. SYS-02's warning describes older storage; the actual current gallery was read. SYS-05 imports no gallery and writes no enrollment. |
| `docs/sys-04-experiment-settings` `4725d11…` | Recognition takes adapter-owned RGB 64×64/ImageNet preprocessing and normalized 512D vectors. Webcam crop margin remains subject to UI-03 and threshold to FAC-05. |
| FAC-03 `61bbf2c…`, FAC-04 `5700e02…` | Models/checkpoint and embedding helpers now exist. `embed()` / `embed_paths()` return raw vectors; the live adapter must L2-normalize and cannot treat path-based evaluator exports as a webcam adapter. `build_meta()` does not select a final threshold. |
| LIV-02 `25eafa2…`, LIV-03 `0a21df3…` | Liveness versions/policy and identifiers remain adapter responsibilities. LIV-03's supplied `_BB.txt` helper uses a dataset-specific 224-coordinate system; it is not the original-frame webcam-box API. |
| Emotion `284c483…` | Current config specifies 48×48 grayscale, divide by 255 and no augmentation. The display document preserves explicit three-class labels. No trained deployment checkpoint was assumed. |

This is read-only compatibility analysis, not execution or approval of another
member's code. Independent shared-interface review remains REVIEWER CONFIRMATION PENDING.
