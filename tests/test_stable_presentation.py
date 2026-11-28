"""Display history never becomes inference, context or persistence evidence."""

from dataclasses import asdict, replace
import json
import unittest
from unittest.mock import Mock

import numpy as np

from facial_attendance.recognition.gallery_matching import CONTROLLED_BOX, controlled_frame
from facial_attendance.liveness.checkpoint_trial import TrialScore
from facial_attendance.ui.identity_overlay import identity_label
from facial_attendance.ui.presentation import StablePresentation, signature
from facial_attendance.ui.live_preview import LiveController, WorkResult, evaluate_adapters
from facial_attendance.ui.webcam_app import LiveWebcamApp
from facial_attendance.ui.webcam_prototype import Completion, Emotion, Liveness, Recognition, StubAdapters
import test_webcam_live as legacy
import test_emotion_presentation as emotion_fixtures


class PresentationTests(unittest.TestCase):
    def setUp(self):
        self.display=StablePresentation();self.live=LiveController();self.frame=controlled_frame()
        self.time=10.;self.sequence=0;self.last_preview=None

    def previews(self, until, frame=None):
        frame=self.frame if frame is None else frame
        if self.last_preview is not None:
            step=self.last_preview+.04
            while step<until:
                self.display.observe_preview(frame,step);step+=.04
        self.display.observe_preview(frame,until);self.last_preview=until

    def begin(self, timestamp, frame=None, boxes=None):
        frame=self.frame if frame is None else frame
        self.previews(timestamp,frame);self.sequence+=1;token=(1,self.sequence)
        self.display.begin_sample(token,frame,timestamp);self.live.begin_sample(token,timestamp)
        boxes=[CONTROLLED_BOX] if boxes is None else boxes
        request=self.live.detected(WorkResult(token,boxes),frame)
        self.display.detected(token,frame,boxes,timestamp)
        return token,request

    def complete(self, timestamp, label=2, confidence=.72):
        token,request=self.begin(timestamp)
        adapters=StubAdapters('Live match')
        adapters.emotion=lambda req:Completion(req.context,Emotion(label,('Negative','Neutral','Positive')[label],confidence))
        batch=evaluate_adapters((request,adapters));self.live.completed(WorkResult(token,batch))
        self.assertTrue(self.display.accept(token,request,self.live.controller.state,timestamp,timestamp+.03))
        return self.display.view(self.live.controller.state,timestamp+.03)

    def warm(self):
        self.complete(10.);return self.complete(10.3)

    def test_routine_processing_retains_explicitly_previous_text_but_raw_fields_clear(self):
        previous=self.warm();self.begin(10.6)
        raw=self.live.controller.state;shown=self.display.view(raw,10.63)
        self.assertEqual((shown.identity,shown.emotion,shown.liveness),(previous.identity,previous.emotion,previous.liveness))
        self.assertTrue(shown.historical);self.assertIn('Previous sample',shown.status)
        self.assertIn('UNVERIFIED',shown.status)
        self.assertIsNone(raw.recognition);self.assertIsNone(raw.emotion);self.assertIsNone(raw.liveness)
        self.assertFalse(raw.can_save);self.assertNotEqual(shown.source_context,raw.context)

    def test_first_single_detection_is_not_enough_to_hold_while_waiting_for_next_detection(self):
        self.complete(10.)
        self.previews(10.3);self.display.begin_sample((1,2),self.frame,10.3)
        self.live.begin_sample((1,2),10.3)
        shown=self.display.view(self.live.controller.state,10.31)
        self.assertEqual(shown.identity,'Unavailable')

    def test_repeated_rendering_and_fresh_previews_do_not_renew_display_expiration(self):
        self.warm();self.begin(10.6)
        self.previews(11.09)
        self.assertIn('Demo Person',self.display.view(self.live.controller.state,11.09).identity)
        self.previews(11.11)
        self.assertEqual(self.display.view(self.live.controller.state,11.11).identity,'Unavailable')
        self.assertIsNone(self.display._sample)


    def test_preview_gap_expires_display_before_sample_ttl(self):
        self.warm()
        self.assertEqual(self.display.view(self.live.controller.state,10.46).identity,'Unavailable')

    def test_same_box_different_pixels_clears_history_immediately(self):
        self.warm();replacement=(255-self.frame).astype(np.uint8)
        self.assertFalse(self.display.observe_preview(replacement,10.34))
        shown=self.display.view(self.live.controller.state,10.35)
        self.assertEqual(shown.identity,'Unavailable');self.assertIsNone(self.display._sample)

    def test_large_box_jump_cannot_inherit_identity_or_emotion(self):
        self.warm();self.begin(10.6,boxes=[(20,20,200,200)])
        self.assertEqual(self.display.view(self.live.controller.state,10.63).identity,'Unavailable')
        self.assertFalse(self.display._emotions)

    def test_changed_frame_dimensions_cannot_inherit_even_an_identical_crop(self):
        self.warm()
        larger=np.zeros((720,1280,3),dtype=np.uint8);larger[:360,:640]=self.frame
        self.assertFalse(self.display.observe_preview(larger,10.34))
        self.assertIsNone(self.display._sample)

    def test_model_completion_must_retain_the_detected_source_region_and_timestamp(self):
        token,request=self.begin(10.)
        self.live.completed(WorkResult(token,evaluate_adapters((request,StubAdapters('Live match')))))
        state=self.live.controller.state
        self.assertFalse(self.display.accept(token,replace(request,bbox=(20,20,200,200)),state,10.,10.03))
        self.assertFalse(self.display.accept(token,replace(request,frame=255-self.frame),state,10.,10.03))
        self.assertFalse(self.display.accept(token,request,state,9.99,10.03))

    def test_uninformative_uniform_color_cannot_establish_continuity(self):
        uniform=np.full((360,640,3),(25,100,170),dtype=np.uint8)
        self.assertIsNone(signature(uniform,CONTROLLED_BOX))
        self.warm();self.display.observe_preview(uniform,10.34)
        self.assertIsNone(self.display._sample)

    def test_no_face_and_multiple_faces_clear_history_and_votes(self):
        for boxes in ([],[CONTROLLED_BOX,(20,20,100,100)]):
            self.setUp();self.warm();self.begin(10.6,boxes=boxes)
            self.assertIsNone(self.display._sample);self.assertFalse(self.display._emotions)
            self.assertEqual(self.display.view(self.live.controller.state,10.63).identity,'Unavailable')

    def test_spoof_uncertain_and_camera_error_clear_history(self):
        for code in ('spoof','uncertain','error'):
            self.setUp();self.warm()
            state=replace(self.live.controller.state,ui_state=code,recognition=None,
                          liveness=Liveness(code if code!='error' else 'uncertain','test',.5))
            shown=self.display.view(state,10.34)
            self.assertEqual(shown.identity,'Unavailable');self.assertIsNone(self.display._sample)
            self.assertFalse(state.can_save)

    def test_retry_and_changed_generation_cannot_retain_history(self):
        self.warm();self.display.reset();self.live.reset()
        self.assertEqual(self.display.view(self.live.controller.state,10.34).identity,'Unavailable')
        self.setUp();self.warm();self.previews(10.4)
        self.display.begin_sample((2,1),self.frame,10.4)
        self.assertIsNone(self.display._sample)

    def test_action_change_cannot_inherit_an_attendance_display(self):
        self.warm();self.begin(10.6)
        state=self.live.controller.state
        state=replace(state,context=replace(state.context,action='registration'))
        self.assertEqual(self.display.view(state,10.63).identity,'Unavailable')

    def test_old_completion_cannot_repopulate_after_replacement(self):
        self.warm();token,request=self.begin(10.6)
        batch=evaluate_adapters((request,StubAdapters('Live match')))
        self.display.observe_preview(255-self.frame,10.64)
        self.live.completed(WorkResult(token,batch))
        self.assertFalse(self.display.accept(token,request,self.live.controller.state,10.6,10.65))
        self.assertIsNone(self.display._sample)

    def test_stale_context_and_wrong_token_are_rejected(self):
        self.warm();token,request=self.begin(10.6)
        self.assertFalse(self.display.accept((1,99),request,self.live.controller.state,10.6,10.63))
        old=evaluate_adapters((request,StubAdapters('Live match')))
        self.begin(10.9)
        self.assertFalse(self.live.completed(WorkResult(token,old)))
        self.assertFalse(self.display.accept(token,request,self.live.controller.state,10.6,10.93))
        self.assertFalse(self.live.controller.state.can_save)

    def test_duplicate_capture_does_not_amplify_continuity_or_emotion_votes(self):
        token,request=self.begin(10.)
        self.assertFalse(self.display.detected(token,self.frame,[CONTROLLED_BOX],10.))
        self.assertEqual(self.display._confirmations,1)
        self.live.completed(WorkResult(token,evaluate_adapters((request,StubAdapters('Live match')))))
        self.assertTrue(self.display.accept(token,request,self.live.controller.state,10.,10.03))
        self.assertFalse(self.display.accept(token,request,self.live.controller.state,10.,10.04))
        self.assertEqual(len(self.display._emotions),1)

    def test_same_timestamp_cannot_count_as_a_second_distinct_continuity_sample(self):
        self.complete(10.)
        self.display.begin_sample((1,2),self.frame,10.)
        self.display.detected((1,2),self.frame,[CONTROLLED_BOX],10.)
        self.assertEqual(self.display._confirmations,1)
        self.assertFalse(self.display._hold)

    def test_confidence_variations_do_not_change_visible_fields_or_raw_values(self):
        self.complete(10.,confidence=.701);first=self.complete(10.3,confidence=.709)
        second=self.complete(10.6,confidence=.722)
        self.assertEqual(first.emotion,second.emotion)
        self.assertNotIn('%',second.emotion)
        self.assertEqual(self.live.controller.state.emotion.confidence,.722)

    def test_bounded_emotion_majority_does_not_change_raw_class(self):
        self.complete(10.,label=2);self.complete(10.3,label=2)
        shown=self.complete(10.6,label=0)
        self.assertEqual(shown.emotion,'Positive');self.assertEqual(self.live.controller.state.emotion.label,'Negative')
        shown=self.complete(10.9,label=0)
        self.assertEqual(shown.emotion,'Negative');self.assertLessEqual(len(self.display._emotions),3)

    def test_delayed_prediction_cannot_vote_with_expired_emotion_samples(self):
        self.warm();token,request=self.begin(10.6);self.previews(11.3)
        adapters=StubAdapters('Live match')
        adapters.emotion=lambda req:Completion(req.context,Emotion(0,'Negative',.9))
        self.live.completed(WorkResult(token,evaluate_adapters((request,adapters))))
        self.assertTrue(self.display.accept(token,request,self.live.controller.state,10.6,11.3))
        self.assertEqual(self.display.view(self.live.controller.state,11.3).emotion,'Negative')
        self.assertEqual(len(self.display._emotions),1)

    def test_new_identity_decision_discards_previous_emotion_votes(self):
        self.warm();token,request=self.begin(10.6)
        batch=evaluate_adapters((request,StubAdapters('Live unknown')))
        self.live.completed(WorkResult(token,batch))
        self.display.accept(token,request,self.live.controller.state,10.6,10.63)
        shown=self.display.view(self.live.controller.state,10.63)
        self.assertEqual(shown.identity,'Unknown');self.assertEqual(shown.emotion,'Neutral')
        self.assertEqual(len(self.display._emotions),1)

    def test_display_api_cannot_mutate_the_authoritative_envelope(self):
        self.warm();before=asdict(self.live.controller.state)
        for now in (10.33,10.34,10.35):self.display.view(self.live.controller.state,now)
        self.assertEqual(asdict(self.live.controller.state),before)
        self.assertFalse(self.live.controller.state.can_save)
        self.assertFalse(hasattr(self.display.view(self.live.controller.state,10.35),'can_save'))


class StableTrialTests(unittest.TestCase):
    def setUp(self):
        self.display = StablePresentation(unverified_preview=True, stable_trial=True)
        self.live = LiveController(unverified_preview=True)
        self.frame = emotion_fixtures.camera_frame()
        self.previous_time = None
        self.sequence = 0

    def preview(self, at, frame):
        if self.previous_time is not None:
            tick = self.previous_time + .04
            while tick < at:
                self.display.observe_preview(frame, tick)
                tick += .04
        self.display.observe_preview(frame, at)
        self.previous_time = at

    def begin(self, at, frame=None, boxes=None):
        frame = self.frame if frame is None else frame
        boxes = [emotion_fixtures.BOX] if boxes is None else boxes
        self.preview(at, frame)
        self.sequence += 1
        token = (1, self.sequence)
        self.display.begin_sample(token, frame, at)
        self.live.begin_sample(token, at)
        request = self.live.detected(WorkResult(token, boxes), frame)
        self.display.detected(token, frame, boxes, at)
        return token, request

    def publish(self, at, frame=None, boxes=None, score=.199, recognition=None):
        token, request = self.begin(at, frame, boxes)
        adapters = StubAdapters("Live match")
        if recognition is not None:
            adapters.recognition = lambda source: Completion(source.context, recognition)
        batch = evaluate_adapters((request, adapters), unverified_preview=True)
        self.live.completed(WorkResult(token, batch))
        self.assertTrue(self.display.accept(token, request, self.live.controller.state, at, at + .03))
        result = WorkResult(token, (TrialScore("Full", "ee6f1341785d22ad", score, .2),))
        self.display.accept_trial(result, at, at + .03)
        return result

    def warm(self):
        self.publish(10.)
        return self.publish(10.42)

    def test_motion_and_lighting_can_hold_both_labels_while_raw_results_clear(self):
        old = self.warm()
        shifted = (68, 43, 228, 203)
        frame = emotion_fixtures.camera_frame(shifted, noise=7, light=14)
        self.begin(10.84, frame, [shifted])
        state = self.live.controller.state
        shown = self.display.view(state, 10.88)
        self.assertTrue(shown.historical)
        self.assertIn("Demo Person", shown.identity)
        self.assertIs(self.display.trial_view(10.88), old)
        self.assertIsNone(state.recognition)
        self.assertIsNone(state.liveness)
        self.assertFalse(state.can_save)

    def test_face_loss_multiple_faces_and_replacement_clear_both_labels(self):
        for frame, boxes in (
                (self.frame, []),
                (self.frame, [emotion_fixtures.BOX, (230, 30, 300, 110)]),
                (emotion_fixtures.camera_frame(invert=True), [emotion_fixtures.BOX])):
            with self.subTest(box_count=len(boxes)):
                self.setUp()
                self.warm()
                self.begin(10.84, frame, boxes)
                shown = self.display.view(self.live.controller.state, 10.87)
                self.assertEqual(shown.identity, "Checking match" if len(boxes) == 1 else "Unavailable")
                self.assertIsNone(shown.identity_name)
                self.assertIsNone(self.display.trial_view(10.87))

    def test_latest_score_expires_and_reset_rejects_old_completion(self):
        old = self.warm()
        self.preview(11.23, self.frame)
        self.assertEqual(self.display.view(self.live.controller.state, 11.23).identity, "Unavailable")
        self.assertIsNone(self.display.trial_view(11.23))
        self.display.reset()
        self.begin(11.30)
        self.display.accept_trial(old, 10.42, 11.32)
        self.assertIsNone(self.display.trial_view(11.32))

    def test_threshold_change_is_visible_even_when_scores_round_to_the_same_text(self):
        below = self.warm()
        above = self.publish(10.84, score=.201)
        text = LiveWebcamApp._liveness_trial_text
        self.assertIn("0.20 / 0.200 · Below threshold", text(None, below))
        self.assertIs(self.display.trial_view(10.88), above)
        self.assertIn("0.20 / 0.200 · Spoof threshold met", text(None, above))
        self.assertEqual(above.value[0].score, .201)

    def test_alternating_known_and_unknown_never_restore_an_unconfirmed_name(self):
        known = Recognition("test-001", "Example", .469, True, "face-model", "a" * 12,
                            "model", "b" * 64, "fixture")
        unknown = replace(known, person_id=None, name=None, score=.454, accepted=False)
        self.publish(10., recognition=known)
        self.publish(10.42, recognition=known)
        pending, _ = self.begin(10.84)
        state = self.live.controller.state
        shown = self.display.view(state, 10.87)
        label = identity_label(state, shown, face_count=1, confirmed=True, processing=True,
                               trained=True, unverified_preview=True)
        self.assertTrue(shown.historical)
        self.assertEqual(label.text, "Example")
        self.assertIsNone(state.recognition)
        for sequence, decision in enumerate((unknown, known, unknown, known, unknown, unknown), start=3):
            when = 10. + sequence * .42
            self.publish(when, recognition=decision)
            state = self.live.controller.state
            shown = self.display.view(state, when + .03)
            label = identity_label(state, shown, face_count=1, confirmed=True, processing=False,
                                   trained=True, unverified_preview=True)
            expected = "Unknown" if sequence == 8 else "Checking match"
            self.assertEqual((shown.identity, label.text), (expected, expected))
            self.assertFalse(label.known)
            self.assertEqual(state.recognition.score, decision.score)
            self.assertFalse(state.can_save)
        self.assertNotEqual(pending, self.live.token)

    def test_local_outline_filters_small_jitter_without_changing_capture_boxes(self):
        offsets = [0, 3, -2, 2, -3, 0]
        sums = []
        for is_local in (False, True):
            live = LiveController(unverified_preview=is_local)
            displayed = []
            for index, dx in enumerate(offsets):
                token, at = (1, index), 10 + index * .42
                box = (60 + dx, 40, 220 + dx, 200)
                live.begin_sample(token, at)
                request = live.detected(WorkResult(token, [box]), self.frame)
                self.assertEqual(request.bbox, box)
                displayed.append(live.visual_regions(at)[0][0][0])
            sums.append(sum(abs(a-b) for a,b in zip(displayed, displayed[1:])))
            token = (1, len(offsets))
            live.begin_sample(token, 12.52)
            live.detected(WorkResult(token, []), self.frame)
            if is_local:
                self.assertFalse(live.visual_regions(12.52)[0])
        self.assertGreater(sums[0], 0)
        self.assertEqual(sums[1], 0)


class StableGuiTests(unittest.TestCase):
    setUp=legacy.LiveAppTests.setUp
    packet=legacy.LiveAppTests.packet

    def install(self):
        app=self.app;app.presentation=StablePresentation()
        for name in ('status','identity','emotion','liveness'):setattr(app,'display_'+name,Mock())
        return app

    def test_opening_registration_is_navigation_not_authorization(self):
        app=self.install();app._trained_recognition_enabled=True;app.registration_form=Mock()
        app.registration_prompt=Mock();app.registration_button=Mock()
        app._show_registration()
        app.registration_form.grid.assert_called_once()
        self.assertEqual(app.controller.action,'registration');self.assertFalse(app.controller.state.can_save)
        self.assertIsNone(app.controller.state.recognition)

    def test_diagnostic_json_is_raw_while_previous_sample_is_visible(self):
        app=self.install();helper=PresentationTests();helper.setUp();helper.warm();helper.begin(10.6)
        app.presentation=helper.display;app.live=helper.live;app.controller=app.live.controller
        app._ui_time=10.63;app._debug_visible=True
        app._render()
        shown=app.display_identity.set.call_args.args[0];self.assertIn('Demo Person',shown)
        payload=json.loads(app.payload.insert.call_args.args[1])
        self.assertIsNone(payload['recognition']);self.assertIsNone(payload['liveness']);self.assertIsNone(payload['emotion'])
        self.assertFalse(payload['can_save'])

    def test_previous_unknown_and_live_text_cannot_submit_test_enrollment(self):
        app=self.install();helper=PresentationTests();helper.setUp();helper.warm()
        token,request=helper.begin(10.6)
        helper.live.completed(WorkResult(token,evaluate_adapters((request,StubAdapters('Live unknown')))))
        helper.display.accept(token,request,helper.live.controller.state,10.6,10.63)
        helper.begin(10.9)
        app.presentation=helper.display;app.live=helper.live;app.controller=app.live.controller
        self.assertEqual(app.presentation.view(app.controller.state,10.93).identity,'Unknown')
        app._controlled_registration=True;app._registration_worker=Mock(busy=False);app.registration_message=Mock()
        app._start_test_enrollment()
        app._registration_worker.submit.assert_not_called()
        self.assertFalse(app.controller.state.can_save)

    def test_unchanged_stabilized_fields_and_registration_button_are_not_rewritten(self):
        app=self.install();helper=PresentationTests();helper.setUp();helper.warm()
        app.presentation=helper.display;app.live=helper.live;app.controller=app.live.controller
        app._trained_recognition_enabled=True;app.registration_prompt=Mock();app.registration_button=Mock()
        app._ui_time=10.33;app._render()
        for name in ('status','identity','emotion','liveness'):getattr(app,'display_'+name).set.reset_mock()
        helper.begin(10.6);app._ui_time=10.63;app._render()
        for name in ('status','identity','emotion','liveness'):getattr(app,'display_'+name).set.assert_not_called()
        self.assertEqual(app.registration_button.configure.call_count,1)
        self.assertFalse(app.controller.state.can_save)

    def test_render_expiry_runs_even_when_raw_state_is_unchanged(self):
        app=self.install();helper=PresentationTests();helper.setUp();helper.warm()
        app.presentation=helper.display;app.live=helper.live;app.controller=app.live.controller
        app._ui_time=10.33;app._render();helper.previews(11.11)
        app._ui_time=11.11;app._render()
        self.assertNotIn('Demo Person',app.display_identity.set.call_args.args[0])


if __name__=='__main__':unittest.main()
