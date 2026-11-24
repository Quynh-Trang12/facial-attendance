"""Modes format existing state; diagnostics never consume or authorize it."""
from dataclasses import asdict, replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from facial_attendance.ui.dashboard import Dashboard, Result, camera_status, corner_segments, preview_phase, preview_results
from facial_attendance.ui.developer_view import DeveloperDiagnostics, runtime_report, standard_fields
from facial_attendance.ui.diagnostics import Diagnostics
from facial_attendance.ui.presentation import DisplayResult
from facial_attendance.ui.emotion_presentation import EmotionDisplay
from facial_attendance.ui.webcam_prototype import Envelope, Liveness, OperationError
import test_stable_presentation as previous
import test_webcam_live as legacy


class StandardFieldsTests(unittest.TestCase):
    def setUp(self):
        self.view=DisplayResult('Previous sample • UNVERIFIED • display only','Unknown','Neutral','live')
        self.emotion=EmotionDisplay('Neutral','latest_valid_sample')

    def test_simple_headings_omit_internal_values_and_preserve_safety(self):
        fields=standard_fields(Envelope(),self.view,self.emotion)
        self.assertEqual(fields['identity'],'Identity: Unknown person')
        self.assertEqual(fields['emotion'],'Emotion: Neutral')
        self.assertIn('SIMULATED',fields['liveness']);self.assertIn('Unverified',fields['status'])
        for word in ('800','450','confidence','hash','continuity','capture_id','smoothed'):
            self.assertNotIn(word,' '.join(fields.values()))

    def test_emotion_keeps_training_labels_without_reinterpretation(self):
        for label in ('Negative','Neutral','Positive'):
            self.assertEqual(standard_fields(Envelope(),self.view,EmotionDisplay(label,'test'))['emotion'],f'Emotion: {label}')

    def test_no_face_multiple_spoof_and_uncertain_do_not_claim_success(self):
        for state,text in (('no_face','No face detected'),('multiple_faces','Multiple faces detected'),
                           ('spoof','Simulated spoof — blocked'),('uncertain','Uncertain liveness — retry')):
            self.assertEqual(standard_fields(Envelope(ui_state=state),self.view,self.emotion)['status'],text)

    def test_capture_and_model_errors_have_distinct_supported_statuses(self):
        for module,text in (('capture','Camera unavailable'),('recognition','Model unavailable')):
            state=Envelope(ui_state='error',errors=(OperationError(module,'inference_error','failed',True),))
            self.assertEqual(standard_fields(state,self.view,self.emotion)['status'],text)

    def test_empty_state_is_checking_not_fake_attendance(self):
        fields=standard_fields(Envelope(),None,None)
        self.assertEqual(fields['status'],'Checking...');self.assertEqual(fields['identity'],'Identity: Unavailable')

    def test_dashboard_retains_name_and_expression_without_ever_verifying_an_identity(self):
        view = replace(self.view, identity="Vỹ [test-001]", identity_name="Vỹ")
        for decision in ("live", "spoof", "uncertain"):
            with self.subTest(decision=decision):
                scan = SimpleNamespace(decision=decision, local_time="17:14:31")
                identity, liveness, emotion = preview_results(Envelope(ui_state="preview"), view, self.emotion,
                                                               scan=scan, trained_liveness=True, trained_emotion=True)
                self.assertEqual(identity.value, "Vỹ")
                self.assertIn("Unverified", identity.detail)
                self.assertEqual(liveness.value, decision)
                self.assertEqual(liveness.detail, "Last scan at 17:14:31")
                self.assertIn("Neutral", emotion.value)

    def test_dashboard_differentiates_stub_unavailable_and_no_face(self):
        state = Envelope(ui_state="preview", liveness=Liveness("live", "example", .1))
        args = dict(scan=None, trained_liveness=False, trained_emotion=False)
        self.assertEqual(preview_results(state, self.view, None, **args)[1].value, "Simulated")
        self.assertEqual(preview_results(Envelope(ui_state="preview"), self.view, None, **args)[1].value,
                         "Model unavailable")
        identity, liveness, emotion = preview_results(
            Envelope(ui_state="no_face"), self.view, None,
            scan=SimpleNamespace(decision="spoof"), trained_liveness=True, trained_emotion=True)
        self.assertEqual(identity.value, "No face detected")
        self.assertEqual(liveness.value, "uncertain")
        self.assertEqual(liveness.detail, "No current liveness result")
        self.assertEqual(emotion.value, "Waiting for a face")
        self.assertEqual(camera_status(False, True).tone, "error")
        self.assertEqual(camera_status(True, False).tone, "success")
        self.assertEqual(corner_segments((100, 50, 160, 100))[0][2:4], (100, 50))

    def test_dashboard_does_not_clear_guarded_results_between_detector_samples(self):
        view = replace(self.view, identity_name="Vỹ", historical=False)
        args = dict(scan=SimpleNamespace(decision="live", local_time="17:14:31"),
                    trained_liveness=True, trained_emotion=True)
        displayed = preview_results(Envelope(ui_state="preview"), view, self.emotion, **args)
        pending = preview_phase("idle", True, "preview")
        self.assertEqual(preview_results(Envelope(), replace(view, historical=True), self.emotion,
                                         **args, phase=pending), displayed)
        self.assertEqual(preview_phase("idle", True, "no_face"), "no_face")
        lost = preview_results(Envelope(ui_state="no_face"), None, None, **args)
        self.assertEqual(lost[0].value, "No face detected")
        self.assertEqual(lost[1].value, "uncertain")

    def test_dashboard_does_not_redraw_unchanged_cards_or_liveness_value(self):
        dashboard = Dashboard.__new__(Dashboard)
        dashboard.cards = tuple((Mock(), Mock()) for _ in range(3))
        base = (Result("Vỹ", "Unverified"), Result("Live", "Last scan at 12:00:00"),
                Result("Neutral", "Facial expression preview"))
        dashboard._results = base
        dashboard.set_results((base[0], replace(base[1], detail="Last scan at 12:00:03"), base[2]))
        for value, _ in dashboard.cards:
            value.configure.assert_not_called()
        dashboard.cards[0][1].configure.assert_not_called()
        dashboard.cards[1][1].configure.assert_called_once_with(text="Last scan at 12:00:03")
        dashboard.cards[2][1].configure.assert_not_called()


class ModesTests(unittest.TestCase):
    setUp=legacy.LiveAppTests.setUp
    packet=legacy.LiveAppTests.packet

    def install(self):
        app=self.app;app._mode_supported=True;app._developer_mode=False;app._debug_visible=False
        app._controlled_registration=False;app._source_name='Test source';app._runtime_info={'camera_index':7}
        app.mode_button=Mock();app.diagnostics_button=Mock();app.source_label=Mock();app._diagnostics_view=None
        app.presentation=previous.StablePresentation()
        for name in ('status','identity','emotion','liveness'):setattr(app,'display_'+name,Mock())
        return app

    def test_switching_modes_preserves_every_shared_pipeline_object_and_raw_state(self):
        app=self.install();before=asdict(app.controller.state)
        objects=tuple(id(getattr(app,n)) for n in ('live','controller','adapters','feed','detector','inference'))
        for value in (True,False)*10:app._set_mode(value,open_diagnostics=False)
        self.assertFalse(app._developer_mode);self.assertEqual(asdict(app.controller.state),before)
        self.assertEqual(tuple(id(getattr(app,n)) for n in ('live','controller','adapters','feed','detector','inference')),objects)
        app.feed.restart.assert_not_called()

    def test_switching_modes_does_not_clear_registration_inputs(self):
        app=self.install();app.registration_id=Mock();app.registration_name=Mock()
        app._set_mode(True,open_diagnostics=False);app._set_mode(False)
        app.registration_id.set.assert_not_called();app.registration_name.set.assert_not_called()

    def test_standard_mode_hides_and_reuses_diagnostics(self):
        app=self.install();app._diagnostics_view=Mock();app._debug_visible=True
        app._set_mode(False)
        app._diagnostics_view.hide.assert_called_once();self.assertFalse(app._debug_visible)

    def test_diagnostics_control_is_only_packed_in_developer_mode(self):
        app=self.install()
        app._set_mode(False)
        app.diagnostics_button.pack_forget.assert_called_once()
        app.diagnostics_button.pack.assert_not_called()
        app._set_mode(True,open_diagnostics=False)
        app.diagnostics_button.pack.assert_called_once_with(anchor='w',pady=(4,8))
        app.mode_button.pack_forget.assert_not_called()

    def test_return_to_standard_hides_window_and_clears_technical_text(self):
        app=self.install();app._diagnostics_view=Mock();app._debug_visible=True
        app._set_mode(False)
        app._diagnostics_view.hide.assert_called_once()
        app.diagnostics_button.pack_forget.assert_called_once()
        self.assertFalse(app._debug_visible)
        self.assertNotIn('800',app.hint.set.call_args.args[0])
        self.assertNotIn('450',app.hint.set.call_args.args[0])
        self.assertEqual(app.source_label.set.call_args.args[0],'Live preview')
        self.assertTrue(app.display_emotion.set.call_args.args[0].startswith('Emotion: '))

    def test_return_to_developer_reopens_existing_window_without_reset(self):
        app=self.install();app._render=Mock();inspector=Mock(window=Mock(),payload=Mock())
        app._diagnostics_view=inspector;app._debug_visible=True
        before=asdict(app.controller.state)
        app._set_mode(False);app._set_mode(True)
        inspector.hide.assert_called_once();inspector.show.assert_called_once()
        self.assertTrue(app._debug_visible);self.assertIs(app._diagnostics_view,inspector)
        self.assertEqual(asdict(app.controller.state),before)
        app.feed.restart.assert_not_called()

    def test_repeated_open_close_uses_one_window_and_leaves_main_running(self):
        app=self.install();app._render=Mock();app._debug_visible=False
        inspector=Mock(window=Mock(),payload=Mock())
        with patch('facial_attendance.ui.developer_view.DeveloperDiagnostics',return_value=inspector) as factory:
            app._toggle_debug();app._toggle_debug();app._toggle_debug()
        factory.assert_called_once();self.assertEqual(inspector.show.call_count,2)
        inspector.hide.assert_called_once();self.assertFalse(app._closed)
        app.feed.close.assert_not_called();app.feed.restart.assert_not_called()

    def test_runtime_report_keeps_current_raw_and_historical_display_separate(self):
        app=self.install();helper=previous.PresentationTests();helper.setUp();helper.warm();helper.begin(10.6)
        app.live=helper.live;app.controller=helper.live.controller;app.presentation=helper.display
        app._last_display_view=helper.display.view(app.controller.state,10.63)
        app.emotion_presentation=None;app.metrics=Diagnostics(False);app._drawn_sequence=1
        app.adapters=SimpleNamespace();before=asdict(app.controller.state)
        report=runtime_report(app,10.63)
        self.assertIsNone(report['raw_current_sys02']['recognition'])
        self.assertIn('Demo Person',report['presentation']['identity_liveness_display']['identity'])
        self.assertFalse(report['authorization']['can_save']);self.assertFalse(report['authorization']['genuine_registration_available'])
        self.assertIsNone(report['liveness']['trained_threshold']);self.assertIsNone(report['liveness']['trained_model_metadata'])
        self.assertEqual(asdict(app.controller.state),before)

    def test_runtime_report_has_no_gallery_embeddings_or_checkpoint_paths(self):
        app=self.install();app.metrics=Diagnostics(False);app._drawn_sequence=1
        app.adapters=SimpleNamespace(matcher=SimpleNamespace(_gallery=SimpleNamespace(people=[object()])))
        report=runtime_report(app,self.now)
        self.assertEqual(report['recognition']['gallery_people_count'],1)
        self.assertNotIn('embeddings',str(report));self.assertNotIn('/Users/',str(report))

    def test_runtime_does_not_fabricate_unmeasured_latencies(self):
        app=self.install();app.metrics=Diagnostics(False);app._drawn_sequence=None;app.adapters=SimpleNamespace()
        self.assertIsNone(runtime_report(app,self.now)['measured_performance'])

    def test_diagnostics_refresh_is_bounded_and_hidden_refresh_does_nothing(self):
        panel=DeveloperDiagnostics.__new__(DeveloperDiagnostics)
        panel.visible=False;panel._next_refresh=0;panel.runtime=Mock();panel.refresh_count=0;panel._last_runtime=None
        panel.paused=False;panel._snapshot_meta={};panel.status=Mock();panel.notebook=Mock()
        panel.notebook.index.return_value=0
        panel.runtime.yview.return_value=(0.,.5)
        panel.runtime.xview.return_value=(0.,1.);panel.runtime.tag_ranges.return_value=()
        with patch('facial_attendance.ui.developer_view.runtime_report',side_effect=lambda *a:{'raw_current_sys02':{},'value':1}) as builder:
            panel.refresh(Mock(),0.);builder.assert_not_called();panel.visible=True
            for i in range(100):panel.refresh(Mock(),i/100)
            self.assertLessEqual(builder.call_count,4)
            self.assertEqual(panel.runtime.insert.call_count,1)

    def test_standard_and_developer_keep_same_false_save_decision_on_blocked_states(self):
        app=self.install()
        for state in ('no_face','multiple_faces','spoof','uncertain','error'):
            app.controller.state=Envelope(ui_state=state)
            for value in (True,False):
                app._set_mode(value,open_diagnostics=False)
                self.assertFalse(app.controller.state.can_save)


class ReadOnlyMetricsTests(unittest.TestCase):
    def test_peek_does_not_reset_or_change_logged_measurements(self):
        clock=[10.];metrics=Diagnostics(True,clock=lambda:clock[0])
        metrics.count('captured',10);metrics.duration('emotion',.002);clock[0]=11.
        first=metrics.peek();second=metrics.peek();logged=metrics.snapshot(force=True)
        self.assertEqual(first,second);self.assertEqual(first,logged)
        clock[0]=12.
        self.assertEqual(metrics.peek()['counts'],{})


if __name__=='__main__':unittest.main()
