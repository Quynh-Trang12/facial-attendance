"""Identity drawing is context-bound, non-authoritative and transform-aware."""
from dataclasses import asdict, replace
import unittest
from unittest.mock import Mock, patch

from facial_attendance.recognition.gallery_matching import CONTROLLED_BOX, controlled_frame
from facial_attendance.ui.identity_overlay import (
    CanvasIdentityOverlay, IdentityLabel, fit_text, identity_label, label_bounds, preview_box,
)
from facial_attendance.ui.live_preview import CameraPacket, WorkResult, evaluate_adapters
from facial_attendance.ui.presentation import DisplayResult, StablePresentation
from facial_attendance.ui.webcam_prototype import Completion, Envelope, Liveness, PrototypeController, Recognition, StubAdapters
import test_webcam_live as legacy


def recognition(request, known=True):
    return Completion(request.context, Recognition('generated-id' if known else None,
        'Generated Person' if known else None,.9,known,'fixture','a'*12,'model','b'*64,'fixture'))


class OverlayDecisionTests(unittest.TestCase):
    def setUp(self):
        ctrl=PrototypeController();req=ctrl.begin_capture(controlled_frame(),[CONTROLLED_BOX])
        ctrl.receive(StubAdapters('Live match').liveness(req));ctrl.receive(recognition(req))
        self.state=ctrl.state;self.view=DisplayResult('Previous sample','Generated Person [generated-id]',
            'Neutral','live',req.context,False)

    def label(self,state=None,view=None,**kw):
        options=dict(face_count=1,confirmed=True,processing=False,trained=True);options.update(kw)
        return identity_label(self.state if state is None else state,self.view if view is None else view,**options)

    def test_current_known_identity_comes_from_model_name(self):
        self.assertEqual(self.label(),IdentityLabel('Generated Person',True))
        self.assertEqual(fit_text(self.label(),1000,len),'Generated Person · UNVERIFIED')

    def test_unknown_uses_the_existing_rejected_match_decision(self):
        state=replace(self.state,recognition=replace(self.state.recognition,accepted=False,name=None,person_id=None))
        self.assertEqual(self.label(state,replace(self.view,identity='Unknown')),IdentityLabel('Unknown'))

    def test_processing_never_carries_a_previous_name(self):
        self.assertEqual(self.label(Envelope(ui_state='checking'),processing=True),IdentityLabel('Checking...'))
        self.assertEqual(self.label(Envelope(),confirmed=False,processing=True),IdentityLabel('Checking...'))

    def test_retained_outline_cannot_receive_a_known_name(self):
        self.assertIsNone(self.label(confirmed=False))

    def test_no_face_and_multiple_faces_have_no_identity_association(self):
        for count in (0,2,3):self.assertIsNone(self.label(face_count=count))
        self.assertIsNone(self.label(replace(self.state,ui_state='no_face')))

    def test_spoof_uncertain_error_and_missing_recognition_are_unlabelled(self):
        for state in ('spoof','uncertain','error'):
            self.assertIsNone(self.label(replace(self.state,ui_state=state)))
        self.assertIsNone(self.label(replace(self.state,recognition=None)))

    def test_old_context_and_historical_side_panel_cannot_supply_a_name(self):
        self.assertIsNone(self.label(view=replace(self.view,source_context=replace(self.state.context,face_id='old'))))
        self.assertIsNone(self.label(view=replace(self.view,historical=True)))

    def test_visual_guard_loss_clears_name_even_if_raw_result_remains_live(self):
        self.assertIsNone(self.label(view=DisplayResult('Checking','Unavailable','Unavailable','Unavailable')))

    def test_stub_identity_is_not_presented_as_a_trained_gallery_match(self):
        self.assertIsNone(self.label(trained=False))
        self.assertIsNone(self.label(replace(self.state,recognition=replace(self.state.recognition,source='stub'))))

    def test_changed_same_context_decision_must_agree_with_validated_view(self):
        state=replace(self.state,recognition=replace(self.state.recognition,name='Replacement'))
        self.assertIsNone(self.label(state))

    def test_label_computation_cannot_mutate_or_authorize_raw_state(self):
        before=asdict(self.state)
        for _ in range(10):self.label()
        self.assertEqual(asdict(self.state),before);self.assertFalse(self.state.can_save)
        self.assertFalse(hasattr(self.label(),'can_save'))


class OverlayGeometryTests(unittest.TestCase):
    def test_half_open_resize_coordinates_are_shared_with_outline(self):
        self.assertEqual(preview_box((100,80,400,300),(800,400),(400,200)),(50,40,200,150))

    def test_mirroring_maps_both_box_edges_in_preview_space(self):
        self.assertEqual(preview_box((100,80,400,300),(800,400),(400,200),mirrored=True),(200,40,350,150))

    def test_horizontal_and_vertical_motion_move_label_anchor(self):
        first=label_bounds((40,80,140,180),(640,360),100,24)
        moved=label_bounds((70,100,170,200),(640,360),100,24)
        self.assertEqual(moved[0]-first[0],30);self.assertEqual(moved[1]-first[1],20)

    def test_label_prefers_above_then_below_at_top_edge(self):
        self.assertLess(label_bounds((30,100,160,220),(640,360),140,24)[3],100)
        self.assertGreater(label_bounds((30,0,160,220),(640,360),140,24)[1],220)

    def test_label_stays_inside_all_preview_edges(self):
        for box in ((0,0,640,360),(600,0,640,60),(600,300,640,360),(0,300,40,360)):
            x1,y1,x2,y2=label_bounds(box,(640,360),180,24)
            self.assertGreaterEqual(x1,0);self.assertGreaterEqual(y1,0)
            self.assertLessEqual(x2,640);self.assertLessEqual(y2,360)

    def test_small_preview_omits_impossible_labels(self):
        self.assertIsNone(label_bounds((0,0,5,5),(5,5),100,24))
        self.assertIsNone(fit_text(IdentityLabel('Name',True),5,len))

    def test_long_name_is_ellipsized_without_losing_unverified_marker(self):
        result=fit_text(IdentityLabel('Very long registered name '*4,True),40,len)
        self.assertLessEqual(len(result),40);self.assertTrue(result.endswith('… · UNVERIFIED'))

    def test_invalid_regions_are_rejected_before_rendering(self):
        for box in ((-1,0,20,30),(20,10,20,30),(0,0,900,500)):
            with self.assertRaises(ValueError):preview_box(box,(800,400),(400,200))

    def test_unchanged_content_reuses_canvas_items_and_font_measurement(self):
        canvas=Mock();font=Mock();font.measure.side_effect=len;font.metrics.return_value=16
        with patch('tkinter.font.Font',return_value=font):painter=CanvasIdentityOverlay(canvas)
        painter.update(IdentityLabel('Unknown'),(40,80,140,180),(640,360))
        calls=font.measure.call_count;canvas.reset_mock()
        for _ in range(20):painter.update(IdentityLabel('Unknown'),(40,80,140,180),(640,360))
        canvas.create_rectangle.assert_not_called();canvas.create_text.assert_not_called();canvas.itemconfigure.assert_not_called()
        self.assertEqual(font.measure.call_count,calls)
        painter.hide();painter.update(IdentityLabel('Checking...'),(40,80,140,180),(640,360))
        canvas.create_text.assert_not_called()

    def test_mode_change_redraws_the_same_name_with_or_without_suffix(self):
        canvas = Mock()
        font = Mock()
        font.measure.side_effect = len
        font.metrics.return_value = 16
        with patch('tkinter.font.Font', return_value=font):
            painter = CanvasIdentityOverlay(canvas)
        label = IdentityLabel('Generated Person', known=True)
        box, size = (40, 80, 140, 180), (640, 360)

        for show_unverified, text in ((False, 'Generated Person'),
                                      (True, 'Generated Person · UNVERIFIED'),
                                      (False, 'Generated Person')):
            painter.update(label, box, size, show_unverified=show_unverified,
                           match_box_width=not show_unverified)
            canvas.itemconfigure.assert_called_with(painter.items[1], text=text, state='normal',
                                                   anchor='nw' if show_unverified else 'n')
            if not show_unverified:
                canvas.coords.assert_any_call(painter.items[0], 40, 53, 140, 77)
                canvas.coords.assert_called_with(painter.items[1], 90, 57)
        canvas.create_text.assert_called_once()

    def test_narrow_outline_ellipsizes_name_and_preserves_exact_width_at_edge(self):
        canvas = Mock()
        font = Mock()
        font.measure.side_effect = len
        font.metrics.return_value = 16
        with patch('tkinter.font.Font', return_value=font):
            painter = CanvasIdentityOverlay(canvas)
        painter.update(IdentityLabel('Generated Person', known=True), (618, 80, 640, 180), (640, 360),
                       show_unverified=False, match_box_width=True)
        canvas.coords.assert_any_call(painter.items[0], 618, 53, 640, 77)
        canvas.coords.assert_called_with(painter.items[1], 629, 57)
        canvas.itemconfigure.assert_called_with(painter.items[1], text='Generated…', state='normal', anchor='n')


class OverlayPipelineTests(unittest.TestCase):
    setUp=legacy.LiveAppTests.setUp

    def install(self):
        app=self.app;app.presentation=StablePresentation();app._identity_overlay=Mock();app._preview_mirrored=False
        app._trained_recognition_enabled=True;app.registration_prompt=Mock();app.registration_button=Mock()
        for name in ('status','identity','emotion','liveness'):setattr(app,'display_'+name,Mock())
        app.adapters.recognition=recognition;self.frame=controlled_frame();self.packet()
        return app

    def packet(self,seq=1,frame=None):
        frame=getattr(self,'frame',controlled_frame()) if frame is None else frame
        packet=CameraPacket(self.app._generation,seq,self.now,frame);self.app.feed.latest.return_value=packet
        return packet

    def complete(self):
        app=self.app;app.detector.finish([CONTROLLED_BOX]);app._tick()
        app.inference.finish(evaluate_adapters(app.inference.job[1]));app._tick()

    def test_known_label_is_replaced_with_checking_on_each_new_capture(self):
        app=self.install();app._tick();self.complete()
        self.assertTrue(app._identity_overlay.update.call_args.args[0].known)
        self.now+=app.ANALYSIS_INTERVAL+.01;self.packet(2);app._tick()
        self.assertEqual(app._identity_overlay.update.call_args.args[0],IdentityLabel('Checking...'))
        self.assertIsNone(app.controller.state.recognition);self.assertFalse(app.controller.state.can_save)
        self.complete();self.assertTrue(app._identity_overlay.update.call_args.args[0].known)

    def test_mirrored_label_reads_normally_and_uses_the_mirrored_box(self):
        app = self.install()
        app._preview_mirrored = True
        app._tick()
        self.complete()
        label, coordinates, size = app._identity_overlay.update.call_args.args
        self.assertEqual(label, IdentityLabel("Generated Person", known=True))
        self.assertEqual(coordinates, preview_box(CONTROLLED_BOX, (640, 360), size, mirrored=True))

    def test_pixel_replacement_hides_old_name_before_next_detection(self):
        app=self.install();app._tick();self.complete();app._identity_overlay.hide.reset_mock()
        self.now+=.04;self.packet(2,255-self.frame);app._tick()
        app._identity_overlay.hide.assert_called();self.assertFalse(app.controller.state.can_save)

    def test_loss_multiple_faces_and_return_never_inherit_old_name(self):
        for boxes in ([],[CONTROLLED_BOX,(20,20,100,100)]):
            self.setUp();app=self.install();app._tick();self.complete()
            self.now+=app.ANALYSIS_INTERVAL+.01;self.packet(2);app._tick();app.detector.finish(boxes);app._tick()
            app._identity_overlay.hide.assert_called()
            self.now+=app.ANALYSIS_INTERVAL+.01;self.packet(3);app._tick()
            self.assertEqual(app._identity_overlay.update.call_args.args[0],IdentityLabel('Checking...'))

    def test_retry_clears_overlay_items_and_old_capture(self):
        app=self.install();app._tick();self.complete();app.retry()
        app._identity_overlay.reset.assert_called_once();app._identity_overlay.hide.assert_called()
        self.assertIsNone(app.controller.state.recognition);self.assertFalse(app.controller.state.can_save)

    def test_stale_model_completion_cannot_restore_a_name(self):
        app=self.install();app._tick();app.detector.finish([CONTROLLED_BOX]);app._tick()
        old=app.inference.job
        self.now+=app.ANALYSIS_INTERVAL+.01;self.packet(2);app._tick()
        app.inference.finish(evaluate_adapters(old[1]));app._tick()
        self.assertIsNone(app.controller.state.recognition)
        self.assertFalse(app._identity_overlay.update.call_args.args[0].known)

    def test_both_modes_use_same_overlay_and_same_authorization(self):
        app=self.install();app._tick();self.complete();before=asdict(app.controller.state)
        app._developer_mode=False;app._render();label=app._identity_overlay.update.call_args.args[0]
        app._developer_mode=True;app._render()
        self.assertEqual(app._identity_overlay.update.call_args.args[0],label)
        self.assertEqual(asdict(app.controller.state),before);self.assertFalse(app.controller.state.can_save)

    def test_local_workspace_centers_name_without_suffix_in_both_modes(self):
        app = self.install()
        app._local_workspace = True
        app._display_regions = Mock(return_value=([CONTROLLED_BOX], (640, 360), False))
        request = PrototypeController().begin_capture(self.frame, [CONTROLLED_BOX])
        state = Envelope(ui_state='preview', context=request.context)
        view = DisplayResult('Preview', 'Generated Person', 'Neutral', 'live',
                             source_context=request.context, identity_name='Generated Person')
        for developer in (False, True, False):
            app._developer_mode = developer
            app._draw_identity_overlay(state, view, self.now)
            call = app._identity_overlay.update.call_args
            self.assertEqual(call.args[0], IdentityLabel('Generated Person', known=True))
            self.assertFalse(call.kwargs['show_unverified'])
            self.assertTrue(call.kwargs['match_box_width'])
        self.assertFalse(state.can_save)


if __name__=='__main__':unittest.main()
