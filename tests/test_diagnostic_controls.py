"""Frozen diagnostic views never freeze or authorize the running application."""
from dataclasses import asdict
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from facial_attendance.ui.developer_view import DeveloperDiagnostics
from facial_attendance.ui.diagnostic_snapshot import CompletedDiagnostic, privacy_safe_snapshot, replace_text
from facial_attendance.ui.live_preview import WorkResult, evaluate_adapters
from facial_attendance.ui.webcam_prototype import Emotion, Envelope, StubAdapters
import test_webcam_live as legacy

STAMP='2026-11-18T13:00:00.000+00:00'


class Text:
    def __init__(self, value=''):
        self.value=value;self.selection=None;self.y=(.3,.7);self.x=(.2,.8);self.writes=0
    def offset(self,index):
        if index in ('end','end-1c'):return len(self.value)
        if index=='1.0':return 0
        return int(re.fullmatch(r'1\.0\+(\d+)c',str(index))[1])
    def get(self,start,end):return self.value[self.offset(start):self.offset(end)]
    def configure(self,**kw):pass
    def delete(self,*a):self.value='';self.selection=None
    def insert(self,index,value):self.value=value;self.writes+=1
    def tag_ranges(self,name):return tuple(f'1.0+{v}c' for v in self.selection) if self.selection else ()
    def tag_add(self,name,start,end):self.selection=(self.offset(start),self.offset(end))
    def yview(self):return self.y
    def xview(self):return self.x
    def yview_moveto(self,value):self.y=(value,self.y[1])
    def xview_moveto(self,value):self.x=(value,self.x[1])


def panel():
    view=DeveloperDiagnostics.__new__(DeveloperDiagnostics)
    view.visible=True;view.paused=False;view.raw_refresh_requested=False
    view.runtime=Text('{"runtime": true}');view.payload=Text('{"ui_state":"checking","can_save":false}')
    view.completed=Text();view.texts=[view.runtime,view.payload,view.completed]
    view.notebook=Mock();view.notebook.index.side_effect=lambda n:n
    view.selected=0
    view.notebook.select.side_effect=lambda index=None: view.selected if index is None else setattr(view,'selected',index)
    view.status=Mock();view.notice=Mock();view.pause_button=Mock();view.window=Mock()
    view._snapshot_meta={0:{'source':'runtime','timestamp':STAMP},1:{'source':'current_sys02','timestamp':STAMP}}
    view._next_refresh=0.;view._last_runtime=None;view.refresh_count=0;view._latest_completed=lambda:None
    return view


class DiagnosticControlTests(unittest.TestCase):
    def test_pause_freezes_both_buffers_and_snapshot_timestamps(self):
        view=panel();before=[v.value for v in view.texts];metadata=dict(view._snapshot_meta)
        view.toggle_pause()
        with patch('facial_attendance.ui.developer_view.runtime_report') as builder:
            view.refresh(Mock(),100.);builder.assert_not_called()
        self.assertFalse(view.update_raw(Envelope(ui_state='no_face')))
        self.assertEqual([v.value for v in view.texts],before);self.assertEqual(view._snapshot_meta,metadata)
        text=view.status.set.call_args.args[0];self.assertIn('PAUSED',text);self.assertIn(STAMP,text)
        self.assertIn('not current authorization',text)

    def test_resume_requests_latest_raw_even_if_state_has_not_changed(self):
        view=panel();view.toggle_pause();view.toggle_pause()
        self.assertFalse(view.paused);self.assertTrue(view.raw_refresh_requested)
        self.assertEqual(view._next_refresh,0.)
        self.assertTrue(view.update_raw(Envelope(ui_state='no_face')))
        self.assertFalse(view.raw_refresh_requested)
        self.assertEqual(json.loads(view.payload.value)['ui_state'],'no_face')

    def test_live_status_has_the_displayed_snapshot_timestamp(self):
        view=panel();view.selected=1;view._update_status()
        self.assertIn('LIVE',view.status.set.call_args.args[0]);self.assertIn(STAMP,view.status.set.call_args.args[0])

    def test_copy_copies_the_exact_selected_tab_without_refetching_state(self):
        view=panel();view.selected=1;view.toggle_pause()
        self.assertTrue(view.copy_json())
        view.window.clipboard_append.assert_called_once_with(view.payload.value)
        self.assertTrue(view.paused)

    def test_copy_works_for_completed_historical_json(self):
        view=panel();view.completed.value='{"ui_state":"live","can_save":false}';view.selected=2
        self.assertTrue(view.copy_json());view.window.clipboard_append.assert_called_once_with(view.completed.value)

    def test_hidden_copy_export_and_capture_have_no_side_effects(self):
        view=panel();view.visible=False
        self.assertFalse(view.copy_json());self.assertFalse(view.export_json());self.assertFalse(view.capture_completed())
        view.window.clipboard_clear.assert_not_called()

    def test_empty_copy_does_not_clear_existing_clipboard(self):
        view=panel();view.runtime.value=''
        self.assertFalse(view.copy_json());view.window.clipboard_clear.assert_not_called()

    def test_export_sanitizes_the_displayed_snapshot_and_retains_its_timestamp(self):
        view=panel();view.selected=1;view.paused=True
        raw={'ui_state':'live','can_save':False,'recognition':{'source':'model','accepted':True,'score':.9,
             'person_id':'PRIVATE-ID','name':'PRIVATE-NAME','checkpoint_sha256':'a'*64},
             'errors':[{'message':'/Users/PRIVATE/file.pt'}],'frame':'PRIVATE-IMAGE','embeddings':[1,2,3]}
        view.payload.value=json.dumps(raw)
        with TemporaryDirectory() as directory:
            path=Path(directory)/'snapshot.json'
            with patch('tkinter.filedialog.asksaveasfilename',return_value=str(path)):
                self.assertTrue(view.export_json())
            text=path.read_text();export=json.loads(text)
        self.assertNotIn('PRIVATE',text);self.assertNotIn('embeddings',text)
        self.assertFalse(export['authorization']['can_save']);self.assertTrue(export['snapshot']['historical'])
        self.assertEqual(export['snapshot']['timestamp'],STAMP)
        self.assertEqual(export['recognition']['checkpoint_sha256'],'a'*64)
        self.assertIn('PRIVATE-NAME',view.payload.value)  # On-screen raw data is not rewritten.

    def test_export_cancel_does_not_write_or_change_pause_state(self):
        view=panel();view.paused=True
        with patch('tkinter.filedialog.asksaveasfilename',return_value=''),patch.object(Path,'write_text') as write:
            self.assertFalse(view.export_json());write.assert_not_called()
        self.assertTrue(view.paused)

    def test_export_error_is_reported_without_leaking_destination(self):
        view=panel()
        with patch('tkinter.filedialog.asksaveasfilename',return_value='/private/SECRET.json'), \
                patch.object(Path,'write_text',side_effect=OSError('SECRET')):
            self.assertFalse(view.export_json())
        self.assertNotIn('SECRET',view.notice.set.call_args.args[0])

    def test_invalid_displayed_json_cannot_be_exported(self):
        view=panel();view.runtime.value='not json'
        with patch('tkinter.filedialog.asksaveasfilename') as dialog:
            self.assertFalse(view.export_json());dialog.assert_not_called()

    def test_capture_completed_is_frozen_historical_and_not_current_checking(self):
        view=panel();state=Envelope(ui_state='live',emotion=Emotion(1,'Neutral',.8))
        view._latest_completed=lambda:CompletedDiagnostic(state,STAMP)
        self.assertTrue(view.capture_completed())
        self.assertEqual(json.loads(view.completed.value)['emotion']['label'],'Neutral')
        self.assertEqual(json.loads(view.payload.value)['ui_state'],'checking')
        self.assertTrue(view.paused);self.assertEqual(view.selected,2)
        self.assertIn('HISTORICAL',view.status.set.call_args.args[0]);self.assertIn(STAMP,view.status.set.call_args.args[0])
        self.assertFalse(state.can_save)

    def test_no_completed_result_does_not_fabricate_one_or_pause(self):
        view=panel();self.assertFalse(view.capture_completed());self.assertFalse(view.paused)
        self.assertEqual(view.completed.value,'')

    def test_resume_from_completed_tab_returns_to_live_raw_tab(self):
        view=panel();view.selected=2;view.paused=True;view.toggle_pause()
        self.assertEqual(view.selected,1);self.assertTrue(view.raw_refresh_requested)

    def test_live_refresh_preserves_selection_and_both_scroll_axes(self):
        text=Text('prefix Neutral suffix');text.selection=(7,14)
        replace_text(text,'prefix Neutral changed')
        self.assertEqual(text.selection,(7,14));self.assertEqual(text.y[0],.3);self.assertEqual(text.x[0],.2)

    def test_live_refresh_can_relocate_unique_selected_text(self):
        text=Text('Neutral');text.selection=(0,7)
        replace_text(text,'more text Neutral')
        self.assertEqual(text.get(*text.tag_ranges('sel')),'Neutral')

    def test_removed_selection_is_not_silently_applied_to_different_text(self):
        text=Text('Neutral');text.selection=(0,7);replace_text(text,'Positive')
        self.assertIsNone(text.selection)

    def test_export_allowlist_excludes_arbitrary_nested_strings_and_arrays(self):
        data={'runtime':{'camera_status':'/Users/SECRET','face_detector':'SECRET'},
              'recognition':{'checkpoint':{'model_id':'SECRET','path':'SECRET'},'raw':{'name':'SECRET','person_id':'SECRET','embedding':[1]*512}},
              'emotion':{'display_reason':'SECRET','raw_current_prediction':{'label':'SECRET'}},
              'liveness':{'raw':{'reason':'SECRET'}},'unknown':{'credentials':'SECRET','image':'SECRET'},
              'measured_performance':{'timings_ms':{'SECRET':{'mean':99}}}}
        exported=privacy_safe_snapshot(data,{'timestamp':STAMP,'source':'runtime'},paused=False)
        self.assertNotIn('SECRET',json.dumps(exported));self.assertNotIn('embedding',json.dumps(exported).replace('embedding_dimensions',''))

    def test_export_rejects_nonfinite_values_and_never_serializes_paths_from_metadata(self):
        data={'ui_state':'live','can_save':False,'emotion':{'source':'model','label':'Neutral','confidence':float('nan')}}
        exported=privacy_safe_snapshot(data,{'source':'/Users/SECRET','timestamp':'SECRET'})
        self.assertIsNone(exported['emotion']['confidence']);self.assertIsNone(exported['snapshot']['source'])
        self.assertNotIn('SECRET',json.dumps(exported,allow_nan=False))


class PausedPipelineTests(unittest.TestCase):
    setUp=legacy.LiveAppTests.setUp
    packet=legacy.LiveAppTests.packet
    run_one_face=legacy.LiveAppTests.run_one_face

    def test_preview_and_new_inference_continue_while_both_diagnostic_tabs_pause(self):
        app=self.app;app._diagnostics_view=Mock(paused=True,raw_refresh_requested=False);app._debug_visible=True
        self.run_one_face();first=app._completed_diagnostic
        self.now+=app.ANALYSIS_INTERVAL+.01;self.packet(2);app._tick();app.detector.finish([legacy.BOX]);app._tick()
        app.inference.finish(evaluate_adapters(app.inference.job[1]));app._tick()
        self.assertGreater(app._draw_frame.call_count,1)
        self.assertNotEqual(first.state.context,app._completed_diagnostic.state.context)
        self.assertFalse(app.controller.state.can_save);app.payload.insert.assert_not_called()

    def test_paused_diagnostics_do_not_delay_authoritative_face_loss_invalidation(self):
        self.run_one_face();app=self.app;record=app._completed_diagnostic
        app._diagnostics_view=Mock(paused=True,raw_refresh_requested=False);app._debug_visible=True
        self.now+=app.ANALYSIS_INTERVAL+.01;self.packet(2);app._tick();app.detector.finish([]);app._tick()
        self.assertEqual(app.controller.state.ui_state,'no_face');self.assertIsNone(app.controller.state.recognition)
        self.assertIsNone(app.controller.state.liveness);self.assertFalse(app.controller.state.can_save)
        self.assertIs(app._completed_diagnostic,record)  # Diagnostic history only.

    def test_stale_completion_cannot_replace_latest_completed_diagnostic(self):
        self.run_one_face();app=self.app;record=app._completed_diagnostic
        app.live.begin_sample((1,99),self.now)
        app.inference.result=WorkResult((1,1),())
        app._tick();self.assertIs(app._completed_diagnostic,record)


if __name__=='__main__':unittest.main()
