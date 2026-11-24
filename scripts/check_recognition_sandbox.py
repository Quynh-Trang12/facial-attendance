"""Generated-input native sandbox verification; never physical evaluation evidence."""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from threading import Event

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face-checkpoint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite evidence')
    import cv2
    import tkinter as tk
    from facial_attendance.recognition.inference import FaceInference
    from facial_attendance.recognition.gallery_matching import controlled_frame,CONTROLLED_BOX
    from facial_attendance.ui.recognition_sandbox import RecognitionSandboxApp
    from facial_attendance.ui.webcam_prototype import PrototypeController
    cv2.setNumThreads(1);model=FaceInference(args.face_checkpoint).load()
    scene={'sample':1,'boxes':[CONTROLLED_BOX],'hold':False,'camera_ok':True};gate=Event();gate.set();cameras=[]
    original=model.embed
    def embed(*a,**kw):
        if scene['hold']:gate.wait(2)
        return original(*a,**kw)
    model.embed=embed
    class Camera:
        def __init__(self):cameras.append(self);self.reads=0
        def isOpened(self):return True
        def read(self):self.reads+=1;return scene['camera_ok'],controlled_frame(scene['sample'])
        def release(self):pass
    temporary=TemporaryDirectory();root=tk.Tk()
    app=RecognitionSandboxApp(root,model,Camera,lambda frame:list(scene['boxes']),source_kind='generated',repository=temporary.name)
    model_id=id(model.model);report={'source':'generated only','physical_webcam_tested':False,'checks':[],'callback_errors':[]}
    export=[None];ref_target=[1];old_result=[None];original_geometry=[None]
    def panel_geometry():
        root.update_idletasks()
        def descendants(w):
            for child in w.winfo_children():
                yield child
                yield from descendants(child)
        return tuple((str(w),w.winfo_rootx(),w.winfo_rooty(),w.winfo_width(),w.winfo_height())
                     for w in (app.status_panel.frame,*descendants(app.status_panel.frame)))
    def check(name,condition):
        if not condition:raise AssertionError(name)
        report['checks'].append(name)
    def wait(predicate,callback):
        if predicate():callback()
        else:root.after(15,lambda:wait(predicate,callback))
    def ready():return app.reference_button.instate(['!disabled'])
    def finish(error=None):
        scene['hold']=False;gate.set()
        if error:report['error']=str(error)
        report['final_state']={'references':len(app.session.references),'records':len(app.session.records),
                               'status':app.status.get(),'consent':app.session.consent,'camera_active':app.feed is not None}
        report['source_sha256']={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')}
        app.close()
    def failed(kind,value,tb):report['callback_errors'].append(f'{kind.__name__}: {value}');finish(value)
    root.report_callback_exception=failed
    def start():
        original_geometry[0]=panel_geometry()
        check('status_panel_fixed_420_by_360',app.status_panel.frame.winfo_width()==420 and app.status_panel.frame.winfo_height()==360)
        check('explicit_sandbox_title_and_empty_isolated_gallery','SANDBOX' in root.title() and app.session.test_ids==())
        check('startup_guides_to_consent_and_camera',app.status_panel.variables['guide'].get().startswith('Confirm consent')
              and app.start_button.winfo_ismapped() and app.reference_button.instate(['disabled']))
        check('settings_hidden_on_startup',not app.details.winfo_ismapped() and not app.status_panel.metadata.winfo_ismapped())
        app.details_button.invoke()
        root.update_idletasks()
        check('details_available_without_moving_main_controls',app.details.winfo_ismapped()
              and app.status_panel.metadata.winfo_ismapped() and panel_geometry()==original_geometry[0])
        app.details.withdraw()
        check('camera_not_opened_without_consent',not app.start_camera() and cameras==[])
        check('normal_production_authorization_unchanged',PrototypeController().state.can_save is False and app.session.can_save is False)
        app.consent.set(True);app._consent_changed();app.start_button.invoke();wait(ready,capture_ref)
    def capture_ref():
        if ref_target[0] == 1:
            check('camera_start_guides_reference_capture','5 reference photos' in app.status_panel.variables['guide'].get())
        app.reference_button.invoke();wait(lambda:len(app.session.references)==ref_target[0],reference_done)
    def reference_done():
        if ref_target[0]==3:check('three_explicit_distinct_reference_samples',len(app.session.reference_hashes)==3)
        if ref_target[0]==5:
            check('reference_target_guides_to_save','Save the test profile' in app.status_panel.variables['guide'].get()
                  and app.status_panel.variables['sample_progress'].get()=='Reference photos: 5 / 5 recommended')
        check(f'reference_{ref_target[0]}_preserves_panel_geometry',panel_geometry()==original_geometry[0])
        if ref_target[0]<10:
            ref_target[0]+=1;scene['sample']+=1
            if ref_target[0]==10:scene['sample']=19  # Keep later held-out fixtures distinct.
            wait(lambda:ready() and app._detected is not None and int(app._detected[1][0,0,0])==scene['sample'],capture_ref)
        else:
            check('ten_explicit_references_are_visible',app.status_panel.variables['references'].get().startswith('References: 10/'))
            wait(lambda:not app.worker.busy,commit)
    def commit():
        app.profile_name_entry.insert(0,'Tên thử')
        app.commit_button.invoke()
        check('test_alias_committed_only_to_memory',app.session.test_ids==('test-001',) and app.session.matcher._gallery.people[0].name=='Test subject 001')
        check('display_name_keeps_generated_id_and_clears_draft',
              app.profile_labels=={'test-001':'Tên thử'} and not app.profile_name.get()
              and tuple(app.truth_picker.cget('values'))==('unspecified','unknown','Tên thử (test-001)'))
        check('saved_profile_replaces_empty_reference_progress',
              app.status_panel.variables['sample_progress'].get()=='Test profile saved in memory')
        check('new_phase_reconfirms_consent',not app.consent.get() and app.feed is None)
        check('saved_profile_guides_to_consent_and_restart','Confirm consent again' in app.status_panel.variables['guide'].get())
        app.truth_picker.current(2)
        app.truth_picker.event_generate('<<ComboboxSelected>>')
        check('named_profile_selection_uses_generated_id_for_scoring',app.truth.get()=='test-001')
        scene['sample']=10;app.consent.set(True);app._consent_changed();app.start_camera()
        wait(lambda:ready() and app._decision is not None and app._decision.accepted,evaluate)
    def evaluate():
        check('saved_profile_guides_to_live_match_and_record','Record a test' in app.status_panel.variables['guide'].get())
        check('live_known_score_and_selected_threshold',app._decision.score is not None and app._decision.threshold==0.4619967254593038 and 'Cosine' in app.live_text.get())
        check('known_fields_show_display_name_and_keep_geometry_fixed',app.status_panel.variables['decision'].get()=='Known: Tên thử (test-001)' and app.status_panel.variables['cosine'].get()==f'Cosine: {app._decision.score:.6f}' and panel_geometry()==original_geometry[0])
        check('commit_updates_alias_and_reference_count',app.status_panel.variables['alias'].get()=='Next alias: test-002' and app.status_panel.variables['references'].get().startswith('References: 0/'))
        app.evaluation_button.invoke();wait(lambda:len(app.session.records)==1,first_trial)
    def first_trial():
        check('held_out_sample_recorded_as_correct_accept',app.session.records[0]['outcome']=='correct_accept')
        app.conditions['presentation'].set('photograph');scene['sample']=11
        wait(lambda:ready() and int(app._detected[1][0,0,0])==11,photo)
    def photo():
        app.evaluation_button.invoke();wait(lambda:len(app.session.records)==2,photo_done)
    def photo_done():
        check('photograph_condition_is_separate_not_liveness',app.session.report()['photo_accepts']==1 and app.session.report()['liveness_evaluated'] is False)
        wait(ready,duplicate)
    def duplicate():
        app.evaluation_button.invoke();wait(lambda:'already recorded' in app.status.get(),duplicate_done)
    def duplicate_done():
        check('same_frame_evaluation_is_not_counted_twice',len(app.session.records)==2)
        app.truth.set('unknown');app.conditions['presentation'].set('live_person');scene['sample']=12
        wait(lambda:ready() and int(app._detected[1][0,0,0])==12,unknown_truth)
    def unknown_truth():
        app.evaluation_button.invoke();wait(lambda:len(app.session.records)==3,unknown_done)
    def unknown_done():
        check('ground_truth_errors_are_recorded_not_hidden',app.session.records[-1]['outcome']=='false_accept')
        scene['boxes']=[];wait(lambda:app._detected is not None and app._detected[2]==[],lost)
    def lost():
        check('face_loss_clears_result_and_disables_capture',app._decision is None and app.reference_button.instate(['disabled']) and app.evaluation_button.instate(['disabled']))
        check('face_loss_panel_has_no_stale_score_or_shift','no current result' in app.status_panel.variables['cosine'].get() and panel_geometry()==original_geometry[0])
        check('no_face_is_shown_only_in_the_live_panel',
              app.status_panel.variables['decision'].get()=='No face detected'
              and app.status.get()=='Test result recorded in the table below.')
        scene['boxes']=[CONTROLLED_BOX];scene['sample']=14
        wait(lambda:app._decision is not None and app._decision.accepted,reentered)
    def reentered():
        check('face_reentry_restores_current_known_without_shift',app.status_panel.variables['decision'].get().startswith('Known:') and panel_geometry()==original_geometry[0])
        check('known_does_not_leave_a_stale_no_face_message',
              app.status.get()=='Test result recorded in the table below.')
        scene['boxes']=[CONTROLLED_BOX,(10,10,100,100)]
        wait(lambda:app._detected is not None and len(app._detected[2])==2,multiple)
    def multiple():
        check('multiple_faces_block_reference_and_evaluation',app._decision is None and app.reference_button.instate(['disabled']))
        check('multiple_faces_is_shown_only_in_the_live_panel',
              app.status_panel.variables['decision'].get().startswith('Multiple faces')
              and app.status.get()=='Test result recorded in the table below.')
        app.consent.set(False);app._consent_changed();wait(lambda:not app.worker.busy,reload)
    def reload():
        before=app.session.matcher._gallery.to_dict();app.reload_button.invoke()
        check('optional_gallery_reload_is_in_memory_only',app.session.matcher._gallery.to_dict()==before and app.session.reload_checks==1)
        app.export_button.invoke();export[0]=app.reports.base/app.session.run_id/'scores.json'
        data=json.loads(export[0].read_text());text=export[0].read_text()
        check('score_export_omits_biometric_data_and_paths',all(s not in text for s in ('embeddings','fingerprint','/Users/')))
        check('score_export_uses_only_test_ids_and_omits_display_name',
              'Tên thử' not in json.dumps(data,ensure_ascii=False)
              and data['trials'][0]['predicted_test_id']=='test-001'
              and data['trials'][0]['ground_truth']=='test-001')
        check('generated_tests_cannot_claim_physical_evaluation',data['source']=='generated' and data['physical_samples_recorded'] is False)
        scene['boxes']=[CONTROLLED_BOX];scene['sample']=13;app.consent.set(True);app._consent_changed();app.start_camera()
        wait(ready,slow_reset)
    def slow_reset():
        scene['hold']=True;gate.clear();app.evaluation_button.invoke()
        wait(lambda:app.worker.busy,reset)
    def reset():
        app.reset_button.invoke();scene['hold']=False;gate.set()
        wait(lambda:not app.worker.busy,reset_done)
    def reset_done():
        check('reset_deletes_gallery_records_drafts_and_local_export',app.session.test_ids==() and app.session.records==[] and not export[0].exists() and app.table.get_children()==())
        check('reset_removes_display_names_and_named_choices',not app.profile_labels
              and tuple(app.truth_picker.cget('values'))==('unspecified','unknown'))
        check('reset_rejects_stale_completion_and_stops_camera',app._decision is None and app.feed is None and app.session.pending is None)
        check('models_not_reloaded_or_retrained',id(model.model)==model_id)
        check('sandbox_still_cannot_authorize_saving',app.session.can_save is False)
        check('reset_panel_counts_zero_and_geometry_fixed','recorded trials: 0' in app.status_panel.variables['counts'].get() and panel_geometry()==original_geometry[0])
        scene['sample']=15;app.consent.set(True);app._consent_changed();app.start_camera()
        wait(lambda:app._decision is not None and not app._decision.accepted,empty_unknown)
    def empty_unknown():
        check('live_unknown_empty_gallery_is_truthful_and_fixed',app._decision.score is None and app.status_panel.variables['decision'].get()=='Unknown' and app.status_panel.variables['cosine'].get()=='Cosine: N/A (empty gallery)' and panel_geometry()==original_geometry[0])
        scene['camera_ok']=False
        wait(lambda:app.feed is None,camera_failed)
    def camera_failed():
        check('sandbox_camera_failure_visible_and_current_result_cleared','Camera failure' in app.status.get() and app._decision is None and 'no current result' in app.status_panel.variables['cosine'].get())
        scene['camera_ok']=True;app.start_button.invoke()
        wait(lambda:app._decision is not None,camera_recovered)
    def camera_recovered():
        check('sandbox_camera_retry_recovers_without_panel_shift',app.feed is not None and panel_geometry()==original_geometry[0] and not app._decision.can_save)
        app.consent.set(False);app._consent_changed()
        check('camera_stop_preserves_panel_geometry',panel_geometry()==original_geometry[0])
        finish()
    root.after(200,start);root.after(30000,lambda:finish('Sandbox native check timeout'))
    root.mainloop()
    for worker in (app.detector,app.worker):worker.thread.join(2)
    if app._retiring_feed is not None:app._retiring_feed.thread.join(2)
    report['workers_stopped']=not app.detector.thread.is_alive() and not app.worker.thread.is_alive() and (app._retiring_feed is None or not app._retiring_feed.thread.is_alive())
    temporary.cleanup();report['passed']=not report.get('error') and not report['callback_errors'] and report['workers_stopped']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
