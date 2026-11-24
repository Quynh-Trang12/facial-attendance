"""Mapped identity-overlay checks using generated pixels and actual trained models.

No webcam, person image or genuine gallery is used. Test scenes and mirrors
exist only in this harness; no extra UI mode or inference pipeline is created.
"""
import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys
from threading import Event
from time import monotonic

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--face-checkpoint',required=True)
    parser.add_argument('--emotion-checkpoint',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():parser.error('Refusing to overwrite evidence')
    import cv2
    import tkinter as tk
    from PIL import Image,ImageTk
    from facial_attendance.ui import webcam_app as ui
    from facial_attendance.ui.identity_overlay import preview_box
    from facial_attendance.ui.recognition_adapter import TrainedPreviewAdapters
    from facial_attendance.recognition.gallery_matching import CONTROLLED_BOX,controlled_frame,empty_gallery
    cv2.setNumThreads(1)
    adapters=TrainedPreviewAdapters(args.face_checkpoint,emotion_checkpoint=args.emotion_checkpoint,controlled_registration=True)
    adapters.matcher.commit(adapters.matcher.prepare(('overlay-generated','Generated Overlay Person')))
    gallery=adapters.matcher._gallery
    frame=controlled_frame();frame_hash=sha256(frame.tobytes()).hexdigest()
    scene={'boxes':[CONTROLLED_BOX],'inverse':False,'hold':False}
    gate=Event();gate.set()
    embed=adapters.face_inference.embed
    def held_embed(*a,**kw):
        if scene['hold']:gate.wait(2.)
        return embed(*a,**kw)
    adapters.face_inference.embed=held_embed
    class Camera:
        def isOpened(self):return True
        def release(self):pass
        def read(self):return True,255-frame if scene['inverse'] else frame
    root=tk.Tk()
    app=ui.LiveWebcamApp(root,Camera,source_name='Generated identity overlay check',adapters=adapters,
                         detector=lambda _:scene['boxes'],diagnostics=False)
    report={'input':'generated pixels, controlled regions, actual trained checkpoints','physical_webcam_tested':False,
            'checks':[],'callback_errors':[]}
    geometry=[None];items=[None];old_context=[None];sequence=[None];original_preview_size=[None]
    model_ids=(id(adapters.face_inference.model),id(adapters.emotion_adapter.inference.model),id(app.inference),id(app.feed))
    def check(name,condition):
        if not condition:raise AssertionError(name)
        report['checks'].append(name)
    def shown():
        painter=app._identity_overlay
        if painter is None or painter.key is None:return None
        return app.canvas.itemcget(painter.items[1],'text')
    def known():return shown()=='Generated Overlay Person · UNVERIFIED'
    def wait_for(predicate,callback):
        if predicate():callback()
        else:root.after(10,lambda:wait_for(predicate,callback))
    def finish(error=None):
        gate.set();scene['hold']=False
        if error:report['error']=str(error)
        report['source_sha256']={str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').rglob('*.py')}
        app.close()
    def failed(kind,value,tb):
        report['callback_errors'].append(f'{kind.__name__}: {value}');finish(value)
    root.report_callback_exception=failed
    def visible_bounds():
        painter=app._identity_overlay
        if painter is None or painter.key is None:return False
        x1,y1,x2,y2=app.canvas.coords(painter.items[0]);w,h=app.preview_size
        return 0<=x1<x2<=w and 0<=y1<y2<=h
    def outline_aligned():
        boxes,size,_=app.live.visual_regions(monotonic())
        if len(boxes)!=1 or size is None:return False
        expected=preview_box(boxes[0],size,app.preview_size,mirrored=app._preview_mirrored)
        coords=app.canvas.coords('region')
        return len(coords)==4 and all(abs(a-b)<.01 for a,b in zip(coords,expected))
    def window_layout():
        return root.winfo_width(),root.winfo_height(),app.canvas.winfo_x(),app.canvas.winfo_y()
    def begin():wait_for(known,known_standard)
    def known_standard():
        root.update_idletasks();geometry[0]=window_layout()
        report['initial_preview_size']=app.preview_size
        check('preview_starts_directly_below_its_heading',app.canvas.winfo_y()==0)
        items[0]=app._identity_overlay.items
        check('known_gallery_match_label_in_standard',not app._developer_mode and known())
        check('known_name_explicitly_unverified',shown().endswith('UNVERIFIED') and not app.controller.state.can_save)
        check('readable_contrasting_label_background',app.canvas.itemcget(items[0][0],'fill')=='#14202b' and app.canvas.itemcget(items[0][1],'fill')=='#fff6cf')
        check('label_above_current_face_box',app.canvas.coords(items[0][0])[3]<app.canvas.coords('region')[1])
        app.mode_button.invoke();wait_for(known,known_developer)
    def known_developer():
        check('same_label_in_developer',app._developer_mode and known())
        app.mode_button.invoke()
        with adapters.matcher._lock:adapters.matcher._gallery=empty_gallery()
        app.retry();wait_for(lambda:shown()=='Unknown',unknown)
    def unknown():
        check('empty_gallery_yields_unknown_overlay',shown()=='Unknown' and app.controller.state.recognition.accepted is False)
        check('side_panel_preserved',app.display_identity.get().startswith('Identity: ') and app.display_emotion.get().startswith('Emotion: ') and 'SIMULATED' in app.display_liveness.get())
        scene['hold']=True;gate.clear();wait_for(lambda:shown()=='Checking...',checking)
    def checking():
        check('pending_inference_uses_neutral_checking',shown()=='Checking...' and app.controller.state.recognition is None)
        gate.set();scene['hold']=False
        with adapters.matcher._lock:adapters.matcher._gallery=gallery
        app.retry();wait_for(known,move)
    def move():
        before=app.canvas.coords(app._identity_overlay.items[0]);report['initial_label_bounds']=before
        scene['boxes']=[(190,100,510,330)]
        wait_for(lambda:known() and app.canvas.coords(app._identity_overlay.items[0])!=before,moved)
    def moved():
        check('label_follows_horizontal_and_vertical_motion',visible_bounds() and outline_aligned())
        # Wait for the smoothed outline to settle close to an image edge.
        scene['boxes']=[(5,0,325,230)]
        wait_for(lambda:shown() is not None and outline_aligned() and app.canvas.coords('region')[1]<10,top_edge)
    def top_edge():
        check('top_edge_label_remains_inside_preview',visible_bounds())
        scene['boxes']=[(315,130,635,360)]
        wait_for(lambda:shown() is not None and outline_aligned()
                 and app.canvas.coords('region')[2]>app.preview_size[0]-20,right_edge)
    def right_edge():
        check('right_bottom_edge_label_is_clamped',visible_bounds())
        scene['boxes']=[CONTROLLED_BOX];app.retry();wait_for(known,mirror)
    def mirror():
        app._preview_mirrored=True;sequence[0]=app._drawn_sequence
        wait_for(lambda:app._drawn_sequence!=sequence[0] and known(),mirrored)
    def mirrored():
        actual=ImageTk.getimage(app.photo).convert('RGB')
        expected=ui.prepare_preview_image(frame, viewport=app._preview_viewport_size).transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        check('mirrored_image_box_and_label_share_transform',actual.tobytes()==expected.tobytes() and outline_aligned() and visible_bounds())
        original_preview_size[0]=app.preview_size
        width,height=geometry[0][:2]
        root.geometry(f'{width+240}x{height+180}')
        wait_for(lambda:app.preview_size[0]>original_preview_size[0][0] and shown() is not None,resized)
    def resized():
        root.update_idletasks()
        canvas=app.canvas
        viewport=canvas.master
        width,height=canvas.winfo_width(),canvas.winfo_height()
        view_width,view_height=viewport.winfo_width(),viewport.winfo_height()
        x,y=canvas.winfo_x(),canvas.winfo_y()
        report['resized_preview']={'image':app.preview_size,'canvas':(x,y,width,height),
                                   'viewport':(view_width,view_height)}
        check('resized_image_fits_and_stays_top_aligned',
              app.preview_size==(width,height) and width<=view_width and height<=view_height
              and abs(2*x+width-view_width)<=2 and y==0)
        check('resized_preview_box_and_label_stay_aligned',outline_aligned() and visible_bounds())
        width,height=geometry[0][:2]
        root.geometry(f'{width}x{height}')
        app._preview_mirrored=False;app.retry()
        wait_for(lambda:window_layout()==geometry[0] and known(),replacement)
    def replacement():
        old_context[0]=app.controller.state.context;scene['hold']=True;gate.clear();scene['inverse']=True
        root.after(100,replacement_cleared)
    def replacement_cleared():
        check('ambiguous_pixels_cannot_keep_old_name',shown() in (None,'Checking...'))
        check('replacement_does_not_authorize_saving',app.controller.state.can_save is False)
        scene['inverse']=False;scene['hold']=False;gate.set();app.retry();wait_for(known,loss)
    def loss():
        scene['boxes']=[];wait_for(lambda:app.controller.state.ui_state=='no_face',lost)
    def lost():
        check('face_loss_clears_identity_label',shown() is None)
        scene['boxes']=[CONTROLLED_BOX];wait_for(known,returned)
    def returned():
        check('return_requires_a_fresh_context',app.controller.state.context!=old_context[0])
        scene['boxes']=[CONTROLLED_BOX,(20,20,100,100)]
        wait_for(lambda:app.controller.state.ui_state=='multiple_faces',multiple)
    def multiple():
        check('multiple_faces_have_no_arbitrary_identity_label',shown() is None)
        scene['boxes']=[CONTROLLED_BOX];app.retry();wait_for(known,final)
    def final():
        check('overlay_uses_two_canvas_items',len(app.canvas.find_withtag('identity-overlay'))==2)
        check('fixed_window_and_preview_position_preserved',geometry[0]==window_layout())
        check('models_workers_and_source_pixels_unchanged',model_ids==(id(adapters.face_inference.model),id(adapters.emotion_adapter.inference.model),id(app.inference),id(app.feed)) and sha256(frame.tobytes()).hexdigest()==frame_hash)
        check('genuine_registration_and_attendance_remain_disabled',app.controller.state.can_save is False)
        finish()
    root.after_idle(app.start);root.after(250,begin)
    root.after(25000,lambda:finish('Overlay GUI check timeout'))
    root.mainloop()
    for worker in (app.feed,app.detector,app.inference,app._registration_worker):worker.thread.join(2)
    report['workers_stopped']=all(not w.thread.is_alive() for w in (app.feed,app.detector,app.inference,app._registration_worker))
    report['passed']=not report.get('error') and not report['callback_errors'] and report['workers_stopped']
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    return 0 if report['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
