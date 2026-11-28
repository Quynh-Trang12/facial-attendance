"""Bounded native emotion-display audit. Explicit --camera opens a physical camera.

Records aggregate reasons/timings only. Never saves frames, crops or identities.
Generated workloads use controlled detections and actual trained model weights.
"""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from hashlib import sha256
import json
from pathlib import Path
import sys
from time import monotonic


def stats(values):
    if not values:return {'n':0}
    values=sorted(values)
    return {'n':len(values),'mean':sum(values)/len(values),'p95':values[min(len(values)-1,int(.95*len(values)))],'max':values[-1]}


def main(argv=None):
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root',type=Path,default=root)
    group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--camera',type=int)
    group.add_argument('--generated',choices=('stationary','noise','motion','illumination','latency','loss'))
    p.add_argument('--duration',type=float,default=15)
    p.add_argument('--request-camera-access',action='store_true',help='Request normal macOS camera consent on the main thread before capture worker startup')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--face-checkpoint',default='/Users/elvyx/Downloads/best (Que An).pt')
    p.add_argument('--emotion-checkpoint',default='/Users/elvyx/Downloads/best(Dung).pt')
    p.add_argument('--yunet-model',default='/Users/elvyx/Downloads/fa-sys03-mac/artifacts/sys05-yunet/face_detection_yunet_2023mar.onnx')
    args=p.parse_args(argv)
    if args.output.exists():p.error('Refusing to overwrite a report')
    sys.path.insert(0,str(args.source_root/'src'))
    import numpy as np
    import cv2
    import tkinter as tk
    from facial_attendance.ui.webcam_app import LiveWebcamApp
    from facial_attendance.ui.recognition_adapter import TrainedPreviewAdapters
    from facial_attendance.ui.face_detection import ScaledFaceDetector,YuNetFaceDetector
    from facial_attendance.ui.presentation import signature,similar_box
    cv2.setNumThreads(1)
    if args.camera is not None and args.request_camera_access:
        probe=cv2.VideoCapture(args.camera)
        opened=probe.isOpened();probe.release()
        if not opened:
            report={'physical_camera':True,'measurement_valid':False,'reason':'camera_permission_or_device_unavailable','captured_frames':0}
            args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report));return 2
    adapters=TrainedPreviewAdapters(args.face_checkpoint,emotion_checkpoint=args.emotion_checkpoint)
    counts=Counter();samples=defaultdict(list)
    started=[None];last_text=[None];last_available=[False];last_seen=[None]
    y,x=np.indices((720,1280));base=np.stack(((x//3+y//2)%256,(x//2+y//4)%256,(x//4+y//3)%256),axis=-1).astype(np.uint8)
    t0=monotonic();rng=np.random.default_rng(42)
    def box_at(t):
        dx=round(20*np.sin(t*2)) if args.generated=='motion' else 0
        scale=1+.08*np.sin(t) if args.generated=='motion' else 1
        w,h=round(640*scale),round(460*scale)
        return (640+dx-w//2,370-h//2,640+dx+w//2,370+h//2)
    class GeneratedCamera:
        def isOpened(self):return True
        def release(self):pass
        def read(self):
            t=monotonic()-t0;image=base
            if args.generated=='motion':image=np.roll(image,round(20*np.sin(t*2)),axis=1)
            if args.generated=='noise':image=np.clip(image.astype(np.int16)+rng.normal(0,7,image.shape).astype(np.int16),0,255).astype(np.uint8)
            if args.generated=='illumination':image=np.clip(image.astype(np.int16)+round(18*np.sin(t*1.5)),0,255).astype(np.uint8)
            return True,image
    def open_camera():
        cap=cv2.VideoCapture(args.camera)
        cap.set(cv2.CAP_PROP_BUFFERSIZE,1)
        return cap
    detector=ScaledFaceDetector(YuNetFaceDetector(args.yunet_model)) if args.camera is not None else lambda frame: ([] if args.generated=='loss' and 5<monotonic()-t0<6 else [box_at(monotonic()-t0)])
    if args.generated=='latency':
        original=adapters.face_inference.embed
        from time import sleep
        def slow(*a,**k):
            sleep(.15 if int((monotonic()-t0)*2)%2 else .38)
            return original(*a,**k)
        adapters.face_inference.embed=slow
    root_tk=tk.Tk()
    app=LiveWebcamApp(root_tk,open_camera if args.camera is not None else GeneratedCamera,
                      source_name='15-second diagnostic; no recording' if args.camera is not None else 'Generated emotion diagnostic',
                      detector=detector,adapters=adapters,diagnostics=True)
    app.metrics.interval=3600;app.metrics.report=lambda **k:None
    raw_detector=app.detector.function;previous_box=[None];miss_streak=[0];miss_started=[None]
    def counted_detector(frame):
        boxes=raw_detector(frame)
        if started[0] is not None:
            counts['raw_detection_batches']+=1;counts['raw_faces_'+str(min(len(boxes),2))]+=1
            phase=min(2,int((monotonic()-started[0])/5))
            counts['phase_'+str(phase)+'_faces_'+str(min(len(boxes),2))]+=1
            if not boxes:
                miss_streak[0]+=1
                counts['max_no_face_streak']=max(counts['max_no_face_streak'],miss_streak[0])
                if miss_started[0] is None:miss_started[0]=monotonic()
                samples['no_face_run_ms'].append((monotonic()-miss_started[0])*1000)
            else:
                miss_streak[0]=0;miss_started[0]=None
            if len(boxes)==1 and previous_box[0] is not None:
                a,b=previous_box[0],boxes[0]
                overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                aa=(a[2]-a[0])*(a[3]-a[1]);ab=(b[2]-b[0])*(b[3]-b[1])
                samples['box_iou'].append(overlap/(aa+ab-overlap));samples['box_area_ratio'].append(ab/aa)
            previous_box[0]=boxes[0] if len(boxes)==1 else None
        return boxes
    app.detector.function=counted_detector
    view=app.presentation
    observe=view.observe_preview
    def audited_observe(frame,ts):
        if started[0] is not None and view._box is not None:
            counts['identity_preview_guard_evaluations']+=1
            try:
                old=view._pixels;new=signature(frame,view._box)
                if old is None or new is None:counts['identity_preview_uninformative']+=1
                else:
                    d=np.abs(old.astype(np.int16)-new.astype(np.int16))
                    samples['preview_pixel_mad'].append(float(d.mean()));samples['preview_pixel_p95'].append(float(np.quantile(d,.95)))
                    samples['preview_mean_illumination_change'].append(abs(float(old.mean())-float(new.mean())))
                    if d.mean()>6 or np.quantile(d,.95)>18:counts['identity_preview_pixel_rejection']+=1
            except (ValueError,TypeError,IndexError):counts['identity_preview_invalid']+=1
        return observe(frame,ts)
    view.observe_preview=audited_observe
    detected=view.detected
    def audited_detect(token,frame,boxes,ts):
        if started[0] is not None:
            counts['detection_batches']+=1;counts['detected_faces_'+str(min(len(boxes),2))]+=1
            if len(boxes)==1 and view._box is not None:
                counts['identity_box_comparisons']+=1
                if not similar_box(view._box,boxes[0]):counts['identity_box_rejection']+=1
        return detected(token,frame,boxes,ts)
    view.detected=audited_detect
    accept=view.accept
    def audited_accept(*a,**k):
        result=accept(*a,**k)
        if started[0] is not None:counts['identity_display_accept' if result else 'identity_display_reject']+=1
        return result
    view.accept=audited_accept
    def text_changed(*a):
        text=app.display_emotion.get();available='Unavailable' not in text and 'Checking' not in text
        if started[0] is not None:
            if text!=last_text[0]:counts['emotion_text_changes']+=1
            if last_available[0] and not available:counts['emotion_to_unavailable']+=1
        last_text[0]=text;last_available[0]=available
    app.display_emotion.trace_add('write',text_changed)
    geometry=[];errors=[];report={}
    def poll():
        now=monotonic()
        if started[0] is not None:
            if last_seen[0] is not None:samples['unavailable_time_s' if not last_available[0] else 'available_time_s'].append(now-last_seen[0])
            last_seen[0]=now
            g=(root_tk.winfo_width(),root_tk.winfo_height())
            if not geometry or geometry[-1]!=g:geometry.append(g)
        root_tk.after(20,poll)
    def begin():
        started[0]=monotonic();app.metrics.snapshot(force=True);root_tk.after(round(args.duration*1000),finish)
    def finish():
        metrics=app.metrics.snapshot(force=True)
        report.update(created_utc=datetime.now(timezone.utc).isoformat(),physical_camera=args.camera is not None,
                      generated_workload=args.generated,duration_s=monotonic()-started[0],counts=dict(counts),
                      measurements={k:stats(v) for k,v in samples.items() if not k.endswith('_time_s')},
                      available_time_s=sum(samples['available_time_s']),unavailable_time_s=sum(samples['unavailable_time_s']),
                      metrics=metrics,geometry_transitions=len(geometry)-1,can_save=app.controller.state.can_save,
                      callback_errors=errors,source_sha256={str(f.relative_to(args.source_root)):sha256(f.read_bytes()).hexdigest() for f in (args.source_root/'src').rglob('*.py')})
        if hasattr(app,'emotion_presentation'):report['emotion_display_audit']=app.emotion_presentation.audit()
        app.close()
    def failure(kind,value,tb):
        errors.append(f'{kind.__name__}: {value}');finish()
    root_tk.report_callback_exception=failure
    root_tk.after_idle(app.start);root_tk.after(20,poll);root_tk.after(1500,begin);root_tk.mainloop()
    for worker in (app.feed,app.detector,app.inference):worker.thread.join(2)
    report['workers_stopped']=all(not w.thread.is_alive() for w in (app.feed,app.detector,app.inference))
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='source_sha256'},indent=2))
    return 0 if not errors and report['workers_stopped'] and not report['can_save'] and report['metrics']['counts'].get('captured',0)>0 else 1


if __name__=='__main__':raise SystemExit(main())
