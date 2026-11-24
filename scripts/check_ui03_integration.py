"""Actual trained Face/Emotion validation on generated inputs, never webcam evidence.

Optional native Tk exercises the real registration form and bounded workers.
Test-gallery persistence is confined to a newly created temporary directory.
No evaluation embeddings, personal images, attendance or genuine enrollment.
"""

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from tempfile import TemporaryDirectory
from time import monotonic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))

from facial_attendance.recognition.gallery_matching import CONTROLLED_BOX, GalleryMatcher, controlled_frame
from facial_attendance.ui.gallery import FaceGallery
from facial_attendance.ui.recognition_adapter import ControlledCamera, TrainedPreviewAdapters
from facial_attendance.ui.live_preview import evaluate_adapters
from facial_attendance.ui.webcam_prototype import Completion, PrototypeController


def native_check(adapters, *, blocked_form=False):
    import tkinter as tk
    from facial_attendance.ui.webcam_app import LiveWebcamApp
    from facial_attendance.ui.face_detection import FaceDetectorUnavailable

    class UnavailableCamera:
        def isOpened(self): return False
        def release(self): pass

    phase=["unknown"]
    camera_missing=[False]
    def camera():
        return UnavailableCamera() if camera_missing[0] else ControlledCamera()
    def detector(frame):
        if phase[0]=="loss": return []
        if phase[0]=="multiple": return [CONTROLLED_BOX,(20,20,110,120)]
        if phase[0]=="detector_error": raise FaceDetectorUnavailable("Controlled detector failure")
        return [CONTROLLED_BOX]

    root=tk.Tk();root.withdraw()
    app=LiveWebcamApp(root,camera,source_name="Generated UI-03 test input",detector=detector,adapters=adapters)
    report={"checks":[],"callback_errors":[],"input":"Generated pixels and controlled detections; actual trained weights"}
    started=monotonic()
    original_embed=adapters.face_inference.model.embed
    late=[None]
    def finish(error=None):
        if error: report["error"]=error
        adapters.face_inference.model.embed=original_embed
        app.close()
    def callback_error(kind,value,trace):
        report["callback_errors"].append(f"{kind.__name__}: {value}")
        finish("Native callback failed")
    root.report_callback_exception=callback_error

    def check():
        state=app.controller.state
        current=(state.ui_state=="live" and state.recognition is not None and state.emotion is not None)
        if state.can_save:
            finish("Mixed integration enabled genuine saving");return
        if phase[0]=="unknown" and current and not state.recognition.accepted:
            if "Unknown" not in app.registration_prompt.get():
                finish("Unknown registration prompt missing");return
            app.registration_button.invoke()
            if app.controller.action!="registration":
                finish("Registration did not start a distinct action");return
            report["checks"].append("unknown_prompt_and_registration_form")
            if blocked_form:
                if str(app.registration_submit.cget("state"))!="disabled":
                    finish("Webcam enrollment was not blocked");return
                report["checks"].append("genuine_enrollment_disabled")
                finish();return
            app.registration_id.set("test-native-001");app.registration_name.set("Generated Native Person")
            phase[0]="registration_ready"
        elif phase[0]=="registration_ready" and current and state.context.action=="registration":
            app.registration_submit.invoke();phase[0]="enrolling"
        elif phase[0]=="enrolling" and current and state.recognition.accepted:
            if (state.recognition.name!="Generated Native Person" or state.recognition.source!="model"
                    or state.emotion.source!="model" or state.liveness.source!="stub"
                    or "UNVERIFIED" not in app.summary.get()):
                finish("Trained/simulated provenance mismatch");return
            app._toggle_debug()
            report["known_payload"]=json.loads(app.payload.get("1.0","end"))
            report["checks"].append("controlled_registration_then_trained_known_name")
            late[0]=Completion(state.context,state.recognition)
            app.retry()
            if app.controller.receive(late[0]):
                finish("Retry accepted stale Face completion");return
            report["checks"].append("retry_rejected_stale_face_completion")
            phase[0]="retry"
        elif phase[0]=="retry" and current and state.recognition.accepted:
            report["checks"].append("retry_restored_fresh_trained_results");phase[0]="loss"
        elif phase[0]=="loss" and state.ui_state=="no_face":
            if any(getattr(state,key) is not None for key in ("recognition","liveness","emotion")):
                finish("Face loss retained results");return
            report["checks"].append("face_loss_cleared_all_results");phase[0]="multiple"
        elif phase[0]=="multiple" and state.ui_state=="multiple_faces":
            if state.recognition is not None or state.emotion is not None:
                finish("Multiple faces retained results");return
            report["checks"].append("multiple_faces_blocked_and_cleared");phase[0]="model_error"
            def fail_embed(images): raise RuntimeError("Controlled Face model failure")
            adapters.face_inference.model.embed=fail_embed
        elif phase[0]=="model_error" and state.ui_state=="error":
            if not any(error.module=="recognition" for error in state.errors):
                finish("Expected typed recognition error");return
            report["checks"].append("model_error_cleared_results")
            adapters.face_inference.model.embed=original_embed
            app.retry();phase[0]="model_recovery"
        elif phase[0]=="model_recovery" and current and state.recognition.accepted:
            report["checks"].append("model_failure_retry_recovered")
            phase[0]="camera_error";camera_missing[0]=True;app.retry()
        elif phase[0]=="camera_error" and state.ui_state=="error":
            if state.recognition is not None or state.emotion is not None:
                finish("Camera error retained predictions");return
            report["checks"].append("camera_error_cleared_results")
            camera_missing[0]=False;app.retry();phase[0]="camera_recovery"
        elif phase[0]=="camera_recovery" and current and state.recognition.accepted:
            report["checks"].append("camera_retry_recovered")
            phase[0]="detector_error"
        elif phase[0]=="detector_error" and state.ui_state=="error":
            report["checks"].append("detector_failure_cleared_results")
            phase[0]="spoof";adapters.scenario="Spoof";app.retry()
        elif phase[0]=="spoof" and state.ui_state=="spoof":
            if state.recognition is not None or state.emotion is not None:
                finish("Simulated spoof did not stop downstream results");return
            report["checks"].append("simulated_spoof_blocks_operations")
            phase[0]="uncertain";adapters.scenario="Uncertain";app.retry()
        elif phase[0]=="uncertain" and state.ui_state=="uncertain" and state.emotion is not None:
            if state.recognition is not None:
                finish("Uncertain liveness exposed identity");return
            report["checks"].append("uncertain_blocks_identity_emotion_display_only")
            finish();return
        if monotonic()-started>30:
            finish("Timed out at phase "+phase[0]);return
        root.after(25,check)

    root.after_idle(app.start);root.after(25,check);root.mainloop()
    workers=[app.feed,app.detector,app.inference]
    if app._registration_worker is not None: workers.append(app._registration_worker)
    for worker in workers: worker.thread.join(timeout=2)
    report["workers_stopped"]=all(not worker.thread.is_alive() for worker in workers)
    report["passed"]=(len(report["checks"])==(2 if blocked_form else 13) and report["workers_stopped"]
                      and not report["callback_errors"] and "error" not in report)
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--face-checkpoint",required=True,type=Path)
    parser.add_argument("--emotion-checkpoint",required=True,type=Path)
    parser.add_argument("--native-gui",action="store_true")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if args.output and args.output.exists(): parser.error("Refusing to overwrite existing report")
    report={"created_utc":datetime.now(timezone.utc).isoformat(),"system":platform.platform(),
            "python":platform.python_version(),"physical_webcam_tested":False,"accuracy_measured":False,
            "versions":{key:version(key) for key in ("torch","torchvision","numpy","Pillow","opencv-python")},
            "source_sha256":{str(p.relative_to(ROOT)):sha256(p.read_bytes()).hexdigest()
                             for folder in ("src","configs","scripts","tests") for p in sorted((ROOT/folder).rglob("*"))
                             if p.is_file() and p.suffix in (".py",".toml",".json")}}
    try:
        adapters=TrainedPreviewAdapters(args.face_checkpoint,emotion_checkpoint=args.emotion_checkpoint,controlled_registration=True)
        report["face"]=adapters.face_inference.provenance
        report["emotion"]=adapters.emotion_adapter.inference.provenance
        frame=controlled_frame();before=frame.copy()
        a=adapters.face_inference.embed(frame,CONTROLLED_BOX);b=adapters.face_inference.embed(frame,CONTROLLED_BOX)
        report.update(embedding_shape=[1,len(a)],embedding_deterministic=a==b,
                      source_unchanged=bool((frame==before).all()))
        report["empty_gallery_unknown"]=not adapters.matcher.match(a).accepted
        adapters.matcher.commit(adapters.matcher.prepare(("test-headless-001","Generated Headless Person")))
        report["subsequent_match"]=asdict(adapters.matcher.match(a))
        with TemporaryDirectory(prefix="ui03-generated-gallery-") as directory:
            path=Path(directory)/"generated-test-gallery.json"
            adapters.matcher._gallery.save(path)
            gallery=FaceGallery.load(path,model_id=report["face"]["model_id"],model_version=report["face"]["model_version"],embedding_dim=512)
            report["reload_match"]=asdict(GalleryMatcher(gallery).match(a))
        c=PrototypeController();request=c.begin_capture(frame,[CONTROLLED_BOX])
        completions=evaluate_adapters((request,adapters))
        for result in completions: c.receive(result)
        report["mixed_payload"]=asdict(c.state);c.reset()
        report["stale_rejected"]=not c.receive(completions[-1])
        report["passed"]=bool(a==b and report["source_unchanged"] and report["empty_gallery_unknown"]
                              and report["subsequent_match"]["accepted"] and report["reload_match"]==report["subsequent_match"]
                              and report["stale_rejected"] and not report["mixed_payload"]["can_save"])
        if args.native_gui:
            # Fresh galleries keep the native Unknown -> registration transition independent.
            controlled=TrainedPreviewAdapters(args.face_checkpoint,emotion_checkpoint=args.emotion_checkpoint,controlled_registration=True)
            report["native_controlled"]=native_check(controlled)
            readonly=TrainedPreviewAdapters(args.face_checkpoint,emotion_checkpoint=args.emotion_checkpoint)
            report["native_blocked_enrollment"]=native_check(readonly,blocked_form=True)
            report["passed"] &= report["native_controlled"]["passed"] and report["native_blocked_enrollment"]["passed"]
    except Exception as error:
        report.update(passed=False,error=f"{type(error).__name__}: {error}")
    output=json.dumps(report,indent=2,allow_nan=False)+"\n"
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(output)
    print(output)
    return 0 if report["passed"] else 1


if __name__=="__main__": raise SystemExit(main())
