"""Informational Emotion display, independent of identity/liveness authorization.

Only original context-bound Emotion completions enter this cache. Continuity is
a display guard, not identity tracking. No envelope or persistence decision is changed.
"""
from collections import Counter,deque
from dataclasses import dataclass,asdict
import math
import numpy as np

from .webcam_prototype import Completion,Emotion,Envelope,Liveness,OperationError,PrototypeController,box_in_frame,frame_size


def appearance(frame,box):
    """Area-averaged, brightness-normalized display descriptor; never model input."""
    import cv2
    width,height=frame_size(frame)
    if not isinstance(frame,np.ndarray) or not box_in_frame(box,width,height):
        raise ValueError('Invalid Emotion display region')
    x1,y1,x2,y2=box;dx=max(0,(x2-x1)//10);dy=max(0,(y2-y1)//10)
    crop=frame[y1+dy:y2-dy,x1+dx:x2-dx]
    small=cv2.resize(crop,(24,24),interpolation=cv2.INTER_AREA).astype(np.float32).mean(axis=2)
    brightness=float(small.mean());centered=small-brightness;norm=float(np.linalg.norm(centered))
    return (None if float(small.std())<5 else centered/norm),brightness


def geometry(left,right):
    aw,ah=left[2]-left[0],left[3]-left[1];bw,bh=right[2]-right[0],right[3]-right[1]
    intersection=max(0,min(left[2],right[2])-max(left[0],right[0]))*max(0,min(left[3],right[3])-max(left[1],right[1]))
    iou=intersection/(aw*ah+bw*bh-intersection)
    ratio=(bw*bh)/(aw*ah)
    distance=math.hypot((left[0]+left[2]-right[0]-right[2])/2,(left[1]+left[3]-right[1]-right[3])/2)/min(aw,ah)
    return iou,ratio,distance


@dataclass(frozen=True)
class EmotionDisplay:
    label: str
    reason: str
    source_context: object=None
    captured_at: float|None=None
    sampled: bool = False


class EmotionPresentation:
    FRESH_SECONDS=.8
    LOSS_GRACE=.35
    FACE_GAP=.75

    def __init__(self,metrics=None, *, prediction_interval=0.0):
        self.prediction_interval=prediction_interval
        self.display_seconds=prediction_interval+self.FRESH_SECONDS
        self.metrics=metrics;self.counts=Counter();self.values={}
        self._epoch=0;self._visible=False
        self.reset('startup')

    def count(self,reason):
        self.counts[reason]+=1
        if self.metrics is not None:self.metrics.count('emotion_display_'+reason)

    def measure(self,name,value):
        self.values.setdefault(name,deque(maxlen=256)).append(float(value))

    def reset(self,reason='retry'):
        self._epoch+=1;self._sample=None;self._pending=None;self._last_accepted=float('-inf')
        self._box=self._descriptor=self._shape=self._one_at=self._lost_at=self._preview_at=None
        self._token=self._frame_id=None;self._detection_at=float('-inf');self._faces=0
        self._next_prediction_at=float('-inf')
        self.reason=reason;self.count('reset_'+reason)

    def clear(self,reason,*,cancel_pending=True):
        if cancel_pending:
            self._epoch+=1;self._pending=None
        self._sample=None;self.reason=reason
        self.count('clear_'+reason)

    def observe_preview(self,frame,timestamp):
        shape=tuple(frame.shape)
        if self._shape is not None and shape!=self._shape:
            self.clear('source_dimensions_changed');self._box=self._descriptor=self._one_at=None
            self._shape=shape
        self._preview_at=timestamp

    def new_context(self):
        self.count('routine_context')  # deliberately does not clear informational data

    def detected(self,token,frame,boxes,captured_at):
        if captured_at<=self._detection_at:
            self.count('detection_out_of_order');return False
        self._detection_at=captured_at;self._faces=len(boxes)
        if self._token is not None and token[0]!=self._token[0]:
            self.clear('camera_generation_changed');self._box=self._descriptor=self._one_at=None
        if len(boxes)>1:
            self.clear('multiple_faces');self._box=self._descriptor=self._one_at=None;return False
        if not boxes:
            self.count('detector_miss')
            if self.prediction_interval:
                self.clear('no_face')
                self._box=self._descriptor=self._one_at=None
            if self._lost_at is None:self._lost_at=captured_at
            return False
        box=tuple(boxes[0])
        try:descriptor,brightness=appearance(frame,box)
        except (ValueError,TypeError,IndexError):
            self.clear('invalid_face_region');return False
        if self._box is not None:
            iou,ratio,distance=geometry(self._box,box)
            self.measure('box_iou',iou);self.measure('box_area_ratio',ratio);self.measure('box_center_shift',distance)
            if captured_at-self._one_at>self.FACE_GAP:
                self.clear('face_evidence_gap')
            elif iou<.4 or not .6<=ratio<=1.7 or distance>.35:
                self.clear('ambiguous_box_jump')
            elif descriptor is None or self._descriptor is None:
                self.clear('uninformative_appearance')
            else:
                correlation=float(np.sum(descriptor*self._descriptor));self.measure('appearance_correlation',correlation)
                self.measure('illumination_change',abs(brightness-self._brightness))
                if correlation<.55:self.clear('ambiguous_replacement')
        self._box,self._descriptor,self._brightness=box,descriptor,brightness
        self._shape=tuple(frame.shape);self._one_at=captured_at;self._lost_at=None
        self._token,self._frame_id=token,id(frame)
        self.count('single_face');return True

    def due(self,captured_at):
        return captured_at>=self._next_prediction_at

    def register(self,token,request,captured_at):
        if (token!=self._token or self._faces!=1 or tuple(request.bbox)!=self._box
                or captured_at!=self._one_at or id(request.frame)!=self._frame_id):
            self.count('register_source_mismatch');return False
        self._pending=(token,request.context,captured_at,self._epoch,id(request.frame),tuple(request.bbox))
        self._next_prediction_at=captured_at+self.prediction_interval
        return True

    def accept(self,token,request,completions,captured_at,now):
        pending=self._pending
        if pending is None or pending!=(token,request.context,captured_at,self._epoch,id(request.frame),tuple(request.bbox)):
            self.count('reject_obsolete_source');return False
        self._pending=None
        if not 0<=now-captured_at<=self.FRESH_SECONDS:
            self.count('reject_result_expired');return False
        if captured_at<=self._last_accepted:
            self.count('reject_out_of_order');return False
        if self._lost_at is not None and now-self._lost_at>self.LOSS_GRACE:
            self.clear('sustained_face_loss');return False
        if not isinstance(completions,(tuple,list)):
            self.count('reject_invalid_batch');return False
        emotion=None
        for completion in completions:
            if not isinstance(completion,Completion) or completion.context!=request.context:
                self.count('reject_context_mismatch');return False
            if isinstance(completion.result,Liveness) and completion.result.status=='spoof':
                self.clear('spoof');return False
            if isinstance(completion.result,OperationError) and completion.result.module=='emotion':
                self.clear('model_error_'+completion.result.code);return False
            if isinstance(completion.result,Emotion):emotion=completion
        if emotion is None:
            self.count('completion_without_emotion');return False
        # Reuse SYS-02 output validation without supplying liveness or authorization.
        validator=PrototypeController();validator.state=Envelope(context=request.context,ui_state='checking')
        validator.receive(emotion)
        if validator.state.emotion is None:
            self.clear('invalid_prediction');return False
        self._sample=(emotion.result,request.context,captured_at)
        self._last_accepted=captured_at;self.reason='latest_prediction';self.count('accepted')
        return True

    def view(self,now):
        if self._sample is not None:
            if now-self._sample[2]>self.display_seconds:self.clear('freshness_expired',cancel_pending=False)
            elif self._lost_at is not None and now-self._lost_at>self.LOSS_GRACE:self.clear('sustained_face_loss')
            elif self._preview_at is None or now-self._preview_at>1.:self.clear('camera_frame_expired')
            elif self.prediction_interval and now-self._one_at>self.FACE_GAP:self.clear('face_evidence_gap')
        visible=self._sample is not None
        if self._visible and not visible:self.count('unavailable_'+self.reason)
        self._visible=visible
        if not visible:return EmotionDisplay('Unavailable',self.reason,sampled=bool(self.prediction_interval))
        prediction,context,timestamp=self._sample
        return EmotionDisplay(prediction.label,'detector_gap_grace' if self._lost_at is not None else 'latest_valid_sample',
                              context,timestamp,sampled=bool(self.prediction_interval))

    def audit(self):
        sample=self._sample
        return {'reason':self.reason,'counts':dict(self.counts),
                'recent_measurements':{name:{'n':len(v),'mean':sum(v)/len(v),'min':min(v),'max':max(v)} for name,v in self.values.items() if v},
                'source_context':asdict(sample[1]) if sample else None,
                'raw_prediction':asdict(sample[0]) if sample else None,
                'captured_at':sample[2] if sample else None,'freshness_limit_ms':round(self.display_seconds*1000)}
