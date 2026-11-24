"""Bounded previous-sample display. This module never authorizes an operation.

Visual continuity is only a conservative reason to retain clearly marked text.
It is not identity tracking or liveness evidence. No raw envelope is modified.
"""

from collections import Counter, deque
from dataclasses import dataclass
from math import hypot

import numpy as np

from .webcam_prototype import Context, box_in_frame, frame_size
from .emotion_presentation import appearance, geometry

CHECKING_MATCH = "Checking match"


@dataclass(frozen=True)
class DisplayResult:
    status: str
    identity: str
    emotion: str
    liveness: str
    source_context: Context | None = None
    historical: bool = False
    identity_name: str | None = None


@dataclass(frozen=True)
class _Sample:
    context: Context
    captured_at: float
    identity: str
    emotion: str
    liveness: str
    identity_name: str | None = None


def signature(frame, box):
    """Copy 16x16 BGR samples for a display guard, never a model input."""
    width, height = frame_size(frame)
    if not isinstance(frame, np.ndarray) or not box_in_frame(box, width, height):
        raise ValueError("Invalid display source region")
    x1,y1,x2,y2 = box
    rows = np.linspace(y1,y2-1,16,dtype=int)
    cols = np.linspace(x1,x2-1,16,dtype=int)
    pixels = frame[rows[:,None],cols[None,:],:].copy()
    # Uninformative patches cannot establish continuity, even with equal boxes.
    if float(pixels.astype(np.float32).std(axis=(0,1)).mean()) < 8:
        return None
    return pixels


def similar_pixels(left, right):
    if left is None or right is None:
        return False
    delta = np.abs(left.astype(np.int16)-right.astype(np.int16))
    return float(delta.mean()) <= 6 and float(np.quantile(delta,.95)) <= 18


def similar_box(left, right):
    aw,ah = left[2]-left[0], left[3]-left[1]
    bw,bh = right[2]-right[0], right[3]-right[1]
    overlap = max(0,min(left[2],right[2])-max(left[0],right[0])) * max(0,min(left[3],right[3])-max(left[1],right[1]))
    union = aw*ah+bw*bh-overlap
    centers = hypot((left[0]+left[2]-right[0]-right[2])/2,
                    (left[1]+left[3]-right[1]-right[3])/2)
    return union > 0 and overlap/union >= .9 and .9 <= (bw*bh)/(aw*ah) <= 1.1 and centers <= .04*min(aw,ah)


class StablePresentation:
    HOLD_SECONDS = .8
    DETECTION_GAP = .45
    TRIAL_DETECTION_GAP = .55
    PREVIEW_GAP = .15

    def __init__(self, *, unverified_preview=False, stable_trial=False):
        self.unverified_preview = unverified_preview
        self.stable_trial = stable_trial
        self.reset()

    def reset(self):
        self._sample = self._trial = None
        self._box = self._pixels = self._detected_at = self._preview_at = None
        self._shape = None
        self._frame_id = None
        self._token = self._confirmed_token = None
        self._confirmations = 0
        self._identity_candidate = None
        self._identity_at = None
        self._processing = False
        self._hold = False
        self._barrier = float('-inf')
        self._emotions = deque(maxlen=3)

    def invalidate(self, timestamp):
        barrier = max(self._barrier,timestamp)
        self.reset()
        self._barrier = barrier

    def _image_signature(self, frame, box):
        return appearance(frame, box)[0] if self.stable_trial else signature(frame, box)

    def _similar_image(self, current):
        if not self.stable_trial:
            return similar_pixels(self._pixels, current)
        return self._pixels is not None and current is not None and float(np.sum(self._pixels * current)) >= .80

    def _similar_region(self, box):
        if not self.stable_trial:
            return similar_box(self._box, box)
        iou, area_ratio, center_shift = geometry(self._box, box)
        return iou >= .65 and .8 <= area_ratio <= 1.25 and center_shift <= .18

    def region_visible(self, token, now, *, previous):
        """Require current preview continuity before drawing a retained outline."""
        if (self._preview_at is None or self._detected_at is None or now-self._preview_at > self.PREVIEW_GAP
                or now-self._detected_at > self.TRIAL_DETECTION_GAP):
            return False
        return self._pixels is not None if previous else self._confirmed_token == token

    def observe_preview(self, frame, timestamp):
        if self._preview_at is not None and timestamp < self._preview_at:
            return False
        gap = self._preview_at is not None and timestamp-self._preview_at > self.PREVIEW_GAP
        if self._box is not None:
            try:
                matches = tuple(frame.shape) == self._shape and self._similar_image(self._image_signature(frame,self._box))
            except (ValueError,TypeError,IndexError):
                matches = False
            if gap or not matches:
                self.invalidate(timestamp)
                self._preview_at = timestamp
                return False
        self._preview_at = timestamp
        return True

    def begin_sample(self, token, frame, captured_at):
        if self._token is not None and token[0] != self._token[0]:
            self.reset()
        self.observe_preview(frame,captured_at)
        self._token = token
        self._confirmed_token = None
        self._processing = True
        self._hold = (self._sample is not None or self._trial is not None) and self._confirmations >= 2

    def detected(self, token, frame, boxes, captured_at):
        if token != self._token or token == self._confirmed_token or captured_at < self._barrier:
            return False
        if len(boxes) != 1:
            self.invalidate(captured_at)
            return False
        box = boxes[0]
        try:
            pixels = self._image_signature(frame,box)
        except (ValueError,TypeError,IndexError):
            self.invalidate(captured_at)
            return False
        detection_gap = self.TRIAL_DETECTION_GAP if self.stable_trial else self.DETECTION_GAP
        continuous = (self._box is not None and self._detected_at is not None
                      and tuple(frame.shape) == self._shape
                      and 0 < captured_at-self._detected_at <= detection_gap
                      and self._similar_region(box) and self._similar_image(pixels))
        if not continuous:
            self._sample = self._trial = None
            self._identity_candidate = self._identity_at = None
            self._emotions.clear()
            self._confirmations = 1 if pixels is not None else 0
        else:
            self._confirmations = min(2,self._confirmations+1)
        self._box,self._pixels,self._detected_at = tuple(box),pixels,captured_at
        self._shape = tuple(frame.shape)
        self._frame_id = id(frame)
        self._confirmed_token = token
        self._hold = continuous and self._confirmations >= 2
        return True

    def accept(self, token, request, state, captured_at, now):
        preview_only = self.unverified_preview and state.ui_state == 'preview' and state.liveness is None
        live = state.ui_state == 'live' and state.liveness is not None and state.liveness.status == 'live'
        if (token != self._token or token != self._confirmed_token or state.context != request.context
                or not (preview_only or live)
                or (self._sample is not None and self._sample.context == state.context)
                or captured_at != self._detected_at or tuple(request.bbox) != self._box
                or captured_at < self._barrier or not 0 <= now-captured_at <= self.HOLD_SECONDS):
            return False
        if self.stable_trial:
            if id(request.frame) != self._frame_id:
                return False
        else:
            try:
                source_pixels = signature(request.frame,request.bbox)
                if tuple(request.frame.shape) != self._shape or not np.array_equal(source_pixels,self._pixels):
                    return False
            except (ValueError,TypeError,IndexError):
                return False
        recognition = state.recognition
        identity = ('Unavailable' if recognition is None else
                    f'{recognition.name} [{recognition.person_id}]' if recognition.accepted else 'Unknown')
        name = None
        if self.unverified_preview and self.stable_trial:
            candidate = None if recognition is None else (recognition.accepted, recognition.person_id)
            repeated = (candidate is not None and candidate == self._identity_candidate
                        and self._identity_at is not None
                        and 0 < captured_at-self._identity_at <= self.TRIAL_DETECTION_GAP)
            self._identity_candidate, self._identity_at = candidate, captured_at
            if not repeated:
                identity = CHECKING_MATCH
            elif recognition.accepted and recognition.source == 'model':
                name = recognition.name
        elif self._sample is not None and identity != self._sample.identity:
            # A changed identity decision cannot inherit another result's votes.
            self._emotions.clear()
            self._confirmations = 1
            if self._trial is not None and self._trial[1].token != token:
                self._trial = None
        emotion = state.emotion.label if state.emotion is not None else 'Unavailable'
        while self._emotions and now-self._emotions[0][0] > self.HOLD_SECONDS:
            self._emotions.popleft()
        if state.emotion is None:
            self._emotions.clear()
        else:
            self._emotions.append((captured_at,emotion))
            votes = Counter(label for _,label in self._emotions)
            majority,count = votes.most_common(1)[0]
            if count >= 2:
                emotion = majority
        self._sample = _Sample(state.context,captured_at,identity,emotion,
                               'Unavailable' if preview_only else state.liveness.status,name)
        self._processing = False
        return True

    def accept_trial(self, result, captured_at, now):
        if (not self.stable_trial or result.token != self._confirmed_token or captured_at != self._detected_at
                or not 0 <= now-captured_at <= self.HOLD_SECONDS):
            return
        self._trial = None if result.error is not None else (captured_at, result)

    def trial_view(self, now):
        if self._trial is None:
            return None
        captured_at, result = self._trial
        if (now-captured_at > self.HOLD_SECONDS or self._preview_at is None
                or now-self._preview_at > self.PREVIEW_GAP):
            self._trial = None
            return None
        if result.token == self._token or self._hold:
            return result
        return None

    def view(self, state, now):
        blocked = state.ui_state in {'no_face','multiple_faces','spoof','uncertain','error'}
        if blocked:
            self.invalidate(now)
        sample = self._sample
        if sample is not None:
            expired = (now-sample.captured_at > self.HOLD_SECONDS or self._preview_at is None
                       or now-self._preview_at > self.PREVIEW_GAP
                       or (state.context is not None and state.context.action != sample.context.action)
                       or (not self._processing and state.context != sample.context))
            if expired:
                if self.stable_trial:
                    self._sample = None
                else:
                    self.invalidate(now)
                sample = None
        if sample is not None and (not self._processing or self._hold):
            return DisplayResult('Previous sample • UNVERIFIED • display only',sample.identity,
                                 sample.emotion,sample.liveness,sample.context,
                                 self._processing or sample.context != state.context,sample.identity_name)
        status = {'no_face':'No face detected','multiple_faces':'Multiple faces — one person only',
                  'spoof':'SIMULATED spoof — blocked','uncertain':'SIMULATED uncertain — retry',
                  'error':'Camera / model unavailable'}.get(state.ui_state,'Checking a fresh sample…')
        emotion = state.emotion.label if state.ui_state == 'uncertain' and state.emotion is not None else 'Unavailable'
        liveness = state.liveness.status if blocked and state.liveness is not None else 'Unavailable'
        checking = (self.unverified_preview and self.stable_trial and not blocked
                    and self.region_visible(self._token, now, previous=True))
        return DisplayResult(status, CHECKING_MATCH if checking else 'Unavailable', emotion, liveness)
