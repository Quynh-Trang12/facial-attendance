"""Convert a face crop or camera frame into a live/spoof decision.

Face crops are PIL RGB images. Camera frames are uint8 arrays with shape
[height, width, 3] in OpenCV BGR order. No detected face or multiple faces returns
uncertain with allow_attendance=False; the caller must also verify identity.
"""

from contextlib import nullcontext

import cv2
import numpy as np
from PIL import Image, ImageEnhance
import torch

from facial_attendance.liveness.binary_model import LivenessPredictor
from facial_attendance.liveness.face_detector import YuNetFaceDetector
from facial_attendance.liveness.temporal_liveness import TemporalLivenessGate


class LivenessAdapter:
    def __init__(self, checkpoint, device='cuda', *, detector_model=None,
                 brightness_factor=1.0, crop_margin=0.0):
        if not np.isfinite(brightness_factor) or brightness_factor <= 0:
            raise ValueError('Brightness factor must be positive and finite')
        if not np.isfinite(crop_margin) or not 0 <= crop_margin <= .5:
            raise ValueError('Crop margin must be between 0 and 0.5 per side')
        self.brightness_factor, self.crop_margin = brightness_factor, crop_margin
        if str(device).startswith('cuda') and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA inference is unavailable; no CPU fallback')
        self.predictor = LivenessPredictor(checkpoint, device)
        self.detector = None
        self.detector_model = detector_model
        self.temporal_gate = None

    def predict_face(self, image):
        # Enable CUDA autocast only for checkpoints that declare float16 inference.
        # Float32 checkpoints and explicit CPU inference use full precision.
        """Score a PIL RGB face crop using the checkpoint's saved threshold.
        """
        half = (self.predictor.device.type == 'cuda'
                and self.predictor.meta.get('inference_precision', 'float16') == 'float16')
        context = torch.autocast('cuda', dtype=torch.float16) if half else nullcontext()
        with context:
            result = self.predictor.predict_face(image)
        precision = 'cuda_float16_autocast' if half else f'{self.predictor.device.type}_float32'
        return {**result, 'precision': precision}

    def predict_stream_frame(self, frame, *, stream_id, timestamp=None):
        """Apply the consecutive-frame policy to a camera-frame prediction.

        The caller supplies a stream_id that changes when the tracked person changes.
        """
        if self.temporal_gate is None:
            self.temporal_gate = TemporalLivenessGate()
        return self.temporal_gate.observe(self.predict_frame(frame),stream_id=stream_id,timestamp=timestamp)

    def predict_frame(self, frame):
        """Detect one face in a uint8 BGR frame, convert its crop to RGB and score it.

        Returned face_box values are [x, y, width, height] in the original frame's pixels.
        """
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Frame input must be uint8 HWC BGR')
        # Face-crop callers supply their own upstream detection and do not need
        # a locator asset. Only the full-frame route opens the pinned YuNet model.
        if self.detector is None:
            self.detector = YuNetFaceDetector(self.detector_model)
        boxes = self.detector.detect(frame)
        if len(boxes) != 1:
            return {'status': 'uncertain', 'allow_attendance': False,
                    'reason': 'no_face' if not len(boxes) else 'multiple_faces', 'spoof_score': None,
                    'live_score': None, 'threshold': self.predictor.meta['threshold'],
                    'face_box': None, 'face_box_format': 'source_pixels_xywh',
                    'model_id': self.predictor.meta['model_id'],
                    'model_version': self.predictor.meta['model_version']}
        x, y, width, height = map(int, boxes[0])
        # Margin is a fraction of the detected width/height on each side.
        # Detect on the original frame; brighten only the model's RGB crop.
        frame_height, frame_width = frame.shape[:2]
        left = max(0, int(np.floor(x - width * self.crop_margin)))
        top = max(0, int(np.floor(y - height * self.crop_margin)))
        right = min(frame_width, int(np.ceil(x + width * (1 + self.crop_margin))))
        bottom = min(frame_height, int(np.ceil(y + height * (1 + self.crop_margin))))
        crop = Image.fromarray(cv2.cvtColor(frame[top:bottom, left:right], cv2.COLOR_BGR2RGB))
        if self.brightness_factor != 1.0:
            crop = ImageEnhance.Brightness(crop).enhance(self.brightness_factor)
        return {**self.predict_face(crop), 'reason': None, 'face_box': [left, top, right-left, bottom-top],
                'face_box_format': 'source_pixels_xywh'}
