"""Call submit/poll from the UI thread; inference runs off the UI thread."""

import time
from concurrent.futures import ThreadPoolExecutor

try:
    from .inference import EmotionPredictor
except ImportError:
    from inference import EmotionPredictor


class EmotionController:
    def __init__(self):
        self.predictor = EmotionPredictor()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.future = None
        self.future_generation = None
        self.generation = 0
        self.face_key = None
        self.next_prediction = 0.0

    def submit(self, face_bgr, face_key=None):
        """Call for each camera frame. Pass None if no valid single face exists.

        face_key is the host's current face/track ID. A changed ID invalidates
        earlier results. Return status is for the current frame, not an emotion.
        """
        if face_bgr is None:
            if self.face_key is not None:
                self.generation += 1
            self.face_key = None
            self.next_prediction = 0.0
            return {"status": "no_valid_face"}
        key = face_key if face_key is not None else "single-face"
        if key != self.face_key:
            self.generation += 1
            self.face_key = key
            self.next_prediction = 0.0
        if self.future is not None:
            return {"status": "pending"}
        now = time.monotonic()
        if now < self.next_prediction:
            return {"status": "waiting_interval"}
        self.next_prediction = now + 3.0
        self.future_generation = self.generation
        self.future = self.executor.submit(self.predictor.predict_bgr, face_bgr.copy())
        return {"status": "pending"}

    def poll(self):
        """Call periodically from the UI thread. None means no NEW result."""
        if self.future is None or not self.future.done():
            return None
        future = self.future
        generation = self.future_generation
        self.future = None
        if generation != self.generation or self.face_key is None:
            return None
        try:
            return future.result()
        except Exception as exc:
            return {"status": "unavailable", "message": str(exc)}

    def close(self):
        self.generation += 1
        self.face_key = None
        self.executor.shutdown(wait=False, cancel_futures=True)
