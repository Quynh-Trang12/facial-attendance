"""Bounded live-preview workers and main-thread SYS-05 coordination.

Workers never access Tk or the active controller. Each analysis sample
supersedes the previous sample; raw preview frames are not inference requests.
"""

from dataclasses import dataclass, field
from threading import Condition, Event, Lock, Thread
from time import monotonic

from .face_detection import FaceDetectorUnavailable
from .diagnostics import Diagnostics
from .webcam_prototype import (
    Completion, Envelope, PrototypeController, box_in_frame, frame_size, run_stubs,
)


@dataclass(frozen=True)
class WorkResult:
    token: tuple[int, int]
    value: object = None
    error: Exception | None = None
    duration: float = field(default=0.0, compare=False)


class SingleWorker:
    """One persistent daemon, one job/result slot, no pending backlog."""

    def __init__(self, function, name, metrics=None):
        self.function = function
        self.metrics = metrics or Diagnostics()
        self.name = name
        self._condition = Condition()
        self._job = self._result = None
        self._busy = self._closed = False
        self.thread = Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def submit(self, token, value):
        with self._condition:
            if self._busy or self._closed:
                return False
            self._busy = True
            self._job = (token, value)
            self._condition.notify()
            return True

    def take(self):
        with self._condition:
            result, self._result = self._result, None
            if result is not None:
                self._busy = False
            return result

    @property
    def busy(self):
        with self._condition:
            return self._busy

    def _run(self):
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._closed or self._job is not None)
                if self._closed:
                    return
                (token, value), self._job = self._job, None
            started = monotonic()
            try:
                result = WorkResult(token, self.function(value), duration=monotonic() - started)
            except Exception as error:
                result = WorkResult(token, error=error, duration=monotonic() - started)
            self.metrics.duration(self.name, result.duration)
            with self._condition:
                if not self._closed:
                    self._result = result
                    self._condition.notify_all()

    def close(self):
        with self._condition:
            self._closed = True
            self._job = self._result = None
            self._condition.notify()
        # A hung native model cannot block Tk shutdown. No replacement thread
        # is spawned; running daemon work can only finish into a closed slot.
        self.thread.join(timeout=0.1)


@dataclass(frozen=True)
class CameraPacket:
    generation: int
    sequence: int
    captured_at: float
    frame: object = None
    error: str | None = None


class CameraFeed:
    """Own camera open/read/release on one thread; keep only the latest frame."""

    def __init__(self, open_camera, metrics=None):
        self.open_camera = open_camera
        self.metrics = metrics or Diagnostics()
        self._lock = Lock()
        self._wake = Event()
        self._stop = Event()
        self._generation = 0
        self._latest = None
        self.thread = Thread(target=self._run, name="sys05-camera", daemon=True)
        self.thread.start()

    def restart(self):
        with self._lock:
            self._generation += 1
            generation = self._generation
            self._latest = None
        self._wake.set()
        return generation

    def latest(self):
        with self._lock:
            return self._latest

    def _current(self, generation):
        with self._lock:
            return not self._stop.is_set() and generation == self._generation

    def _publish(self, packet):
        with self._lock:
            if not self._stop.is_set() and packet.generation == self._generation:
                self._latest = packet
                return True
            return False

    def _run(self):
        while not self._stop.is_set():
            self._wake.wait()
            self._wake.clear()
            if self._stop.is_set():
                break
            with self._lock:
                generation = self._generation
            camera = None
            sequence = 0
            try:
                camera = self.open_camera()
                if not camera.isOpened():
                    raise OSError("Camera unavailable. Check permissions/device and retry.")
                while self._current(generation):
                    cycle_started = monotonic()
                    ok, frame = camera.read()
                    captured_at = monotonic()
                    self.metrics.duration("camera_read", captured_at - cycle_started)
                    if not ok:
                        raise OSError("Camera read failed. Reconnect the camera and retry.")
                    frame_size(frame)
                    # Some backends reuse their read buffer. Publish an owned,
                    # immutable BGR snapshot for both preview and analysis.
                    frame = frame.copy()
                    frame.flags.writeable = False
                    self.metrics.duration("camera_copy", monotonic() - captured_at)
                    sequence += 1
                    published = self._publish(CameraPacket(generation, sequence, captured_at, frame))
                    self.metrics.count("captured" if published else "capture_discarded")
                    # Blocking reads already consume the frame period. Adding
                    # a fixed sleep after them nearly halves a 30-Hz source.
                    self._stop.wait(max(0.0, 1 / 30 - (monotonic() - cycle_started)))
            except Exception:
                self._publish(CameraPacket(
                    generation, sequence + 1, monotonic(),
                    error="Camera unavailable or unusable BGR frame. Check access and retry.",
                ))
            finally:
                if camera is not None:
                    try:
                        camera.release()
                    except Exception:
                        pass

    def close(self):
        self._stop.set()
        self._wake.set()
        with self._lock:
            self._generation += 1
            self._latest = None
        self.thread.join(timeout=0.1)


def evaluate_adapters(request_and_adapters):
    """Use an isolated controller for worker-side gating; return original contexts."""
    request, adapters = request_and_adapters
    local = PrototypeController()
    local.action = request.context.action
    local.attempt_id = request.context.attempt_id
    local.state = Envelope(context=request.context, ui_state="checking")
    return run_stubs(local, request, adapters)


def prepare_preview_image(frame):
    """Resize BGR before RGB conversion/Pillow allocation, on the GUI thread."""
    import cv2
    from PIL import Image
    width, height = frame_size(frame)
    scale = min(1.0, 640 / width, 360 / height)
    size = max(1, round(width * scale)), max(1, round(height * scale))
    small = frame if size == (width, height) else cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))


class BoxStability:
    """One display-only box and timestamp; never used as adapter evidence."""

    HOLD_SECONDS = 0.45

    def __init__(self):
        self.clear()

    def clear(self):
        self.box = self.size = self.time = None

    def confirm(self, box, size, timestamp):
        if self.box is not None and size == self.size and timestamp - self.time <= self.HOLD_SECONDS:
            a, b = self.box, box
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            area_a = (a[2] - a[0]) * (a[3] - a[1])
            area_b = (b[2] - b[0]) * (b[3] - b[1])
            if overlap / (area_a + area_b - overlap) > 0.5:
                box = tuple(round(0.7 * current + 0.3 * previous) for current, previous in zip(box, a))
        self.box, self.size, self.time = box, size, timestamp

    def visual(self, now):
        if self.box is None or now - self.time > self.HOLD_SECONDS:
            return [], None
        return [self.box], self.size


class LiveController:
    """Called exclusively by the GUI thread (also usable without Tk in tests)."""

    def __init__(self):
        self.controller = PrototypeController()
        self.stability = BoxStability()
        self.reset()

    def reset(self):
        self.controller.reset("attendance")
        self.stability.clear()
        self.token = None
        self.boxes = []
        self.sample_time = None
        self.sample_size = None
        self.status = "Idle / opening camera"

    def begin_sample(self, token, captured_at):
        self.controller.reset("attendance")
        self.token = token
        self.boxes = []
        self.sample_size = None
        self.sample_time = captured_at
        self.status = "Processing — detecting a face"

    def fail(self, code, message, module="capture"):
        self.stability.clear()
        self.token = None
        self.boxes = []
        self.sample_size = None
        self.sample_time = None
        self.controller.capture_error(code, message, module)
        self.status = message

    def detected(self, result, frame):
        if result.token != self.token:
            return None
        if result.error is not None:
            code = "model_unavailable" if isinstance(result.error, FaceDetectorUnavailable) else (
                "invalid_input" if isinstance(result.error, ValueError) else "inference_error"
            )
            self.fail(code, f"Detector unavailable: {result.error}"
                      if code == "model_unavailable" else "Face detector failed; retry.", "detector")
            return None
        try:
            width, height = frame_size(frame)
            boxes = list(result.value)
            if not all(box_in_frame(box, width, height) for box in boxes):
                raise ValueError("Invalid detector box")
        except (TypeError, ValueError):
            self.fail("invalid_input", "Invalid face detection coordinates.", "detector")
            return None
        self.boxes = boxes
        self.sample_size = width, height
        if len(boxes) == 1:
            self.stability.confirm(boxes[0], self.sample_size, self.sample_time)
        elif len(boxes) > 1:
            self.stability.clear()
        request = self.controller.begin_capture(frame, boxes)
        self.status = {
            "no_face": "No face detected",
            "multiple_faces": "Multiple faces — show one face only",
            "checking": "Processing — simulated adapters",
        }.get(self.controller.state.ui_state, "Face input unavailable")
        return request

    def visual_regions(self, now):
        if len(self.boxes) == 1:
            return [self.stability.box], self.sample_size, False
        if self.boxes:
            return self.boxes, self.sample_size, False
        boxes, size = self.stability.visual(now)
        return boxes, size, bool(boxes)

    def completed(self, result):
        if result.token != self.token or self.controller.state.context is None:
            return False
        if result.error is not None:
            self.fail("inference_error", "Demo adapter unavailable; retry.", "adapters")
            return False
        if (not isinstance(result.value, (tuple, list))
                or not all(isinstance(value, Completion) for value in result.value)):
            self.fail("invalid_input", "Malformed adapter completion batch.", "adapters")
            return False
        applied = False
        for completion in result.value:
            applied = self.controller.receive(completion) or applied
        state = self.controller.state
        self.status = {
            "live": "One face — SIMULATED results",
            "spoof": "Simulated spoof — attempt blocked",
            "uncertain": "Simulated uncertain liveness — retry / use a fresh frame",
            "error": "Demo adapter error — retry",
        }.get(state.ui_state, self.status)
        return applied

    def expire(self, now, max_age=1.0):
        if self.sample_time is not None and now - self.sample_time > max_age:
            self.fail("stale_result", "Analysis expired — waiting for a fresh frame.")
            return True
        return False
