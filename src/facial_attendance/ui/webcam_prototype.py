"""SYS-05 capture context, controller and deterministic demo adapters."""

from dataclasses import dataclass, field, replace
from math import isfinite
from numbers import Integral
from typing import TYPE_CHECKING, Literal
from uuid import uuid4

if TYPE_CHECKING:
    from numpy import uint8
    from numpy.typing import NDArray

CONTRACT_VERSION = "sys02-v1"
EMOTION_LABELS = ("Negative", "Neutral", "Positive")
SCENARIOS = ("Live match", "Live unknown", "Spoof", "Uncertain")
Action = Literal["attendance", "registration"]
Box = tuple[int, int, int, int]


@dataclass(frozen=True)
class Context:
    contract_version: str
    attempt_id: str
    action: Action
    capture_id: str
    face_id: str


@dataclass(frozen=True)
class FaceRequest:
    context: Context
    bbox: Box
    frame: "NDArray[uint8]"

    @property
    def face_crop(self) -> "NDArray[uint8]":
        x1, y1, x2, y2 = self.bbox
        return self.frame[y1:y2, x1:x2]


@dataclass(frozen=True)
class Recognition:
    person_id: str | None
    name: str | None
    score: float | None
    accepted: bool | None
    model_id: str | None = None
    model_version: str | None = None


@dataclass(frozen=True)
class Liveness:
    status: Literal["live", "spoof", "uncertain"]
    reason: str
    p_spoof: float | None
    model_version: str | None = None
    preprocessing_version: str | None = None
    decision_policy_version: str | None = None
    source: Literal["stub", "model"] = "stub"


@dataclass(frozen=True)
class Emotion:
    class_id: int
    label: str
    confidence: float | None = None


@dataclass(frozen=True)
class OperationError:
    module: str
    code: str
    message: str
    blocks_persistence: bool


@dataclass(frozen=True)
class Completion:
    context: Context
    result: Recognition | Liveness | Emotion | OperationError


@dataclass(frozen=True)
class Envelope:
    context: Context | None = None
    recognition: Recognition | None = None
    liveness: Liveness | None = None
    emotion: Emotion | None = None
    ui_state: str = "idle"
    can_save: Literal[False] = field(default=False, init=False)
    errors: tuple[OperationError, ...] = ()


def frame_size(frame: "NDArray[uint8]") -> tuple[int, int]:
    shape = getattr(frame, "shape", ())
    if len(shape) != 3 or shape[2] != 3 or min(shape) < 1 or str(frame.dtype) != "uint8":
        raise ValueError("Expected a nonempty uint8 BGR frame with shape H×W×3.")
    height, width, _ = shape
    return width, height


def box_in_frame(box: Box, width: int, height: int) -> bool:
    if len(box) != 4 or not all(
        isinstance(value, Integral) and not isinstance(value, bool) for value in box
    ):
        return False
    x1, y1, x2, y2 = box
    return 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height


def _valid_score(value: float | None, minimum: float = 0.0) -> bool:
    return (
        isinstance(value, (float, int))
        and not isinstance(value, bool)
        and minimum <= value <= 1.0
        and isfinite(value)
    )


class PrototypeController:
    """Own the active capture and reject results from superseded requests."""

    def __init__(self):
        self.action: Action = "attendance"
        self.reset()

    def reset(self, action: Action | None = None) -> None:
        if action is not None:
            self.action = action
        self.attempt_id = uuid4().hex
        self.state = Envelope()

    def capture_error(self, code: str, message: str, module: str = "capture") -> None:
        self.reset()
        ui_state = code if code in {"no_face", "multiple_faces"} else "error"
        self.state = Envelope(
            ui_state=ui_state, errors=(OperationError(module, code, message, True),)
        )

    def begin_capture(self, frame: "NDArray[uint8]", boxes: list[Box]) -> FaceRequest | None:
        if self.state.ui_state == "spoof":
            return None  # Explicit retry or operation change starts a fresh attempt.
        try:
            width, height = frame_size(frame)
        except ValueError as error:
            self.capture_error("invalid_input", str(error))
            return None
        if not boxes:
            self.capture_error("no_face", "No face region available.")
            return None
        if len(boxes) != 1:
            self.capture_error("multiple_faces", "Select one face region at a time.")
            return None
        if not box_in_frame(boxes[0], width, height):
            self.capture_error("invalid_input", "Face region is empty or outside the source frame.")
            return None

        context = Context(
            CONTRACT_VERSION, self.attempt_id, self.action, uuid4().hex, uuid4().hex
        )
        self.state = Envelope(context=context, ui_state="checking")
        return FaceRequest(context, tuple(int(value) for value in boxes[0]), frame)

    def receive(self, completion: Completion) -> bool:
        """Apply a current completion, returning False for a stale or aborted one."""
        state = self.state
        if completion.context != state.context or state.ui_state in {"spoof", "error"}:
            return False
        result = completion.result
        if isinstance(result, OperationError):
            self._apply_error(result)
        elif isinstance(result, Liveness):
            if result.source != "stub" or result.status not in {"live", "spoof", "uncertain"}:
                self._invalid("liveness")
            elif not _valid_score(result.p_spoof):
                self._invalid("liveness")
            elif result.status == "spoof":
                self.state = Envelope(context=state.context, liveness=result, ui_state="spoof")
            else:
                self.state = replace(state, liveness=result, ui_state=result.status)
        elif isinstance(result, Recognition):
            if (
                type(result.accepted) is not bool
                or (result.score is not None and not _valid_score(result.score, -1.0))
                or (result.accepted and not (result.person_id and result.name and result.score is not None))
                or (not result.accepted and (result.person_id is not None or result.name is not None))
            ):
                self._invalid("recognition")
            else:
                self.state = replace(state, recognition=result)
        elif isinstance(result, Emotion):
            if (
                type(result.class_id) is not int
                or not 0 <= result.class_id < len(EMOTION_LABELS)
                or result.label != EMOTION_LABELS[result.class_id]
                or (result.confidence is not None and not _valid_score(result.confidence))
            ):
                self._invalid("emotion")
            else:
                self.state = replace(state, emotion=result)
        return True

    def _invalid(self, module: str) -> None:
        self._apply_error(OperationError(
            module, "invalid_input", f"Invalid {module} stub result.", module != "emotion"
        ))

    def _apply_error(self, error: OperationError) -> None:
        state = self.state
        errors = (*state.errors, error)
        if error.blocks_persistence:
            self.state = Envelope(context=state.context, ui_state="error", errors=errors)
        else:
            self.state = replace(state, emotion=None, errors=errors)

    @property
    def demo_identity(self) -> str:
        result = self.state.recognition
        if self.state.ui_state != "live" or result is None:
            return "Unavailable"
        return result.name if result.accepted else "Unknown"


class StubAdapters:
    """Emit illustrative results for a selected scenario using the SYS-02 fields."""

    def __init__(self, scenario: str):
        if scenario not in SCENARIOS:
            raise ValueError(f"Unknown stub scenario: {scenario}")
        self.scenario = scenario

    def liveness(self, request: FaceRequest) -> Completion:
        if self.scenario == "Spoof":
            result = Liveness("spoof", "demo_result", 0.96)
        elif self.scenario == "Uncertain":
            result = Liveness("uncertain", "demo_result", 0.45)
        else:
            result = Liveness("live", "demo_result", 0.08)
        return Completion(request.context, result)

    def recognition(self, request: FaceRequest) -> Completion:
        result = Recognition(None, None, None, False)
        if self.scenario == "Live match":
            result = Recognition("demo-001", "Demo Person", 0.91, True)
        return Completion(request.context, result)

    def emotion(self, request: FaceRequest) -> Completion:
        class_id = {"Live match": 2, "Live unknown": 1, "Uncertain": 0}[self.scenario]
        return Completion(request.context, Emotion(class_id, EMOTION_LABELS[class_id]))


def run_stubs(
    controller: PrototypeController, request: FaceRequest, adapters: StubAdapters
) -> None:
    """Dispatch current-capture calls serially. Each adapter receives the source and crop."""
    for module, adapter in (
        ("liveness", adapters.liveness),
        ("recognition", adapters.recognition),
        ("emotion", adapters.emotion),
    ):
        if controller.state.context != request.context or controller.state.ui_state in {"spoof", "error"}:
            break
        if module == "recognition" and controller.state.ui_state != "live":
            continue
        try:
            result = adapter(request)
        except Exception:
            result = Completion(request.context, OperationError(
                module, "inference_error", f"The {module} adapter failed.", module != "emotion"
            ))
        if not controller.receive(result):
            break
