"""ResNet-18 sampling, provenance, and isolation from authorization."""

import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import numpy as np

from facial_attendance.emotion.model_contracts import RESNET18_MODEL_ID, RESNET18_SHA256
from facial_attendance.emotion.resnet_inference import ResNetEmotionInference
from facial_attendance.ui.diagnostic_snapshot import prediction
from facial_attendance.ui.emotion_adapter import TrainedEmotionAdapter
from facial_attendance.ui.emotion_presentation import EmotionPresentation
from facial_attendance.ui.live_preview import evaluate_adapters
from facial_attendance.ui.webcam_prototype import (
    Completion,
    Emotion,
    PrototypeController,
    StubAdapters,
)
from tests import test_emotion_presentation as fixtures

BOX, camera_frame = fixtures.BOX, fixtures.camera_frame

CHECKPOINT = Path(__file__).resolve().parents[1] / "artifacts/models/emotion/best.pt"
HAPPY = Emotion(
    1,
    "Happy",
    0.6,
    "model",
    RESNET18_MODEL_ID,
    "a" * 64,
    RESNET18_SHA256,
    "pillow-bgr-rgb-gray3-224-bilinear-imagenet-v1",
    10,
)


class ResNetContractTests(unittest.TestCase):
    def test_label_order_requires_matching_provenance_and_survives_export(self):
        controller = PrototypeController()
        invalid = [
            replace(HAPPY, label="Neutral"),
            replace(HAPPY, model_id="legacy"),
            replace(HAPPY, checkpoint_sha256="b" * 64),
        ]
        for result in [HAPPY, *invalid]:
            with self.subTest(result=result):
                request = controller.begin_capture(camera_frame(), [BOX])
                controller.receive(Completion(request.context, result))
                self.assertEqual(
                    controller.state.emotion, HAPPY if result is HAPPY else None
                )
                self.assertFalse(controller.state.can_save)
        self.assertEqual(
            prediction({"class_id": 1, "label": "Happy", "source": "model"}, "emotion")[
                "label"
            ],
            "Happy",
        )

    def test_skipped_samples_do_not_call_emotion_even_when_recognition_fails(self):
        request = PrototypeController().begin_capture(camera_frame(), [BOX])
        adapters = StubAdapters("Live match")
        adapters.trained_recognition = adapters.trained_emotion = True
        adapters.emotion = Mock()
        self.assertEqual(len(evaluate_adapters((request, adapters, False))), 2)
        adapters.recognition = Mock(side_effect=RuntimeError("model failed"))
        self.assertEqual(len(evaluate_adapters((request, adapters, False))), 2)
        adapters.emotion.assert_not_called()

    @unittest.skipUnless(
        CHECKPOINT.is_file(), "Supplied ResNet-18 handover is not installed"
    )
    def test_actual_checkpoint_is_selected_and_returns_native_class_order(self):
        adapter = TrainedEmotionAdapter(CHECKPOINT)
        self.assertIsInstance(adapter.inference, ResNetEmotionInference)
        y, x = np.indices((109, 137))
        frame = np.stack(
            ((x * 5 + y) % 256, (x + y * 3) % 256, (2 * x + y) % 256), axis=-1
        ).astype("uint8")
        box = (2, 3, 131, 103)
        request = PrototypeController().begin_capture(frame, [box])
        result = adapter(request).result
        self.assertEqual(
            (result.source, result.checkpoint_sha256, result.selected_epoch),
            ("model", RESNET18_SHA256, 10),
        )
        self.assertEqual(result.label, ("Sad", "Happy", "Neutral")[result.class_id])
        self.assertGreater(result.confidence, 1 / 3)
        self.assertFalse(adapter.inference.model.training)
        self.assertIs(adapter.inference.model, adapter.inference.load().model)


class ResNetGuiTests(unittest.TestCase):
    setUp = fixtures.EmotionGuiTests.setUp
    packet = fixtures.EmotionGuiTests.packet
    finish = fixtures.EmotionGuiTests.finish

    def start_sample(self, at, sequence):
        self.now = at
        self.packet(sequence, camera_frame(noise=6, light=sequence % 12, seed=sequence))
        self.app._tick()

    def install(self):
        app = fixtures.EmotionGuiTests.install(self)
        app._trained_emotion_enabled = True
        app.model_info = Mock()
        app.emotion_presentation = EmotionPresentation(prediction_interval=3.0)
        app.adapters.emotion = Mock(
            side_effect=lambda req: Completion(req.context, HAPPY)
        )
        self.start_sample(10.0, 1)
        self.finish()
        return app

    def test_emotion_samples_every_three_seconds_without_skipping_recognition(self):
        app = self.install()
        app.display_emotion.set.reset_mock()
        for sequence in range(2, 9):
            self.start_sample(10 + (sequence - 1) * 0.42, sequence)
            self.finish()
            self.assertIsNone(app.controller.state.emotion)
            self.assertIsNotNone(app.controller.state.recognition)
            self.assertFalse(app.controller.state.can_save)
        app.display_emotion.set.assert_not_called()
        app.adapters.emotion.assert_called_once()
        self.start_sample(13.36, 9)
        self.finish()
        self.assertEqual(app.adapters.emotion.call_count, 2)
        self.assertEqual(app.controller.state.emotion, HAPPY)

    def test_no_face_and_multiple_faces_immediately_clear_previous_prediction(self):
        for boxes in ([], [BOX, (230, 30, 300, 110)]):
            with self.subTest(boxes=boxes):
                app = self.install()
                self.start_sample(10.42, 2)
                app.detector.finish(boxes)
                app._tick()
                self.assertEqual(
                    app.emotion_presentation.view(self.now).label, "Unavailable"
                )
                self.assertIsNone(app.controller.state.emotion)
                self.assertFalse(app.controller.state.can_save)
                app.retry()

    def test_previous_prediction_expires_while_camera_and_detections_continue(self):
        app = self.install()
        for seq in range(2, 11):
            self.start_sample(10 + (seq - 1) * 0.43, seq)
            # Continuous valid detections with no second successful Emotion prediction.
            app.detector.finish([BOX])
            app._tick()
            app.inference.finish(
                evaluate_adapters((app.inference.job[1][0], app.adapters, False))
            )
            app._tick()
        view = app.emotion_presentation.view(self.now)
        self.assertEqual(
            (view.label, view.reason), ("Unavailable", "freshness_expired")
        )
