"""Exercise the Haar adapter contract without requiring an OpenCV install."""

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch, sentinel

from facial_attendance.ui.face_detection import FaceDetectorUnavailable, HaarFaceDetector


class HaarFaceDetectorTests(unittest.TestCase):
    def setUp(self):
        self.frame = SimpleNamespace(shape=(720, 1280, 3), dtype="uint8")
        self.classifier = Mock()
        self.classifier.empty.return_value = False
        self.cv2 = SimpleNamespace(
            CascadeClassifier=Mock(return_value=self.classifier),
            data=SimpleNamespace(haarcascades="bundle"),
            cvtColor=Mock(return_value=sentinel.gray),
            COLOR_BGR2GRAY=6,
            error=type("OpenCVError", (Exception,), {}),
        )
        self.detector = HaarFaceDetector()
        self.enterContext(patch.dict(sys.modules, {"cv2": self.cv2}))
        self.xml = self.enterContext(
            patch("facial_attendance.ui.face_detection.Path.is_file", return_value=True)
        )

    def test_converts_bgr_to_gray_and_keeps_source_coordinates(self):
        self.classifier.detectMultiScale.return_value = [(700, 300, 200, 100), (1230, 650, 50, 70)]
        self.assertEqual(self.detector(self.frame), [(700, 300, 900, 400), (1230, 650, 1280, 720)])
        self.cv2.cvtColor.assert_called_once_with(self.frame, self.cv2.COLOR_BGR2GRAY)
        self.classifier.detectMultiScale.assert_called_once_with(
            sentinel.gray, scaleFactor=1.1, minNeighbors=5
        )

    def test_empty_detection_and_classifier_is_reused(self):
        self.classifier.detectMultiScale.return_value = ()
        self.assertEqual(self.detector(self.frame), [])
        self.assertEqual(self.detector(self.frame), [])
        self.cv2.CascadeClassifier.assert_called_once()

    def test_enclosing_face_wins_over_nested_eye_box_in_either_order(self):
        face = (100, 100, 200, 200)
        eye = (140, 140, 70, 60)
        for detections in ((eye, face), (face, eye)):
            with self.subTest(detections=detections):
                self.classifier.detectMultiScale.return_value = detections
                self.assertEqual(self.detector(self.frame), [(100, 100, 300, 300)])

    def test_exact_and_near_duplicate_windows_keep_one_original_box(self):
        face = (100, 100, 200, 200)
        shifted = (108, 105, 196, 195)
        self.classifier.detectMultiScale.return_value = [shifted, face, face]
        self.assertEqual(self.detector(self.frame), [(100, 100, 300, 300)])

    def test_separate_face_survives_suppression_of_nested_and_duplicate_boxes(self):
        face = (100, 100, 200, 200)
        other_face = (500, 130, 120, 120)
        self.classifier.detectMultiScale.return_value = [(140, 140, 70, 60), other_face, face, face]
        self.assertEqual(self.detector(self.frame), [(100, 100, 300, 300), (500, 130, 620, 250)])

    def test_ambiguous_overlap_or_containment_keeps_both_boxes(self):
        face = (100, 100, 200, 200)
        for candidate in ((155, 100, 200, 200), (120, 120, 160, 160), (80, 140, 70, 60)):
            with self.subTest(candidate=candidate):
                self.classifier.detectMultiScale.return_value = [face, candidate]
                x, y, w, h = candidate
                self.assertEqual(self.detector(self.frame), [(100, 100, 300, 300), (x, y, x + w, y + h)])

    def test_invalid_detection_blocks_entire_batch(self):
        for invalid in (None, (0, 0, 1), (-1, 10, 5, 5), (1279, 0, 2, 3),
                        (0, 0, 0, 4), (0, 0, 3, -1), (0, 0, True, 2), (0, 0, 2.0, 2)):
            with self.subTest(invalid=invalid):
                self.classifier.detectMultiScale.return_value = [(10, 10, 40, 40), invalid]
                with self.assertRaisesRegex(ValueError, "Haar returned"):
                    self.detector(self.frame)

    def test_invalid_frame_is_rejected_before_loading_classifier(self):
        self.frame.dtype = "float32"
        with self.assertRaises(ValueError):
            self.detector(self.frame)
        self.cv2.CascadeClassifier.assert_not_called()

    def test_missing_xml_and_unusable_classifier_are_unavailable(self):
        self.xml.return_value = False
        with self.assertRaisesRegex(FaceDetectorUnavailable, "XML"):
            self.detector(self.frame)
        self.cv2.CascadeClassifier.assert_not_called()
        self.xml.return_value = True
        self.classifier.empty.return_value = True
        with self.assertRaisesRegex(FaceDetectorUnavailable, "could not load"):
            self.detector(self.frame)

    def test_missing_binding_is_unavailable(self):
        del self.cv2.CascadeClassifier
        with self.assertRaisesRegex(FaceDetectorUnavailable, "lacks"):
            self.detector(self.frame)


if __name__ == "__main__":
    unittest.main()
