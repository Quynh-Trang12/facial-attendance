"""The top-1/top-2 margin in the GUI: check-in names a face only past the margin,
registration still refuses duplicates on tau alone, and attendance treats a close
runner-up as unknown rather than as a broken result."""

import math
import unittest

import numpy as np

from facial_attendance.attendance.service import AttendanceService, VerificationPolicy
from facial_attendance.attendance.store import AttendanceStore
from facial_attendance.recognition.contracts import (
    SELECTED_COSINE_THRESHOLD, SELECTED_MODEL_ID, SELECTED_MODEL_VERSION, FaceInferenceContract,
)
from facial_attendance.recognition.decision import load_margin
from facial_attendance.recognition.evaluation import RecognitionEvaluation, SandboxInference
from facial_attendance.recognition.gallery_matching import (
    DECISION_CONFIG, ControlledTestEnrollment, GalleryMatcher, PreparedTestEnrollment, empty_gallery,
)
from facial_attendance.ui.webcam_prototype import Recognition

TAU = SELECTED_COSINE_THRESHOLD
MARGIN = 0.15


def at_angle(degrees):
    """A unit 512-D vector in the plane of the first two axes."""
    radians = math.radians(degrees)
    return (math.cos(radians), math.sin(radians)) + (0.0,) * 510


def with_cosine(score):
    """A unit vector whose cosine with at_angle(0) is score."""
    return at_angle(math.degrees(math.acos(score)))


# A sits at 0 degrees and B at about 72.5 degrees, so cos(A, B) = 0.3, below tau.
# NEAR_BOTH is about 0.84 from A and 0.77 from B: above tau for both, within the margin.
A, B = at_angle(0), with_cosine(0.3)
NEAR_BOTH = at_angle(33)


class MatcherMarginTests(unittest.TestCase):
    def setUp(self):
        self.gallery = empty_gallery()
        self.matcher = GalleryMatcher(self.gallery, margin=MARGIN)

    def test_default_margin_comes_from_the_recognition_config(self):
        self.assertEqual(GalleryMatcher(empty_gallery()).margin, load_margin(DECISION_CONFIG))
        self.assertEqual(load_margin(DECISION_CONFIG), 0.15)

    def test_clear_winner_is_named(self):
        self.gallery.register("test-A", "Person A", A, n_images=5)
        self.gallery.register("test-B", "Person B", B, n_images=5)
        result = self.matcher.match(A)
        self.assertEqual((result.accepted, result.person_id, result.name), (True, "test-A", "Person A"))
        self.assertAlmostEqual(result.score, 1.0)

    def test_close_runner_up_is_unknown_but_keeps_its_score(self):
        self.gallery.register("test-A", "Person A", A, n_images=5)
        self.gallery.register("test-B", "Person B", B, n_images=5)
        result = self.matcher.match(NEAR_BOTH)
        self.assertEqual((result.accepted, result.person_id, result.name), (False, None, None))
        self.assertGreaterEqual(result.score, TAU)
        # The old rule would have named A for the same probe.
        self.assertEqual(self.matcher.match_threshold_only(NEAR_BOTH).person_id, "test-A")

    def test_one_profile_needs_tau_plus_margin(self):
        self.gallery.register("test-A", "Person A", A, n_images=5)
        between = self.matcher.match(with_cosine(TAU + MARGIN - 0.01))
        self.assertEqual((between.accepted, between.person_id), (False, None))
        above = self.matcher.match(with_cosine(TAU + MARGIN + 0.01))
        self.assertEqual((above.accepted, above.person_id), (True, "test-A"))

    def test_below_tau_is_unknown_under_both_rules(self):
        self.gallery.register("test-A", "Person A", A, n_images=5)
        probe = with_cosine(TAU - 0.01)
        for result in (self.matcher.match(probe), self.matcher.match_threshold_only(probe)):
            self.assertEqual((result.accepted, result.person_id, result.name), (False, None, None))

    def test_negative_margin_is_refused(self):
        self.gallery.register("test-A", "Person A", A, n_images=5)
        with self.assertRaises(ValueError):
            GalleryMatcher(self.gallery, margin=-0.1).match(A)


class FixedInference:
    def __init__(self, vector):
        self.vector = vector

    def embed(self, frame, box):
        return self.vector


class ControlledEnrollmentDuplicateTests(unittest.TestCase):
    def test_a_face_close_to_two_profiles_is_still_a_duplicate(self):
        enrollment = ControlledTestEnrollment(FixedInference(A))
        enrollment.commit(PreparedTestEnrollment("test-a", "Person A", (A,) * 5))
        enrollment.commit(PreparedTestEnrollment("test-b", "Person B", (B,) * 5))
        self.assertFalse(enrollment.match(NEAR_BOTH).accepted)
        with self.assertRaisesRegex(ValueError, "Already enrolled"):
            enrollment.commit(PreparedTestEnrollment("test-c", "Person C", (NEAR_BOTH,) * 5))
        self.assertEqual([person.person_id for person in enrollment._gallery.people], ["test-a", "test-b"])


class SandboxInference512:
    """The sandbox's own fake: a fixed embedding with the selected model's provenance."""

    contract = FaceInferenceContract()
    provenance = {"model_id": contract.model_id, "model_version": contract.model_version,
                  "threshold": contract.cosine_threshold, "checkpoint_sha256": "a" * 64,
                  "preprocessing_version": "fixture"}

    def __init__(self):
        self.vector = A

    def embed(self, frame, box):
        return self.vector


class WebcamRegistrationDuplicateTests(unittest.TestCase):
    """RecognitionEvaluation.commit_reference is the webcam registration path."""

    def setUp(self):
        self.session = RecognitionEvaluation(source_kind="generated")
        self.inference = SandboxInference512()
        self.worker = SandboxInference(self.inference, self.session)
        self.sequence = 0

    def complete(self, kind, **kw):
        self.sequence += 1
        frame = np.full((240, 320, 3), self.sequence, np.uint8)
        job = self.session.job(kind, (1, self.sequence), frame, [(40, 40, 180, 180)], 10., now=10., **kw)
        return self.session.accept(self.worker(job), current_source=job.source, now=10.05)

    def enroll(self, vector):
        self.inference.vector = vector
        self.session.set_consent(True)
        for _ in range(3):
            self.complete("reference")
        test_id = self.session.commit_reference()
        self.session.set_consent(True)
        return test_id

    def test_a_face_close_to_two_profiles_is_refused_at_registration(self):
        self.enroll(A)
        self.enroll(B)
        self.inference.vector = NEAR_BOTH
        for _ in range(3):
            self.complete("reference")
        with self.assertRaisesRegex(ValueError, "existing test subject"):
            self.session.commit_reference()
        self.assertEqual(self.session.test_ids, ("test-001", "test-002"))

    def test_a_close_runner_up_is_recorded_as_unknown(self):
        first = self.enroll(A)
        self.enroll(B)
        self.inference.vector = NEAR_BOTH
        decision = self.complete("evaluation", truth=first)
        self.assertEqual((decision.accepted, decision.test_id, decision.test_name), (False, None, None))
        self.assertGreaterEqual(decision.score, TAU)
        self.assertEqual(self.session.records[-1]["decision"], "Unknown")


POLICY = VerificationPolicy("fixture-live", "fixture-rgb", "fixture-capture-v1", "a" * 64,
                            "fixture-face-rgb", 1.0)


def recognition(person_id, name, score, accepted):
    return Recognition(person_id, name, score, accepted, SELECTED_MODEL_ID, SELECTED_MODEL_VERSION,
                       "model", POLICY.recognition_checkpoint, POLICY.recognition_preprocessing)


class AttendanceStatusTests(unittest.TestCase):
    def setUp(self):
        self.service = AttendanceService(AttendanceStore(":memory:"), policy=POLICY,
                                         registered_name={"fixture-person": "Attendance Test"}.get)

    def test_close_runner_up_above_tau_is_unknown(self):
        self.assertEqual(self.service.recognition_status(recognition(None, None, 0.80, False)), "unknown")

    def test_accepted_result_still_needs_tau(self):
        self.assertEqual(self.service.recognition_status(
            recognition("fixture-person", "Attendance Test", TAU - 0.01, True)), "invalid_verification")
        self.assertEqual(self.service.recognition_status(
            recognition("fixture-person", "Attendance Test", 0.80, True)), "known")

    def test_unknown_result_must_not_carry_an_identity(self):
        for score in (0.30, 0.80):
            self.assertEqual(self.service.recognition_status(
                recognition("fixture-person", "Attendance Test", score, False)), "invalid_verification")


if __name__ == "__main__":
    unittest.main()
