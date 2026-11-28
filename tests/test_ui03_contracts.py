"""UI-03.1 policy/design tests, not trained face recognition integration."""

import json
import math
from pathlib import Path
import unittest

from facial_attendance.recognition.contracts import (
    EnrollmentContract, FaceInferenceContract, SELECTED_COSINE_THRESHOLD,
    SELECTED_MODEL_ID, SELECTED_MODEL_VERSION, identity_display_state,
)


class FaceContractTests(unittest.TestCase):
    def setUp(self):
        self.contract = FaceInferenceContract()
        self.meta = {"model_id": SELECTED_MODEL_ID, "model_version": SELECTED_MODEL_VERSION,
                     "architecture": "resnet18-3x3-stem-no-maxpool", "approach": "metric",
                     "input_size": 64, "embedding_dim": 512, "mean": [0.485, 0.456, 0.406],
                     "std": [0.229, 0.224, 0.225], "threshold": None}

    def test_selected_policy_matches_the_published_fac05_record(self):
        path = Path(__file__).resolve().parents[1] / "configs/ui03-recognition-selection.json"
        policy = json.loads(path.read_text())
        self.assertEqual(policy["selected_model_id"], self.contract.model_id)
        self.assertEqual(policy["selected_model_version"], self.contract.model_version)
        self.assertEqual(policy["threshold"], 0.4619967254593038)
        self.assertEqual(policy["measure"], "cosine similarity of raw embeddings, accept when score >= threshold")

    def test_matching_metadata_binds_external_selection_without_editing_checkpoint(self):
        self.contract.validate_checkpoint_metadata(self.meta)
        self.assertIsNone(self.meta["threshold"])
        self.meta["threshold"] = SELECTED_COSINE_THRESHOLD
        self.contract.validate_checkpoint_metadata(self.meta)

    def test_wrong_checkpoint_identity_preprocessing_or_threshold_is_rejected(self):
        for key, value in (("model_id", "other"), ("model_version", "other"), ("input_size", 112),
                           ("embedding_dim", 128), ("approach", "classification"), ("threshold", .5),
                           ("mean", None), ("std", [1, 1, 1])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.contract.validate_checkpoint_metadata({**self.meta, key: value})

    def test_threshold_equality_is_accepted_without_recalibration(self):
        threshold = self.contract.cosine_threshold
        self.assertTrue(self.contract.accepts_cosine(threshold))
        self.assertTrue(self.contract.accepts_cosine(math.nextafter(threshold, math.inf)))
        self.assertFalse(self.contract.accepts_cosine(math.nextafter(threshold, -math.inf)))
        for invalid in (True, None, float("nan"), float("inf"), -1.01, 1.01):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                self.contract.accepts_cosine(invalid)

    def test_gallery_contract_requires_finite_normalized_512d_vectors(self):
        self.contract.validate_normalized_embedding([1.0] + [0.0] * 511)
        for invalid in (None, [0.0] * 512, [1.0] * 511, [float("nan")] * 512, [True] + [0.0] * 511):
            with self.subTest(invalid=type(invalid).__name__), self.assertRaises(ValueError):
                self.contract.validate_normalized_embedding(invalid)

    def test_identity_display_is_gated_and_unknown_has_explicit_registration_state(self):
        self.assertEqual(identity_display_state(liveness="live", accepted=True), "known_name")
        self.assertEqual(identity_display_state(liveness="live", accepted=False), "unknown_registration_prompt")
        for status in ("spoof", "uncertain"):
            self.assertEqual(identity_display_state(liveness=status, accepted=True), "blocked")
        self.assertEqual(identity_display_state(liveness="live", accepted=None), "processing")


class EnrollmentContractTests(unittest.TestCase):
    def test_id_and_name_rules_match_gallery_v02_without_any_write(self):
        contract = EnrollmentContract()
        contract.validate_identity("emp-002", "Vỹ", ["emp-001"])
        for person_id, name in (("", "Name"), ("emp-002", ""), (" emp-002", "Name"),
                                ("EMP-001", "Other"), ("emp-002", "x" * 65)):
            with self.subTest(person_id=person_id), self.assertRaises(ValueError):
                contract.validate_identity(person_id, name, ["emp-001"])
        self.assertEqual((contract.gallery_schema, contract.target_samples), ("0.2", 5))

    def test_sample_count_rules_preserve_individually_live_requirement(self):
        contract = EnrollmentContract()
        for count in (3, 5, 10): contract.validate_sample_count(count)
        for count in (True, 2, 11, 3.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                contract.validate_sample_count(count)


if __name__ == "__main__":
    unittest.main()
