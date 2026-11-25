"""Live check-in logic with generated frames, fake detections and a fake embedder."""

import importlib.util
import unittest

# The fused mode measures frames with OpenCV; the dependency jobs have both.
if importlib.util.find_spec("numpy") is None or importlib.util.find_spec("cv2") is None:
    raise unittest.SkipTest("numpy or OpenCV is not installed in this environment")

import csv
from pathlib import Path
import tempfile

import cv2
import numpy as np

from facial_attendance.recognition.check_in_live import (
    SCHEMA_RULE,
    STUDY_RULES,
    TRACE_COLUMNS,
    CheckIn,
    Face,
    Gallery,
    append_log,
    append_rows,
    check_limits,
    crop,
    dataset_limits,
    diagnose,
    enroll_study,
    faces_from_yunet,
    image_paths,
    limits_toml,
    load_study,
    photo_samples,
    read_image,
    replay_gate,
    study_embeddings,
    summarize,
    template_cosines,
    trace_row,
    with_limits,
)
from facial_attendance.recognition.quality import FaceQuality
from facial_attendance.recognition.probe_fusion import CheckInSettings
from facial_attendance.recognition.quality import QualityLimits

TAU = 0.4620
META = {"model_id": "run-m", "model_version": "abc123abc123", "threshold": TAU}
LIMITS = QualityLimits(min_sharpness=100.0, min_brightness=60.0, max_brightness=200.0, min_contrast=20.0,
                       min_size=64, max_yaw=0.25, max_roll=15.0)
SETTINGS = CheckInSettings(enabled=True, k=3, min_frames=2, window_s=2.0, limits=LIMITS)
FRONTAL = np.array([[40, 50], [80, 50], [60, 70], [45, 90], [75, 90]], dtype=np.float64)


def unit(*values):
    vector = np.asarray(values, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def frame(blur=False):
    """A 160x160 RGB frame whose face area is a sharp mid-grey checkerboard."""
    tiles = (np.indices((160, 160)) // 8).sum(axis=0) % 2
    grey = np.where(tiles == 1, 200, 60).astype(np.uint8)
    image = np.repeat(grey[:, :, None], 3, axis=2)
    return cv2.GaussianBlur(image, (0, 0), 6) if blur else image


FACE = Face((10, 10, 138, 138), FRONTAL + 10)


def gallery_with(*people, rule="box"):
    gallery = Gallery(META["model_id"], META["model_version"], TAU, rule)
    for label, direction in people:
        gallery.enroll(label, [direction] * 5)
    return gallery


class DetectionTests(unittest.TestCase):
    def test_yunet_rows_are_clipped_like_the_gui(self):
        row = [-3.4, 5.2, 50.0, 60.0, *FRONTAL.ravel(), 0.95]
        (face,) = faces_from_yunet(np.array([row]), width=40, height=200)
        self.assertEqual(face.box, (0, 5, 40, 66))
        np.testing.assert_array_equal(face.landmarks, FRONTAL)

    def test_empty_or_outside_boxes_are_dropped(self):
        self.assertEqual(faces_from_yunet(None, 10, 10), [])
        outside = [20.0, 20.0, 5.0, 5.0, *FRONTAL.ravel(), 0.95]
        self.assertEqual(faces_from_yunet(np.array([outside]), 10, 10), [])

    def test_crop_uses_the_box(self):
        image = np.arange(100 * 100 * 3, dtype=np.uint8).reshape(100, 100, 3)
        self.assertEqual(crop(image, Face((10, 20, 30, 60), None)).shape, (40, 20, 3))
        self.assertEqual(crop(image, Face((10, 20, 30, 60), None), "box").shape, (40, 20, 3))

    def test_schema_crop_is_a_square_around_the_box_centre(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        image[100, 60] = 255  # the box centre
        # A tall box, 40 wide and 60 high, centred on (60, 100).
        face = Face((40, 70, 80, 130), None, (40.0, 70.0, 40.0, 60.0))
        square = crop(image, face, "square-1.2")
        self.assertEqual(square.shape, (72, 72, 3))
        self.assertEqual(square[36, 36, 0], 255)
        self.assertEqual(crop(image, face, "square-1.0").shape, (60, 60, 3))
        self.assertEqual(crop(image, face, "square-1.4").shape, (84, 84, 3))

    def test_schema_crop_is_clipped_to_the_frame_around_the_unclipped_box(self):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # YuNet's box starts left of the frame; its centre is (10, 50), side 1.2 x 40 = 48.
        (face,) = faces_from_yunet(np.array([[-10.0, 30.0, 40.0, 40.0, *FRONTAL.ravel(), 0.95]]), 100, 100)
        self.assertEqual(face.box, (0, 30, 30, 70))
        self.assertEqual(crop(image, face, "square-1.2").shape, (48, 34, 3))

    def test_crop_rules_are_checked(self):
        image = np.zeros((50, 50, 3), dtype=np.uint8)
        for bad in ("square", "square-0", "square--1", "margin-1.2", ""):
            with self.assertRaises(ValueError):
                crop(image, FACE, bad)
        self.assertEqual(SCHEMA_RULE, "square-1.2")
        self.assertEqual(STUDY_RULES, ("box", "square-1.0", "square-1.2", "square-1.4"))


class GalleryTests(unittest.TestCase):
    def test_enrollment_averages_and_matches(self):
        gallery = gallery_with(("An", unit(1, 0, 0)))
        match = gallery.match(unit(1, 0.1, 0))
        self.assertEqual((match.label, match.accepted), ("An", True))

    def test_raw_embedding_length_does_not_change_the_score(self):
        gallery = gallery_with(("An", unit(1, 0, 0)))
        raw = 0.08 * unit(1, 0.1, 0)  # the model's embeddings are not unit length
        self.assertAlmostEqual(gallery.match(raw).score, gallery.match(unit(1, 0.1, 0)).score)
        self.assertTrue(gallery.match(raw).accepted)

    def test_mixed_capture_duplicate_face_and_taken_name_are_refused(self):
        gallery = gallery_with(("An", unit(1, 0, 0)))
        with self.assertRaisesRegex(ValueError, "do not agree"):
            gallery.enroll("Trang", [unit(0, 1, 0)] * 4 + [unit(0, -1, 0.2)])
        with self.assertRaisesRegex(ValueError, "already matches 'An'"):
            gallery.enroll("Someone", [unit(1, 0.05, 0)] * 5)
        with self.assertRaisesRegex(ValueError, "already registered"):
            gallery.enroll("an", [unit(0, 0, 1)] * 5)
        self.assertEqual(gallery.labels, ["An"])

    def test_save_load_and_refuse_another_checkpoint(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-gallery-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "gallery.npz"
        self.assertEqual(Gallery.load_or_new(path, META, "box").labels, [])
        gallery = gallery_with(("An", unit(1, 0, 0)), ("Vy", unit(0, 1, 0)))
        gallery.save(path)
        loaded = Gallery.load_or_new(path, META, "box")
        self.assertEqual(loaded.labels, ["An", "Vy"])
        np.testing.assert_allclose(loaded.templates, gallery.templates)
        with self.assertRaisesRegex(ValueError, "another checkpoint"):
            Gallery.load_or_new(path, META | {"model_version": "000000000000"}, "box")
        Gallery(META["model_id"], META["model_version"], TAU, "box").save(path)
        self.assertEqual(Gallery.load_or_new(path, META, "box").labels, [])

    def test_gallery_records_its_crop_rule_and_refuses_another(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-gallery-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "gallery.npz"
        gallery_with(("An", unit(1, 0, 0)), rule="square-1.2").save(path)
        self.assertEqual(Gallery.load_or_new(path, META, "square-1.2").crop_rule, "square-1.2")
        with self.assertRaisesRegex(ValueError, "crop rule 'square-1.2'.*'box'"):
            Gallery.load_or_new(path, META, "box")
        self.assertEqual(Gallery.load_or_new(Path(directory.name) / "new.npz", META, "square-1.4").crop_rule,
                         "square-1.4")

    def test_gallery_without_a_crop_rule_is_refused(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-gallery-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "gallery.npz"
        np.savez(path, model_id=META["model_id"], model_version=META["model_version"], tau=TAU,
                 labels=np.array(["An"]), templates=unit(1, 0, 0)[None])
        with self.assertRaisesRegex(ValueError, "no crop rule"):
            Gallery.load_or_new(path, META, "box")


class CropStudyTests(unittest.TestCase):
    """One gallery per crop rule, all built from the same frames."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-study-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_study_embeddings_embed_each_rule_s_crop(self):
        image = np.zeros((200, 200, 3), dtype=np.uint8)
        face = Face((40, 70, 80, 130), None, (40.0, 70.0, 40.0, 60.0))
        shapes = {}

        def embed(face_crop):
            shapes[face_crop.shape[:2]] = True
            return unit(face_crop.shape[0], 1, 0)

        embeddings = study_embeddings(image, face, embed)
        self.assertEqual(tuple(embeddings), STUDY_RULES)
        self.assertEqual(set(shapes), {(60, 40), (60, 60), (72, 72), (84, 84)})

    def test_enrollment_fills_every_rule_or_none(self):
        galleries = load_study(self.root, META)
        self.assertEqual(sorted(galleries), sorted(STUDY_RULES))
        an = {rule: [unit(1, 0, 0)] * 5 for rule in STUDY_RULES}
        enroll_study(galleries, "An", an)
        for rule, gallery in galleries.items():
            gallery.save(self.root / f"gallery-{rule}.npz")
        # Under one rule Que looks like An, so the whole registration is refused.
        que = {rule: [unit(0, 1, 0)] * 5 for rule in STUDY_RULES} | {"box": [unit(1, 0.05, 0)] * 5}
        with self.assertRaisesRegex(ValueError, "box: This face already matches 'An'"):
            enroll_study(galleries, "Que", que)
        self.assertTrue(all(gallery.labels == ["An"] for gallery in galleries.values()))
        # The study may keep a look-alike, so the crop that confuses them can be measured.
        closest = enroll_study(galleries, "Que", que, allow_similar=True)
        self.assertEqual(closest["box"][0], "An")
        self.assertGreater(closest["box"][1], TAU)
        self.assertAlmostEqual(closest["square-1.2"][1], 0.0)
        self.assertEqual(galleries["box"].labels, ["An", "Que"])
        with self.assertRaisesRegex(ValueError, "already registered"):
            enroll_study(galleries, "que", que, allow_similar=True)
        loaded = load_study(self.root, META)
        self.assertEqual({rule: g.crop_rule for rule, g in loaded.items()}, {rule: rule for rule in STUDY_RULES})

    def test_study_galleries_with_different_people_are_refused(self):
        galleries = load_study(self.root, META)
        enroll_study(galleries, "An", {rule: [unit(1, 0, 0)] * 5 for rule in STUDY_RULES})
        galleries["box"].enroll("Que", [unit(0, 1, 0)] * 5)
        for rule, gallery in galleries.items():
            gallery.save(self.root / f"gallery-{rule}.npz")
        with self.assertRaisesRegex(ValueError, "do not hold the same people"):
            load_study(self.root, META)

    def test_trace_records_own_and_best_other_score_per_rule(self):
        galleries = {rule: gallery_with(("An", unit(1, 0, 0)), ("Que", unit(0, 1, 0)), ("Vy", unit(0, 0, 1)),
                                        rule=rule) for rule in STUDY_RULES}
        study = {rule: (galleries[rule], unit(1, 0.5, 0.2)) for rule in STUDY_RULES}
        study["box"] = (galleries["box"], 0.05 * unit(0.3, 1, 0))
        row = dict(zip(TRACE_COLUMNS, trace_row("a1", "An", 0.1, 1, gallery=galleries[SCHEMA_RULE],
                                                embedding=unit(1, 0.5, 0.2), study=study, source="image")))
        self.assertEqual(row["source"], "image")
        self.assertAlmostEqual(float(row["own:square-1.2"]), float(unit(1, 0.5, 0.2)[0]), places=5)
        self.assertAlmostEqual(float(row["other:square-1.2"]), float(unit(1, 0.5, 0.2)[1]), places=5)
        self.assertEqual(row["other_label:square-1.2"], "Que")
        self.assertGreater(float(row["other:box"]), float(row["own:box"]))
        # Someone not enrolled has no own score; the best other is the best of all.
        stranger = dict(zip(TRACE_COLUMNS, trace_row("a2", "impostor", 0.1, 1, study=study)))
        self.assertEqual(stranger["own:box"], "")
        self.assertAlmostEqual(float(stranger["other:box"]), float(unit(0.3, 1, 0)[1]), places=5)

    def test_diagnosis_reports_each_rule_per_person(self):
        galleries = {rule: gallery_with(("An", unit(1, 0, 0)), ("Que", unit(0, 1, 0)), rule=rule)
                     for rule in STUDY_RULES}
        rows = []
        # The bare box confuses An with Que in two of three frames; the square crops do not.
        for index, box_direction in enumerate((unit(1, 0.2, 0), unit(0.3, 1, 0), unit(0.4, 1, 0))):
            study = {rule: (galleries[rule], unit(1, 0.1 * index, 0)) for rule in STUDY_RULES}
            study["box"] = (galleries["box"], box_direction)
            rows.append(trace_row("a1", "An", index / 10, 1, gallery=galleries[SCHEMA_RULE],
                                  embedding=study[SCHEMA_RULE][1], study=study))
        study = {rule: (galleries[rule], unit(0.2, 1, 0)) for rule in STUDY_RULES}
        rows.append(trace_row("p1", "Que", 0, 1, gallery=galleries[SCHEMA_RULE], embedding=unit(0.2, 1, 0),
                              study=study, source="image"))
        # An's photos from another day: the box crop gets one wrong.
        for index, box_direction in enumerate((unit(1, 0.1, 0), unit(0.2, 1, 0))):
            study = {rule: (galleries[rule], unit(1, 0.3, 0)) for rule in STUDY_RULES}
            study["box"] = (galleries["box"], box_direction)
            rows.append(trace_row(f"p{index + 2}", "An", 0, 1, study=study, source="image"))
        # Lynn is not enrolled: the box crop accepts her as Que, the squares do not.
        study = {rule: (galleries[rule], unit(0.2, 0.3, 1)) for rule in STUDY_RULES}
        study["box"] = (galleries["box"], unit(0.1, 1, 0.3))
        rows.append(trace_row("p4", "Lynn", 0, 1, study=study, source="image"))
        path = self.root / "trace.csv"
        append_rows(path, TRACE_COLUMNS, rows)
        with path.open(newline="") as handle:
            trace = list(csv.DictReader(handle))
        result = diagnose(trace, [], tau=TAU)["crop_margins"]
        box, schema = result["rules"]["box"]["per_person"]["An"], result["rules"]["square-1.2"]["per_person"]["An"]
        self.assertEqual((box["frames"], box["sources"]), (5, ["camera", "image"]))
        self.assertAlmostEqual(box["correct_identity_rate"], 2 / 5)
        self.assertAlmostEqual(box["per_source"]["camera"]["correct_identity_rate"], 1 / 3)
        self.assertEqual((box["per_source"]["image"]["frames"], box["per_source"]["image"]["correct_identity_rate"]),
                         (2, 0.5))
        self.assertEqual(box["confused_with"], {"Que": 3})
        self.assertEqual(schema["correct_identity_rate"], 1.0)
        self.assertEqual(schema["accept_rate"], 1.0)
        self.assertGreater(schema["own_minus_other"]["mean"], 0.5)
        self.assertLess(box["own_minus_other"]["mean"], 0)
        self.assertEqual(set(schema["own_score"]), {"n", "mean", "median", "std", "min", "p10"})
        que = result["rules"]["box"]["per_person"]["Que"]
        self.assertEqual((que["frames"], que["sources"], que["correct_identity_rate"]), (1, ["image"], 1.0))
        self.assertEqual(result["ranking"][-1], "box")
        self.assertIn("balanced_correct_identity_rate", result["rules"]["box"])
        self.assertNotIn("Lynn", result["rules"]["box"]["per_person"])
        lynn_box, lynn_schema = (result["rules"][rule]["impostors"]["Lynn"] for rule in ("box", "square-1.2"))
        self.assertEqual((lynn_box["frames"], lynn_box["false_accept_rate"], lynn_box["accepted_as"]),
                         (1, 1.0, {"Que": 1}))
        self.assertEqual((lynn_schema["false_accept_rate"], lynn_schema["accepted_as"]), (0.0, {}))
        self.assertEqual(set(lynn_schema["best_score"]), {"n", "mean", "median"})

    def test_template_cosines_between_people_per_rule(self):
        galleries = load_study(self.root, META)
        enroll_study(galleries, "An", {rule: [unit(1, 0, 0)] * 5 for rule in STUDY_RULES})
        enroll_study(galleries, "Que", {rule: [unit(0.4, 0.8, 0)] * 5 for rule in STUDY_RULES})
        cosines = template_cosines(galleries)
        self.assertEqual(list(cosines), list(STUDY_RULES))
        self.assertAlmostEqual(cosines["square-1.2"]["An / Que"], float(unit(0.4, 0.8, 0)[0]))


class PhotoTests(unittest.TestCase):
    """Enrollment and probes from image files go through the same detection and crops."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-photos-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def test_images_are_listed_in_order_and_large_ones_are_shrunk(self):
        (self.root / "sub").mkdir()
        cv2.imwrite(str(self.root / "b.jpg"), np.full((2000, 1000, 3), 128, np.uint8))
        cv2.imwrite(str(self.root / "a.png"), np.full((300, 400, 3), 128, np.uint8))
        (self.root / "notes.txt").write_text("not an image")
        cv2.imwrite(str(self.root / "sub" / "c.JPG"), np.full((10, 10, 3), 128, np.uint8))
        paths = image_paths([self.root])
        self.assertEqual([path.name for path in paths], ["a.png", "b.jpg", "c.JPG"])
        self.assertEqual(read_image(paths[1]).shape, (1280, 640, 3))
        self.assertEqual(read_image(paths[0]).shape, (300, 400, 3))
        with self.assertRaisesRegex(ValueError, "cannot be read"):
            read_image(self.root / "notes.txt")
        with self.assertRaisesRegex(ValueError, "no image"):
            image_paths([self.root / "missing"])

    def test_photo_samples_need_one_face_of_the_model_size(self):
        for name in ("one.png", "none.png", "two.png", "small.png"):
            cv2.imwrite(str(self.root / name), np.full((200, 200, 3), 128, np.uint8))
        big = Face((20, 20, 120, 140), FRONTAL + 20, (20.0, 20.0, 100.0, 120.0))
        small = Face((20, 20, 60, 60), FRONTAL, (20.0, 20.0, 40.0, 40.0))
        found = {"one.png": [big], "none.png": [], "two.png": [big, big], "small.png": [small]}
        paths = image_paths([self.root])
        calls = iter(paths)

        def detect(frame_bgr):
            self.assertEqual(frame_bgr.shape, (200, 200, 3))
            return found[next(calls).name]

        samples = {sample.path.name: sample for sample in photo_samples(paths, detect, lambda c: unit(1, 0, 0))}
        self.assertEqual(samples["none.png"].reason, "no face")
        self.assertEqual(samples["two.png"].reason, "2 faces")
        self.assertEqual(samples["small.png"].reason, "face smaller than 64 px")
        self.assertIsNone(samples["one.png"].reason)
        self.assertEqual(tuple(samples["one.png"].embeddings), STUDY_RULES)
        self.assertEqual(samples["one.png"].quality.size, 100)


class CheckInTests(unittest.TestCase):
    def setUp(self):
        self.gallery = gallery_with(("An", unit(1, 0, 0)), ("Vy", unit(0, 1, 0)))
        self.calls = 0

    def embed_as(self, direction):
        def embed(face_crop):
            self.calls += 1
            return direction
        return embed

    def test_single_mode_decides_on_the_first_face_frame(self):
        check_in = CheckIn("single", self.gallery, self.embed_as(unit(1, 0.2, 0)), margin=0.0)
        self.assertEqual(check_in.step(frame(), [FACE], 0.0).prompt, "Press SPACE to check in")
        check_in.start(1.0)
        self.assertEqual(check_in.step(frame(), [], 1.1).prompt, "No face found")
        display = check_in.step(frame(blur=True), [FACE], 1.3)
        self.assertEqual((display.decision.label, display.decision.frames_used, display.decision.end), ("An", 1, "match"))
        self.assertAlmostEqual(display.decision.elapsed_s, 0.3)
        self.assertFalse(check_in.active)

    def test_fused_mode_skips_bad_frames_without_embedding_them(self):
        check_in = CheckIn("fused", self.gallery, self.embed_as(unit(1, 0.2, 0)), margin=0.0, settings=SETTINGS)
        check_in.start(0.0)
        self.assertEqual(check_in.step(frame(blur=True), [FACE], 0.1).prompt, "Hold still")
        self.assertEqual(self.calls, 0)
        self.assertEqual(check_in.step(frame(), [FACE], 0.2).kept, 1)
        self.assertEqual(check_in.step(frame(), [FACE, FACE], 0.3).prompt, "One person at a time")
        check_in.step(frame(), [FACE], 0.4)
        check_in.step(frame(), [FACE], 0.5)
        display = check_in.step(frame(), [FACE], 0.6)
        self.assertEqual((display.decision.label, display.decision.frames_used), ("An", 3))
        self.assertEqual(self.calls, 4)

    def test_embedding_uses_the_gallery_crop_and_the_gate_measures_the_box(self):
        gallery = gallery_with(("An", unit(1, 0, 0)), rule="square-1.2")
        shapes = []

        def embed(face_crop):
            shapes.append(face_crop.shape[:2])
            return unit(1, 0, 0)

        image = np.zeros((300, 300, 3), dtype=np.uint8)
        image[60:220, 60:220] = frame()
        face = Face((80, 80, 200, 200), FRONTAL + 80, (80.0, 80.0, 120.0, 120.0))
        single = CheckIn("single", gallery, embed, margin=0.0)
        single.start(0.0)
        self.assertEqual(single.step(image, [face], 0.1).decision.label, "An")
        self.assertEqual(shapes, [(144, 144)])
        fused = CheckIn("fused", gallery, embed, margin=0.0, settings=SETTINGS)
        fused.start(0.0)
        self.assertEqual(fused.step(image, [face], 0.1).kept, 1)
        self.assertEqual(shapes[-1], (144, 144))

    def test_turned_head_is_refused_and_named(self):
        check_in = CheckIn("fused", self.gallery, self.embed_as(unit(1, 0, 0)), margin=0.0, settings=SETTINGS)
        check_in.start(0.0)
        turned = Face(FACE.box, FACE.landmarks + np.array([[0, 0], [0, 0], [20, 0], [0, 0], [0, 0]]))
        self.assertEqual(check_in.step(frame(), [turned], 0.1).prompt, "Face the camera")

    def test_unknown_person_is_not_recognised(self):
        check_in = CheckIn("fused", self.gallery, self.embed_as(unit(0, 0, 1)), margin=0.0, settings=SETTINGS)
        check_in.start(0.0)
        for moment in (0.1, 0.2):
            check_in.step(frame(), [FACE], moment)
        decision = check_in.step(frame(), [FACE], 0.3).decision
        self.assertEqual((decision.label, decision.accepted, decision.end), (None, False, "unknown"))

    def test_retry_continues_until_the_attempt_times_out(self):
        check_in = CheckIn("fused", self.gallery, self.embed_as(unit(1, 0, 0)), margin=0.0, settings=SETTINGS,
                           timeout_s=5.0)
        check_in.start(0.0)
        check_in.step(frame(blur=True), [FACE], 0.0)
        self.assertEqual(check_in.step(frame(blur=True), [FACE], 2.0).prompt, "Hold still")
        self.assertTrue(check_in.active)
        decision = check_in.step(frame(), [FACE], 5.0).decision
        self.assertEqual((decision.end, decision.retries), ("timeout", 1))

    def decide_once(self, mode, direction, margin, gallery=None):
        check_in = CheckIn(mode, gallery or self.gallery, self.embed_as(direction), margin=margin,
                           settings=SETTINGS)
        check_in.start(0.0)
        for moment in (0.1, 0.2, 0.3):
            display = check_in.step(frame(), [FACE], moment)
            if display.decision:
                return display.decision
        raise AssertionError("no decision")

    def test_a_clear_winner_is_named_and_the_runner_up_kept(self):
        for mode in ("single", "fused"):
            decision = self.decide_once(mode, unit(1, 0.5, 0), 0.15)
            self.assertEqual((decision.label, decision.accepted, decision.best_label, decision.margin),
                             ("An", True, "An", 0.15))
            self.assertAlmostEqual(decision.score, float(unit(1, 0.5, 0)[0]))
            self.assertAlmostEqual(decision.runner_up, float(unit(1, 0.5, 0)[1]))

    def test_a_close_runner_up_makes_the_face_unknown_in_both_modes(self):
        # An 0.781, Vy 0.625: above tau, but only 0.156 apart.
        probe = unit(1, 0.8, 0)
        for mode in ("single", "fused"):
            self.assertTrue(self.decide_once(mode, probe, 0.15).accepted)
            decision = self.decide_once(mode, probe, 0.20)
            self.assertEqual((decision.label, decision.accepted, decision.end, decision.best_label),
                             (None, False, "unknown", "An"))

    def test_one_enrolled_person_needs_tau_plus_the_margin(self):
        alone = gallery_with(("An", unit(1, 0, 0)))
        probe = unit(1, 1.5, 0)  # cosine 0.555: above tau 0.462, below tau + 0.15
        self.assertTrue(self.decide_once("single", probe, 0.0, alone).accepted)
        decision = self.decide_once("single", probe, 0.15, alone)
        self.assertEqual((decision.accepted, decision.best_label, decision.runner_up), (False, "An", None))

    def test_a_negative_margin_is_refused(self):
        with self.assertRaises(ValueError):
            CheckIn("single", self.gallery, self.embed_as(unit(1, 0, 0)), margin=-0.1)

    def test_fused_mode_needs_limits(self):
        with self.assertRaises(ValueError):
            CheckIn("fused", self.gallery, self.embed_as(unit(1, 0, 0)), margin=0.0,
                    settings=CheckInSettings(False, 3, 2, 2.0, None))


class LogTests(unittest.TestCase):
    def test_log_and_summary(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-log-")
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "attempts.csv"
        gallery = gallery_with(("An", unit(1, 0, 0)))
        for mode, person, direction in (("single", "An", unit(1, 0, 0)), ("single", "An", unit(0, 1, 0)),
                                        ("single", "impostor", unit(1, 0.1, 0)), ("fused", "An", unit(1, 0, 0))):
            check_in = CheckIn("single", gallery, lambda _crop, d=direction: d, margin=0.0)
            check_in.start(0.0)
            decision = check_in.step(frame(), [FACE], 0.5).decision
            append_log(path, person, decision.__class__(**{**decision.__dict__, "mode": mode}))
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 4)
        # One registered person: there is no runner-up to log.
        self.assertEqual({row["runner_up"] for row in rows}, {""})
        self.assertEqual([row["best_label"] for row in rows], ["An"] * 4)
        self.assertEqual({row["margin"] for row in rows}, {"0"})
        summary = summarize(rows)
        self.assertEqual(summary["single"]["genuine_accept_rate"], 0.5)
        self.assertEqual(summary["single"]["impostor_accept_rate"], 1.0)
        self.assertEqual(summary["single"]["decision_ms_median"], 500.0)
        self.assertEqual(summary["fused"]["people"], ["An"])
        self.assertIsNone(summary["fused"]["impostor_accept_rate"])


class PairedTraceTests(unittest.TestCase):
    """A paired run: both modes decide on the same frames, and every frame is traced."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix="check-in-trace-")
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.gallery = gallery_with(("An", unit(1, 0, 0)), ("Vy", unit(0, 1, 0)))

    def quality(self, sharpness):
        return FaceQuality(sharpness=sharpness, brightness=128.0, contrast=50.0, size=120, yaw=0.0, roll=0.0)

    def test_trace_row_scores_raw_embeddings_as_cosines(self):
        row = dict(zip(TRACE_COLUMNS, trace_row("a1", "An", 0.25, 1, self.quality(500.0), ("blur",),
                                                self.gallery, 0.08 * unit(1, 0.2, 0))))
        self.assertEqual((row["attempt"], row["t_ms"], row["reasons"], row["best_label"]), ("a1", 250, "blur", "An"))
        self.assertAlmostEqual(float(row["own_score"]), float(unit(1, 0.2, 0) @ unit(1, 0, 0)), places=5)
        empty = dict(zip(TRACE_COLUMNS, trace_row("a1", "impostor", 0.3, 0)))
        self.assertEqual((empty["faces"], empty["own_score"], empty["sharpness"]), (0, "", ""))

    def test_paired_log_needs_its_own_file(self):
        decision = CheckIn("single", self.gallery, lambda _crop: unit(1, 0, 0), margin=0.0)
        decision.start(0.0)
        result = decision.step(frame(), [FACE], 0.1).decision
        plain, paired = self.root / "plain.csv", self.root / "paired.csv"
        append_log(plain, "An", result)
        append_log(paired, "An", result, attempt="a1")
        with self.assertRaisesRegex(ValueError, "other columns"):
            append_log(plain, "An", result, attempt="a2")
        with paired.open(newline="") as handle:
            self.assertEqual(next(csv.DictReader(handle))["attempt"], "a1")

    def test_diagnosis_compares_kept_and_rejected_frames_and_pairs_decisions(self):
        rows = []
        # Gate-passing frames score well; blurred frames score badly, so the gate helps.
        for index, (sharpness, reasons, direction) in enumerate(
                [(900.0, (), unit(1, 0.1, 0)), (950.0, (), unit(1, 0.05, 0)), (50.0, ("blur",), unit(1, 1, 0)),
                 (40.0, ("blur",), unit(1, 1.2, 0))]):
            rows.append(trace_row("a1", "An", index / 10, 1, self.quality(sharpness), reasons, self.gallery, direction))
        rows.append(trace_row("a1", "An", 0.5, 0))
        trace_file = self.root / "trace.csv"
        append_rows(trace_file, TRACE_COLUMNS, rows)
        log_file = self.root / "paired.csv"
        for attempt, single, fused in (("a1", unit(1, 0, 0), unit(1, 0.1, 0)), ("a2", unit(0, 1, 0), unit(1, 0, 0))):
            for mode, direction in (("single", single), ("fused", fused)):
                check_in = CheckIn("single", self.gallery, lambda _crop, d=direction: d, margin=0.0)
                check_in.start(0.0)
                decision = check_in.step(frame(), [FACE], 0.2).decision
                append_log(log_file, "An", decision.__class__(**{**decision.__dict__, "mode": mode}), attempt)
        with trace_file.open(newline="") as handle:
            trace = list(csv.DictReader(handle))
        with log_file.open(newline="") as handle:
            log = list(csv.DictReader(handle))
        result = diagnose(trace, log)
        self.assertEqual(result["face_frames"], 4)
        self.assertGreater(result["own_score_gate_passed"]["mean"], result["own_score_gate_failed"]["mean"])
        self.assertEqual(result["own_score_by_reason"]["blur"]["n"], 2)
        self.assertGreater(result["spearman_measure_vs_own_score"]["sharpness"], 0.9)
        self.assertEqual(result["per_person"]["An"]["gate_pass_rate"], 0.5)
        self.assertEqual(result["per_person"]["An"]["failed_by"], {"blur": 2})
        self.assertEqual(result["paired_genuine_outcomes"],
                         {"both_correct": 1, "only_single": 0, "only_fused": 1, "neither": 0})


class GateLimitTests(unittest.TestCase):
    PERCENTILES = {"sharpness": {"p5": 700.0, "p95": 4000.0}, "brightness": {"p5": 60.0, "p95": 160.0},
                   "contrast": {"p5": 40.0, "p95": 80.0}, "size": {"p5": 64.0, "p95": 64.0}}

    def row(self, sharpness=1500.0, brightness=125.0, contrast=50.0, size=300, yaw=0.0, roll=0.0, **extra):
        return {"sharpness": str(sharpness), "brightness": str(brightness), "contrast": str(contrast),
                "size": str(size), "yaw": str(yaw), "roll": str(roll)} | extra

    def test_limits_follow_the_dataset_range_and_fixed_pose(self):
        limits = dataset_limits(self.PERCENTILES, min_size=64)
        self.assertEqual(limits, {"min_sharpness": 700.0, "min_brightness": 60.0, "max_brightness": 160.0,
                                  "min_contrast": 40.0, "min_size": 64, "max_yaw": 0.35, "max_roll": 15.0})
        QualityLimits.from_mapping(limits)
        with self.assertRaises(ValueError):
            dataset_limits({"sharpness": {"p5": 1.0}}, min_size=64)

    def test_a_lighting_change_inside_the_dataset_range_passes(self):
        # The old webcam-percentile limits refused brightness above 122; the dataset range does not.
        limits = dataset_limits(self.PERCENTILES, min_size=64)
        good = [self.row(brightness=value) for value in (110, 125, 139)]
        bad = [self.row(sharpness=500), self.row(brightness=40), self.row(yaw=0.5), self.row(size=40)]
        result = check_limits(limits, good, bad)
        self.assertEqual(result["good"]["pass_rate"], 1.0)
        self.assertEqual(result["bad"]["pass_rate"], 0.0)
        self.assertEqual({name: count for name, count in result["bad"]["rejected_by"].items() if count},
                         {"blur": 1, "dark": 1, "turned": 1, "small": 1})

    def test_toml_table_round_trips_and_replaces_the_old_table(self):
        import tomllib

        base = "[check_in]\nenabled = false\nk = 3\nmin_frames = 2\nwindow_s = 2.0\n"
        old = with_limits(base, limits_toml(dataset_limits(self.PERCENTILES, min_size=64), "first"))
        wider = dict(self.PERCENTILES, brightness={"p5": 50.0, "p95": 170.0})
        text = with_limits(old, limits_toml(dataset_limits(wider, min_size=64), ["second", "two lines"]))
        self.assertEqual(text.count("[check_in.limits]"), 1)
        self.assertNotIn("# first", text)
        table = tomllib.loads(text)["check_in"]["limits"]
        self.assertEqual(QualityLimits.from_mapping(table).max_brightness, 170.0)


class GateReplayTests(unittest.TestCase):
    def frame(self, attempt, t_ms, reasons="", brightness=125.0, faces="1", own="0.8"):
        if faces != "1":
            return {"attempt": attempt, "source": "camera", "t_ms": str(t_ms), "faces": faces, "sharpness": "",
                    "reasons": "", "own_score": ""}
        return {"attempt": attempt, "source": "camera", "t_ms": str(t_ms), "faces": "1", "size": "300",
                "sharpness": "1500", "brightness": str(brightness), "contrast": "50", "yaw": "0.0", "roll": "0.0",
                "reasons": reasons, "own_score": own}

    def test_frames_refused_live_for_glare_now_reach_a_decision(self):
        limits = QualityLimits.from_mapping(dataset_limits(GateLimitTests.PERCENTILES, min_size=64))
        trace = [self.frame("a", 100 * step, reasons="bright") for step in range(1, 30)]
        trace += [self.frame("b", 100, faces="0"), self.frame("b", 200), self.frame("b", 300),
                  self.frame("b", 400, brightness=200.0, own="0.3"), self.frame("b", 500)]
        trace.append({"attempt": "c", "source": "image", "t_ms": "0", "faces": "1"})
        result = replay_gate(trace, limits, k=3, min_frames=2, window_s=2.0)
        self.assertEqual(result["face_frames"], 33)
        # Live, attempt a never got past the glare check; b had no recorded problems.
        self.assertEqual(result["recorded_gate"]["attempts"]["decided"], 1)
        self.assertEqual(result["recorded_gate"]["attempts"]["attempts"], 2)
        self.assertEqual(result["new_gate"]["attempts"]["decided"], 2)
        self.assertEqual(result["new_gate"]["frames"]["rejected_by"]["bright"], 1)
        self.assertEqual(result["new_gate"]["frames"]["own_score_rejected"]["n"], 1)
        self.assertAlmostEqual(result["new_gate"]["frames"]["own_score_rejected"]["mean"], 0.3)
        self.assertEqual(result["new_gate"]["attempts"]["mean_frames_used"], 3.0)


if __name__ == "__main__":
    unittest.main()
