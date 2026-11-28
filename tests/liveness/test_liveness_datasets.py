"""Behavior contracts for liveness datasets."""

import importlib.util
import io
import json
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from facial_attendance.liveness.audit.reproduction_protocol import (
    inspect_public_protocol,
)
from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.datasets.face_crop import read_face
from facial_attendance.liveness.datasets.reviewed_cache import (
    ReviewedCacheView,
    TransportFaces,
    row_order_sha256,
)
from facial_attendance.liveness.datasets.source_dataset import read_reproduction_face

RUNTIME_0 = all(
    importlib.util.find_spec(package) for package in ("torch", "torchvision")
)


if RUNTIME_0:
    import torch
    from PIL import Image

    from facial_attendance.liveness.dataset_files import file_sha256
    from facial_attendance.liveness.datasets.semantic_dataset import (
        balanced_cohort,
        read_annotation_targets,
    )
    from facial_attendance.liveness.models.binary_model import (
        RESNET_ARCHITECTURE,
        face_transform,
        normalization,
    )
    from facial_attendance.liveness.models.multitask_model import SOURCE_TYPES


@unittest.skipUnless(RUNTIME_0, "Requires the existing project runtime")
class MultitaskDatasetTests(unittest.TestCase):
    def test_bound_annotations_match_inventory_and_reject_tampering(self):
        attributes = [0] * 44
        row = SimpleNamespace(relative_path="live.jpg", binary_label=0, source_type=0)
        dataset = SimpleNamespace(samples=[row])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "annotations.json"
            path.write_text(json.dumps({"live.jpg": attributes}), encoding="utf-8")
            receipt = Path(directory) / "readiness.json"
            receipt.write_text(
                json.dumps({"source_bindings": {str(path): file_sha256(path)}}),
                encoding="utf-8",
            )
            target = read_annotation_targets(dataset, path, receipt)[0]
            self.assertEqual(target["binary"], 0)
            row.binary_label = 1
            with self.assertRaisesRegex(ValueError, "mismatch"):
                read_annotation_targets(dataset, path, receipt)
            row.binary_label = 0
            path.write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "not bound"):
                read_annotation_targets(dataset, path, receipt)

    def test_cohort_balances_attack_and_illumination_without_replacement(self):
        rows, targets = [], []
        for source, illumination, count in [(0, 0, 30)] + [
            (s, i, 3) for s in SOURCE_TYPES[1:] for i in range(1, 5)
        ]:
            for _ in range(count):
                rows.append(SimpleNamespace(source_type=source))
                targets.append({"illumination": illumination})

        class Fixture:
            samples = rows

            def __len__(self):
                return len(self.samples)

        dataset = Fixture()
        first = balanced_cohort(dataset, targets, 48, 42)
        self.assertEqual(first, balanced_cohort(dataset, targets, 48, 42))
        self.assertNotEqual(first, balanced_cohort(dataset, targets, 48, 43))
        self.assertEqual(len(set(first)), 48)
        self.assertEqual(sum(rows[i].source_type == 0 for i in first), 24)
        self.assertEqual(
            len({(rows[i].source_type, targets[i]["illumination"]) for i in first}), 25
        )
        with self.assertRaises(ValueError):
            balanced_cohort(dataset, targets, 96, 42)

    def test_checkpoint_specific_normalization(self):
        image = Image.new("RGB", (50, 60), (255, 0, 128))
        tensor = face_transform(architecture=RESNET_ARCHITECTURE)(image)
        mean, std = normalization(RESNET_ARCHITECTURE)
        expected = (
            torch.tensor([1.0, 0.0, 128 / 255]) - torch.tensor(mean)
        ) / torch.tensor(std)
        self.assertTrue(torch.allclose(tensor[:, 0, 0], expected))
        legacy = face_transform()(image)
        self.assertTrue(
            torch.allclose(
                legacy[:, 0, 0], torch.tensor([1.0, -1.0, 128 / 255 * 2 - 1]), atol=1e-7
            )
        )


class ReproductionCropTests(unittest.TestCase):
    def test_source_integer_conversion_and_scaling_order(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "face.png"
            box = image.with_name("face_BB.txt")
            Image.new("RGB", (448, 600)).save(image)
            box.write_text("78 52 78 84 0.99")
            self.assertEqual(read_reproduction_face(image, box).size, (156, 224))
            box.write_text("1.9 2.9 3.9 4.9 0.99")
            self.assertEqual(read_reproduction_face(image, box).size, (6, 10))
            box.write_text("78  52 78 84 0.99")
            self.assertEqual(read_reproduction_face(image, box).size, (448, 600))
            # The author tests the clipped origin plus height before choosing
            # the lower edge. Negative origins can therefore change much more
            # than a single pixel; retain this distinct clipping branch.
            box.write_text("12 -37 199 241 0.99")
            self.assertEqual(read_reproduction_face(image, box).size, (398, 600))

    def test_empty_source_box_retains_image_only_in_reproduction(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "face.png"
            box = image.with_name("face_BB.txt")
            Image.new("RGB", (224, 224), (20, 40, 60)).save(image)
            box.write_text("")
            self.assertEqual(read_reproduction_face(image, box).size, (224, 224))
            with self.assertRaisesRegex(ValueError, "Expected x y"):
                read_face(image)
            box.write_text("10 20 80 90 0.99")
            self.assertEqual(
                read_reproduction_face(image, box).tobytes(), read_face(image).tobytes()
            )
            self.assertEqual(read_reproduction_face(image, box).size, (80, 90))
            box.write_text("-100 20 150 90 0.99")
            with self.assertRaisesRegex(ValueError, "loses over half"):
                read_face(image)
            self.assertEqual(read_reproduction_face(image, box).size, (50, 90))

    def test_invalid_geometry_and_missing_file_still_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "face.png"
            box = image.with_name("face_BB.txt")
            Image.new("RGB", (224, 224)).save(image)
            box.write_text("0 0 -10 80 0.99")
            with self.assertRaises(ValueError):
                read_reproduction_face(image, box)
            with self.assertRaises(FileNotFoundError):
                read_reproduction_face(image, box.with_name("missing.txt"))


class Source:
    def __init__(self):
        self.samples = [
            SimpleNamespace(
                relative_path=f"face-{i}.png", binary_label=i % 2, source_type=i % 2
            )
            for i in range(4)
        ]
        self.targets = [{"binary": i % 2} for i in range(4)]

    def __len__(self):
        return len(self.samples)

    def bindings(self):
        return {
            "inventory_sha256": "inventory",
            "annotations_sha256": "annotations",
            "samples": 4,
        }


def fixture(root, pending=0):
    source = Source()
    cache = root / "transport"
    view = root / "view"
    cache.mkdir()
    view.mkdir()
    shards = []
    for first, stop in ((0, 2), (2, 4)):
        path = cache / f"{first}.tar"
        with tarfile.open(path, "w", format=tarfile.USTAR_FORMAT) as archive:
            for index in range(first, stop):
                buffer = io.BytesIO()
                Image.new("RGB", (224, 224), (index, 20, 40)).save(buffer, format="PNG")
                member = tarfile.TarInfo(f"{index:09d}.png")
                member.size = buffer.tell()
                buffer.seek(0)
                archive.addfile(member, buffer)
        shards.append(
            {
                "file": path.name,
                "first_row": first,
                "stop_row": stop,
                "bytes": path.stat().st_size,
                "sha256": file_sha256(path),
            }
        )
    (cache / "face-cache-shards.json").write_text(
        json.dumps(
            {
                "schema": "liveness-face-cache-transport-v1",
                "data": source.bindings(),
                "shape": [4, 224, 224, 3],
                "dtype": "uint8",
                "shards": shards,
            }
        )
    )
    np.save(view / "keep.npy", np.array([0, 2, 3], dtype=np.int64))
    (view / "cache-view.json").write_text(
        json.dumps(
            {
                "schema": "liveness-reviewed-cache-view-v1",
                "inventory_sha256": "inventory",
                "splits": {
                    "train": {
                        "annotations_sha256": "annotations",
                        "base_samples": 4,
                        "base_row_order_sha256": row_order_sha256(source.samples),
                        "retained_samples": 3,
                        "pending_review": pending,
                        "keep_indices": "keep.npy",
                        "keep_indices_sha256": file_sha256(view / "keep.npy"),
                    }
                },
            }
        )
    )
    return source, cache, view


class ReviewedCacheTests(unittest.TestCase):
    def test_training_adapter_uses_filtered_image_and_geometric_targets(self):
        from facial_attendance.liveness.datasets.source_dataset import (
            GeometricTrainingImages,
        )

        with tempfile.TemporaryDirectory() as temporary:
            source, cache, directory = fixture(Path(temporary))
            view = ReviewedCacheView(TransportFaces(source, cache), directory, "train")
            maps = np.arange(4 * 4 * 14 * 14).reshape(4, 4, 14, 14)
            dataset = GeometricTrainingImages(
                view, view.select_maps(maps), lambda image, target: (image, target)
            )
            for filtered, original in enumerate((0, 2, 3)):
                image, target = dataset[filtered]
                self.assertEqual(image.getpixel((0, 0))[0], original)
                self.assertEqual(target["binary"], original % 2)
                np.testing.assert_array_equal(
                    target["depth_map"].numpy(), maps[original, :1]
                )
                np.testing.assert_array_equal(
                    target["reflection_map"].numpy(), maps[original, 1:]
                )

    def test_image_label_and_teacher_rows_use_the_same_original_indices(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, cache, directory = fixture(root)
            original_bytes = {path: path.read_bytes() for path in cache.iterdir()}
            base = TransportFaces(source, cache)
            view = ReviewedCacheView(base, directory, "train")
            maps = np.arange(4 * 4 * 14 * 14).reshape(4, 4, 14, 14)
            selected_maps = view.select_maps(maps)
            for filtered, original in enumerate((0, 2, 3)):
                image, target = view[filtered]
                self.assertEqual(image.mode, "RGB")
                self.assertEqual(np.asarray(image).dtype, np.uint8)
                self.assertEqual(image.getpixel((0, 0)), (original, 20, 40))
                self.assertEqual(target["binary"], original % 2)
                np.testing.assert_array_equal(selected_maps[filtered], maps[original])
                self.assertEqual(
                    view.samples[filtered].relative_path,
                    source.samples[original].relative_path,
                )
            self.assertEqual(selected_maps.shape, (3, 4, 14, 14))
            self.assertEqual(
                {p: p.read_bytes() for p in cache.iterdir()}, original_bytes
            )
            with self.assertRaisesRegex(ValueError, "Teacher maps"):
                view.select_maps(maps[:2])

    def test_pending_review_requires_explicit_diagnostic_access(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, cache, directory = fixture(Path(temporary), pending=1)
            base = TransportFaces(source, cache)
            with self.assertRaisesRegex(ValueError, "Unresolved"):
                ReviewedCacheView(base, directory, "train")
            self.assertEqual(
                len(
                    ReviewedCacheView(
                        base, directory, "train", allow_pending_review=True
                    )
                ),
                3,
            )

    def test_wrong_source_order_and_changed_cache_bytes_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, cache, directory = fixture(Path(temporary))
            base = TransportFaces(source, cache)
            source.samples.reverse()
            with self.assertRaisesRegex(ValueError, "row order"):
                ReviewedCacheView(base, directory, "train")
            source.samples.reverse()
            image, _ = base[0]
            self.assertEqual(image.getpixel((0, 0)), (0, 20, 40))
            with (cache / "0.tar").open("r+b") as stream:
                stream.seek(600)
                stream.write(b"x")
            with self.assertRaisesRegex(ValueError, "checksum"):
                base[0]

    def test_notebook_filters_every_model_before_validation_plots(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, _, directory = fixture(Path(temporary))
            path = directory / "cache-view.json"
            config = json.loads(path.read_text())
            config["splits"]["validation"] = config["splits"]["train"]
            path.write_text(json.dumps(config))
            rows = [vars(row) for row in source.samples]
            model = {
                "rows": rows,
                "labels": np.array([0, 1, 0, 1]),
                "scores": np.array([0.1, 0.9, 0.2, 0.8]),
                "correct": np.array([True, True, False, True]),
                "receipt": {
                    "inference_metadata": {
                        "data": {
                            "inventory_sha256": "inventory",
                            "validation": {"annotations_sha256": "annotations"},
                        }
                    }
                },
            }
            notebook = json.loads(
                (
                    Path(__file__).resolve().parents[2]
                    / "notebooks/liveness-training-and-evaluation.ipynb"
                ).read_text()
            )
            cell = next(
                c
                for c in notebook["cells"]
                if c.get("id") == "active-validation-selection"
            )
            namespace = {
                "IS_COLAB": False,
                "REVIEWED_VIEW": directory,
                "visual_models": {"binary-classification": model, "aenet-full": model},
            }
            exec("".join(cell["source"]), namespace)
            for selected in namespace["visual_models"].values():
                self.assertEqual(
                    [r["relative_path"] for r in selected["rows"]],
                    ["face-0.png", "face-2.png", "face-3.png"],
                )
                for field in ("labels", "scores", "correct"):
                    np.testing.assert_array_equal(
                        selected[field], model[field][[0, 2, 3]]
                    )
            self.assertEqual(len(model["rows"]), 4)
            model["receipt"]["inference_metadata"]["data"]["inventory_sha256"] = "wrong"
            namespace["visual_models"] = {"binary-classification": model}
            with self.assertRaises(ValueError):
                exec("".join(cell["source"]), namespace)

    def test_test_predictions_use_test_indices_and_reject_changed_labels(self):
        from facial_attendance.liveness.datasets.reviewed_cache import (
            reviewed_prediction_indices,
        )

        with tempfile.TemporaryDirectory() as temporary:
            source, _, directory = fixture(Path(temporary))
            np.save(directory / "test-keep.npy", np.array([1, 3], dtype=np.int64))
            path = directory / "cache-view.json"
            config = json.loads(path.read_text())
            config["splits"]["test"] = {
                **config["splits"]["train"],
                "keep_indices": "test-keep.npy",
                "keep_indices_sha256": file_sha256(directory / "test-keep.npy"),
                "retained_samples": 2,
            }
            path.write_text(json.dumps(config))
            data = {
                "inventory_sha256": "inventory",
                "test": {"annotations_sha256": "annotations"},
            }
            rows = [vars(row).copy() for row in source.samples]
            indices = reviewed_prediction_indices(rows, data, directory, "test")
            np.testing.assert_array_equal(indices, [1, 3])
            rows[0]["binary_label"] = 1
            with self.assertRaisesRegex(ValueError, "row order"):
                reviewed_prediction_indices(rows, data, directory, "test")

    def test_changed_selection_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            source, cache, directory = fixture(Path(temporary))
            np.save(directory / "keep.npy", np.array([0, 0, 3], dtype=np.int64))
            with self.assertRaisesRegex(ValueError, "changed"):
                ReviewedCacheView(TransportFaces(source, cache), directory, "train")


class ReproductionProtocolTests(unittest.TestCase):
    def fixture(self, root, split, subject, source_type=0, directory=None):
        name = f"Data/{split}/{subject}/{directory or ('live' if source_type == 0 else 'spoof')}/000001.jpg"
        image = root / name
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"File presence fixture; no image decoding claim")
        attributes = [0] * 44
        attributes[40], attributes[41], attributes[43] = (
            source_type,
            int(source_type != 0),
            int(source_type != 0),
        )
        labels = root / f"metas/intra_test/{split}_label.json"
        labels.parent.mkdir(parents=True, exist_ok=True)
        labels.write_text(json.dumps({name: attributes}), encoding="utf-8")
        return name

    def test_report_keeps_source_overlap_and_missing_validation_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root, "train", "7")
            self.fixture(root, "test", "7", source_type=10)
            report = inspect_public_protocol(root)
            self.assertEqual(report["public_train_test_subject_overlap"], ["7"])
            self.assertEqual(report["public_validation_metadata_files"], [])
            self.assertEqual(report["public_annotated_images"], 2)
            self.assertEqual(report["recipe"]["epochs"], 50)
            self.assertIn(
                "training normalization",
                report["recipe"]["unspecified_training_details"],
            )
            self.assertNotIn("ready", report)
            project = Path(__file__).resolve().parents[2]
            completed = subprocess.run(
                [
                    sys.executable,
                    str(project / "scripts/run_liveness_notebook.py"),
                    "--action",
                    "inspect-protocol",
                    "--data-root",
                    str(root),
                ],
                cwd=project,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(
                json.loads(completed.stdout), json.loads(json.dumps(report))
            )

    def test_public_path_label_conflict_is_reported_without_relabelling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = self.fixture(root, "train", "1", source_type=7, directory="live")
            self.fixture(root, "test", "2")
            report = inspect_public_protocol(root)
            self.assertEqual(
                report["public_metadata"]["train"]["path_label_conflicts"], [name]
            )
            self.assertEqual(
                report["public_metadata"]["train"]["binary_label_counts"], {1: 1}
            )

    def test_missing_or_unsafe_source_path_does_not_pass_inspection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            name = self.fixture(root, "train", "1")
            self.fixture(root, "test", "2")
            (root / name).unlink()
            with self.assertRaisesRegex(ValueError, "Missing file"):
                inspect_public_protocol(root)


if __name__ == "__main__":
    unittest.main()
