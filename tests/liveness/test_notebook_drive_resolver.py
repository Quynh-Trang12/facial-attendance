"""Shared project discovery and Commenter output routing without a Colab VM."""

import ast
import hashlib
import json
import re
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import urlparse

NOTEBOOKS = Path(__file__).resolve().parents[2] / "notebooks"
TRAINING = NOTEBOOKS / "liveness-training-and-evaluation.ipynb"
AUDIT = NOTEBOOKS / "liveness-data-preparation-and-audit.ipynb"
FOLDER = "application/vnd.google-apps.folder"
SHORTCUT = "application/vnd.google-apps.shortcut"


def notebook_source(path, key):
    cells = json.loads(path.read_text())["cells"]
    return "".join(next(c for c in cells if c.get("id") == key)["source"])


class HttpError(Exception):
    def __init__(self, status, reason="insufficientFilePermissions"):
        self.resp = SimpleNamespace(status=status)
        self.content = json.dumps({"error": {"errors": [{"reason": reason}]}}).encode()


def bootstrap_scope(path=TRAINING):
    cells = json.loads(path.read_text())["cells"]
    definitions = []
    for cell in cells:
        if cell["cell_type"] == "code" and cell.get("id", "").startswith("drive-"):
            definitions.extend(
                node
                for node in ast.parse("".join(cell["source"])).body
                if isinstance(node, ast.FunctionDef)
            )
    scope = {
        "HttpError": HttpError,
        "time": time,
        "Path": Path,
        "urlparse": urlparse,
        "json": json,
        "hashlib": hashlib,
    }
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), "bootstrap", "exec"),
        scope,
    )
    return scope


def folder(identifier, name, parent=None, writable=True):
    return {
        "id": identifier,
        "name": name,
        "mimeType": FOLDER,
        "parents": [parent] if parent else [],
        "capabilities": {"canAddChildren": writable},
    }


class API:
    def __init__(self, nodes, *, page_size=100):
        self.nodes = {"mine": folder("mine", "My Drive"), **nodes}
        self.created, self.events = [], []
        self.page_size = page_size

    def files(self):
        return self

    def get(self, fileId, **kwargs):
        def execute():
            item = self.nodes.get(
                "mine" if fileId == "root" else fileId, HttpError(404)
            )
            if isinstance(item, Exception):
                raise item
            return dict(item)

        return SimpleNamespace(execute=execute)

    def list(self, q, pageToken=None, **kwargs):
        def execute():
            items = [x for x in self.nodes.values() if isinstance(x, dict)]
            parent = re.search(r"'([^']+)' in parents", q)
            if parent:
                items = [x for x in items if parent[1] in x.get("parents", [])]
            name = re.search(r"name='((?:\\.|[^'])*)'", q)
            if name:
                value = name[1].replace("\\'", "'").replace("\\\\", "\\")
                items = [x for x in items if x["name"] == value]
            if "mimeType='" + SHORTCUT + "'" in q:
                items = [x for x in items if x["mimeType"] == SHORTCUT]
            start = int(pageToken or 0)
            page = {"files": items[start : start + self.page_size]}
            if start + self.page_size < len(items):
                page["nextPageToken"] = str(start + self.page_size)
            return page

        return SimpleNamespace(execute=execute)

    def create(self, body, **kwargs):
        def execute():
            item = {
                "id": "new" + str(len(self.created)),
                **body,
                "capabilities": {"canAddChildren": True},
            }
            self.nodes[item["id"]] = item
            self.created.append(body)
            self.events.append("create")
            return item

        return SimpleNamespace(execute=execute)


class ResolverTests(unittest.TestCase):
    def test_runtime_bootstrap_uses_local_python_or_loaded_colab(self):
        display = ModuleType("IPython.display")
        display.HTML, display.display = Mock(), Mock()
        for path in (TRAINING, AUDIT):
            source = notebook_source(path, "native-runtime-route")
            for directory in (NOTEBOOKS.parent, NOTEBOOKS):
                scope = {"sys": sys, "Path": Path, "LOCAL_PROJECT": directory}
                with (
                    patch.object(sys, "path", list(sys.path)),
                    patch.dict(sys.modules, {"google.colab": None}),
                ):
                    exec(compile(source, "runtime", "exec"), scope)
                self.assertFalse(scope["IS_COLAB"])
                self.assertEqual(scope["LOCAL_PROJECT"], NOTEBOOKS.parent)
            scope = {"sys": sys}
            with patch.dict(
                sys.modules,
                {
                    "google.colab": ModuleType("google.colab"),
                    "IPython.display": display,
                },
            ):
                exec(compile(source, "runtime", "exec"), scope)
            self.assertTrue(scope["IS_COLAB"])
            self.assertIs(scope["display"], display.display)

    def test_local_connect_uses_project_without_google_imports_or_mounting(self):
        project = Path("/tmp/local-notebook-project")
        scope = {"IS_COLAB": False, "LOCAL_PROJECT": project}
        with patch.dict(sys.modules, {"google.colab": None, "googleapiclient": None}):
            for path in (TRAINING, AUDIT):
                exec(
                    compile(notebook_source(path, "drive-connect"), "connect", "exec"),
                    scope,
                )
                self.assertEqual(scope["DRIVE_PROJECT"], project)
                self.assertEqual(scope["OUTPUT_PROJECT"], project)

    def setUp(self):
        self.scope = bootstrap_scope()

    def resolve(self, api, value="project"):
        return self.scope["resolve_drive_folder"](api, value, "mine")

    def test_folder_url_and_id_are_the_only_selection_inputs(self):
        parse = self.scope["drive_folder_id"]
        for value in (
            "project",
            "https://drive.google.com/drive/folders/project?usp=sharing",
        ):
            self.assertEqual(parse(value), "project")
        for value in (
            "",
            "../project",
            "https://evil.test/drive/folders/project",
            "https://colab.research.google.com/drive/book",
        ):
            with self.assertRaises(ValueError):
                parse(value)

    def test_owned_nested_path_does_not_create_a_shortcut(self):
        for notebook in (TRAINING, AUDIT):
            with self.subTest(notebook=notebook.name):
                scope = bootstrap_scope(notebook)
                api = API(
                    {
                        "project": folder("project", "facial-attendance", "outer"),
                        "outer": folder("outer", "AI", "mine"),
                    }
                )
                _, parts = scope["resolve_drive_folder"](api, "project", "mine")
                self.assertEqual(parts, ("AI", "facial-attendance"))
                self.assertFalse(api.created)

    def test_nested_shortcut_reused_by_target_id_across_pages(self):
        api = API(
            {
                "project": folder("project", "source", "private"),
                "outer": folder("outer", "AI", "mine"),
                "wrong": {
                    "id": "wrong",
                    "name": "source",
                    "mimeType": SHORTCUT,
                    "parents": ["mine"],
                    "shortcutDetails": {"targetId": "other"},
                },
                "shortcut": {
                    "id": "shortcut",
                    "name": "my project",
                    "mimeType": SHORTCUT,
                    "parents": ["outer"],
                    "shortcutDetails": {"targetId": "project"},
                },
            },
            page_size=1,
        )
        _, parts = self.resolve(api)
        self.assertEqual(parts, ("AI", "my project"))
        self.assertFalse(api.created)

    def test_name_collision_gets_an_unambiguous_id_alias(self):
        api = API(
            {
                "project": folder("project", "source", "private"),
                "collision": folder("collision", "colab-folder-project", "mine"),
            }
        )
        _, parts = self.resolve(api)
        self.assertEqual(parts, ("colab-folder-project-1",))
        self.assertEqual(api.created[0]["shortcutDetails"]["targetId"], "project")
        self.assertEqual(self.resolve(api)[1], parts)
        self.assertEqual(len(api.created), 1)

    def test_duplicate_owned_names_use_a_shortcut(self):
        api = API(
            {
                "project": folder("project", "same", "mine"),
                "other": folder("other", "same", "mine"),
            }
        )
        self.assertEqual(self.resolve(api)[1], ("colab-folder-project",))

    def test_permission_errors_are_friendly_and_quota_errors_are_preserved(self):
        for error in (HttpError(404), HttpError(403)):
            api = API({"project": error})
            with self.assertRaisesRegex(PermissionError, "sharing permissions"):
                self.resolve(api)
            self.assertFalse(api.created)
        for status, reason in ((403, "rateLimitExceeded"), (500, "backendError")):
            api = API({"project": HttpError(status, reason)})
            with self.assertRaises(HttpError):
                self.resolve(api)
            api.nodes["project"] = folder("project", "source", "parent")
            api.nodes["parent"] = HttpError(status, reason)
            with self.assertRaises(HttpError):
                self.resolve(api)

    def test_cycle_and_nonfolder_stop(self):
        api = API({"project": folder("project", "source", "project")})
        with self.assertRaisesRegex(ValueError, "cycle"):
            self.resolve(api)
        api.nodes["project"]["mimeType"] = "text/plain"
        with self.assertRaisesRegex(ValueError, "folder"):
            self.resolve(api)

    def test_output_capability_uses_actual_evidence_destination(self):
        api = API(
            {
                "project": folder("project", "source", "mine"),
                "evidence": folder("evidence", "evidence", "project"),
                "liveness": folder("liveness", "liveness", "evidence", writable=False),
            }
        )
        select = self.scope["output_project_id"]
        selected = select(api, api.nodes["project"], "mine")
        self.assertNotEqual(selected, "project")
        self.assertEqual(api.created[0]["parents"], ["mine"])
        self.assertEqual(select(api, api.nodes["project"], "mine"), selected)
        self.assertEqual(len(api.created), 2)

    def test_commenter_setup_creates_shortcut_before_mount_and_private_outputs(self):
        api = API({"project": folder("project", "source", "private", writable=False)})
        with tempfile.TemporaryDirectory() as temporary:
            mount = Path(temporary)
            scope = {
                **self.scope,
                "IS_COLAB": True,
                "PROJECT_FOLDER": "project",
                "TRAINING_OUTPUT_NAME": "run",
            }
            scope["Path"] = lambda p: (
                mount if p == "/content/drive/MyDrive" else Path(p)
            )

            def mount_drive(*args, **kwargs):
                api.events.append("mount")
                for key, value in api.nodes.items():
                    if isinstance(value, dict):
                        parts = self.scope["my_drive_folder_path"](api, key, "mine")
                        if parts is not None:
                            mount.joinpath(*parts).mkdir(parents=True, exist_ok=True)

            google = ModuleType("google")
            colab = ModuleType("google.colab")
            colab.auth = SimpleNamespace(authenticate_user=Mock())
            colab.drive = SimpleNamespace(mount=mount_drive)
            discovery = ModuleType("googleapiclient.discovery")
            discovery.build = Mock(return_value=api)
            with patch.dict(
                sys.modules,
                {
                    "google": google,
                    "google.colab": colab,
                    "googleapiclient": ModuleType("googleapiclient"),
                    "googleapiclient.discovery": discovery,
                },
            ):
                exec(
                    compile(
                        notebook_source(TRAINING, "drive-connect"), "connect", "exec"
                    ),
                    scope,
                )
            self.assertEqual(api.events[-1], "mount")
            self.assertTrue(
                scope["OUTPUT_PROJECT"].is_relative_to(
                    mount / "facial-attendance-outputs"
                )
            )
            self.assertNotEqual(scope["DRIVE_PROJECT"], scope["OUTPUT_PROJECT"])
            self.assertTrue(scope["DRIVE_PROJECT"].is_dir())

    def test_mount_wait_is_bounded(self):
        with patch.object(time, "monotonic", side_effect=[0, 31]):
            with self.assertRaisesRegex(FileNotFoundError, "Reconnect Drive"):
                self.scope["wait_for_drive_folder"](Path("/missing-folder"))

    def test_release_is_checked_before_source_is_loaded(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            with self.assertRaises(FileNotFoundError):
                self.scope["load_project_release"](project)
            for name in ("src", "scripts", "configs"):
                (project / name).mkdir()
            source = project / "scripts/safe.py"
            source.write_text("raise RuntimeError('not executed')\n")
            record = {
                "delivery_path": "scripts/safe.py",
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
            release = {
                "schema": "liveness-training-release-v1",
                "files": {"scripts/safe.py": record},
            }
            (project / "configs/liveness-release.json").write_text(json.dumps(release))
            self.scope["load_project_release"](project)
            self.assertEqual(
                self.scope["verified_source"](project, record), source.read_bytes()
            )
            source.write_text("changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                self.scope["verified_source"](project, record)
            record["delivery_path"] = "../escape.py"
            (project / "configs/liveness-release.json").write_text(json.dumps(release))
            with self.assertRaisesRegex(ValueError, "escapes"):
                self.scope["load_project_release"](project)


class CameraFolderTests(unittest.TestCase):
    def test_local_camera_action_uses_explicit_cpu_without_evaluation_bootstrap(self):
        project = Path("/tmp/project path")
        process = SimpleNamespace(run=Mock())
        scope = dict(
            ACTION="inference",
            IS_COLAB=False,
            PROJECT=project,
            CHECKPOINT_RUN=project / "evidence/run",
            subprocess=process,
            sys=SimpleNamespace(executable="project-python"),
        )
        exec(
            compile(
                notebook_source(TRAINING, "camera-inference-run"), "camera", "exec"
            ),
            scope,
        )
        command = process.run.call_args.args[0]
        self.assertEqual(command[0], "project-python")
        self.assertEqual(command[command.index("--device") + 1], "cpu")
        self.assertEqual(command[command.index("--action") + 1], "inference")
        self.assertNotIn("evaluate", command)
        self.assertTrue(process.run.call_args.kwargs["check"])

    def test_camera_reads_shared_checkpoint_and_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            project = Path(temporary)
            (project / "data/liveness/camera-trials").mkdir(parents=True)
            stage = Mock(return_value="script")
            scope = {
                "ACTION": "inference",
                "IS_COLAB": True,
                "DRIVE_PROJECT": project,
                "CHECKPOINT_RUN": project / "shared-evidence",
                "evaluation_module": SimpleNamespace(
                    selected_checkpoint=lambda p: (None, p / "model.pt", "sha")
                ),
                "release": {
                    "files": {"scripts/liveness_camera_inputs.py": {}},
                    "evaluation": {"camera_evaluator": {}},
                },
                "evaluation_bootstrap": project / "bootstrap",
                "stage_release_script": stage,
            }
            exec(
                compile(
                    notebook_source(TRAINING, "evaluation-stage-51"), "camera", "exec"
                ),
                scope,
            )
            self.assertEqual(scope["checkpoint"], project / "shared-evidence/model.pt")
            self.assertEqual(
                scope["camera_folder"], project / "data/liveness/camera-trials"
            )
            self.assertEqual(scope["batch_script"], "script")
            self.assertEqual(stage.call_args.args[0], project)


if __name__ == "__main__":
    unittest.main()
