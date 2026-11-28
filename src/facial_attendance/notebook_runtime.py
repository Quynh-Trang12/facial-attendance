"""Optional rich display and sequential notebook execution without a Jupyter kernel.

IPython is an optional presentation boundary. Project Python remains responsible
for ML dependencies; this module neither installs an environment nor chooses a GPU.
"""

import base64
import contextlib
import hashlib
import io
import json
import sys
import traceback
from contextvars import ContextVar
from datetime import UTC, datetime
from html.parser import HTMLParser
from pathlib import Path

# A real notebook kernel renders these objects directly. Plain Python execution
# captures the same MIME payloads without requiring IPython to be installed.
try:
    from IPython import get_ipython
    from IPython.display import display as ipython_display
except ModuleNotFoundError:
    get_ipython = ipython_display = None

_outputs = ContextVar("notebook_outputs", default=None)


class _PlainHTML(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts, self.hidden = [], False

    def handle_starttag(self, tag, attrs):
        if tag in ("style", "script"):
            self.hidden = True

    def handle_endtag(self, tag):
        if tag in ("style", "script"):
            self.hidden = False
        if tag in ("td", "th"):
            self.parts.append(" | ")
        elif tag in ("tr", "p"):
            self.parts.append("\n")

    def handle_data(self, data):
        if not self.hidden:
            self.parts.append(data)


class HTML:
    def __init__(self, data):
        self.data = data

    def _repr_html_(self):
        return self.data

    def __str__(self):
        parser = _PlainHTML()
        parser.feed(self.data)
        return "".join(parser.parts).strip()


def display(value):
    target = _outputs.get()
    if target is None:
        if get_ipython is not None and get_ipython() is not None:
            ipython_display(value)
        else:
            print(value)
        return
    data = {}
    if hasattr(value, "_repr_html_"):
        data["text/html"] = value._repr_html_()
    elif hasattr(value, "_repr_png_"):
        data["image/png"] = base64.b64encode(value._repr_png_()).decode()
    else:
        data["text/plain"] = repr(value)
    target.append({"output_type": "display_data", "metadata": {}, "data": data})


def _capture_figures(*_args, **_kwargs):
    # Matplotlib is loaded by the analytical notebook, not by the lightweight
    # audit route. Access its existing module instead of introducing a dependency.
    pyplot = sys.modules.get("matplotlib.pyplot")
    if pyplot is None:
        return
    for number in pyplot.get_fignums():
        stream = io.BytesIO()
        pyplot.figure(number).savefig(
            stream, format="png", dpi=130, bbox_inches="tight"
        )
        _outputs.get().append(
            {
                "output_type": "display_data",
                "metadata": {},
                "data": {"image/png": base64.b64encode(stream.getvalue()).decode()},
            }
        )
    pyplot.close("all")


class _OutputStream(io.StringIO):
    """Preserve stream/display ordering rather than appending all text last."""

    def __init__(self, outputs, name):
        super().__init__()
        self.outputs, self.name = outputs, name

    def write(self, value):
        if value:
            if (
                self.outputs
                and self.outputs[-1].get("output_type") == "stream"
                and self.outputs[-1].get("name") == self.name
            ):
                self.outputs[-1]["text"] += value
            else:
                self.outputs.append(
                    {"output_type": "stream", "name": self.name, "text": value}
                )
        return super().write(value)


def execute_notebook(notebook_path, *, project, action="inspect", output=None):
    """Run every code cell in order and retain genuine outputs on failure too."""
    notebook_path, project = Path(notebook_path).resolve(), Path(project).resolve()
    destination = Path(output).resolve() if output else notebook_path
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    scope = {
        "__name__": "__notebook__",
        "ACTION": action,
        "LOCAL_PROJECT": project,
    }
    failure = None
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] != "code":
            continue
        identifier = cell.get("id", str(index))
        outputs = []
        stdout, stderr = (
            _OutputStream(outputs, "stdout"),
            _OutputStream(outputs, "stderr"),
        )
        token = _outputs.set(outputs)
        pyplot = sys.modules.get("matplotlib.pyplot")
        old_show = pyplot.show if pyplot is not None else None
        try:
            if pyplot is not None:
                pyplot.show = _capture_figures
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                exec(compile("".join(cell["source"]), identifier, "exec"), scope)
                # Also captures a plot created in the same cell that first
                # imports pyplot, before its show hook could be installed.
                _capture_figures()
        except Exception as error:
            failure = error
            outputs.append(
                {
                    "output_type": "error",
                    "ename": type(error).__name__,
                    "evalue": str(error),
                    "traceback": traceback.format_exc().splitlines(),
                }
            )
        finally:
            if pyplot is not None:
                pyplot.show = old_show
            _outputs.reset(token)
        cell["outputs"] = outputs
        cell["execution_count"] = None
        print(f"{identifier}: {'FAIL' if failure else 'PASS'}", flush=True)
        if failure is not None:
            break
    notebook["metadata"]["local_execution"] = {
        "runtime": sys.version,
        "executed_at": datetime.now(UTC).isoformat(),
        "project": str(project),
        "action": action,
        "kernel_required": False,
        "completed": failure is None,
        "code_source_sha256": hashlib.sha256(
            json.dumps(
                [
                    (cell.get("id"), cell["source"])
                    for cell in notebook["cells"]
                    if cell["cell_type"] == "code"
                ]
            ).encode()
        ).hexdigest(),
        "execution_counts": "Unset: sequential Python execution without a Jupyter kernel",
    }
    destination.write_text(
        json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8"
    )
    if failure is not None:
        raise RuntimeError(
            f"Notebook stopped at {identifier}; outputs retained"
        ) from failure
    return notebook
