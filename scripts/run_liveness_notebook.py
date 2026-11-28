"""One local entry point for saved inspection, frozen-data audit and inference.

ML imports occur only at the explicit inference boundary. Help and inspect setup
remain available without importing training, evaluation or GPU runtime staging.
"""

import argparse
import json
import os
import sys
import tempfile
from importlib import import_module
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from liveness_camera_inputs import selected_checkpoint

from facial_attendance.liveness.dataset_files import file_sha256

TRAINING_ACTIONS = {
    "train-baseline": "training.train_liveness_binary",
    "train-multitask": "training.train_liveness_multitask",
    "train-source": "training.train_liveness_aenet",
}


def print_result(result, output):
    content = json.dumps(result, indent=2, allow_nan=False) + "\n"
    if output is not None:
        with output.open("x", encoding="utf-8") as handle:
            handle.write(content)
    print(content, end="")


def infer(args, project):
    """Score one image or a recording folder using a fixed, verified checkpoint.

    This route never selects a model or fits a threshold on the supplied input.
    """
    # The inference action alone needs the installed model/image dependencies.
    import numpy as np
    from PIL import Image

    from facial_attendance.liveness.inference.liveness_adapter import LivenessAdapter

    if args.input is None:
        raise ValueError(
            "Inference requires --input (an image or camera recording folder)"
        )
    if not args.input.exists():
        raise FileNotFoundError(args.input)
    checkpoint = args.checkpoint
    if checkpoint is None:
        if args.run_dir is not None:
            _, checkpoint, _ = selected_checkpoint(args.run_dir)
        else:
            # Source submissions bundle the validation-selected inference weights;
            # archived run directories remain an explicit reproduction option.
            asset = json.loads(
                (project / "evidence/liveness/submission-artifacts.json").read_text()
            )["selected_model"]
            checkpoint = project / asset["local_path"]
            if file_sha256(checkpoint) != asset["sha256"]:
                raise ValueError("Bundled liveness checkpoint checksum differs")
    args.run_dir = args.run_dir or project / "evidence/liveness/liveness-continuation"
    if args.input.is_dir():
        if args.output is not None:
            raise ValueError("Camera folders use --cache, not --output")
        if args.face_crop:
            raise ValueError("--face-crop applies to a single RGB image")
        from evaluation.score_liveness_mp4_folder import evaluate_batch, recordings

        release = json.loads((project / "configs/liveness-release.json").read_text())
        args.checkpoint = checkpoint
        args.trials = args.input
        args.inventory = project / "configs/liveness-release.json"
        args.cache = args.cache or args.run_dir / "camera-evaluation" / file_sha256(
            checkpoint
        )
        clips = recordings(args.input)
        evaluate_batch(
            args, clips, release["evaluation"]["camera_evaluator"]["compatible_results"]
        )
        return
    if args.cache is not None:
        raise ValueError("--cache applies to camera folders")
    if args.output is not None and args.output.exists():
        raise FileExistsError(args.output)

    adapter = LivenessAdapter(checkpoint, args.device, detector_model=args.detector)
    with Image.open(args.input) as source:
        rgb = source.convert("RGB")
        if args.face_crop:
            result = adapter.predict_face(rgb)
        else:
            # Image files are RGB; the frame adapter's public contract is BGR.
            result = adapter.predict_frame(np.asarray(rgb)[:, :, ::-1].copy())

    print_result(result, args.output)


def main(argv=None):
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--action",
        choices=(
            "inspect",
            "audit",
            "inference",
            "diagnose-run",
            "inspect-protocol",
            *TRAINING_ACTIONS,
        ),
        default="inspect",
    )
    parser.add_argument("--notebook", type=Path)
    parser.add_argument("--output", "--output-dir", type=Path)
    parser.add_argument(
        "--input",
        "--data-root",
        type=Path,
        help="Image file or existing flat MP4 camera folder",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        help="Otherwise use the checksum-verified validation winner",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Explicit inference device; no automatic fallback",
    )
    parser.add_argument(
        "--detector",
        type=Path,
        default=project / "artifacts/detectors/face_detection_yunet_2023mar.onnx",
    )
    parser.add_argument(
        "--face-crop",
        action="store_true",
        help="Input image is already an RGB face crop",
    )
    parser.add_argument("--cache", type=Path, help="Camera-folder prediction cache")
    parser.add_argument(
        "arguments",
        nargs=argparse.REMAINDER,
        help="Specialized training options follow --",
    )

    args = parser.parse_args(argv)

    if args.action in TRAINING_ACTIONS:
        arguments = (
            args.arguments[1:] if args.arguments[:1] == ["--"] else args.arguments
        )
        module = import_module(TRAINING_ACTIONS[args.action])
        if (
            args.action == "train-source"
            and "--retain-epoch-checkpoints" not in arguments
        ):
            arguments.append("--retain-epoch-checkpoints")
        module.main(arguments)
        return
    if args.arguments:
        parser.error("Extra options after -- are only for training actions")

    if args.action == "inspect-protocol":
        from facial_attendance.liveness.audit.reproduction_protocol import (
            inspect_public_protocol,
        )

        if args.input is None:
            parser.error("inspect-protocol requires --data-root")
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        print_result(inspect_public_protocol(args.input), args.output)
        return

    if args.action == "diagnose-run":
        from facial_attendance.liveness.evaluation.quarantine_analysis import (
            inspect_run,
        )

        output = args.output or (
            Path(os.environ["GPU_JOB_OUTPUT"]) / "analysis"
            if "GPU_JOB_OUTPUT" in os.environ
            else None
        )
        if output is None or args.run_dir is None:
            parser.error("diagnose-run requires --run-dir and --output-dir")
        print(inspect_run(args.run_dir, output))
        return
    cache = Path(tempfile.mkdtemp(prefix="liveness-notebook-"))
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache))
    os.environ.setdefault("MPLBACKEND", "Agg")

    if args.action == "inference":
        if args.notebook is not None:
            parser.error("Inference runs directly; omit --notebook")
        infer(args, project)
        return
    if (
        args.input is not None
        or args.checkpoint is not None
        or args.face_crop
        or args.cache is not None
    ):
        parser.error("Inference options require --action inference")
    default = (
        "liveness-data-preparation-and-audit.ipynb"
        if args.action == "audit"
        else "liveness-training-and-evaluation.ipynb"
    )
    # Notebook rendering is loaded only for inspect/audit, never for direct inference.
    from facial_attendance.notebook_runtime import execute_notebook

    execute_notebook(
        args.notebook or project / "notebooks" / default,
        project=project,
        action=args.action,
        output=args.output,
    )


if __name__ == "__main__":
    main()
