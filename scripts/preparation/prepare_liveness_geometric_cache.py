"""Generate real PRNet/reflection pseudo-labels, resuming only identical inputs."""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import os

from facial_attendance.liveness.datasets.geometric_cache import generate_cache
from facial_attendance.liveness.datasets.source_dataset import ReproductionImages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-root",
        "inventory",
        "annotations",
        "depth-checkpoint",
        "reflection-checkpoint",
        "mesh-directory",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--flush-images", type=int, default=64)
    parser.add_argument("--resume-cache", type=Path)

    args = parser.parse_args()

    output = args.output_dir or (
        Path(os.environ["GPU_JOB_OUTPUT"]) if "GPU_JOB_OUTPUT" in os.environ else None
    )

    if output is None:
        parser.error("--output-dir is required unless GPU_JOB_OUTPUT is set")
    source = ReproductionImages(args.data_root, args.inventory, args.annotations)

    result = generate_cache(
        source,
        args.depth_checkpoint,
        args.reflection_checkpoint,
        args.mesh_directory,
        output,
        flush_images=args.flush_images,
        resume=args.resume_cache,
    )
    print(
        json.dumps(
            {"completed": result["completed"], "rows": result["completed_rows"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
