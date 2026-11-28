"""Download/check the pinned ImageNet initialization used for ResNet18 training."""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
from urllib.request import urlopen

from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.models.binary_model import (
    RESNET_INITIALIZATION_SHA256,
    RESNET_INITIALIZATION_URL,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "artifacts/pretrained/resnet18-imagenet1k-initialization.pth",
    )

    args = parser.parse_args()
    if args.output.exists():
        if file_sha256(args.output) != RESNET_INITIALIZATION_SHA256:
            raise ValueError(
                "Existing initialization file does not match the pinned weights"
            )
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(".download")
        with (
            urlopen(RESNET_INITIALIZATION_URL, timeout=60) as source,
            temporary.open("xb") as target,
        ):
            while block := source.read(1024 * 1024):
                target.write(block)
        if file_sha256(temporary) != RESNET_INITIALIZATION_SHA256:
            raise ValueError("Downloaded initialization checksum mismatch")
        temporary.replace(args.output)
    print(f"Verified ImageNet initialization: {args.output}")


if __name__ == "__main__":
    main()
