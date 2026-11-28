"""Fetch the pinned official YuNet auxiliary locator and its MIT license."""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import urllib.request
from hashlib import sha256

from facial_attendance.liveness.models.face_detector import (
    YUNET_FILENAME,
    YUNET_REVISION,
    YUNET_SHA256,
    YUNET_URL,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/detectors"))

    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model = args.output_dir / YUNET_FILENAME
    if model.exists():
        data = model.read_bytes()
    else:
        with urllib.request.urlopen(YUNET_URL, timeout=60) as response:
            data = response.read()
    if sha256(data).hexdigest() != YUNET_SHA256:
        raise ValueError("Downloaded/existing detector checksum differs")
    if not model.exists():
        with model.open("xb") as handle:
            handle.write(data)
    license_file = args.output_dir / "LICENSE-YuNet.txt"
    license_url = f"https://huggingface.co/opencv/face_detection_yunet/resolve/{YUNET_REVISION}/LICENSE"
    if not license_file.exists():
        with urllib.request.urlopen(license_url, timeout=60) as response:
            license_text = response.read()
        with license_file.open("xb") as handle:
            handle.write(license_text)
    record = args.output_dir / "yunet-source.json"
    if not record.exists():
        record.write_text(
            json.dumps(
                {
                    "url": YUNET_URL,
                    "revision": YUNET_REVISION,
                    "sha256": YUNET_SHA256,
                    "size": len(data),
                    "license": "MIT",
                    "license_url": license_url,
                },
                indent=2,
            )
            + "\n"
        )
    print(f"Verified auxiliary face detector: {model}")


if __name__ == "__main__":
    main()
