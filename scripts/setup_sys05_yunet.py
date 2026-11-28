"""Explicit optional download/check of the OpenCV Zoo YuNet model and licence.

No network is used unless --download is supplied. No shared dependency changes.
"""

import argparse
from hashlib import sha256
from pathlib import Path
import sys
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from facial_attendance.ui.face_detection import YUNET_FILENAME, YUNET_SHA256

BASE = "https://raw.githubusercontent.com/opencv/opencv_zoo/main/models/face_detection_yunet/"
FILES = (
    (YUNET_FILENAME,
     "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/face_detection_yunet/" + YUNET_FILENAME,
     YUNET_SHA256, 232589),
    ("LICENSE", BASE + "LICENSE", "c83b8120c50ccbd4c4f96edf53141bdd566ebb8f8e9227e415326aa1b1aba958", 1085),
)


def setup(destination, download=False):
    for name, url, digest, size in FILES:
        path = destination / name
        if path.is_file():
            data = path.read_bytes()
        elif download:
            print(f"Downloading {url}\nTo {path}", flush=True)
            with urlopen(url, timeout=30) as response:
                data = response.read(size + 1)
        else:
            raise ValueError(f"Missing {path}. Run this script with --download explicitly.")
        if len(data) != size or sha256(data).hexdigest() != digest:
            raise ValueError(f"Size/checksum mismatch for {name}; no model installed or overwritten.")
        if not path.is_file():
            destination.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".download")
            temporary.write_bytes(data)
            temporary.replace(path)
        print(f"Verified {name}: sha256={digest}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--download", action="store_true", help="Explicitly download missing model/licence files.")
    args = parser.parse_args(argv)
    try:
        setup(ROOT / "artifacts/sys05-yunet", args.download)
    except (OSError, ValueError) as error:
        parser.exit(1, f"YuNet setup failed: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
