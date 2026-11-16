"""CPU-only synthetic SYS-05 microbenchmarks; never opens Tk or a webcam."""

import argparse
import json
import platform
from pathlib import Path
from statistics import median
import sys
from time import perf_counter, sleep

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facial_attendance.ui.face_detection import HaarFaceDetector, ScaledFaceDetector, YuNetFaceDetector
from facial_attendance.ui.live_preview import CameraFeed, prepare_preview_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=8)
    args = parser.parse_args(argv)
    if args.runs < 1:
        parser.error("--runs must be positive")
    import cv2
    import numpy as np
    from PIL import Image

    cv2.setNumThreads(1)
    source = np.random.default_rng(42).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)

    def legacy_prepare():
        image = Image.fromarray(cv2.cvtColor(source, cv2.COLOR_BGR2RGB))
        image.thumbnail((640, 360))

    operations = {
        "legacy_preview_prepare": legacy_prepare,
        "resize_first_prepare": lambda: prepare_preview_image(source),
    }
    # Reuse loaded detectors like the application (exclude model load after warmup).
    haar = HaarFaceDetector()
    small_haar = ScaledFaceDetector(HaarFaceDetector())
    yunet = ScaledFaceDetector(YuNetFaceDetector())
    operations.update(haar_1920=lambda: haar(source), haar_640=lambda: small_haar(source),
                      yunet_640=lambda: yunet(source))
    timings = {}
    for name, operation in operations.items():
        samples = []
        for i in range(args.runs + 2):
            started = perf_counter()
            operation()
            if i >= 2:
                samples.append((perf_counter() - started) * 1000)
        timings[name] = round(median(samples), 3)

    class PacedCamera:
        def isOpened(self): return True
        def read(self):
            sleep(1 / 30)
            return True, source
        def release(self): pass

    # This legacy loop reproduces the previous read/copy/fixed-wait timing.
    camera = PacedCamera()
    count = 0
    started = perf_counter()
    while perf_counter() - started < 2.5:
        _, frame = camera.read()
        frame = frame.copy()
        frame.flags.writeable = False
        count += 1
        sleep(.03)
    legacy_fps = count / (perf_counter() - started)

    feed = CameraFeed(PacedCamera)
    started = perf_counter()
    try:
        feed.restart()
        sleep(2.5)
        packet = feed.latest()
        current_fps = packet.sequence / (perf_counter() - started)
    finally:
        feed.close()
    print(json.dumps({
        "scope": "Synthetic noise / paced fake camera only. No native Tk, physical webcam or glasses accuracy.",
        "system": platform.platform(), "python": platform.python_version(), "opencv": cv2.__version__,
        "source": [1920, 1080], "opencv_threads": 1, "warmup": 2, "runs": args.runs,
        "median_ms": timings,
        "synthetic_capture_fps": {"legacy_fixed_wait": round(legacy_fps, 2), "current": round(current_fps, 2)},
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
