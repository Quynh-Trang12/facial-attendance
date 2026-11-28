"""Controlled SYS-05 interface walkthrough; no GUI, webcam, model download or record writes."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from facial_attendance.ui.live_preview import (
    LiveController,
    WorkResult,
    evaluate_adapters,
)
from facial_attendance.ui.webcam_prototype import SCENARIOS, StubAdapters


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenario", choices=(*SCENARIOS, "all"), default="all")
    args = parser.parse_args(argv)
    import numpy as np

    frame = np.empty((360, 640, 3), dtype=np.uint8)
    frame[:] = (11, 22, 33)  # Deliberately distinct BGR channels; synthetic data only.
    frame.flags.writeable = False
    rows = []
    for index, scenario in enumerate(
        SCENARIOS if args.scenario == "all" else [args.scenario], 1
    ):
        live = LiveController()
        token = 1, index
        live.begin_sample(token, 1.0)
        request = live.detected(WorkResult(token, [(160, 70, 480, 300)]), frame)
        batch = evaluate_adapters((request, StubAdapters(scenario)))
        live.completed(WorkResult(token, batch))
        payload = asdict(live.controller.state)
        identity = live.controller.demo_identity
        live.reset()
        stale_rejected = not live.completed(WorkResult(token, batch))
        rows.append(
            {
                "scenario": scenario,
                "frame_dtype": str(frame.dtype),
                "frame_shape": list(frame.shape),
                "bbox_half_open": list(request.bbox),
                "crop_shape": list(request.face_crop.shape),
                "synthetic_bgr_pixel": request.face_crop[0, 0].tolist(),
                "source_shared_with_crop": bool(
                    np.shares_memory(frame, request.face_crop)
                ),
                "demo_identity": identity,
                "payload": payload,
                "stale_rejected_after_reset": stale_rejected,
            }
        )
    print(
        json.dumps(
            {
                "scope": "Developer supplied regions on a generated BGR frame; all downstream values simulated",
                "actual_face_detection": False,
                "records_written": False,
                "scenarios": rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
