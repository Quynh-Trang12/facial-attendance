"""Export the backbone and binary classifier weights from one archived epoch.

--receipt selects the epoch's JSON record containing its checksum and validation
threshold. --output names the new inference file. The epoch archive, including
its auxiliary heads and optimizer state, is read without modification.
"""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import tempfile

import torch

from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.models.binary_model import (
    build_model,
    load_checkpoint,
    normalization,
    save_checkpoint,
    state_version,
)


def export_epoch(receipt_path, output):
    receipt_path, output = Path(receipt_path), Path(output)
    if output.exists():
        raise FileExistsError("Inference output already exists")
    receipt = json.loads(receipt_path.read_text())
    name = receipt["checkpoint"]
    if Path(name).name != name:
        raise ValueError("Archived checkpoint must be beside its receipt")
    checkpoint = receipt_path.parent / name
    if (
        checkpoint.stat().st_size != receipt["bytes"]
        or file_sha256(checkpoint) != receipt["sha256"]
    ):
        raise ValueError("Archived checkpoint checksum or size mismatch")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    epoch = receipt["epoch"]
    if (
        saved["epoch"] != epoch
        or saved["resume_epoch"] != epoch + 1
        or saved["progress"] is not None
        or receipt["inference_metadata"]["epoch"] != epoch
    ):
        raise ValueError("Expected a complete validated epoch")
    prefix = "binary_model."
    state = {
        name.removeprefix(prefix): value
        for name, value in saved["state_dict"].items()
        if name.startswith(prefix)
    }
    metadata = dict(receipt["inference_metadata"])
    if not state or state_version(state) != metadata["model_version"]:
        raise ValueError(
            "Archived inference weights differ from their validation metadata"
        )
    if metadata["optimizer_updates"] != saved["updates"]:
        raise ValueError("Archived inference update count differs")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Validate the entire inference contract before creating the output.
    normalization(metadata["architecture"])
    build_model(metadata["architecture"]).load_state_dict(state, strict=True)
    with tempfile.TemporaryDirectory(
        dir=output.parent, prefix="liveness-export-"
    ) as directory:
        pending = Path(directory) / "inference.pt"
        save_checkpoint(
            pending, state, metadata, archived_checkpoint_sha256=receipt["sha256"]
        )
        load_checkpoint(pending, "cpu")
        pending.replace(output)
    return {
        "epoch": epoch,
        "checkpoint_sha256": file_sha256(output),
        "source_checkpoint_sha256": receipt["sha256"],
        "threshold": metadata["threshold"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()

    print(json.dumps(export_epoch(args.receipt, args.output), indent=2))


if __name__ == "__main__":
    main()
