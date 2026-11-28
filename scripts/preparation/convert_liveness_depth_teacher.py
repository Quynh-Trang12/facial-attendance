"""Convert the author's PRNet checkpoint to native inference-only weights."""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json

import torch

from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.models.depth_teacher import PRNetDepthTeacher
from facial_attendance.liveness.models.tensorflow_checkpoint import TensorFlowCheckpoint


def convert(prefix, tensor_data, output):
    source = TensorFlowCheckpoint(prefix, data_file=tensor_data)
    model = PRNetDepthTeacher()
    state = model.state_dict()
    copied = set()

    def assign(key, source_name, *, convolution=False):
        value = source.tensor(source_name)
        if convolution:
            value = value.transpose(3, 2, 0, 1)
        tensor = torch.from_numpy(value.copy())
        if tensor.shape != state[key].shape or not torch.isfinite(tensor).all():
            raise ValueError(
                f"PRNet source/target tensor mismatch: {source_name} -> {key}"
            )
        state[key] = tensor
        copied.add(key)

    def norm(key, scope):
        for target, original in (
            ("gamma", "gamma"),
            ("beta", "beta"),
            ("mean", "moving_mean"),
            ("variance", "moving_variance"),
        ):
            assign(key + "." + target, scope + "/" + original)
        if (state[key + ".variance"] < 0).any():
            raise ValueError("PRNet running variance must be nonnegative")

    def conv(key, scope, normalized=True):
        assign(key + ".conv.weight", scope + "/weights", convolution=True)
        if normalized:
            norm(key + ".norm", scope + "/BatchNorm")

    conv("initial", "resfcn256/Conv")
    for index, block in enumerate(model.encoder):
        scope = "resfcn256/resBlock" + (f"_{index}" if index else "")
        for target, original in (
            ("first", "Conv"),
            ("second", "Conv_1"),
            ("third", "Conv_2"),
        ):
            conv(
                f"encoder.{index}.{target}",
                scope + "/" + original,
                normalized=target != "third",
            )
        if block.shortcut is not None:
            conv(f"encoder.{index}.shortcut", scope + "/shortcut", normalized=False)
        norm(f"encoder.{index}.norm", scope + "/BatchNorm")
    for index, _ in enumerate(model.decoder):
        scope = "resfcn256/Conv2d_transpose" + (f"_{index}" if index else "")
        conv(f"decoder.{index}", scope)
    if copied != set(state):
        raise ValueError("PRNet conversion left initialized tensors")
    model.load_state_dict(state, strict=True)
    meta = {
        "architecture": "prnet-resfcn256-teacher-v1",
        "source": "https://github.com/yfeng95/PRNet",
        "index_sha256": file_sha256(Path(str(prefix) + ".index")),
        "tensor_data_sha256": file_sha256(tensor_data),
        "converted_tensors": len(copied),
        "random_initialized_tensors_remaining": 0,
        "tensorflow_reference_parity": "unverified; CRC-checked weight conversion is not an executed TensorFlow output comparison",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save({"state_dict": state, "meta": meta}, output)

    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-prefix", type=Path, required=True)
    parser.add_argument("--tensor-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    convert(args.checkpoint_prefix, args.tensor_data, args.output)


if __name__ == "__main__":
    main()
