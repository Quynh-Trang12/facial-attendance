"""Convert the original reflection teacher checkpoint/graph without TensorFlow."""

import sys
from pathlib import Path

# Direct CLI execution needs package sources and the shared workflow helpers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import argparse
import json
import tarfile

import numpy as np
import torch

from facial_attendance.liveness.dataset_files import file_sha256
from facial_attendance.liveness.models.reflection_teacher import (
    GENERATOR_LAYERS,
    ReflectionTeacher,
)
from facial_attendance.liveness.models.tensorflow_checkpoint import (
    TensorFlowCheckpoint,
    protobuf_fields,
    tensor_shape,
)


def graph_vgg_constants(archive):
    with tarfile.open(archive, "r:gz") as source:
        # Read the known graph member in memory; never extract arbitrary archive
        # paths, links or the large graph to the nearly-full project partition.
        member = source.getmember("pre-trained/model.ckpt.meta")
        if not member.isfile():
            raise ValueError("Expected the author graph file")
        meta = source.extractfile(member).read()
    graph = next(value for field, _, value in protobuf_fields(meta) if field == 2)

    result = {}
    for field, _, node in protobuf_fields(graph):
        if field != 1:
            continue
        fields = list(protobuf_fields(node))
        name = next((value.decode() for number, _, value in fields if number == 1), "")
        op = next((value.decode() for number, _, value in fields if number == 2), "")
        if not name.startswith("vgg19/") or op != "Const":
            continue
        for number, _, entry in fields:
            if number != 5:
                continue
            attribute = dict((key, value) for key, _, value in protobuf_fields(entry))
            if attribute.get(1) != b"value":
                continue
            value = dict(
                (key, value) for key, _, value in protobuf_fields(attribute[2])
            )
            tensor = dict((key, value) for key, _, value in protobuf_fields(value[8]))
            if tensor.get(1) != 1:
                raise ValueError("Expected float32 VGG constant")
            shape = tensor_shape(tensor.get(2, b""))
            content = tensor.get(4, b"")
            expected = int(np.prod(shape)) * 4
            if len(content) != expected:
                raise ValueError("VGG graph constant shape/byte mismatch")
            result[name] = np.frombuffer(content, "<f4").reshape(shape).copy()
    return result


def convert(archive, prefix, output):
    source = TensorFlowCheckpoint(prefix)
    constants = graph_vgg_constants(archive)
    model = ReflectionTeacher()
    state = model.state_dict()
    copied = []

    def assign(key, array):
        tensor = torch.from_numpy(array.copy())
        if (
            tuple(tensor.shape) != tuple(state[key].shape)
            or not torch.isfinite(tensor).all()
        ):
            raise ValueError(
                f"Source/target tensor shape or finite check failed: {key}"
            )
        state[key] = tensor
        copied.append(key)

    constant_index = 0
    for block_index, block in enumerate(model.vgg):
        for layer_index, _ in enumerate(block):
            name = "vgg19/Const" + (f"_{constant_index}" if constant_index else "")
            assign(
                f"vgg.{block_index}.{layer_index}.weight",
                constants[name].transpose(3, 2, 0, 1),
            )
            constant_index += 1
            assign(
                f"vgg.{block_index}.{layer_index}.bias",
                constants[f"vgg19/Const_{constant_index}"],
            )
            constant_index += 1
    assign("vgg_mean", constants["vgg19/sub/y"].transpose(0, 3, 1, 2))
    for index, name in enumerate(GENERATOR_LAYERS):
        assign(
            f"generator.{index}.weight",
            source.tensor(name + "/weights").transpose(3, 2, 0, 1),
        )
        assign(f"beta.{index}", source.tensor(name + "/BatchNorm/beta"))
        for weight in ("w0", "w1"):
            assign(f"{weight}.{index}", source.tensor(name + "/" + weight))
    assign("output.weight", source.tensor("g_conv_last/weights").transpose(3, 2, 0, 1))
    assign("output.bias", source.tensor("g_conv_last/biases"))
    if set(copied) != set(state):
        raise ValueError("Conversion left initialized values in the inference model")
    model.load_state_dict(state, strict=True)
    meta = {
        "architecture": "cvpr2018-hypercolumn-reflection-teacher-v1",
        "source": "https://github.com/ceciliavision/perceptual-reflection-removal",
        "archive_sha256": file_sha256(archive),
        "index_sha256": file_sha256(Path(str(prefix) + ".index")),
        "tensor_data_sha256": file_sha256(Path(str(prefix) + ".data-00000-of-00001")),
        "converted_tensors": len(copied),
        "random_initialized_tensors_remaining": 0,
        "tensorflow_reference_parity": "unverified; source-weight conversion and native execution are separate checks",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    torch.save({"state_dict": state, "meta": meta}, output)

    print(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--checkpoint-prefix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    convert(args.archive, args.checkpoint_prefix, args.output)


if __name__ == "__main__":
    main()
