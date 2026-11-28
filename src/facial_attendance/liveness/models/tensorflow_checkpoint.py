"""Read dense numeric TensorFlow V2 teacher checkpoints without TensorFlow.

Supports the LevelDB table / BundleEntryProto format used by the two source
teachers. Partitioned/string tensors and multi-shard bundles are rejected.
Format references: tensorflow/core/protobuf/tensor_bundle.proto and
Google LevelDB table/format.{h,cc}. This is weight I/O, not a TensorFlow runtime.
"""

import math
import struct
from pathlib import Path

import numpy as np


def varint(data, offset):
    value = 0
    for shift in range(0, 70, 7):
        if offset >= len(data):
            raise ValueError("Truncated checkpoint varint")
        byte = data[offset]
        offset += 1
        value |= (byte & 127) << shift
        if byte < 128:
            return value, offset
    raise ValueError("Invalid checkpoint varint")


def protobuf_fields(data):
    """Decode wire values without depending on a TensorFlow-generated schema."""
    offset = 0
    while offset < len(data):
        tag, offset = varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if not field:
            raise ValueError("Invalid protobuf field")
        if wire == 0:
            value, offset = varint(data, offset)
        elif wire in (1, 5):
            length = 8 if wire == 1 else 4
            value = data[offset : offset + length]
            offset += length
            if len(value) != length:
                raise ValueError("Truncated protobuf fixed value")
        elif wire == 2:
            length, offset = varint(data, offset)
            value = data[offset : offset + length]
            offset += length
            if len(value) != length:
                raise ValueError("Truncated protobuf bytes")
        else:
            raise ValueError("Unsupported protobuf wire type")
        yield field, wire, value


def crc32c(content):
    """Castagnoli CRC used by TensorFlow/LevelDB; no optional native package."""
    value = 0xFFFFFFFF
    for byte in content:
        value = CRC32C_TABLE[(value ^ byte) & 255] ^ (value >> 8)
    return value ^ 0xFFFFFFFF


def _crc_table():
    result = []
    for index in range(256):
        value = index
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        result.append(value)
    return tuple(result)


CRC32C_TABLE = _crc_table()


def masked_crc32c(content):
    value = crc32c(content)
    return (((value >> 15) | (value << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def table_entries(block):
    if len(block) < 4:
        raise ValueError("Truncated checkpoint table block")
    restart_count = struct.unpack("<I", block[-4:])[0]
    end = len(block) - 4 * (restart_count + 1)
    if end < 0:
        raise ValueError("Invalid checkpoint restart table")
    offset, previous = 0, b""
    while offset < end:
        shared, offset = varint(block, offset)
        unique, offset = varint(block, offset)
        length, offset = varint(block, offset)
        if shared > len(previous) or offset + unique + length > end:
            raise ValueError("Invalid checkpoint table entry")
        key = previous[:shared] + block[offset : offset + unique]
        offset += unique
        value = block[offset : offset + length]
        offset += length
        yield key, value
        previous = key


def read_index(path):
    data = Path(path).read_bytes()
    if len(data) < 48 or data[-8:] != bytes.fromhex("57fb808b247547db"):
        raise ValueError("Unsupported TensorFlow checkpoint index format")

    def handle(value, start=0):
        offset, cursor = varint(value, start)
        size, cursor = varint(value, cursor)
        return offset, size, cursor

    def block(offset, size):
        if offset + size + 5 > len(data) - 48:
            raise ValueError("Checkpoint block lies outside the table")
        content, compression = data[offset : offset + size], data[offset + size]
        stored_crc = int.from_bytes(
            data[offset + size + 1 : offset + size + 5], "little"
        )
        if masked_crc32c(content + bytes([compression])) != stored_crc:
            raise ValueError("Checkpoint index block checksum mismatch")
        # TensorFlow's bundle writer explicitly uses kNoCompression. Reject
        # other formats rather than implementing unused LevelDB variants.
        if compression != 0:
            raise ValueError(
                "Only uncompressed TensorFlow bundle indexes are supported"
            )
        return content

    _, _, cursor = handle(data, len(data) - 48)
    index_offset, index_size, _ = handle(data, cursor)
    result = {}
    for _, value in table_entries(block(index_offset, index_size)):
        offset, size, _ = handle(value)
        for key, metadata in table_entries(block(offset, size)):
            name = key.decode("utf-8")
            if name in result:
                raise ValueError("Duplicate checkpoint tensor name")
            result[name] = metadata
    return result


def tensor_shape(data):
    dimensions = []
    for field, _, value in protobuf_fields(data):
        if field == 2:
            dimension = dict(
                (number, value) for number, _, value in protobuf_fields(value)
            ).get(1, 0)
            if dimension >= 2**63:
                raise ValueError("Unknown checkpoint tensor dimension")
            dimensions.append(dimension)
        elif field == 3 and value:
            raise ValueError("Unknown checkpoint tensor rank")
    return tuple(dimensions)


class TensorFlowCheckpoint:
    def __init__(self, prefix, *, data_file=None):
        self.prefix = Path(prefix)
        self.entries = read_index(str(prefix) + ".index")
        header = dict(
            (field, value) for field, _, value in protobuf_fields(self.entries.pop(""))
        )
        if header.get(1, 0) != 1 or header.get(2, 0) != 0:
            raise ValueError(
                "Only single-shard little-endian source checkpoints are supported"
            )
        self.data_file = (
            Path(data_file)
            if data_file is not None
            else Path(str(prefix) + ".data-00000-of-00001")
        )
        self.metadata = {}
        for name, raw in self.entries.items():
            fields = dict((field, value) for field, _, value in protobuf_fields(raw))
            if 7 in fields or fields.get(3, 0) != 0:
                raise ValueError("Partitioned/multi-shard tensor unsupported")
            self.metadata[name] = {
                "dtype": fields.get(1, 0),
                "shape": tensor_shape(fields.get(2, b"")),
                "offset": fields.get(4, 0),
                "bytes": fields.get(5, 0),
                "crc32c": int.from_bytes(fields.get(6, b""), "little"),
            }

    def tensor(self, name):
        entry = self.metadata[name]
        dtypes = {1: "<f4", 2: "<f8", 3: "<i4", 9: "<i8", 10: "?", 19: "<f2"}
        if entry["dtype"] not in dtypes:
            raise ValueError("Only dense numeric teacher tensors are supported")
        dtype = np.dtype(dtypes[entry["dtype"]])
        expected = math.prod(entry["shape"]) * dtype.itemsize
        if (
            expected != entry["bytes"]
            or entry["offset"] + expected > self.data_file.stat().st_size
        ):
            raise ValueError("Checkpoint tensor shape/byte range mismatch")
        with self.data_file.open("rb") as handle:
            handle.seek(entry["offset"])
            content = handle.read(expected)
        if len(content) != expected:
            raise ValueError("Truncated checkpoint tensor data")
        if masked_crc32c(content) != entry["crc32c"]:
            raise ValueError(f"Checkpoint tensor checksum mismatch: {name}")
        return np.frombuffer(content, dtype=dtype).reshape(entry["shape"]).copy()
