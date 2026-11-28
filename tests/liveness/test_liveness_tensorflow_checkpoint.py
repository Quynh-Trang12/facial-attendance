"""Known numeric bundle fixture and checksum failure boundaries."""

import importlib.util
import struct
import tempfile
import unittest
from pathlib import Path

FULL = importlib.util.find_spec("numpy") is not None
if FULL:
    import numpy as np

    from facial_attendance.liveness.models.tensorflow_checkpoint import (
        TensorFlowCheckpoint,
        crc32c,
        masked_crc32c,
        varint,
    )


def encode(value):
    result = bytearray()
    while value >= 128:
        result.append((value & 127) | 128)
        value >>= 7
    result.append(value)
    return bytes(result)


def field(number, value):
    return encode(number << 3) + encode(value)


def byte_field(number, value):
    return encode(number << 3 | 2) + encode(len(value)) + value


@unittest.skipUnless(FULL, "Requires numpy in the project runtime")
class TensorFlowCheckpointTests(unittest.TestCase):
    def fixture(self, directory):
        prefix = Path(directory) / "teacher"
        values = np.array([[1.0, 2.0], [3.0, 4.0]], dtype="<f4")
        raw = values.tobytes()
        Path(str(prefix) + ".data-00000-of-00001").write_bytes(raw)
        header = field(1, 1)
        shape = byte_field(2, field(1, 2)) * 2
        metadata = (
            field(1, 1)
            + byte_field(2, shape)
            + field(5, len(raw))
            + bytes([53])
            + struct.pack("<I", masked_crc32c(raw))
        )

        def block(entries):
            content = b"".join(
                encode(0) + encode(len(k)) + encode(len(v)) + k + v for k, v in entries
            )
            content += struct.pack("<II", 0, 1)
            return content + b"\0" + struct.pack(
                "<I", masked_crc32c(content + b"\0")
            ), len(content)

        data_block, data_size = block([(b"", header), (b"weights", metadata)])
        index_block, index_size = block([(b"z", encode(0) + encode(data_size))])
        footer = encode(0) + encode(0) + encode(len(data_block)) + encode(index_size)
        footer += bytes(40 - len(footer)) + bytes.fromhex("57fb808b247547db")
        Path(str(prefix) + ".index").write_bytes(data_block + index_block + footer)
        return prefix, values

    def test_known_crc_and_numeric_tensor(self):
        self.assertEqual(crc32c(b"123456789"), 0xE3069283)
        with tempfile.TemporaryDirectory() as directory:
            prefix, values = self.fixture(directory)
            reader = TensorFlowCheckpoint(prefix)
            self.assertTrue(np.array_equal(reader.tensor("weights"), values))

    def test_corrupted_tensor_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _ = self.fixture(directory)
            path = Path(str(prefix) + ".data-00000-of-00001")
            path.write_bytes(bytes(16))
            with self.assertRaisesRegex(ValueError, "tensor checksum"):
                TensorFlowCheckpoint(prefix).tensor("weights")

    def test_corrupted_index_and_truncated_varint_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            prefix, _ = self.fixture(directory)
            path = Path(str(prefix) + ".index")
            content = bytearray(path.read_bytes())
            content[5] ^= 1
            path.write_bytes(content)
            with self.assertRaisesRegex(ValueError, "block checksum"):
                TensorFlowCheckpoint(prefix)
        with self.assertRaisesRegex(ValueError, "Truncated"):
            varint(b"\x80", 0)


if __name__ == "__main__":
    unittest.main()
