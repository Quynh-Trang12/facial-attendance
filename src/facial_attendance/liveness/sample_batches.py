"""Save and verify representative diagnostic batches with source alignment."""

from dataclasses import asdict
from pathlib import Path

from PIL import Image, ImageDraw

from facial_attendance.liveness.audit_report import BatchEvidence
from facial_attendance.liveness.dataset_files import file_sha256, local_path
from facial_attendance.liveness.face_crop import bbox_path, read_face


def sample_batch(
    dataset, output: Path, batch_size: int, image_size: int
) -> BatchEvidence:
    # This boundary keeps CLI help and annotation failures usable without Torch,
    # matching the project's Pillow-only source-check job.
    import torch
    from torch.utils.data import DataLoader, Subset

    # Include each selected source type that exists, then fill in inventory order.
    indices, types = [], set()
    for index, row in enumerate(dataset.samples):
        if row.source_type not in types:
            indices.append(index)
            types.add(row.source_type)
    if batch_size < len(indices):
        raise ValueError(f"batch-size must cover {len(indices)} source types")
    indices += [i for i in range(len(dataset)) if i not in indices][
        : batch_size - len(indices)
    ]

    images, targets = next(
        iter(
            DataLoader(
                Subset(dataset, indices),
                batch_size=len(indices),
                shuffle=False,
                num_workers=0,
            )
        )
    )

    if (
        images.shape != (len(indices), 3, image_size, image_size)
        or images.dtype != torch.float32
        or targets.dtype != torch.int64
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
    ):
        raise ValueError(f"Invalid batch in {dataset.split}")

    records = []
    sheet = Image.new("RGB", (image_size * len(indices), image_size + 64), "white")
    draw = ImageDraw.Draw(sheet)
    for position, index in enumerate(indices):
        row = dataset.samples[index]
        face = read_face(
            local_path(dataset.root, row.relative_path),
            box_file=local_path(dataset.root, str(bbox_path(Path(row.relative_path)))),
        )
        expected = dataset.transform(face)
        if int(targets[position]) != row.binary_label or not torch.equal(
            images[position], expected
        ):
            raise ValueError(f"Image/target alignment failed: {row.relative_path}")

        sheet.paste(
            face.resize((image_size, image_size), Image.Resampling.BILINEAR),
            (position * image_size, 0),
        )
        draw.text(
            (position * image_size + 3, image_size + 3),
            f"index={index} target={row.binary_label}\n{row.split} type={row.source_type}\n{row.attack_family}",
            fill="black",
        )
        records.append({"dataset_index": index, **asdict(row)})

    stem = output / dataset.split
    sheet.save(stem.with_suffix(".png"))
    torch.save(
        {
            "images": images,
            "targets": targets,
            "relative_paths": [r["relative_path"] for r in records],
        },
        stem.with_suffix(".pt"),
    )

    saved = torch.load(stem.with_suffix(".pt"), weights_only=True)
    if (
        not torch.equal(saved["images"], images)
        or not torch.equal(saved["targets"], targets)
        or saved["relative_paths"] != [r["relative_path"] for r in records]
    ):
        raise ValueError(f"Saved batch readback failed: {dataset.split}")

    return BatchEvidence(
        saved_batch_readback="PASS",
        shape=list(images.shape),
        image_dtype=str(images.dtype),
        target_dtype=str(targets.dtype),
        range=[images.min().item(), images.max().item()],
        alignment="PASS",
        rows=records,
        tensor_sha256=file_sha256(stem.with_suffix(".pt")),
        preview_sha256=file_sha256(stem.with_suffix(".png")),
    )

