"""Re-evaluate the frozen checkpoint; never train, select or change labels."""

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import torch
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from torch.utils.data import DataLoader, Dataset

from facial_attendance.emotion.final import EmotionPredictor

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    manifest = args.data_root / "manifest.csv"
    digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert (
        digest == "54ac9d2ea037dbb2d879acd9d960f11161bfc353b89f5ca8a0151f221d2ac4ec"
    ), "Wrong dataset version."
    with manifest.open(encoding="utf-8", newline="") as f:
        rows = sorted(
            [
                r
                for r in csv.DictReader(f)
                if r["status"] == "retained" and r["output_split"] == args.split
            ],
            key=lambda r: r["output_path"],
        )
    assert len(rows) == {"val": 3217, "test": 4133}[args.split]
    predictor = EmotionPredictor(args.checkpoint)

    class Faces(Dataset):
        def __len__(self):
            return len(rows)

        def __getitem__(self, i):
            r = rows[i]
            relative = Path(r["output_path"])
            assert not relative.is_absolute() and ".." not in relative.parts
            path = args.data_root / relative
            assert hashlib.sha256(path.read_bytes()).hexdigest() == r["file_sha256"]
            with Image.open(path) as im:
                x = predictor.transform(im)
            return x, int(r["model_id"])

    args.output.mkdir(parents=True, exist_ok=False)
    truth = []
    pred = []
    saved = []
    loss = 0.0
    offset = 0
    with torch.inference_mode():
        for x, y in DataLoader(Faces(), batch_size=32, shuffle=False):
            out = predictor.model(x)
            prob = out.softmax(1)
            ids = out.argmax(1)
            loss += torch.nn.functional.cross_entropy(out, y, reduction="sum").item()
            truth.extend(y.tolist())
            pred.extend(ids.tolist())
            for j in range(len(y)):
                saved.append(
                    {
                        "output_path": rows[offset + j]["output_path"],
                        "true_id": int(y[j]),
                        "predicted_id": int(ids[j]),
                        **{
                            n: float(prob[j, k])
                            for k, n in enumerate(["Sad", "Happy", "Neutral"])
                        },
                    }
                )
            offset += len(y)
            if offset % 512 == 0:
                print(offset, "/", len(rows), flush=True)
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, pred, labels=[0, 1, 2], zero_division=0
    )
    report = {
        "split": args.split,
        "count": len(rows),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f1.mean()),
        "loss": loss / len(rows),
        "label_order": ["Sad", "Happy", "Neutral"],
        "confusion_axes": "rows=true, columns=predicted",
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
        "per_class": [
            {
                "name": n,
                "precision": float(p),
                "recall": float(r),
                "f1": float(f),
                "support": int(s),
            }
            for n, p, r, f, s in zip(
                ["Sad", "Happy", "Neutral"], precision, recall, f1, support
            )
        ],
        "dataset_manifest_sha256": digest,
        "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        "selection": "Frozen epoch 10; no optimization or selection performed.",
    }
    (args.output / "metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    with (args.output / "predictions.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=saved[0])
        w.writeheader()
        w.writerows(saved)
    print(json.dumps(report, indent=2))
