import copy
import csv
import functools
import hashlib
import io
import json
import os
import random
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "4"
import numpy as np
import torch
from model_code import ResNet18, freeze_bn_statistics, set_training_stage
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

ROOT = Path(__file__).resolve().parent
DATA = Path(os.environ["EXPRESSION_RAW_ROOT"])
assert torch.cuda.is_available(), "Connect a GPU runtime before full training."
LABELS = ["sad", "happy", "neutral"]
torch.set_num_threads(4)
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def save_checkpoint(path, value):
    temp = path.with_suffix(".pt.tmp")
    torch.save(value, temp)
    temp.replace(path)


with (ROOT / "sample-manifest.csv").open(encoding="utf-8", newline="") as f:
    selected = list(csv.DictReader(f))
for r in selected:
    r["experiment_id"] = int(r["experiment_id"])
records = {s: [r for r in selected if r["output_split"] == s] for s in ["train", "val"]}
counts = {
    s: {n: sum(r["original_label"] == n for r in records[s]) for n in LABELS}
    for s in records
}
assert counts == {
    "train": {"sad": 3581, "happy": 5541, "neutral": 3781},
    "val": {"sad": 890, "happy": 1384, "neutral": 943},
}
assert len({r["pixel_sha256"] for r in selected}) == 16120
weights = Path("/content/resnet18-f37072fd.pth")
if not weights.exists():
    torch.hub.download_url_to_file(
        "https://download.pytorch.org/models/resnet18-f37072fd.pth", str(weights)
    )
assert (
    sha(weights) == "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
)
model = ResNet18(1000)
model.load_state_dict(
    torch.load(weights, map_location="cpu", weights_only=True), strict=True
)
model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 3))
device = torch.device("cuda")
model.to(device)
config = json.loads((ROOT / "planned-configuration.json").read_text(encoding="utf-8"))
config.update(
    device=str(device),
    torch=str(torch.__version__),
    numpy=str(np.__version__),
    gpu=torch.cuda.get_device_name(0),
    feature_cache="Not used: full-image training on GPU",
    mixed_precision=False,
)
assert config["sample_manifest_sha256"] == sha(ROOT / "sample-manifest.csv")
if (ROOT / "configuration.json").exists():
    assert (
        json.loads((ROOT / "configuration.json").read_text(encoding="utf-8")) == config
    )
write_json(ROOT / "configuration.json", config)
print("CONFIGURATION " + json.dumps(config), flush=True)
transform = transforms.Compose(
    [
        transforms.Grayscale(3),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


@functools.lru_cache(maxsize=20000)
def read_verified_image(relative, expected_hash):
    path = DATA / relative
    data = path.read_bytes()
    assert hashlib.sha256(data).hexdigest() == expected_hash, path
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        assert im.mode == "L" and im.size == (48, 48)
        return im.copy()


class RawFaces(Dataset):
    def __init__(self, rows, flips=False):
        self.rows = rows
        self.flips = flips

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        im = read_verified_image(r["output_path"], r["file_sha256"]).copy()
        if self.flips and bool(torch.rand(1) < 0.5):
            im = im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        x = transform(im)
        return x, int(r["experiment_id"])


def logits_from_features(x, stage):
    return model(x)


def evaluate(loader, stage, raw=False, save_rows=False):
    model.eval()
    truth = []
    pred = []
    loss = 0.0
    predictions = []
    offset = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x) if raw else logits_from_features(x, stage)
            loss += nn.functional.cross_entropy(out, y, reduction="sum").item()
            probabilities = out.softmax(1)
            ids = out.argmax(1)
            truth.extend(y.cpu().tolist())
            pred.extend(ids.cpu().tolist())
            if save_rows:
                for j in range(len(y)):
                    r = records["val"][offset + j]
                    predictions.append(
                        {
                            "output_path": r["output_path"],
                            "true_id": int(y[j]),
                            "predicted_id": int(ids[j]),
                            **{
                                n + "_probability": float(probabilities[j, k])
                                for k, n in enumerate(LABELS)
                            },
                        }
                    )
            offset += len(y)
    p, r, f, s = precision_recall_fscore_support(
        truth, pred, labels=[0, 1, 2], zero_division=0
    )
    metrics = {
        "count": len(truth),
        "accuracy": float(accuracy_score(truth, pred)),
        "macro_f1": float(f.mean()),
        "loss": loss / len(truth),
        "label_order": LABELS,
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
        "confusion_axes": "rows=true, columns=predicted",
        "per_class": [
            {
                "name": n,
                "precision": float(a),
                "recall": float(b),
                "f1": float(c),
                "support": int(d),
            }
            for n, a, b, c, d in zip(LABELS, p, r, f, s)
        ],
    }
    return metrics, predictions


def selection_key(record):
    m = record["validation"]
    return (m["per_class"][2]["f1"], m["macro_f1"], -m["loss"], -record["epoch"])


history = []
best = None
first_epoch = 1
resume = None
if (ROOT / "last.pt").exists():
    resume = torch.load(ROOT / "last.pt", map_location="cpu", weights_only=True)
    assert resume["config"] == config
    model.load_state_dict(resume["state_dict"])
    history = resume["history"]
    best = resume["best"]
    first_epoch = resume["epoch"] + 1
    torch.set_rng_state(resume["torch_rng"])
    torch.cuda.set_rng_state_all(resume["cuda_rng"])
    print(f"RESUME from completed epoch {resume['epoch']}", flush=True)
start = time.time()
for epoch in range(first_epoch, 11):
    stage = "head" if epoch <= 3 else "finetune"
    set_training_stage(model, stage)
    if epoch == first_epoch or epoch == 4:
        groups = [
            {
                "params": model.fc.parameters(),
                "lr": 0.001 if stage == "head" else 0.0003,
            }
        ]
        if stage == "finetune":
            groups.append({"params": model.layer4.parameters(), "lr": 0.0001})
        optimizer = torch.optim.Adam(groups, weight_decay=0.0001)
        if resume is not None and (resume["epoch"] <= 3) == (stage == "head"):
            optimizer.load_state_dict(resume["optimizer"])
    train = DataLoader(
        RawFaces(records["train"], flips=True),
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    val = DataLoader(
        RawFaces(records["val"]), batch_size=32, shuffle=False, num_workers=0
    )
    model.train()
    freeze_bn_statistics(model)
    optimizer.zero_grad(set_to_none=True)
    train_loss = 0.0
    group_count = 0
    updates = 0
    for i, (x, y) in enumerate(train):
        x, y = x.to(device), y.to(device)
        out = logits_from_features(x, stage)
        loss = nn.functional.cross_entropy(out, y, reduction="sum")
        loss.backward()
        train_loss += loss.item()
        group_count += len(y)
        if (i + 1) % 8 == 0 or i + 1 == len(train):
            for p in model.parameters():
                if p.grad is not None:
                    p.grad.div_(group_count)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            group_count = 0
            updates += 1
        if (i + 1) % 128 == 0:
            print(f"TRAIN epoch {epoch}: batch {i + 1}/{len(train)}", flush=True)
    metrics, _ = evaluate(val, stage)
    record = {
        "epoch": epoch,
        "stage": stage,
        "train_loss": train_loss / len(records["train"]),
        "validation": metrics,
        "optimizer_updates": updates,
        "elapsed_seconds_this_process": time.time() - start,
    }
    history.append(record)
    if best is None or selection_key(record) > selection_key(best):
        best = copy.deepcopy(record)
        save_checkpoint(
            ROOT / "best.pt",
            {"state_dict": model.state_dict(), "config": config, "selected": best},
        )
    save_checkpoint(
        ROOT / "last.pt",
        {
            "state_dict": model.state_dict(),
            "config": config,
            "epoch": epoch,
            "history": history,
            "best": best,
            "optimizer": optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
        },
    )
    write_json(ROOT / "history.json", history)
    print(
        f"EPOCH {epoch}/10: macro F1={metrics['macro_f1']:.4f}; Neutral F1={metrics['per_class'][2]['f1']:.4f}; loss={metrics['loss']:.4f}",
        flush=True,
    )
payload = torch.load(ROOT / "best.pt", map_location="cpu", weights_only=True)
model.load_state_dict(payload["state_dict"])
best = payload["selected"]
read_verified_image.cache_clear()
raw_loader = DataLoader(
    RawFaces(records["val"]), batch_size=32, shuffle=False, num_workers=0
)
raw, predictions = evaluate(raw_loader, best["stage"], raw=True, save_rows=True)
expected = best["validation"]
assert (
    raw["confusion_matrix"] == expected["confusion_matrix"]
    and abs(raw["macro_f1"] - expected["macro_f1"]) < 1e-12
)
assert abs(raw["loss"] - expected["loss"]) < 1e-5
with (ROOT / "validation-predictions.csv").open("w", encoding="utf-8", newline="") as f:
    w = csv.DictWriter(f, fieldnames=predictions[0])
    w.writeheader()
    w.writerows(predictions)
result = {
    "configuration": config,
    "selected": best,
    "original_image_validation": raw,
    "checkpoint_reload_verified": True,
    "validation_predictions_reproduced_after_reload": True,
    "checkpoint_sha256": sha(ROOT / "best.pt"),
    "validation_predictions_sha256": sha(ROOT / "validation-predictions.csv"),
    "test_accessed": False,
    "limitations": "One seed. Validation includes images used in pilot comparisons. No new held-out assessment or real attendance evaluation.",
}
write_json(ROOT / "result.json", result)
print("COMPLETE " + json.dumps(result), flush=True)
