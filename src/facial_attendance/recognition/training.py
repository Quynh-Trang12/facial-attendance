"""Train the classification-based face embedding with resumable, traceable runs.

One run directory holds everything a run produces: its configuration, the validation
pair list, `last.pt` (resume state, saved every epoch because Colab sessions end
mid-run), `best.pt` (the deployable checkpoint of the selected epoch), `history.json` and
`manifest.json`. Selection follows the experiment settings: the epoch with the highest
cosine AUC on the seeded val_data pairs; top-1 on val_data is monitored only, and top-1
on test_data is measured once, on the selected checkpoint.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import platform
import subprocess
import time
import tomllib

import torch
from torch import nn

from .data_loader import ClassificationFaces, FaceLayout
from .model import (
    ARCHITECTURE,
    FaceEmbeddingNet,
    build_meta,
    eval_transform,
    load_checkpoint,
    save_checkpoint,
    train_transform,
)
from .validation_pairs import build_validation_pairs, read_pairs, write_pairs

REPOSITORY = Path(__file__).resolve().parents[3]
APPROACH = "classification"
# Settings that change speed only; a resumed run may use different values.
RUNTIME_ONLY = {"num_workers", "eval_batch_size"}


@dataclass(frozen=True)
class TrainConfig:
    epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    seed: int
    embedding_dim: int
    input_size: int
    eval_batch_size: int
    num_workers: int
    mixed_precision: bool
    max_steps_per_epoch: int | None = None  # smoke runs only; never a reported result


def load_config(file: Path, **overrides) -> TrainConfig:
    with Path(file).open("rb") as handle:
        content = tomllib.load(handle)
    if "training" not in content:
        raise ValueError(f"{file} has no [training] section")
    section = dict(content["training"])
    names = {f.name for f in fields(TrainConfig)}
    unknown = set(section) - names
    if unknown:
        raise ValueError(f"Unknown training settings: {sorted(unknown)}")
    section |= {k: v for k, v in overrides.items() if v is not None}
    return TrainConfig(**section)


def config_sha256(config: TrainConfig) -> str:
    return sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    return sha256(Path(path).read_bytes()).hexdigest()


def git_state() -> dict:
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(REPOSITORY), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {"git_commit": run("rev-parse", "HEAD"), "git_dirty": bool(run("status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}


def samples_sha256(layout: FaceLayout, datasets: dict[str, ClassificationFaces]) -> str:
    """Hash of the exact image paths and targets a run used, after exclusions."""
    digest = sha256()
    for split, dataset in datasets.items():
        for sample in dataset.samples:
            relative = sample.path.relative_to(layout.root).as_posix()
            digest.update(f"{split}\t{relative}\t{sample.target}\n".encode())
    return digest.hexdigest()


def _loader(dataset, batch_size, config, device, *, shuffle=False, generator=None):
    from torch.utils.data import DataLoader

    workers = config.num_workers
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=False,
        prefetch_factor=4 if workers else None,
    )


def _autocast(device, enabled):
    return torch.autocast(device.type, dtype=torch.float16, enabled=enabled)


@torch.no_grad()
def evaluate_split(model, dataset, config, device, amp) -> dict:
    """One pass: top-1, cross-entropy and the raw embeddings, in dataset order."""
    model.eval()
    correct, loss_sum, embeddings = 0, 0.0, []
    criterion = nn.CrossEntropyLoss(reduction="sum")
    for images, targets in _loader(dataset, config.eval_batch_size, config, device):
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        targets = targets.to(device, non_blocking=True)
        with _autocast(device, amp):
            embedding = model.embed(images)
            logits = model.head(embedding)
        loss_sum += criterion(logits.float(), targets).item()
        correct += (logits.argmax(1) == targets).sum().item()
        embeddings.append(embedding.float().cpu())
    count = len(dataset)
    return {
        "top1": correct / count,
        "loss": loss_sum / count,
        "images": count,
        "embeddings": torch.cat(embeddings),
    }


def pair_scores(embeddings: torch.Tensor, paths: list[Path], pairs) -> dict:
    """Cosine similarity and negated Euclidean distance on the same raw vectors."""
    from sklearn.metrics import roc_auc_score

    row = {path: index for index, path in enumerate(paths)}
    left = embeddings[[row[p.left] for p in pairs]]
    right = embeddings[[row[p.right] for p in pairs]]
    labels = [p.label for p in pairs]
    cosine = nn.functional.cosine_similarity(left, right).tolist()
    # Higher must mean "same person" for ROC, so distance is negated.
    negative_euclidean = (-(left - right).norm(dim=1)).tolist()
    return {
        "pairs": len(pairs),
        "cosine_auc": roc_auc_score(labels, cosine),
        "euclidean_auc": roc_auc_score(labels, negative_euclidean),
    }


def _write_json(file: Path, content):
    temporary = file.with_suffix(".tmp")
    temporary.write_text(json.dumps(content, indent=2, default=str) + "\n")
    temporary.replace(file)


def _new_run_id(seed: int, commit: str | None) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    return f"{stamp}-{APPROACH}-{(commit or 'nogit')[:7]}-s{seed}"


def _environment(device) -> dict:
    import torchvision

    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
    }


def train(
    config: TrainConfig,
    layout: FaceLayout,
    *,
    output_root: Path,
    exclusions_file: Path | None,
    device: torch.device,
    resume: Path | None = None,
    changes: list[dict] | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Train or resume one run and return its directory."""
    torch.manual_seed(config.seed)
    train_set = ClassificationFaces(layout, "train_data", transform=train_transform(config.input_size))
    mapping = train_set.class_to_idx
    evaluation = eval_transform(config.input_size)
    val_set = ClassificationFaces(layout, "val_data", class_to_idx=mapping, transform=evaluation)
    test_set = ClassificationFaces(layout, "test_data", class_to_idx=mapping, transform=evaluation)
    datasets = {"train_data": train_set, "val_data": val_set, "test_data": test_set}

    git = git_state()
    if resume is None:
        run_id = _new_run_id(config.seed, git["git_commit"])
        run_dir = Path(output_root) / run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", asdict(config))
        _write_json(run_dir / "changes.json", changes or [])
        write_pairs(run_dir / "val-pairs.txt", layout, build_validation_pairs(val_set, seed=config.seed))
    else:
        run_dir = Path(resume)
        run_id = run_dir.name
        stored = json.loads((run_dir / "config.json").read_text())
        differing = {
            k for k, v in asdict(config).items() if stored.get(k) != v and k not in RUNTIME_ONLY
        }
        if differing:
            raise ValueError(f"Resume must keep the run's settings; differs in {sorted(differing)}")
        config = TrainConfig(**(stored | {k: getattr(config, k) for k in RUNTIME_ONLY}))
        changes = json.loads((run_dir / "changes.json").read_text())
    pairs = read_pairs(run_dir / "val-pairs.txt", layout)
    val_paths = [sample.path for sample in val_set.samples]

    model = FaceEmbeddingNet(len(mapping), config.embedding_dim).to(device)
    model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    criterion = nn.CrossEntropyLoss()

    state = {"epoch": 0, "history": [], "best": None, "sessions": []}
    if resume is not None:
        saved = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        state = saved["state"]
        log(f"Resuming {run_id} after epoch {state['epoch']}")
    state["sessions"].append(
        {"started_utc": datetime.now(timezone.utc).isoformat(), "from_epoch": state["epoch"], **git}
    )

    base = {
        "run_id": run_id,
        "module": "face",
        "approach": APPROACH,
        "seed": config.seed,
        "architecture": {
            "name": ARCHITECTURE,
            "backbone": "torchvision resnet18, weights=None; conv1 3x3 stride 1; maxpool removed",
            "embedding": f"global average pool (512) -> linear {config.embedding_dim}; raw output, before the head",
            "head": f"linear {config.embedding_dim} -> {len(mapping)} identities, softmax via cross-entropy",
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_layers": "all; trained from scratch, no frozen layers, no pre-trained weights",
        },
        "loss": "cross-entropy over train_data identities",
        "optimizer": "SGD with momentum, cosine learning-rate decay per epoch",
        "augmentation": "horizontal flip; 4-pixel pad and random crop; train only",
        "config": asdict(config),
        "config_sha256": config_sha256(config),
        "changes": changes or [],
        "data": {
            "root": str(layout.root),
            "exclusions_file": str(exclusions_file) if exclusions_file else None,
            "exclusions_sha256": file_sha256(exclusions_file) if exclusions_file else None,
            "samples_sha256": samples_sha256(layout, datasets),
            "identity_mapping_sha256": sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
            "val_pairs_sha256": file_sha256(run_dir / "val-pairs.txt"),
            "official_pairs_sha256": file_sha256(layout.pairs_file),
            "images": {split: len(d) for split, d in datasets.items()},
            "identities": len(mapping),
        },
        "environment": _environment(device),
    }

    def manifest(status: str, extra: dict | None = None):
        best = state["best"]
        content = base | git | {
            "status": status,
            "sessions": state["sessions"],
            "epochs_completed": state["epoch"],
            "selection": None if best is None else {
                "metric": "cosine AUC on val_data pairs",
                "selected_epoch": best["epoch"],
                "value": best["cosine_auc"],
                "threshold": None,
                "threshold_source": "chosen by FAC-05 on the same validation pairs",
            },
            "output": {
                "model_id": run_id,
                "model_version": None if best is None else best["model_version"],
                "checkpoint": str(run_dir / "best.pt"),
            },
        } | (extra or {})
        _write_json(run_dir / "manifest.json", content)

    while state["epoch"] < config.epochs:
        epoch = state["epoch"] + 1
        model.train()
        generator = torch.Generator().manual_seed(config.seed + epoch)
        loader = _loader(train_set, config.batch_size, config, device, shuffle=True, generator=generator)
        started, seen, loss_sum, correct = time.time(), 0, 0.0, 0
        for step, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            with _autocast(device, amp):
                logits = model(images)
                loss = criterion(logits, targets)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            seen += len(targets)
            loss_sum += loss.item() * len(targets)
            correct += (logits.argmax(1) == targets).sum().item()
            if step % 200 == 0:
                log(f"epoch {epoch} step {step}/{len(loader)} loss {loss_sum / seen:.4f}")
            if config.max_steps_per_epoch and step >= config.max_steps_per_epoch:
                break
        if not torch.isfinite(torch.tensor(loss_sum)):
            raise RuntimeError(f"Training loss became non-finite in epoch {epoch}")
        learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()

        val = evaluate_split(model, val_set, config, device, amp)
        scores = pair_scores(val.pop("embeddings"), val_paths, pairs)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": loss_sum / seen,
            "train_top1": correct / seen,
            "train_images_seen": seen,
            "val_top1": val["top1"],
            "val_loss": val["loss"],
            "val_pairs_cosine_auc": scores["cosine_auc"],
            "val_pairs_euclidean_auc": scores["euclidean_auc"],
            "seconds": round(time.time() - started, 1),
        }
        state["history"].append(record)
        log(json.dumps(record))
        # Ties keep the earlier epoch, so selection does not drift toward longer training.
        if state["best"] is None or scores["cosine_auc"] > state["best"]["cosine_auc"]:
            meta = build_meta(model, model_id=run_id, approach=APPROACH)
            save_checkpoint(run_dir / "best.pt", model, meta)
            state["best"] = {"epoch": epoch, "cosine_auc": scores["cosine_auc"], "model_version": meta["model_version"]}
        state["epoch"] = epoch
        temporary = run_dir / "last.pt.tmp"
        torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "state": state,
            },
            temporary,
        )
        temporary.replace(run_dir / "last.pt")
        _write_json(run_dir / "history.json", state["history"])
        manifest("running")

    # The test split is scored once, on the selected checkpoint only.
    best_model, meta = load_checkpoint(run_dir / "best.pt", device)
    best_model = best_model.to(memory_format=torch.channels_last)
    test = evaluate_split(best_model, test_set, config, device, amp)
    val = evaluate_split(best_model, val_set, config, device, amp)
    val.pop("embeddings"), test.pop("embeddings")
    manifest(
        "complete",
        {
            "classification_accuracy": {
                "checkpoint": "best.pt",
                "val_data": {"top1": val["top1"], "images": val["images"]},
                "test_data": {"top1": test["top1"], "images": test["images"]},
            }
        },
    )
    log(f"Done: {meta['model_id']} selected epoch {state['best']['epoch']} val top-1 {val['top1']:.4f} test top-1 {test['top1']:.4f}")
    return run_dir
