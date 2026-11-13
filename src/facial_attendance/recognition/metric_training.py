"""Train the metric-learning face embedding with triplet loss, resumable and traceable.

The run directory has the same layout as a classification run: `config.json`,
`changes.json`, the seeded `val-pairs.txt`, `last.pt` (resume state, every epoch),
`best.pt` (deployable checkpoint of the selected epoch), `history.json` and
`manifest.json`. The network, preprocessing, validation pairs, scoring and run record
come from the classification training code, so the two approaches differ only in the
objective and the batches.

Batches are P identities × K images from `IdentityBatchSampler`. Each batch's triplets
are mined online and checked by `triplet_loss`; the per-epoch totals of that check go
into the history. The best epoch has the highest cosine AUC on the val_data pairs.
"""

from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import time
import tomllib

import torch

from .data_loader import ClassificationFaces, FaceLayout
from .metric import triplet_loss
from .model import ARCHITECTURE, FaceEmbeddingNet, build_meta, eval_transform, save_checkpoint, train_transform
from .sampling import IdentityBatchSampler
from .training import (
    _autocast,
    _environment,
    _write_json,
    config_sha256,
    file_sha256,
    git_state,
    pair_scores,
    samples_sha256,
)
from .validation_pairs import build_validation_pairs, read_pairs, write_pairs

APPROACH = "metric"
# Settings that change speed only; a resumed run may use different values.
RUNTIME_ONLY = {"num_workers", "eval_batch_size"}
COUNTS = ("anchor_positive_pairs", "semi_hard", "fallback", "active")
MEANS = ("mean_positive_distance", "mean_negative_distance", "embedding_std")


@dataclass(frozen=True)
class MetricConfig:
    epochs: int
    identities_per_batch: int
    images_per_identity: int
    margin: float
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

    @property
    def batch_size(self) -> int:
        return self.identities_per_batch * self.images_per_identity


def load_metric_config(file: Path, **overrides) -> MetricConfig:
    with Path(file).open("rb") as handle:
        content = tomllib.load(handle)
    if "metric_training" not in content:
        raise ValueError(f"{file} has no [metric_training] section")
    section = dict(content["metric_training"])
    unknown = set(section) - {f.name for f in fields(MetricConfig)}
    if unknown:
        raise ValueError(f"Unknown metric training settings: {sorted(unknown)}")
    section |= {k: v for k, v in overrides.items() if v is not None}
    return MetricConfig(**section)


@torch.no_grad()
def embed_split(model, dataset, config: MetricConfig, device, amp) -> torch.Tensor:
    """Raw embeddings of a split in dataset order; the network has no head to score."""
    from torch.utils.data import DataLoader

    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=config.eval_batch_size,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    chunks = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        with _autocast(device, amp):
            chunks.append(model.embed(images).float().cpu())
    return torch.cat(chunks)


def load_warm_start(model: FaceEmbeddingNet, checkpoint: Path) -> dict:
    """Copy the backbone and embedding layer of a classification checkpoint; drop its head."""
    content = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    meta = content["meta"]
    if meta.get("architecture") != ARCHITECTURE or meta.get("embedding_dim") != model.embedding_dim:
        raise ValueError("Warm start needs the same architecture and embedding size")
    weights = {k: v for k, v in content["state_dict"].items() if not k.startswith("head.")}
    model.load_state_dict(weights)  # strict: every backbone and embedding weight must be present
    return {"checkpoint": str(checkpoint), "model_id": meta["model_id"], "model_version": meta["model_version"]}


def _new_run_id(seed: int, commit: str | None, warm: bool) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    approach = f"{APPROACH}-warm" if warm else APPROACH
    return f"{stamp}-{approach}-{(commit or 'nogit')[:7]}-s{seed}"


def train_metric(
    config: MetricConfig,
    layout: FaceLayout,
    *,
    output_root: Path,
    exclusions_file: Path | None,
    device: torch.device,
    resume: Path | None = None,
    warm_start: Path | None = None,
    changes: list[dict] | None = None,
    log: Callable[[str], None] = print,
) -> Path:
    """Train or resume one triplet run and return its directory."""
    if resume is not None and warm_start is not None:
        raise ValueError("A warm start begins a new run; resume continues an existing one")
    torch.manual_seed(config.seed)
    train_set = ClassificationFaces(layout, "train_data", transform=train_transform(config.input_size))
    mapping = train_set.class_to_idx
    val_set = ClassificationFaces(
        layout, "val_data", class_to_idx=mapping, transform=eval_transform(config.input_size)
    )
    sampler = IdentityBatchSampler(
        train_set,
        identities_per_batch=config.identities_per_batch,
        images_per_identity=config.images_per_identity,
        seed=config.seed,
    )

    git = git_state()
    if resume is None:
        run_dir = Path(output_root) / _new_run_id(config.seed, git["git_commit"], warm_start is not None)
        run_dir.mkdir(parents=True, exist_ok=False)
        _write_json(run_dir / "config.json", asdict(config))
        _write_json(run_dir / "changes.json", changes or [])
        write_pairs(run_dir / "val-pairs.txt", layout, build_validation_pairs(val_set, seed=config.seed))
    else:
        run_dir = Path(resume)
        stored = json.loads((run_dir / "config.json").read_text())
        differing = {k for k, v in asdict(config).items() if stored.get(k) != v and k not in RUNTIME_ONLY}
        if differing:
            raise ValueError(f"Resume must keep the run's settings; differs in {sorted(differing)}")
        config = MetricConfig(**(stored | {k: getattr(config, k) for k in RUNTIME_ONLY}))
        changes = json.loads((run_dir / "changes.json").read_text())
    run_id = run_dir.name
    pairs = read_pairs(run_dir / "val-pairs.txt", layout)
    val_paths = [sample.path for sample in val_set.samples]

    # No identity head: the triplet objective trains the embedding directly.
    model = FaceEmbeddingNet(None, config.embedding_dim)
    initialisation = load_warm_start(model, warm_start) if warm_start is not None else None
    model = model.to(device).to(memory_format=torch.channels_last)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    amp = config.mixed_precision and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    state = {"epoch": 0, "history": [], "best": None, "sessions": [], "initialisation": initialisation}
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

    trainable = (
        "all; trained from scratch, no frozen layers, no pre-trained weights"
        if state["initialisation"] is None
        else "all; backbone and embedding start from the classification checkpoint named in initialisation, no frozen layers"
    )
    base = {
        "run_id": run_id,
        "module": "face",
        "approach": APPROACH,
        "seed": config.seed,
        "architecture": {
            "name": ARCHITECTURE,
            "backbone": "torchvision resnet18, weights=None; conv1 3x3 stride 1; maxpool removed",
            "embedding": f"global average pool (512) -> linear {config.embedding_dim}; raw output",
            "head": "none; the triplet loss trains the embedding directly",
            "parameters": sum(p.numel() for p in model.parameters()),
            "trainable_layers": trainable,
        },
        "initialisation": state["initialisation"],
        "loss": (
            f"triplet loss, margin {config.margin}, squared Euclidean distance on L2-normalized "
            "embeddings; every same-identity pair in a batch is an anchor-positive pair; "
            "semi-hard negative mining (closest negative farther than the positive, else the "
            "farthest negative); positive and negative identities checked on every batch"
        ),
        "batches": (
            f"{config.identities_per_batch} identities x {config.images_per_identity} images "
            f"from IdentityBatchSampler, {len(sampler)} batches per epoch"
        ),
        "optimizer": "SGD with momentum, cosine learning-rate decay per epoch",
        "augmentation": "horizontal flip; 4-pixel pad and random crop; train only",
        "config": asdict(config) | {"batch_size": config.batch_size},
        "config_sha256": config_sha256(config),
        "changes": changes or [],
        "data": {
            "root": str(layout.root),
            "exclusions_file": str(exclusions_file) if exclusions_file else None,
            "exclusions_sha256": file_sha256(exclusions_file) if exclusions_file else None,
            "samples_sha256": samples_sha256(layout, {"train_data": train_set, "val_data": val_set}),
            "identity_mapping_sha256": sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest(),
            "val_pairs_sha256": file_sha256(run_dir / "val-pairs.txt"),
            "official_pairs_sha256": file_sha256(layout.pairs_file),
            "images": {"train_data": len(train_set), "val_data": len(val_set)},
            "identities": len(mapping),
        },
        "environment": _environment(device),
    }

    def manifest(status: str):
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
        }
        _write_json(run_dir / "manifest.json", content)

    from torch.utils.data import DataLoader

    while state["epoch"] < config.epochs:
        epoch = state["epoch"] + 1
        model.train()
        sampler.set_epoch(epoch)
        loader = DataLoader(
            train_set,
            batch_sampler=sampler,
            num_workers=config.num_workers,
            pin_memory=device.type == "cuda",
            prefetch_factor=4 if config.num_workers else None,
        )
        started, steps, loss_sum = time.time(), 0, 0.0
        totals = dict.fromkeys(COUNTS, 0) | dict.fromkeys(MEANS, 0.0)
        for step, (images, targets) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            targets = targets.to(device, non_blocking=True)
            with _autocast(device, amp):
                embeddings = model.embed(images)
            # Raises if a positive or negative has the wrong identity.
            loss, stats = triplet_loss(embeddings, targets, margin=config.margin)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            steps += 1
            loss_sum += loss.item()
            for key in totals:
                totals[key] += stats[key]
            if step % 200 == 0:
                log(f"epoch {epoch} step {step}/{len(loader)} loss {loss_sum / steps:.4f}")
            if config.max_steps_per_epoch and step >= config.max_steps_per_epoch:
                break
        if not torch.isfinite(torch.tensor(loss_sum)):
            raise RuntimeError(f"Training loss became non-finite in epoch {epoch}")
        learning_rate = scheduler.get_last_lr()[0]
        scheduler.step()

        scores = pair_scores(embed_split(model, val_set, config, device, amp), val_paths, pairs)
        record = {
            "epoch": epoch,
            "learning_rate": learning_rate,
            "train_loss": loss_sum / steps,
            "train_batches": steps,
            "train_images_seen": steps * config.batch_size,
            "triplets": {
                "relationship_check": f"passed on all {steps} batches",
                **{key: totals[key] for key in COUNTS},
                **{key: totals[key] / steps for key in MEANS},
            },
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

    manifest("complete")
    best = state["best"]
    log(f"Done: {run_id} selected epoch {best['epoch']} val-pair cosine AUC {best['cosine_auc']:.4f}")
    return run_dir
