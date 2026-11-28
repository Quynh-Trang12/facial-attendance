"""Run matched multitask epochs and verify the selected binary export."""

import json
import math
import time

import torch

from facial_attendance.liveness.dataset_files import file_sha256, write_json
from facial_attendance.liveness.datasets.image_augmentation import (
    CaptureAugmentation,
    apply_device_color,
)
from facial_attendance.liveness.datasets.semantic_dataset import MultitaskImages
from facial_attendance.liveness.models.binary_model import (
    RESNET_ARCHITECTURE,
    build_model,
    load_checkpoint,
    state_version,
)
from facial_attendance.liveness.models.multitask_model import (
    MultitaskLivenessModel,
    active_auxiliary_tasks,
    multitask_loss,
)
from facial_attendance.liveness.training.batching import make_loader, seed_training
from facial_attendance.liveness.training.checkpoints import (
    checkpoint_metadata,
    save_selected,
)
from facial_attendance.liveness.training.validation import evaluate_validation


def gradients_are_finite(parameters):
    """Reduce on-device before one host check, without changing gradients."""
    checks = [
        torch.isfinite(parameter.grad).all()
        for parameter in parameters
        if parameter.grad is not None
    ]
    return not checks or bool(torch.stack(checks).all())


def train_multitask_epoch(
    model,
    dataset,
    optimizer,
    config,
    *,
    epoch,
    auxiliary,
    loader=None,
    progress=None,
    on_batch=None,
    auxiliary_tasks=None,
):
    """Update only enabled tasks and return losses, coverage and gradient evidence.

    Resume progress must correspond to the supplied remaining-row loader. Missing
    or extra samples/optimizer steps fail the final coverage checks.
    """
    model.train()
    task_names = ("face_attributes", "spoof_type", "illumination")
    if model.geometry:
        task_names += ("depth_map", "reflection_map")
    selected_tasks = active_auxiliary_tasks(
        auxiliary=auxiliary, geometry=model.geometry, tasks=auxiliary_tasks
    )
    loss_names = ("binary", *selected_tasks)
    totals = {name: 0.0 for name in (*loss_names, "total")}
    samples, updates, proof = 0, 0, None

    if progress is not None:
        samples, updates, proof = (
            progress["samples"],
            progress["optimizer_updates"],
            progress["gradient_proof"],
        )
        # Legacy control checkpoints also recorded unused auxiliary diagnostics.
        # Restore only losses still computed, without changing binary totals.
        totals.update({name: progress["totals"][name] for name in totals})

    started = last_progress = time.monotonic()
    batches = (
        loader
        if loader is not None
        else make_loader(dataset, config, training=True, epoch=epoch)
    )

    for images, targets in batches:
        color_order = targets.get("_color_order")
        images = images.cuda(non_blocking=True)
        # The control consumes only binary labels. Keep augmentation order on
        # CPU to group rows without synchronizing CUDA for dynamic indices.
        targets = {
            key: value.cuda(non_blocking=True)
            for key, value in targets.items()
            if key != "_color_order"
            and key in ("binary", "_color_factors", "_color_hue", *selected_tasks)
        }
        if color_order is not None:
            images = apply_device_color(
                images,
                color_order,
                targets.pop("_color_factors"),
                targets.pop("_color_hue"),
            )

        optimizer.zero_grad(set_to_none=True)
        # FP32 avoids unequal skipped AMP updates in this matched short trial.
        outputs = model.forward_tasks(images, auxiliary=auxiliary, tasks=selected_tasks)
        loss, parts = multitask_loss(
            outputs, targets, auxiliary=auxiliary, tasks=selected_tasks
        )
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite multitask loss")

        if epoch == 1 and updates == 0 and auxiliary:
            weighted_auxiliary = loss - parts["binary"]
            shared = model.binary_model.backbone.conv1.weight
            gradient = torch.autograd.grad(
                weighted_auxiliary, shared, retain_graph=True
            )[0]
            proof = {"auxiliary_only_backbone_gradient_norm": float(gradient.norm())}
            if (
                not math.isfinite(proof["auxiliary_only_backbone_gradient_norm"])
                or proof["auxiliary_only_backbone_gradient_norm"] <= 0
            ):
                raise RuntimeError("Auxiliary losses do not reach the backbone")

        loss.backward()
        if epoch == 1 and updates == 0:
            proof = proof or {}
            for name in task_names:
                gradient = getattr(model, name).weight.grad
                norm = float(gradient.norm()) if gradient is not None else 0.0
                proof[f"{name}_head_gradient_norm"] = norm
                if name in selected_tasks and (not math.isfinite(norm) or norm <= 0):
                    raise RuntimeError(f"Auxiliary head has no finite gradient: {name}")
                if name not in selected_tasks and gradient is not None:
                    raise RuntimeError(
                        "Disabled auxiliary head received a gradient: " + name
                    )
        if not gradients_are_finite(model.parameters()):
            raise RuntimeError("Nonfinite training gradient")

        optimizer.step()
        updates += 1
        samples += len(images)

        # One host transfer retains the same FP32 scalar values and Python
        # accumulation order, instead of synchronizing once per loss component.
        names = list(parts)
        values = (
            torch.stack([parts[name].detach() for name in names] + [loss.detach()])
            .cpu()
            .tolist()
        )
        for name, value in zip(names, values):
            totals[name] += value * len(images)
        totals["total"] += values[-1] * len(images)

        if time.monotonic() - last_progress >= 30:
            print(
                json.dumps(
                    {
                        "phase": "training",
                        "epoch": epoch,
                        "samples": samples,
                        "total_samples": len(dataset),
                        "optimizer_updates": updates,
                        "mean_loss": totals["total"] / samples,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
            last_progress = time.monotonic()

        if on_batch is not None:
            on_batch(
                {
                    "samples": samples,
                    "optimizer_updates": updates,
                    "totals": dict(totals),
                    "gradient_proof": proof,
                }
            )

    if samples != len(dataset) or updates != math.ceil(
        len(dataset) / config.batch_size
    ):
        raise RuntimeError("Training did not cover the declared sample/update budget")
    return {
        "samples": samples,
        "optimizer_updates": updates,
        "losses": {name: value / samples for name, value in totals.items()},
        "gradient_proof": proof,
    }


def train_multitask_comparison(
    states,
    train,
    train_targets,
    train_indices,
    clean,
    stress,
    config,
    run,
    output,
    started,
):
    """Update matched configurations, select on validation and verify exports.

    The experiment supplies identical initialization and cohorts. This loop never
    opens test data or changes the predeclared auxiliary-task controls.
    """
    stop_reason = "epoch cap"
    for epoch, indices in enumerate(train_indices, 1):
        pair_started = time.monotonic()
        for name, state in states.items():
            seed_training(config.seed + epoch)
            dataset = MultitaskImages(
                train, indices, train_targets, CaptureAugmentation(RESNET_ARCHITECTURE)
            )
            row = train_multitask_epoch(
                state["model"],
                dataset,
                state["optimizer"],
                config,
                epoch=epoch,
                auxiliary=state["auxiliary"],
            )
            state["updates"] += row["optimizer_updates"]
            key, evaluation = evaluate_validation(state["model"], clean, stress, config)
            row.update(
                epoch=epoch,
                validation=evaluation,
                elapsed_seconds=time.monotonic() - started,
            )
            state["history"].append(row)
            if key < state["best_key"]:
                state.update(
                    best_key=key, best_epoch=epoch, best_validation=evaluation, stale=0
                )
                meta = checkpoint_metadata(
                    state["model"],
                    run,
                    epoch=epoch,
                    updates=state["updates"],
                    auxiliary=state["auxiliary"],
                    threshold=evaluation["pooled_views"]["threshold"],
                )
                save_selected(output / name, state["model"], meta)
            else:
                state["stale"] += 1
            (output / name / "training-history.json").write_text(
                json.dumps(state["history"], indent=2, allow_nan=False) + "\n",
                encoding="utf-8",
            )
            print(
                json.dumps(
                    {
                        "arm": name,
                        "epoch": epoch,
                        "updates": state["updates"],
                        "validation_ACER": key[0],
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )
        if len({state["updates"] for state in states.values()}) != 1:
            raise RuntimeError("Matched actual optimizer updates differ")
        if all(state["stale"] >= config.patience for state in states.values()):
            stop_reason = "paired early stopping"
            break
        # Reserve one measured paired epoch rather than starting work that is
        # likely to exceed the declared budget. The worker adds a hard timeout.
        if (
            time.monotonic() - started + (time.monotonic() - pair_started)
            >= config.max_seconds
        ):
            stop_reason = "resource budget"
            break

    results = {}
    for name, state in states.items():
        model, meta = load_checkpoint(output / name / "liveness-inference.pt", "cuda")
        full = torch.load(
            output / name / "multitask-training.pt",
            map_location="cpu",
            weights_only=True,
        )
        trained = MultitaskLivenessModel(build_model(RESNET_ARCHITECTURE)).cuda().eval()
        trained.load_state_dict(full["state_dict"])
        images = torch.stack([clean[index][0] for index in range(8)]).cuda()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            difference = float((trained(images) - model(images)).abs().max())
        if difference != 0:
            raise RuntimeError(
                "Exported binary inference differs from the trained model"
            )
        auxiliary_version = state_version(
            {
                k: v
                for k, v in state["model"].state_dict().items()
                if not k.startswith("binary_model.")
            }
        )
        if (auxiliary_version != state["initial_auxiliary_version"]) != state[
            "auxiliary"
        ]:
            raise RuntimeError("Auxiliary head update/control invariant failed")
        results[name] = {
            "best_epoch": state["best_epoch"],
            "actual_optimizer_updates": state["updates"],
            "selected_optimizer_updates": meta["optimizer_updates"],
            "validation": state["best_validation"],
            "selection_key": list(state["best_key"]),
            "binary_export_max_logit_difference": difference,
            "auxiliary_heads_changed": state["auxiliary"],
            "checkpoint_sha256": file_sha256(output / name / "liveness-inference.pt"),
        }
        del model, trained

    chosen = min(states, key=lambda name: states[name]["best_key"])
    summary = {
        "chosen": chosen,
        "arms": results,
        "stop_reason": stop_reason,
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
        "test_or_physical_used_for_selection": False,
    }

    write_json(output / "validation-results.json", summary)
    print(json.dumps(summary), flush=True)
    return summary
