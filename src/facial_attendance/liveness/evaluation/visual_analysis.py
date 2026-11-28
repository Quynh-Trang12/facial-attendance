"""Render saved liveness evidence; no inference, selection or training side effects."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

VISUAL_ARMS = (
    "binary-classification",
    "binary-semantic",
    "binary-geometric",
    "aenet-full",
)


VISUAL_LABELS = ("Live", "Spoof")


VISUAL_COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")


VISUAL_SOURCE_TYPES = (
    "Live",
    "Photo",
    "Poster",
    "A4",
    "Face mask",
    "Upper body mask",
    "Region mask",
    "PC",
    "Pad",
    "Phone",
    "3D mask",
)


VISUAL_ILLUMINATION = ("Live", "Normal", "Strong", "Back", "Dark")


VISUAL_ATTRIBUTES = (
    "5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs "
    "Big_Lips Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair "
    "Bushy_Eyebrows Chubby Double_Chin Eyeglasses Goatee Gray_Hair "
    "Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache "
    "Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline "
    "Rosy_Cheeks Sideburns Smiling Straight_Hair Wavy_Hair Wearing_Earrings "
    "Wearing_Hat Wearing_Lipstick Wearing_Necklace Wearing_Necktie Young"
).split()


VISUAL_SHORT_NAMES = {
    "binary-classification": "Baseline",
    "binary-semantic": "+ Semantic",
    "binary-geometric": "+ Geometry",
    "aenet-full": "AENet full",
}


def visual_confusion(ax, labels, scores, threshold, title):
    matrix = confusion_matrix(labels, scores >= threshold, labels=[0, 1])
    ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(matrix.max(), 1))
    for (row, col), count in np.ndenumerate(matrix):
        ax.text(
            col,
            row,
            f"{count:,}",
            ha="center",
            va="center",
            color="white" if count > matrix.max() / 2 else "black",
        )
    ax.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=VISUAL_LABELS,
        yticklabels=VISUAL_LABELS,
        xlabel="Predicted class",
        ylabel="Dataset label",
        title=title,
    )


def visual_roc(ax, labels, scores, threshold, label, color):
    fpr, tpr, _ = roc_curve(labels, scores)
    ax.plot(
        fpr, tpr, color=color, label=f"{label}: AUC {roc_auc_score(labels, scores):.4f}"
    )
    predicted = scores >= threshold
    ax.scatter(
        predicted[labels == 0].mean(),
        predicted[labels == 1].mean(),
        color=color,
        s=35,
        zorder=3,
    )
    ax.set(
        xlabel="Live rejected as spoof (FPR)",
        ylabel="Spoof detected (TPR)",
        xlim=(0, 1),
        ylim=(0, 1.01),
    )
    ax.grid(alpha=0.2)


def visual_histogram(ax, labels, scores, threshold, title):
    bins = np.linspace(0, 1, 41)
    for label, color in enumerate(VISUAL_COLORS[:2]):
        values = scores[labels == label]
        ax.hist(
            values,
            bins=bins,
            weights=np.full(len(values), 100 / len(values)),
            histtype="step",
            linewidth=1.7,
            color=color,
            label=VISUAL_LABELS[label],
        )
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        label="_nolegend_",
    )
    ax.set(
        xlabel="Spoof score",
        ylabel="Images within true class (%)",
        title=f"{title} · threshold {threshold:.3f}",
    )
    ax.legend(
        fontsize=10,
        frameon=True,
        facecolor="white",
        framealpha=1,
        edgecolor="none",
        loc="center right",
    )


def visual_error_scores(ax, labels, scores, threshold, title):
    """Combine the two error classes without letting bins cross the threshold."""
    bins = np.unique(np.r_[np.linspace(0, 1, 21), threshold])
    totals = []
    for label, direction in enumerate(("Live → Spoof", "Spoof → Live")):
        values = scores[labels == label]
        errors = (
            values[values >= threshold] if label == 0 else values[values < threshold]
        )
        counts, edges = np.histogram(errors, bins=bins)
        totals.append(len(errors))
        # Zero-height bars must not cover the other class in shared bins.
        occupied = counts > 0
        ax.bar(
            edges[:-1][occupied],
            counts[occupied],
            width=np.diff(edges)[occupied],
            align="edge",
            color=VISUAL_COLORS[label],
            edgecolor="white",
            linewidth=0.4,
            label=direction,
        )
    ax.axvline(
        threshold, color="black", linestyle="--", linewidth=1.2, label="Threshold"
    )
    ax.set(
        title=f"{title} · {sum(totals):,} errors",
        xlabel="Spoof score",
        ylabel="Images",
        xlim=(0, 1),
        xticks=[0, 0.5, 1],
    )
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
    ax.grid(axis="y", alpha=0.15)
    if not sum(totals):
        ax.text(0.5, 0.5, "No errors", ha="center", transform=ax.transAxes)


def visual_test_error_scores(labels, scores, threshold):
    """Use independent count axes for strongly imbalanced test error classes."""
    fig, axes = plt.subplots(1, 2, figsize=(8, 2.7), constrained_layout=True)
    bins = np.unique(np.r_[np.linspace(0, 1, 21), threshold])
    for label, ax in enumerate(axes):
        values = scores[labels == label]
        errors = (
            values[values >= threshold] if label == 0 else values[values < threshold]
        )
        counts, edges = np.histogram(errors, bins=bins)
        ax.bar(
            edges[:-1],
            counts,
            width=np.diff(edges),
            align="edge",
            color=VISUAL_COLORS[label],
            edgecolor="white",
            linewidth=0.4,
            label="Live → Spoof" if label == 0 else "Spoof → Live",
        )
        ax.axvline(
            threshold, color="black", linestyle="--", linewidth=1.2, label="Threshold"
        )
        ax.set(
            title=f"{len(errors):,} errors",
            xlabel="Spoof score",
            ylabel="Images",
            xlim=(0, 1),
            xticks=[0, 0.5, 1],
            ylim=(0, max(1, counts.max() * 1.15)),
        )
        ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=4))
        ax.grid(axis="y", alpha=0.15)
    handles = [axes[0].patches[0], axes[1].patches[0], axes[0].lines[0]]
    fig.legend(
        handles,
        ["Live → Spoof", "Spoof → Live", "Threshold"],
        loc="outside upper center",
        ncol=3,
        frameon=False,
    )
    return fig


def visual_learning_curves(arm, model):
    history = model["history"]
    epochs = np.array([row["epoch"] for row in history])
    best = model["result"]["selected_epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2), constrained_layout=True)
    axes[0].plot(
        epochs,
        [r["losses"]["binary"] for r in history],
        label="Training",
        color=VISUAL_COLORS[0],
        linewidth=2,
    )
    axes[0].plot(
        epochs,
        [r["validation_BCE"] for r in history],
        label="Validation",
        color=VISUAL_COLORS[1],
        linewidth=2,
    )
    axes[0].set(
        yscale="log",
        ylabel="Binary loss (log scale)",
        title="Training versus validation",
    )
    axes[0].legend(frameon=False)
    acer = np.array([100 * r["validation"]["ACER"] for r in history])
    axes[1].plot(epochs, acer, color=VISUAL_COLORS[0], linewidth=2)
    axes[1].scatter(best, acer[best - 1], color="black", s=35, zorder=3)
    axes[1].set(
        ylabel="Validation ACER (%)", title="Classification error", ylim=(0, None)
    )
    for ax in axes:
        ax.axvline(best, color="0.4", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        f"{VISUAL_SHORT_NAMES[arm]} — selected epoch {best} / budget {len(history)}"
    )
    plt.show()


def visual_example_decisions(number, group, index, reference, models, images):
    row = reference[index]
    fig, (image_ax, table_ax) = plt.subplots(
        1,
        2,
        figsize=(8.5, 3.4),
        gridspec_kw={"width_ratios": [1, 2]},
        constrained_layout=True,
    )
    if index in images:
        image_ax.imshow(images[index])
    else:
        image_ax.text(0.5, 0.5, "Source crop unavailable", ha="center")
    image_ax.axis("off")
    image_ax.set_title(
        f"Dataset label: {VISUAL_LABELS[int(row['binary_label'])]}\n"
        f"Source: {VISUAL_SOURCE_TYPES[int(row['source_type'])]}"
    )
    table_ax.axis("off")
    values = []
    for arm in VISUAL_ARMS:
        model = models[arm]
        score = model["scores"][index]
        threshold = model["result"]["threshold"]
        values.append(
            [
                VISUAL_SHORT_NAMES[arm],
                VISUAL_LABELS[int(score >= threshold)],
                f"{score:.4f}",
                f"{threshold:.4f}",
            ]
        )
    table = table_ax.table(
        cellText=values,
        colLabels=["Model", "Decision", "Spoof score", "Threshold"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        colWidths=[0.28, 0.22, 0.25, 0.25],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.9)
    for position, arm in enumerate(VISUAL_ARMS, start=1):
        color = VISUAL_COLORS[0] if models[arm]["correct"][index] else VISUAL_COLORS[1]
        table[position, 1].get_text().set_color(color)
    fig.suptitle(f"Example {number} · {group}", fontsize=12, weight="bold")
    fig.supxlabel(row["relative_path"], fontsize=9)
    plt.show()


def visual_semantic_rows(example):
    semantic_rows = []
    for arm in ("binary-semantic", "aenet-full"):
        result = example["models"][arm]
        source = np.asarray(result["spoof_type"])
        illumination = np.asarray(result["illumination"])
        semantic_rows.extend(
            [
                [
                    VISUAL_SHORT_NAMES[arm],
                    "Source type",
                    VISUAL_SOURCE_TYPES[source.argmax()],
                    VISUAL_SOURCE_TYPES[example["true_source_type"]],
                ],
                [
                    VISUAL_SHORT_NAMES[arm],
                    "Illumination",
                    VISUAL_ILLUMINATION[illumination.argmax()],
                    VISUAL_ILLUMINATION[example["true_illumination"]],
                ],
            ]
        )
        attributes = np.asarray(result["face_attributes"])
        for attribute in np.argsort(-attributes, kind="stable")[:3]:
            semantic_rows.append(
                [
                    VISUAL_SHORT_NAMES[arm],
                    VISUAL_ATTRIBUTES[attribute].replace("_", " "),
                    f"{'Present' if attributes[attribute] >= 0.5 else 'Absent'} ({attributes[attribute]:.3f})",
                    "Present"
                    if example["true_face_attributes"][attribute]
                    else "Absent",
                ]
            )
    return semantic_rows


def visual_example_geometry(number, example):
    fig, axes = plt.subplots(2, 3, figsize=(7.5, 4.8), constrained_layout=True)
    columns = [
        (
            "Teacher pseudo-target",
            np.asarray(example["pseudo_depth"]),
            np.asarray(example["pseudo_reflection"]),
        ),
        *[
            (
                VISUAL_SHORT_NAMES[arm],
                np.asarray(example["models"][arm]["depth_map"]),
                np.asarray(example["models"][arm]["reflection_map"]),
            )
            for arm in ("binary-geometric", "aenet-full")
        ],
    ]
    for col, (title, depth, reflection) in enumerate(columns):
        depth_image = axes[0, col].imshow(
            depth[0], cmap="viridis", vmin=0, vmax=1, interpolation="nearest"
        )
        axes[0, col].set_title(title, fontsize=11)
        axes[1, col].imshow(
            np.clip(reflection.transpose(1, 2, 0), 0, 1), interpolation="nearest"
        )
    for ax in axes.flat:
        ax.set(xticks=[], yticks=[])
    axes[0, 0].set_ylabel("Depth", fontsize=11)
    axes[1, 0].set_ylabel("Reflection", fontsize=11)
    fig.colorbar(
        depth_image,
        ax=axes[0, :],
        ticks=[0, 0.5, 1],
        shrink=0.75,
        label="Relative depth",
    )
    fig.suptitle(
        f"Example {number} · dataset label {VISUAL_LABELS[example['true_binary']]}"
    )
    plt.show()
