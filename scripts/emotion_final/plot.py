import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent
r = json.loads((root / "result.json").read_text(encoding="utf-8"))
h = json.loads((root / "history.json").read_text(encoding="utf-8"))
m = r["original_image_validation"]
matrix = np.array(m["confusion_matrix"])
normalized = matrix / matrix.sum(1, keepdims=True)
fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5), layout="constrained")
axes[0].plot(
    [v["epoch"] for v in h],
    [v["validation"]["macro_f1"] for v in h],
    "-o",
    label="Validation macro F1",
)
axes[0].plot(
    [v["epoch"] for v in h],
    [v["validation"]["per_class"][2]["f1"] for v in h],
    "-o",
    label="Validation Neutral F1",
)
axes[0].axvline(
    r["selected"]["epoch"], color="gray", linestyle="--", label="Selected by Neutral F1"
)
axes[0].set(
    xlabel="Epoch",
    ylabel="F1",
    ylim=(0, 1),
    title="12,903 train / 3,217 validation images",
)
axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.2)
axes[1].imshow(normalized, cmap="Blues", vmin=0, vmax=1)
axes[1].set(
    xticks=range(3),
    yticks=range(3),
    xticklabels=["Sad", "Happy", "Neutral"],
    yticklabels=["Sad", "Happy", "Neutral"],
    xlabel="Predicted class",
    ylabel="Actual class",
    title="Validation: counts and row percentages",
)
for (i, j), v in np.ndenumerate(matrix):
    axes[1].text(
        j,
        i,
        f"{v}\n{normalized[i, j]:.1%}",
        ha="center",
        va="center",
        color="white" if normalized[i, j] > 0.5 else "black",
    )
fig.savefig(root / "validation-results.png", dpi=160)
print(json.dumps(m, indent=2))
