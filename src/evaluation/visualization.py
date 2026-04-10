"""Visualization utilities for ablation studies and SOTA comparison."""
import os

import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


def plot_accuracy_vs_speed(results, baselines=None, output_path="accuracy_vs_speed.png"):
    """Scatter plot: mAP (y) vs ms/frame (x) for all configurations.

    Args:
        results: list of AblationResult objects
        baselines: list of baseline dicts (from load_baseline_results)
        output_path: path to save PNG
    """
    fig, ax = plt.subplots(figsize=(10, 7))

    if not results and not baselines:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_xlabel("Inference latency (ms/frame)")
        ax.set_ylabel("avg-mAP tight (%)")
        ax.set_title("Accuracy vs Inference Speed")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    # plot baselines as gray circles
    if baselines:
        for b in baselines:
            # baselines don't have latency data, place them at x=0
            ax.scatter(0, b["avg_map"], c="gray", marker="o", s=50, alpha=0.6)
            ax.annotate(b["model"], (0, b["avg_map"]), fontsize=7,
                        xytext=(5, 2), textcoords="offset points")

    # plot project results with different markers by model type
    markers = {"tsm": ("s", "steelblue"), "slowfast": ("^", "seagreen"),
               "two_stage": ("*", "firebrick")}
    for r in results:
        model_type = r.config.get("model", "tsm")
        marker, color = markers.get(model_type, ("o", "gray"))
        ax.scatter(r.latency_ms, r.avg_map, c=color, marker=marker,
                   s=80, zorder=5, label=model_type)
        ax.annotate(r.name, (r.latency_ms, r.avg_map), fontsize=7,
                    xytext=(5, 2), textcoords="offset points")

    ax.set_xlabel("Inference latency (ms/frame)")
    ax.set_ylabel("avg-mAP tight (%)")
    ax.set_title("Accuracy vs Inference Speed — SoccerNet-v3 Action Spotting")
    ax.grid(True, alpha=0.3)

    # deduplicate legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), loc="lower right")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_heatmap(results, output_path="ablation_heatmap.png"):
    """Heatmap of ablation results: configs (rows) vs metrics (cols)."""
    fig, ax = plt.subplots(figsize=(10, max(4, len(results) * 0.5 + 1)))

    if not results:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=14)
        ax.set_title("Ablation Results")
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return

    names = [r.name for r in results]
    cols = ["avg-mAP", "mAP@1s", "mAP@2s", "mAP@5s", "ms/frame"]
    data = np.array([
        [r.avg_map, r.map_at_1s, r.map_at_2s, r.map_at_5s, r.latency_ms]
        for r in results
    ])

    im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)

    # annotate cells
    for i in range(len(names)):
        for j in range(len(cols)):
            val = data[i, j]
            fmt = f"{val:.1f}" if j < 4 else f"{val:.2f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=7)

    ax.set_title("Ablation Results — SoccerNet-v3 Action Spotting")
    fig.colorbar(im, ax=ax, shrink=0.6)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
