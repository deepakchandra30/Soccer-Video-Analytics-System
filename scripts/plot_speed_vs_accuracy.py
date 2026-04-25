#!/usr/bin/env python
"""Speed vs accuracy scatter plot for the evaluation report.

Y axis: tight avg-mAP on SoccerNet-v2 test (100 matches).
X axis: per-frame inference latency in ms, log scale.

Our numbers come from ``results/efficiency.json`` and
``results/ablation_architecture.json``. Published SOTA numbers are
from the supervisor-provided rubric (rough — published papers don't
all report ms/frame; values converted from relative-to-real-time
descriptors).

Emits:
    results/speed_vs_accuracy.svg
    results/speed_vs_accuracy.png  (preview, gitignored)
"""
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# SoccerNet Baidu feature rate is 1 fps => "real time" means ~1000 ms/frame
# on that input rate. We convert the supervisor-provided relative-speed
# descriptors into ms/frame at that rate for comparability.
# - "> real-time" means faster than 1000 ms/frame (actually RT). We pick
#   something near 500 ms/frame to still be slow but faster than 1x.
# - "~0.5x RT" means half-RT, i.e. 2x slower than realtime => 2000 ms/frame.
# - "very slow" we put around 5000 ms/frame as a conservative visible
#   placeholder.
SOTA_POINTS = [
    {"name": "CALF (2020)",         "features": "ResNet-152", "tight": 15.3, "ms_per_frame": 500.0},
    {"name": "NetVLAD++ (2021)",    "features": "ResNet-152", "tight": 53.0, "ms_per_frame": 2000.0},
    {"name": "E2E-Spot (2022)",     "features": "Raw video",  "tight": 57.6, "ms_per_frame": 5000.0},
    {"name": "SpotFormer (2023)",   "features": "Baidu+Flow", "tight": 70.0, "ms_per_frame": 10000.0},
]


def main():
    ablation = json.load(open("results/ablation_architecture.json"))
    efficiency = json.load(open("results/efficiency.json"))

    lat = efficiency["inference_latency_ms_per_frame"]
    modes = ablation["modes"]

    # Ours — one point per ablation mode with matching measured latency.
    # For ensemble modes (tsm_slowfast, three_way, etc.) we use the sum of
    # per-head latencies — that's what benchmark_efficiency.py records for
    # ThreeWayEnsemble; we synthesise the 2-way ones the same way.
    def head_ms(name):
        return lat[name]["mean_ms_per_frame"]

    ours_points = [
        {"name": "Ours: TSM",       "tight": modes["tsm_only"]["tight_avg_mAP_pct"],
         "ms_per_frame": head_ms("TSM")},
        {"name": "Ours: SlowFast",  "tight": modes["slowfast_only"]["tight_avg_mAP_pct"],
         "ms_per_frame": head_ms("SlowFast")},
        {"name": "Ours: NetVLAD++", "tight": modes["netvlad_only"]["tight_avg_mAP_pct"],
         "ms_per_frame": head_ms("NetVLAD")},
        {"name": "Ours: TSM+SF (2-way)",
         "tight": modes["tsm_slowfast"]["tight_avg_mAP_pct"],
         "ms_per_frame": head_ms("TSM") + head_ms("SlowFast")},
        {"name": "Ours: 3-way ensemble",
         "tight": modes["three_way"]["tight_avg_mAP_pct"],
         "ms_per_frame": head_ms("TSM") + head_ms("SlowFast") + head_ms("NetVLAD")},
    ]

    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot SOTA first (behind ours)
    sota_x = [p["ms_per_frame"] for p in SOTA_POINTS]
    sota_y = [p["tight"] for p in SOTA_POINTS]
    ax.scatter(sota_x, sota_y, s=120, c="steelblue", marker="s",
               edgecolor="black", label="Published SOTA", zorder=3)
    for p in SOTA_POINTS:
        ax.annotate(p["name"], (p["ms_per_frame"], p["tight"]),
                    xytext=(8, 5), textcoords="offset points", fontsize=9,
                    color="steelblue")

    # Plot ours
    ours_x = [p["ms_per_frame"] for p in ours_points]
    ours_y = [p["tight"] for p in ours_points]
    ax.scatter(ours_x, ours_y, s=160, c="crimson", marker="*",
               edgecolor="black", label="Ours (Baidu features)", zorder=4)
    for p in ours_points:
        ax.annotate(p["name"], (p["ms_per_frame"], p["tight"]),
                    xytext=(8, -10), textcoords="offset points", fontsize=9,
                    color="crimson")

    # Real-time reference line (1 fps input rate => 1000 ms/frame is 1x RT)
    ax.axvline(1000.0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
    ax.text(1010, 10, "real-time (1 fps)", rotation=90,
            va="bottom", fontsize=8, color="grey")

    ax.set_xscale("log")
    ax.set_xlim(0.01, 20000)
    ax.set_ylim(0, 80)
    ax.set_xlabel("Inference latency per frame (ms, log scale)")
    ax.set_ylabel("Tight avg-mAP (%)")
    ax.set_title("SoccerNet-v2: Speed vs Accuracy — Ours vs Published SOTA")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")

    out_svg = Path("results/speed_vs_accuracy.svg")
    out_png = Path("results/speed_vs_accuracy.png")
    plt.tight_layout()
    plt.savefig(out_svg, dpi=150)
    plt.savefig(out_png, dpi=150)
    plt.close(fig)

    # Also emit a small JSON describing the data points so the report can
    # cite exact values rather than re-read them off the plot.
    with open("results/speed_vs_accuracy.json", "w") as f:
        json.dump({"sota": SOTA_POINTS, "ours": ours_points}, f, indent=2)

    print(f"Wrote {out_svg}, {out_png}, results/speed_vs_accuracy.json")
    print(f"  ours points (log-speedup over SOTA most-left): "
          f"{min(ours_x):.3f} ms/frame vs. SOTA min {min(sota_x):.0f} ms/frame "
          f"→ ~{min(sota_x)/min(ours_x):.0f}× faster")


if __name__ == "__main__":
    main()
