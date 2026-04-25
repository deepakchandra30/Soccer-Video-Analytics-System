#!/usr/bin/env python
"""Per-class precision-recall curves from the cached 3-way scores.

Uses the same per-frame ensemble scores that produced the 61.11% tight
mAP, plus frame-level binary labels synthesised from Labels-v2.json
events with a ±TOLERANCE_S second tolerance. Per-frame binary labels
differ slightly from SoccerNet's internal "closest event" accounting,
so the AP here is a diagnostic, not the same number as the evaluator's
tight avg-mAP — but the shape of the PR curve is the right thing to
plot for the supervisor's rubric.

Emits:
    results/pr_curves_final_baidu.svg    (16-class grid + a combined axis)
    results/pr_curves_final_baidu.png
    results/pr_summary.json              (per-class AP + operating points)
"""
import argparse
import json
from pathlib import Path

import numpy as np

import matplotlib


def precision_recall_curve(y_true, y_score):
    """Drop-in for sklearn.metrics.precision_recall_curve.

    Sorts scores desc, sweeps thresholds, returns (precision, recall,
    thresholds) tuple — precision and recall are length n+1 (with the
    zero-recall terminus) to match sklearn's convention.
    """
    y_true = np.asarray(y_true, dtype=np.int8)
    y_score = np.asarray(y_score, dtype=np.float64)
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true[order]
    y_score_sorted = y_score[order]

    tp_cum = np.cumsum(y_true_sorted)
    fp_cum = np.cumsum(1 - y_true_sorted)

    # Precision at each prefix = TP / (TP + FP)
    denom = tp_cum + fp_cum
    precision = np.where(denom > 0, tp_cum / np.maximum(denom, 1), 1.0)
    total_pos = y_true.sum()
    recall = tp_cum / max(total_pos, 1)

    # Sklearn-style: keep one point per unique threshold (to compress ties).
    # For simplicity, skip compression — curve plots identically either way.
    # Append the (precision=1, recall=0) terminus.
    precision = np.r_[precision[::-1], 1.0]
    recall = np.r_[recall[::-1], 0.0]
    thresholds = y_score_sorted[::-1]
    return precision[::-1], recall[::-1], thresholds


def average_precision_score(y_true, y_score):
    """Sklearn-style step-function AP: AP = Σₙ (Rₙ − Rₙ₋₁) × Pₙ."""
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    # precision/recall are in decreasing-threshold order; sklearn computes
    # AP as the sum over all recall transitions.
    return float(np.sum(np.diff(recall) * precision[:-1]))


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = [
    "Penalty", "Kick-off", "Goal", "Substitution", "Offside",
    "Shots on target", "Shots off target", "Clearance",
    "Ball out of play", "Throw-in", "Foul", "Indirect free-kick",
    "Direct free-kick", "Corner", "Yellow card", "Red card",
    "Yellow->red card",
]
CLASS2IDX = {c: i for i, c in enumerate(CLASSES)}


def load_events(label_path, fps):
    """Return (h1_events, h2_events) as lists of (frame_index, class_idx)."""
    with open(label_path) as f:
        annos = json.load(f)["annotations"]
    h1, h2 = [], []
    ms_per_frame = 1000.0 / fps
    for a in annos:
        half = int(a["gameTime"].split(" ")[0])
        pos_ms = int(a["position"])
        frame = int(pos_ms / ms_per_frame)
        cls = CLASS2IDX.get(a["label"])
        if cls is None:
            continue
        (h1 if half == 1 else h2).append((frame, cls))
    return h1, h2


def events_to_labels(events, T, n_classes, tol_frames):
    """Vectorised: set label[t, c] = 1 iff any event of class c is within
    ±tol_frames of t. Safe O(E + N) per half where E = events, N = T."""
    labels = np.zeros((T, n_classes), dtype=np.uint8)
    for frame, cls in events:
        lo = max(0, frame - tol_frames)
        hi = min(T, frame + tol_frames + 1)
        labels[lo:hi, cls] = 1
    return labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="outputs/sweep3_cache")
    ap.add_argument("--data-dir", default="data/")
    ap.add_argument("--tsm-w", type=float, default=0.10)
    ap.add_argument("--sf-w", type=float, default=0.60)
    ap.add_argument("--nv-w", type=float, default=0.30)
    ap.add_argument("--fps", type=int, default=1, help="Baidu framerate")
    ap.add_argument("--tolerance-s", type=float, default=5.0,
                    help="± seconds for positive frame match (matches loose@5s)")
    ap.add_argument("--grid-out", default="results/pr_curves_final_baidu.svg")
    ap.add_argument("--json-out", default="results/pr_summary.json")
    args = ap.parse_args()

    tol_frames = int(round(args.tolerance_s * args.fps))
    n_classes = len(CLASSES)
    cache_dir = Path(args.cache_dir)
    data_dir = Path(args.data_dir)

    # Pool scores/labels per class across all matches.
    per_class_scores = [[] for _ in range(n_classes)]
    per_class_labels = [[] for _ in range(n_classes)]
    n_matches = 0

    for half_lens in cache_dir.rglob("half_lens.json"):
        mdir = half_lens.parent
        rel = mdir.relative_to(cache_dir)
        label_path = data_dir / rel / "Labels-v2.json"
        if not label_path.exists():
            continue

        try:
            tsm1 = np.load(mdir / "tsm_h1.npy")
            sf1  = np.load(mdir / "slowfast_h1.npy")
            nv1  = np.load(mdir / "netvlad_h1.npy")
            tsm2 = np.load(mdir / "tsm_h2.npy")
            sf2  = np.load(mdir / "slowfast_h2.npy")
            nv2  = np.load(mdir / "netvlad_h2.npy")
        except FileNotFoundError:
            continue

        ens1 = args.tsm_w * tsm1 + args.sf_w * sf1 + args.nv_w * nv1
        ens2 = args.tsm_w * tsm2 + args.sf_w * sf2 + args.nv_w * nv2
        h1_events, h2_events = load_events(label_path, args.fps)
        lab1 = events_to_labels(h1_events, ens1.shape[0], n_classes, tol_frames)
        lab2 = events_to_labels(h2_events, ens2.shape[0], n_classes, tol_frames)

        # Accumulate per-class. Append flattened arrays rather than per-element
        # to keep this O(T) not O(T * n_classes).
        for c in range(n_classes):
            per_class_scores[c].append(ens1[:, c])
            per_class_scores[c].append(ens2[:, c])
            per_class_labels[c].append(lab1[:, c])
            per_class_labels[c].append(lab2[:, c])
        n_matches += 1

    print(f"Accumulated {n_matches} matches, {tol_frames}-frame tolerance "
          f"({args.tolerance_s}s @ {args.fps} fps)")

    # Compute PR per class.
    pr = {}
    for c in range(n_classes):
        scores = np.concatenate(per_class_scores[c])
        labels = np.concatenate(per_class_labels[c])
        if labels.sum() == 0:
            print(f"  {CLASSES[c]:<22s} — no positives in GT — skipping")
            pr[CLASSES[c]] = None
            continue
        p, r, _ = precision_recall_curve(labels, scores)
        ap_val = float(average_precision_score(labels, scores))
        # Downsample to ≤ N points per class so the SVG is plottable —
        # raw curves have millions of points (one per frame), which blows
        # up the SVG to >100 MB. Interpolate precision onto a uniform
        # recall grid instead; that's what the human eye sees anyway.
        N_PLOT = 200
        recall_grid = np.linspace(0.0, 1.0, N_PLOT)
        # precision is piecewise-constant under sklearn's convention; use
        # right-inclusive step interpolation so we preserve the highest
        # precision achievable at each recall level.
        sort_idx = np.argsort(r)
        r_sorted = r[sort_idx]
        p_sorted = p[sort_idx]
        # right=True picks the precision of the segment ending at each
        # target recall — the one that still contains that recall level.
        idx = np.searchsorted(r_sorted, recall_grid, side="right") - 1
        idx = np.clip(idx, 0, len(r_sorted) - 1)
        p_grid = p_sorted[idx]
        pr[CLASSES[c]] = {"precision": p_grid, "recall": recall_grid,
                          "ap": ap_val,
                          "n_positives": int(labels.sum()),
                          "n_frames": int(labels.size)}

    # --- Figure: grid of 17 per-class PR curves + a combined overlay ---
    fig = plt.figure(figsize=(18, 14))
    for c in range(n_classes):
        ax = plt.subplot(5, 4, c + 1)
        d = pr[CLASSES[c]]
        if d is None:
            ax.set_title(f"{CLASSES[c]}  (no GT)", fontsize=9)
            ax.set_xlim(0, 1); ax.set_ylim(0, 1)
            continue
        ax.plot(d["recall"], d["precision"], linewidth=1.4)
        ax.fill_between(d["recall"], d["precision"], alpha=0.15)
        ax.set_title(f"{CLASSES[c]}  (AP={d['ap']:.3f}, N⁺={d['n_positives']})",
                     fontsize=9)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
        ax.grid(True, linestyle="--", alpha=0.3)
        if c % 4 == 0:
            ax.set_ylabel("Precision")
        if c >= 12:
            ax.set_xlabel("Recall")

    # Combined overlay in the empty 18th slot, if you prefer — here we hide.
    for c in range(n_classes, 20):
        plt.subplot(5, 4, c + 1).set_visible(False)

    plt.suptitle("Per-class Precision-Recall — 3-way Baidu ensemble "
                 f"(±{args.tolerance_s:.0f} s tolerance)",
                 fontsize=13, y=0.995)
    plt.tight_layout()
    plt.savefig(args.grid_out, dpi=130)
    plt.savefig(args.grid_out.replace(".svg", ".png"), dpi=130)
    plt.close(fig)

    # Summary JSON.
    summary = {"tolerance_s": args.tolerance_s, "fps": args.fps,
               "weights": {"tsm": args.tsm_w, "sf": args.sf_w, "nv": args.nv_w},
               "n_matches": n_matches, "per_class": {}}
    for name, d in pr.items():
        if d is None:
            summary["per_class"][name] = None
        else:
            summary["per_class"][name] = {
                "ap": d["ap"],
                "n_positives": d["n_positives"],
                "n_frames": d["n_frames"],
            }
    Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.json_out, "w") as f:
        json.dump(summary, f, indent=2)

    # Console report.
    print(f"\nPer-class PR AP (sorted desc):")
    rows = sorted(((n, d["ap"] if d else 0.0) for n, d in pr.items()),
                  key=lambda x: -x[1])
    for name, ap_val in rows:
        print(f"  {name:<22s}  AP = {ap_val:.3f}")
    print(f"\nSaved: {args.grid_out} (+ .png preview), {args.json_out}")


if __name__ == "__main__":
    main()
