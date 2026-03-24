"""Generate report artifacts from saved ablation results."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.evaluation.ablation import AblationRunner
from src.evaluation.comparison import generate_comparison_table, load_baseline_results
from src.evaluation.visualization import plot_accuracy_vs_speed, plot_ablation_heatmap


def main():
    parser = argparse.ArgumentParser(description="Generate report artifacts")
    parser.add_argument("--results-json", required=True,
                        help="Path to ablation_results.json")
    parser.add_argument("--output-dir", default="outputs/report")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # load results
    results = AblationRunner.load_results(args.results_json)
    print(f"Loaded {len(results)} experiment results")

    # comparison table
    baselines = load_baseline_results()
    table = generate_comparison_table(results, baselines)
    table_path = os.path.join(args.output_dir, "comparison_table.md")
    with open(table_path, "w") as f:
        f.write(table)

    # plots
    plot_accuracy_vs_speed(
        results, baselines,
        output_path=os.path.join(args.output_dir, "accuracy_vs_speed.png"),
    )
    plot_ablation_heatmap(
        results,
        output_path=os.path.join(args.output_dir, "ablation_heatmap.png"),
    )

    # compact summary
    best = max(results, key=lambda r: r.avg_map) if results else None
    summary = {
        "num_experiments": len(results),
        "best_config": best.name if best else None,
        "best_avg_map": best.avg_map if best else 0.0,
        "results": [{"name": r.name, "avg_map": r.avg_map,
                      "latency_ms": r.latency_ms} for r in results],
    }
    summary_path = os.path.join(args.output_dir, "results_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Artifacts saved to {args.output_dir}/")


if __name__ == "__main__":
    main()