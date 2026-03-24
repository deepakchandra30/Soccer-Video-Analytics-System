"""Run ablation studies across model configurations."""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.seeds import set_seeds
from config.ablation_config import ABLATION_EXPERIMENTS
from src.evaluation.ablation import AblationRunner
from src.evaluation.comparison import generate_comparison_table, load_baseline_results
from src.evaluation.visualization import plot_accuracy_vs_speed, plot_ablation_heatmap


def main():
    parser = argparse.ArgumentParser(description="Run ablation studies")
    parser.add_argument("--data-dir", default="data/")
    parser.add_argument("--output-dir", default="outputs/ablation")
    parser.add_argument("--checkpoints-dir", default="outputs/",
                        help="Directory containing trained model checkpoints")
    parser.add_argument("--device", default=None)
    parser.add_argument("--experiments", default="all",
                        help="Comma-separated experiment names, or 'all'")
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    set_seeds(42)
    os.makedirs(args.output_dir, exist_ok=True)

    # filter experiments
    if args.experiments == "all":
        experiments = ABLATION_EXPERIMENTS
    else:
        names = [n.strip() for n in args.experiments.split(",")]
        experiments = [e for e in ABLATION_EXPERIMENTS if e["name"] in names]

    if not experiments:
        print("No experiments matched. Available:")
        for e in ABLATION_EXPERIMENTS:
            print(f"  {e['name']}")
        return

    # run ablation
    runner = AblationRunner(data_dir=args.data_dir, output_dir=args.output_dir,
                            device=args.device)
    for exp in experiments:
        runner.add_experiment(exp["name"], {k: v for k, v in exp.items()
                                            if k != "name"})

    print(f"Running {len(experiments)} experiments...")
    results = runner.run(checkpoint_dir=args.checkpoints_dir)

    # save results
    json_path = os.path.join(args.output_dir, "ablation_results.json")
    runner.save_results(json_path)
    print(f"Results saved to {json_path}")

    # generate comparison table
    baselines = load_baseline_results()
    table = generate_comparison_table(results, baselines)
    table_path = os.path.join(args.output_dir, "comparison_table.md")
    with open(table_path, "w") as f:
        f.write(table)
    print(f"Comparison table saved to {table_path}")

    # generate plots
    plot_accuracy_vs_speed(
        results, baselines,
        output_path=os.path.join(args.output_dir, "accuracy_vs_speed.png"),
    )
    plot_ablation_heatmap(
        results,
        output_path=os.path.join(args.output_dir, "ablation_heatmap.png"),
    )
    print("Plots saved.")

    # print summary
    print("\n" + runner.to_markdown_table())


if __name__ == "__main__":
    main()
