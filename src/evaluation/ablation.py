"""Ablation study runner for systematic model comparison."""
import json
import os
from dataclasses import dataclass, asdict, field

import torch

from src.models.temporal.tsm import TSMSpottingHead
from src.models.temporal.slowfast import SlowFastSpotting
from src.evaluation.benchmark import benchmark_latency


@dataclass
class AblationResult:
    name: str
    config: dict
    avg_map: float = 0.0
    map_at_1s: float = 0.0
    map_at_2s: float = 0.0
    map_at_5s: float = 0.0
    latency_ms: float = 0.0
    num_params: int = 0


class AblationRunner:
    """Registers experiment configurations and evaluates them."""

    def __init__(self, data_dir="data/", output_dir="outputs/ablation",
                 device="cpu"):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.device = device
        self.experiments = []
        self.results = []

    def add_experiment(self, name, config):
        self.experiments.append({"name": name, **config})

    def _build_model(self, config):
        """Factory: create model from config."""
        model_type = config.get("model", "tsm")
        feat_dim = config.get("feat_dim", 512)
        hidden_dim = config.get("hidden_dim", 256)
        num_classes = config.get("num_classes", 17)

        if model_type == "tsm":
            return TSMSpottingHead(feat_dim=feat_dim, num_classes=num_classes,
                                  hidden_dim=hidden_dim)
        elif model_type == "slowfast":
            return SlowFastSpotting(feat_dim=feat_dim, num_classes=num_classes,
                                   hidden_dim=hidden_dim)
        elif model_type == "two_stage":
            # return coarse model; fine model loaded separately
            coarse_dim = config.get("coarse_hidden_dim", 128)
            return TSMSpottingHead(feat_dim=feat_dim, num_classes=num_classes,
                                  hidden_dim=coarse_dim)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _count_params(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _evaluate_config(self, name, config, checkpoint_dir=None):
        """Evaluate one configuration. Returns AblationResult."""
        model = self._build_model(config)
        num_params = self._count_params(model)

        if checkpoint_dir:
            ckpt_path = os.path.join(checkpoint_dir, name, "best.pt")
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location="cpu",
                                  weights_only=False)
                model.load_state_dict(ckpt["model_state_dict"])

        feat_dim = config.get("feat_dim", 512)
        chunk_size = config.get("chunk_size", 40)
        dummy = torch.randn(chunk_size, feat_dim)
        try:
            lat = benchmark_latency(model, dummy, device=self.device,
                                    num_runs=5, warmup=2)
            latency_ms = lat["mean_ms_per_frame"]
        except Exception:
            latency_ms = 0.0

        return AblationResult(
            name=name, config=config, avg_map=0.0,
            map_at_1s=0.0, map_at_2s=0.0, map_at_5s=0.0,
            latency_ms=latency_ms, num_params=num_params,
        )

    def run(self, checkpoint_dir=None):
        """Run all registered experiments."""
        self.results = []
        for exp in self.experiments:
            name = exp["name"]
            config = {k: v for k, v in exp.items() if k != "name"}
            result = self._evaluate_config(name, config, checkpoint_dir)
            self.results.append(result)
        return self.results

    def save_results(self, path):
        """Write results to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = [asdict(r) for r in self.results]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_results(cls, path):
        """Read results from JSON."""
        with open(path) as f:
            data = json.load(f)
        return [AblationResult(**d) for d in data]

    def to_markdown_table(self):
        """Format results as a markdown table."""
        lines = ["| Config | avg-mAP | mAP@1s | mAP@2s | mAP@5s | ms/frame | Params |",
                 "|--------|---------|--------|--------|--------|----------|--------|"]
        for r in self.results:
            lines.append(
                f"| {r.name} | {r.avg_map:.1f}% | {r.map_at_1s:.1f}% | "
                f"{r.map_at_2s:.1f}% | {r.map_at_5s:.1f}% | "
                f"{r.latency_ms:.2f} | {r.num_params:,} |"
            )
        return "\n".join(lines)
