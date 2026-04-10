"""SOTA comparison table generator."""
import os
import re


def load_baseline_results(sota_path=None):
    """Parse SOTA_TABLE.md and extract published baseline entries.

    Returns list of dicts: {model, year, avg_map, features, code_available}.
    """
    if sota_path is None:
        sota_path = os.path.join(os.path.dirname(__file__), "..", "..",
                                 "docs", "literature", "SOTA_TABLE.md")
        sota_path = os.path.normpath(sota_path)

    with open(sota_path) as f:
        content = f.read()

    # extract the Action Spotting table (between the two ## headers)
    section = re.search(
        r'## Action Spotting.*?\n(.*?)(?=\n## |\Z)', content, re.DOTALL
    )
    if not section:
        return []

    baselines = []
    for line in section.group(1).strip().split("\n"):
        line = line.strip()
        if not line.startswith("|") or "---" in line or "Model" in line:
            continue

        cells = [c.strip() for c in line.split("|")[1:-1]]
        if len(cells) < 5:
            continue

        model_name = cells[0].replace("**", "").strip()
        # skip our own row
        if "Ours" in model_name or "TBD" in cells[2]:
            continue

        year_str = cells[1].replace("**", "").strip()
        score_str = cells[2].replace("**", "").replace("~", "").replace("%", "").replace("*", "").strip()
        features = cells[3].replace("**", "").strip()
        code = cells[4].replace("**", "").strip()

        try:
            year = int(year_str)
            avg_map = float(score_str)
        except ValueError:
            continue

        baselines.append({
            "model": model_name,
            "year": year,
            "avg_map": avg_map,
            "features": features,
            "code_available": code not in ("", "—", "-"),
        })

    return baselines


def generate_comparison_table(project_results, baselines=None):
    """Generate a merged markdown table with baselines and project results.

    Args:
        project_results: list of AblationResult objects
        baselines: optional list of baseline dicts (loaded from SOTA_TABLE.md)

    Returns markdown string.
    """
    if baselines is None:
        baselines = load_baseline_results()

    # build unified row list
    rows = []
    for b in baselines:
        rows.append({
            "model": b["model"],
            "year": b["year"],
            "avg_map": b["avg_map"],
            "features": b["features"],
            "ms_frame": "—",
            "source": "published",
        })

    for r in project_results:
        rows.append({
            "model": f"**{r.name}**",
            "year": 2026,
            "avg_map": r.avg_map,
            "features": r.config.get("feature_type", "pca512"),
            "ms_frame": f"{r.latency_ms:.2f}" if r.latency_ms > 0 else "—",
            "source": "project",
        })

    # sort by avg_map descending
    rows.sort(key=lambda x: x["avg_map"], reverse=True)

    lines = [
        "| Model | Year | avg-mAP tight | Features | ms/frame |",
        "|-------|------|---------------|----------|----------|",
    ]
    for row in rows:
        map_str = f"{row['avg_map']:.1f}%" if row["avg_map"] > 0 else "TBD"
        lines.append(
            f"| {row['model']} | {row['year']} | {map_str} | "
            f"{row['features']} | {row['ms_frame']} |"
        )

    return "\n".join(lines)
