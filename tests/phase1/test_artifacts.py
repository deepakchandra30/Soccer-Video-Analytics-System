"""
Artifact existence tests for literature review deliverables.

Phase 1 Plan 03 — TDD RED->GREEN cycle.
Tests confirm all required documentation files exist on disk.
These tests will FAIL until Task 2 creates the literature documents.
"""
from pathlib import Path
import pytest


def test_pipeline_diagram_exists():
    """
    At least one pipeline diagram file must exist in docs/literature/.
    Acceptable formats: .drawio, .png, .pdf
    """
    candidates = [
        Path("docs/literature/pipeline_diagram.drawio"),
        Path("docs/literature/pipeline_diagram.png"),
        Path("docs/literature/pipeline_diagram.pdf"),
    ]
    exists = [p for p in candidates if p.exists()]
    assert exists, (
        "No pipeline diagram found. Expected one of: "
        + ", ".join(str(p) for p in candidates)
    )


def test_papers_md_exists():
    """docs/literature/PAPERS.md must exist."""
    assert Path("docs/literature/PAPERS.md").exists(), (
        "docs/literature/PAPERS.md does not exist. "
        "Create it in Task 2 with 10+ paper entries."
    )


def test_sota_table_exists():
    """docs/literature/SOTA_TABLE.md must exist."""
    assert Path("docs/literature/SOTA_TABLE.md").exists(), (
        "docs/literature/SOTA_TABLE.md does not exist. "
        "Create it in Task 2 with separate Action Spotting and Ball Action Spotting sections."
    )


def test_novelty_md_exists():
    """docs/literature/NOVELTY.md must exist."""
    assert Path("docs/literature/NOVELTY.md").exists(), (
        "docs/literature/NOVELTY.md does not exist. "
        "Create it in Task 2 with 2-3 novel contributions."
    )
