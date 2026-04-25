"""
Wave-0 test: Verify requirements.txt has all dependencies pinned with ==.
This test fails if any non-comment, non-blank line lacks a version pin.
"""
import os
import pytest


REQUIREMENTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "requirements.txt"
)


@pytest.mark.unit
def test_requirements_file_exists():
    """requirements.txt must exist at project root."""
    assert os.path.isfile(REQUIREMENTS_PATH), (
        f"requirements.txt not found at {REQUIREMENTS_PATH}"
    )


@pytest.mark.unit
def test_all_versions_pinned():
    """Every non-blank, non-comment line in requirements.txt must contain '=='."""
    assert os.path.isfile(REQUIREMENTS_PATH), (
        "requirements.txt not found — run pip freeze or create it first"
    )
    with open(REQUIREMENTS_PATH) as f:
        lines = f.readlines()

    unpinned = []
    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        # Skip blank lines and comment lines
        if not line or line.startswith("#"):
            continue
        # Skip options lines like --index-url or -r
        if line.startswith("-"):
            continue
        if "==" not in line:
            unpinned.append((lineno, line))

    assert not unpinned, (
        f"Found {len(unpinned)} unpinned dependencies in requirements.txt:\n"
        + "\n".join(f"  line {n}: {l}" for n, l in unpinned)
    )
