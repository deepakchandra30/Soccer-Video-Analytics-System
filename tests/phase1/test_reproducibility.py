"""
Wave-0 test: Verify that set_seeds(42) produces deterministic output across two calls.
Tests numpy and torch random number generation for identical results.
"""
import pytest


@pytest.mark.unit
def test_seeds_module_importable():
    """config.seeds must be importable and export set_seeds."""
    from config.seeds import set_seeds  # noqa: F401
    assert callable(set_seeds), "set_seeds must be a callable function"


@pytest.mark.unit
def test_numpy_determinism():
    """Two calls to set_seeds(42) + np.random must yield identical arrays."""
    import numpy as np
    from config.seeds import set_seeds

    set_seeds(42)
    arr1 = np.random.rand(10)

    set_seeds(42)
    arr2 = np.random.rand(10)

    assert np.array_equal(arr1, arr2), (
        f"numpy random output differs after two set_seeds(42) calls:\n"
        f"  Run 1: {arr1}\n"
        f"  Run 2: {arr2}"
    )


@pytest.mark.unit
def test_torch_determinism():
    """Two calls to set_seeds(42) + torch.rand must yield identical tensors."""
    import torch
    from config.seeds import set_seeds

    set_seeds(42)
    t1 = torch.rand(5)

    set_seeds(42)
    t2 = torch.rand(5)

    assert torch.equal(t1, t2), (
        f"torch.rand output differs after two set_seeds(42) calls:\n"
        f"  Run 1: {t1}\n"
        f"  Run 2: {t2}"
    )


@pytest.mark.unit
def test_different_seeds_differ():
    """Sanity check: seed=42 and seed=99 must produce different outputs."""
    import numpy as np
    from config.seeds import set_seeds

    set_seeds(42)
    arr42 = np.random.rand(10)

    set_seeds(99)
    arr99 = np.random.rand(10)

    assert not np.array_equal(arr42, arr99), (
        "seed=42 and seed=99 produced identical output — seeding is broken"
    )
