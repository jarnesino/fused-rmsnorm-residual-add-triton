import os

import pytest
import torch

SIMULATING = os.environ.get("TRITON_INTERPRET") == "1"
HAS_CUDA = torch.cuda.is_available()


def pytest_collection_modifyitems(config, items):
    """This is a pytest hook. Skips CUDA tests when simulating."""

    if HAS_CUDA and not SIMULATING:
        return

    skip = pytest.mark.skip(reason="needs a real CUDA device")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def device() -> torch.device:
    """CPU under the interpreter, CUDA otherwise. Never hard-code 'cuda' in kernels/wrappers."""

    if SIMULATING or not HAS_CUDA:
        return torch.device("cpu")

    return torch.device("cuda")


@pytest.fixture(autouse=True)
def _seed():
    """Resets RNGs from torch to avoid flaky tests from floating point rounding errors."""
    torch.manual_seed(0)
