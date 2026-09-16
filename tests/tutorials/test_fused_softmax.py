import pytest
import torch

from tutorials.fused_softmax import (
    cpu_softmax,
    gpu_softmax_with_one_program_per_row,
    gpu_softmax_with_persistent_grid,
)


def test_cpu_softmax(device):
    _test_softmax_implementation(cpu_softmax, device)


@pytest.mark.gpu
def test_gpu_softmax_with_one_program_per_row(device):
    _test_softmax_implementation(gpu_softmax_with_one_program_per_row, device)


@pytest.mark.gpu
def test_gpu_softmax_with_persistent_grid(device):
    _test_softmax_implementation(gpu_softmax_with_persistent_grid, device)


def _test_softmax_implementation(implementation, device):
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=device)

    triton_result = implementation(x)

    torch_result = torch.softmax(x, dim=1)
    assert torch.allclose(triton_result, torch_result, rtol=1e-5, atol=1e-5)
