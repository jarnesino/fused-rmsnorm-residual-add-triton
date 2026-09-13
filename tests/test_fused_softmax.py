import pytest
import torch

from tutorials.fused_softmax import cpu_softmax, gpu_softmax


def test_cpu_softmax(device):
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=device)

    triton_result = cpu_softmax(x)

    torch_result = torch.softmax(x, dim=1)
    assert torch.allclose(triton_result, torch_result, rtol=1e-5, atol=1e-5)


@pytest.mark.gpu
def test_gpu_softmax(device):
    torch.manual_seed(0)
    x = torch.randn(1823, 781, device=device)

    triton_result = gpu_softmax(x)

    torch_result = torch.softmax(x, dim=1)
    assert torch.allclose(triton_result, torch_result, rtol=1e-5, atol=1e-5)
