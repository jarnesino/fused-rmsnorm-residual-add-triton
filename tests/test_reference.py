from dataclasses import dataclass
from itertools import product

import pytest
import torch

from fused_rmsnorm_residual_add.reference import functional_on_eager_torch, naive_on_eager_torch


@dataclass(frozen=True)
class RMSNormResidualAddTestCase:
    data_type: torch.dtype
    rows: int
    columns: int
    variance_epsilon: float
    scale: float

    @property
    def id(self) -> str:
        return (
            f"{str(self.data_type).removeprefix('torch.')}-{self.rows}x{self.columns}"
            f"-eps{self.variance_epsilon:g}-x{self.scale:g}"
        )

    @property
    def tolerance(self):
        tolerance_map = {
            torch.float32: 1e-5,
            torch.float16: 1e-3,
            torch.bfloat16: 1e-2,
        }
        return tolerance_map[self.data_type]

    def inputs_on(self, device):
        generator = torch.Generator(device)

        tensor_options = dict(dtype=self.data_type, device=device, generator=generator)

        input_x = torch.randn(self.rows, self.columns, **tensor_options) * self.scale
        input_residual = torch.randn(self.rows, self.columns, **tensor_options) * self.scale
        input_weight = torch.randn(self.columns, **tensor_options)

        return input_x, input_residual, input_weight


RMSNORM_RESIDUAL_ADD_TEST_CASES = [
    RMSNormResidualAddTestCase(*combination)
    for combination in product(
        [torch.float32, torch.float16, torch.bfloat16],
        [1, 7, 64],
        [64, 768, 1000, 1024, 4097, 8192],
        [1e-6, 1e-5],
        [1.0, 100.0],
    )
]


@pytest.mark.parametrize(
    "test_case", RMSNORM_RESIDUAL_ADD_TEST_CASES, ids=lambda test_case: test_case.id
)
def test_naive_rmsnorm_residual_add(test_case, device):
    _test_rmsnorm_residual_add(naive_on_eager_torch, test_case, device)


@pytest.mark.parametrize(
    "test_case", RMSNORM_RESIDUAL_ADD_TEST_CASES, ids=lambda test_case: test_case.id
)
def test_functional_rmsnorm_residual_add(test_case, device):
    _test_rmsnorm_residual_add(functional_on_eager_torch, test_case, device)


def _test_rmsnorm_residual_add(implementation, test_case, device):
    variance_epsilon = test_case.variance_epsilon
    inputs = test_case.inputs_on(device)
    expected_y, expected_residual_out = _oracle(*inputs, variance_epsilon)

    y, residual_out = implementation(*inputs, variance_epsilon)

    data_type = test_case.data_type
    tolerance = test_case.tolerance
    assert y.dtype == data_type and residual_out.dtype == data_type
    torch.testing.assert_close(y.double(), expected_y, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        residual_out.double(), expected_residual_out, atol=tolerance, rtol=tolerance
    )


def _oracle(x, residual, weight, variance_epsilon):
    # Double precision (64-bit floating point)
    residual_out = (x.double() + residual.double()).to(x.dtype).double()

    inverse_rms = torch.rsqrt(residual_out.pow(2).mean(-1, keepdim=True) + variance_epsilon)
    normalized_output = residual_out * inverse_rms * weight.double()

    return normalized_output, residual_out
