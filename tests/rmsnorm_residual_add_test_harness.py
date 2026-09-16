from dataclasses import dataclass
from itertools import product

import pytest
import torch


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

    def state_on(self, device):
        tensor_options = self._tensor_options_on(device)
        input_weight = torch.randn(self.columns, **tensor_options)

        return input_weight, self.variance_epsilon

    def inputs_on(self, device):
        tensor_options = self._tensor_options_on(device)
        input_x = torch.randn(self.rows, self.columns, **tensor_options) * self.scale
        input_residual = torch.randn(self.rows, self.columns, **tensor_options) * self.scale

        return input_x, input_residual

    def _tensor_options_on(self, device):
        generator = torch.Generator(device)
        return dict(dtype=self.data_type, device=device, generator=generator)


DATA_TYPES = [torch.float32, torch.float16, torch.bfloat16]

TOLERANCES = [1e-6, 1e-5]

SCALES = [1.0, 100.0]

RMSNORM_RESIDUAL_ADD_BASE_TEST_CASES = [
    RMSNormResidualAddTestCase(*combination)
    for combination in product(
        DATA_TYPES,
        [1, 7, 64],
        [64, 768, 1000, 1024, 4097, 8192],
        TOLERANCES,
        SCALES,
    )
]

RMSNORM_RESIDUAL_ADD_GPU_ONLY_TEST_CASES = [
    RMSNormResidualAddTestCase(*combination)
    for combination in product(
        DATA_TYPES,
        (4096,),
        (768, 1024, 2048, 3072, 4096, 5120, 8192, 12288, 16384),
        TOLERANCES,
        SCALES,
    )
]

rmsnorm_residual_add_base_test_cases = pytest.mark.parametrize(
    "test_case", RMSNORM_RESIDUAL_ADD_BASE_TEST_CASES, ids=lambda test_case: test_case.id
)

rmsnorm_residual_add_gpu_only_test_cases = pytest.mark.parametrize(
    "test_case", RMSNORM_RESIDUAL_ADD_GPU_ONLY_TEST_CASES, ids=lambda test_case: test_case.id
)


def run_rmsnorm_residual_add_test(implementation, test_case, device):
    if test_case.data_type not in implementation.supported_data_types_on(device):
        pytest.skip(
            f"{implementation.__name__} does not support {test_case.data_type} on {device.type}."
        )

    weight, variance_epsilon = test_case.state_on(device)
    x, residual = test_case.inputs_on(device)

    operation = implementation(weight, variance_epsilon)
    output = operation.forward(x, residual)

    expected_y, expected_residual_out = oracle(x, residual, weight, variance_epsilon)
    data_type = test_case.data_type
    tolerance = test_case.tolerance
    normalized = output.normalized
    residual = output.residual
    assert normalized.dtype == data_type and residual.dtype == data_type
    torch.testing.assert_close(normalized.double(), expected_y, atol=tolerance, rtol=tolerance)
    torch.testing.assert_close(
        residual.double(), expected_residual_out, atol=tolerance, rtol=tolerance
    )


def oracle(x, residual, weight, variance_epsilon):
    # Double precision (64-bit floating point)
    residual_out = (x.double() + residual.double()).to(x.dtype).double()

    inverse_rms = torch.rsqrt(residual_out.pow(2).mean(-1, keepdim=True) + variance_epsilon)
    normalized_output = residual_out * inverse_rms * weight.double()

    return normalized_output, residual_out
