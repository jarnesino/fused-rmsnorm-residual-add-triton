import pytest
import torch

from fused_rmsnorm_residual_add.fused import FusedImplementation
from fused_rmsnorm_residual_add.reference import NaiveLlamaStyleImplementation
from tests.rmsnorm_residual_add_test_harness import (
    RMSNormResidualAddTestCase,
    oracle,
    rmsnorm_residual_add_base_test_cases,
    rmsnorm_residual_add_gpu_only_test_cases,
    run_rmsnorm_residual_add_test,
)


@rmsnorm_residual_add_base_test_cases
def test_fused_rmsnorm_residual_add(test_case, device):
    run_rmsnorm_residual_add_test(FusedImplementation, test_case, device)


@pytest.mark.gpu
@rmsnorm_residual_add_gpu_only_test_cases
def test_fused_rmsnorm_residual_add_on_gpu_cases(test_case, device):
    run_rmsnorm_residual_add_test(FusedImplementation, test_case, device)


@pytest.mark.gpu
@rmsnorm_residual_add_gpu_only_test_cases
def test_fused_rmsnorm_residual_add_on_gpu_cases_is_similarly_accurate_to_llama(test_case, device):
    weight, variance_epsilon = test_case.state_on(device)
    x, residual = test_case.inputs_on(device)

    fused_output = FusedImplementation(weight, variance_epsilon).forward(x, residual)

    port_output = NaiveLlamaStyleImplementation(weight, variance_epsilon).forward(x, residual)
    expected_normalized_output, _ = oracle(x, residual, weight, variance_epsilon)
    fused_error = (fused_output.normalized.double() - expected_normalized_output).abs().max()
    port_error = (port_output.normalized.double() - expected_normalized_output).abs().max()
    assert fused_error <= 1.5 * port_error


SMALL_TEST_CASE = RMSNormResidualAddTestCase(
    data_type=torch.float32, rows=8, columns=1000, variance_epsilon=1e-6, scale=1.0
)


def _fused_operation_and_small_inputs_on(device):
    weight, variance_epsilon = SMALL_TEST_CASE.state_on(device)
    x, residual = SMALL_TEST_CASE.inputs_on(device)

    return FusedImplementation(weight, variance_epsilon), x, residual


def test_supports_row_strided_input(device):
    operation, x, residual = _fused_operation_and_small_inputs_on(device)

    output = operation.forward(x[::2], residual[::2])

    expected_output = operation.forward(x[::2].contiguous(), residual[::2].contiguous())
    assert torch.equal(output.normalized, expected_output.normalized)
    assert torch.equal(output.residual, expected_output.residual)


def test_rejects_column_strided_input(device):
    operation, x, residual = _fused_operation_and_small_inputs_on(device)

    with pytest.raises(ValueError, match="Tensor must have a contiguous last dimension."):
        operation.forward(x[:, ::2], residual[:, ::2])


def test_supports_in_place_residual(device):
    operation, x, residual = _fused_operation_and_small_inputs_on(device)
    original_residual = residual.clone()

    output = operation.forward(x, residual, residual_output=residual)  # Writes residual in-place

    expected_output = operation.forward(x, original_residual)
    assert output.residual.data_ptr() == residual.data_ptr()
    assert torch.equal(output.residual, expected_output.residual)
    assert torch.equal(output.normalized, expected_output.normalized)


def test_uses_pre_allocated_outputs(device):
    operation, x, residual = _fused_operation_and_small_inputs_on(device)
    normalized_output, residual_output = torch.empty_like(x), torch.empty_like(x)

    output = operation.forward(
        x, residual, normalized_output=normalized_output, residual_output=residual_output
    )

    assert output.normalized.data_ptr() == normalized_output.data_ptr()
    assert output.residual.data_ptr() == residual_output.data_ptr()


def test_is_deterministic(device):
    operation, x, residual = _fused_operation_and_small_inputs_on(device)

    first_output = operation.forward(x, residual)
    second_output = operation.forward(x, residual)

    assert torch.equal(first_output.normalized, second_output.normalized)
    assert torch.equal(first_output.residual, second_output.residual)


def test_flattens_leading_dimensions(device):
    operation, x, residual = _fused_operation_and_small_inputs_on(device)

    output = operation.forward(x.view(2, 4, -1), residual.view(2, 4, -1))

    expected_output = operation.forward(x, residual)
    assert output.normalized.shape == (2, 4, 1000)
    assert output.residual.shape == (2, 4, 1000)
    assert torch.equal(output.normalized.view(8, -1), expected_output.normalized)
    assert torch.equal(output.residual.view(8, -1), expected_output.residual)


def test_zero_rows(device):
    operation, x, residual = _fused_operation_and_small_inputs_on(device)

    output = operation.forward(x[:0], residual[:0])

    assert output.normalized.shape == (0, 1000)
    assert output.residual.shape == (0, 1000)


def test_rejects_weight_if_it_is_in_a_different_data_type(device):
    weight, variance_epsilon = SMALL_TEST_CASE.state_on(device)
    x, residual = SMALL_TEST_CASE.inputs_on(device)

    with pytest.raises(ValueError, match="Weight must have the same datatype as input x."):
        FusedImplementation(weight.half(), variance_epsilon).forward(x, residual)


def test_rejects_weight_if_it_is_non_contiguous(device):
    _, variance_epsilon = SMALL_TEST_CASE.state_on(device)
    column_strided_weight = torch.randn(1000, 2, device=device)[:, 0]

    with pytest.raises(ValueError, match="Tensor must have a contiguous last dimension."):
        FusedImplementation(column_strided_weight, variance_epsilon)


@pytest.mark.gpu
def test_keeps_in_place_residual_correct_on_first_autotuned_call(device):
    FusedImplementation.clear_kernel_cache()  # Force tuning
    operation, x, residual = _fused_operation_and_small_inputs_on(device)
    original_residual = residual.clone()

    output = operation.forward(x, residual, residual_output=residual)

    expected_output = operation.forward(x, original_residual)
    assert torch.equal(output.residual, expected_output.residual)
    assert torch.equal(output.normalized, expected_output.normalized)
