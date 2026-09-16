import torch
from transformers.models.llama.modeling_llama import LlamaRMSNorm

from fused_rmsnorm_residual_add.reference import (
    FunctionalOnEagerTorchImplementation,
    HuggingFaceLlamaStyleImplementation,
)
from tests.rmsnorm_residual_add_test_harness import (
    rmsnorm_residual_add_base_test_cases,
    run_rmsnorm_residual_add_test,
)


@rmsnorm_residual_add_base_test_cases
def test_functional_rmsnorm_residual_add(test_case, device):
    run_rmsnorm_residual_add_test(FunctionalOnEagerTorchImplementation, test_case, device)


@rmsnorm_residual_add_base_test_cases
def test_llama_style_rmsnorm_residual_add(test_case, device):
    run_rmsnorm_residual_add_test(HuggingFaceLlamaStyleImplementation, test_case, device)


@rmsnorm_residual_add_base_test_cases
def test_llama_style_port_is_bit_identical_to_the_original_implementation(test_case, device):
    weight, variance_epsilon = test_case.state_on(device)
    x, residual = test_case.inputs_on(device)

    operation = HuggingFaceLlamaStyleImplementation(weight, variance_epsilon)
    output = operation.forward(x, residual)

    original_norm = LlamaRMSNorm(test_case.columns, eps=variance_epsilon).to(
        device=device, dtype=test_case.data_type
    )
    with torch.no_grad():
        original_norm.weight.copy_(weight)
        expected_residual_output = residual + x  # The add method from LlamaDecoderLayer
        expected_normalized_output = original_norm(expected_residual_output)
    assert torch.equal(output.residual, expected_residual_output)
    assert torch.equal(output.normalized, expected_normalized_output)
