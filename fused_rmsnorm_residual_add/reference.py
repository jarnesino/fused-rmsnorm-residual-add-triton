import torch
import torch.nn.functional as functional

from fused_rmsnorm_residual_add.base import BaseRMSNormResidualAdd, RMSNormResidualAddOutput


class FunctionalOnEagerTorchImplementation(BaseRMSNormResidualAdd):
    def forward(self, x, residual):
        residual_output = x + residual

        shape = residual_output.shape[-1:]
        normalized_output = functional.rms_norm(
            residual_output, shape, self._weight, self._variance_epsilon
        )

        return RMSNormResidualAddOutput(normalized_output, residual_output)


class NaiveLlamaStyleImplementation(BaseRMSNormResidualAdd):
    """
    Port of Llama residual-add + RMSNorm (same cast order).
    From huggingface/transformers v5.17.0 (Apache 2.0).
    In src/transformers/models/llama/modeling_llama.py:
        1. LlamaDecoderLayer.forward: hidden_states = residual + hidden_states.
        2. LlamaRMSNorm.forward: upcast to fp32, normalize, cast back, then scale by weight.

    Could have adapted the original implementation with a wrapper, but I wrote this for learning
        purposes.
    """

    def forward(self, x, residual):
        return _llama_style_rmsnorm_residual_add(x, residual, self._weight, self._variance_epsilon)


def _llama_style_rmsnorm_residual_add(x, residual, weight, variance_epsilon):
    """The Llama-style body as a free function, so torch.compile can trace it into one graph."""
    residual_output = residual + x  # From LlamaDecoderLayer

    # From LlamaRMSNorm
    input_data_type = residual_output.dtype
    hidden_states = residual_output.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    normalized_output = weight * hidden_states.to(input_data_type)

    return RMSNormResidualAddOutput(normalized_output, residual_output)


_compiled_llama_style_rmsnorm_residual_add = torch.compile(_llama_style_rmsnorm_residual_add)


class CompiledLlamaStyleImplementation(BaseRMSNormResidualAdd):
    """
    The Llama-style reference under torch.compile. Inductor fuses what it can.
    Compiled once per process.
    """

    def forward(self, x, residual):
        return _compiled_llama_style_rmsnorm_residual_add(
            x, residual, self._weight, self._variance_epsilon
        )

    @classmethod
    def supported_data_types_on(cls, device):
        # Inductor's CPU backend needs a C++ compiler that I don't want to put in the container
        if device.type == "cpu":
            return frozenset()

        return super().supported_data_types_on(device)
