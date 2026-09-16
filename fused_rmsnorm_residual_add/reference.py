import torch
import torch.nn.functional as functional


def functional_on_eager_torch(x, residual, weight, variance_epsilon):
    residual_output = x + residual

    shape = residual_output.shape[-1:]
    output = functional.rms_norm(residual_output, shape, weight, variance_epsilon)

    return output, residual_output


def hugging_face_llama_style(x, residual, weight, variance_epsilon):
    """
    Port of Llama residual-add + RMSNorm (same cast order).
    From huggingface/transformers in src/transformers/models/llama/modeling_llama.py (Apache 2.0):
        1. LlamaDecoderLayer.forward: hidden_states = residual + hidden_states.
        2. LlamaRMSNorm.forward: upcast to fp32, normalize, cast back, then scale by weight.
    """
    residual_output = residual + x  # From LlamaDecoderLayer

    # From LlamaRMSNorm
    input_data_type = residual_output.dtype
    hidden_states = residual_output.to(torch.float32)
    variance = hidden_states.pow(2).mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    normalized_output = weight * hidden_states.to(input_data_type)

    return normalized_output, residual_output
