import torch
import torch.nn.functional as functional


def naive_on_eager_torch(x, residual, weight, variance_epsilon):
    residual_output = x + residual

    mean_squared = residual_output.float().pow(2).mean(-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_squared + variance_epsilon)  # Avoid zero division with epsilon

    normalized_output = (residual_output.float() * inverse_rms).to(x.dtype) * weight
    return normalized_output, residual_output


def functional_on_eager_torch(x, residual, weight, variance_epsilon):
    residual_output = x + residual

    shape = residual_output.shape[-1:]
    output = functional.rms_norm(residual_output, shape, weight, variance_epsilon)

    return output, residual_output
