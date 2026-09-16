import torch


def naive_on_eager_torch(x, residual, weight, variance_epsilon):
    residual_output = x + residual

    mean_squared = residual_output.float().pow(2).mean(-1, keepdim=True)
    inverse_rms = torch.rsqrt(mean_squared + variance_epsilon)  # Avoid zero division with epsilon

    normalized_output = (residual_output.float() * inverse_rms).to(x.dtype) * weight
    return normalized_output, residual_output
