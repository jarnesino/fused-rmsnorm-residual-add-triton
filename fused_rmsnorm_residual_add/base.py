from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class RMSNormResidualAddOutput:
    normalized: torch.Tensor
    residual: torch.Tensor


class BaseRMSNormResidualAdd:
    def __init__(self, weight, variance_epsilon):
        self._weight = weight
        self._variance_epsilon = variance_epsilon

    def forward(self, x, residual) -> RMSNormResidualAddOutput:
        raise NotImplementedError()

    @classmethod
    def supported_data_types_on(cls, device):
        return {torch.float32, torch.float16, torch.bfloat16}
