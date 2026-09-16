import torch
import triton
import triton.language as tl

from fused_rmsnorm_residual_add.base import BaseRMSNormResidualAdd, RMSNormResidualAddOutput


class FusedImplementation(BaseRMSNormResidualAdd):
    def __init__(self, weight, variance_epsilon):
        super().__init__(weight, variance_epsilon)
        self._assert_last_dimensions_are_contiguous([weight])

    def forward(self, x, residual, normalized_output=None, residual_output=None):
        # Allows the caller to pass pre-allocated buffers for the outputs and also alias the
        #   residual output as the next input, which is what a production model would do.

        self._assert_last_dimensions_are_contiguous(
            [x, residual, normalized_output, residual_output]
        )
        self._assert_data_types_match(self._weight, x)

        original_shape = x.shape
        number_of_columns = x.shape[-1]
        x_2d = x.view(-1, number_of_columns)
        residual_2d = residual.view(-1, number_of_columns)
        number_of_rows = x_2d.shape[0]

        block_size = self._block_size_for(number_of_columns)

        # Heuristic to be optimized
        number_of_warps = 4 if block_size < 2048 else 8 if block_size < 8192 else 16

        normalized_output_2d = (
            torch.empty_like(x_2d)
            if normalized_output is None
            else normalized_output.view(-1, number_of_columns)
        )
        residual_output_2d = (
            torch.empty_like(x_2d)
            if residual_output is None
            else residual_output.view(-1, number_of_columns)
        )

        grid = (number_of_rows,)
        fused_rmsnorm_residual_add_kernel[grid](
            x_2d,
            residual_2d,
            self._weight,
            normalized_output_2d,
            residual_output_2d,
            x_2d.stride(0),
            residual_2d.stride(0),
            normalized_output_2d.stride(0),
            residual_output_2d.stride(0),
            number_of_columns,
            self._variance_epsilon,
            block_size=tl.constexpr(block_size),
            num_warps=number_of_warps,
        )

        normalized_output = normalized_output_2d.view(original_shape)
        residual_output = residual_output_2d.view(original_shape)

        return RMSNormResidualAddOutput(normalized_output, residual_output)

    @classmethod
    def supported_data_types_on(cls, device):
        base_supported_data_types = super().supported_data_types_on(device)

        if device.type == "cpu":
            return base_supported_data_types - {torch.bfloat16}
        return base_supported_data_types

    @staticmethod
    def _assert_last_dimensions_are_contiguous(tensors):
        for tensor in tensors:
            if tensor is not None and tensor.stride(-1) != 1:
                raise ValueError("Tensor must have a contiguous last dimension.")

    @staticmethod
    def _assert_data_types_match(weight, x):
        if weight.dtype != x.dtype:
            raise ValueError("Weight must have the same datatype as input x.")

    def _block_size_for(self, number_of_columns):
        block_size = triton.next_power_of_2(number_of_columns)
        if block_size > self._block_size_limit():
            message = f"A row of {number_of_columns} columns is too wide for a single-block kernel."
            raise ValueError(message)
        return block_size

    @staticmethod
    def _block_size_limit() -> int:
        return 65536


@triton.jit
def fused_rmsnorm_residual_add_kernel(
    x_ptr,
    residual_ptr,
    weight_ptr,
    normalized_output_ptr,
    residual_output_ptr,
    x_stride,
    residual_stride,
    normalized_output_stride,
    residual_output_stride,  # row strides; last dim is contiguous
    number_of_columns,
    variance_epsilon,
    block_size: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.arange(0, block_size)
    mask = columns < number_of_columns

    x = tl.load(x_ptr + row * x_stride + columns, mask=mask, other=0.0)
    residual = tl.load(residual_ptr + row * residual_stride + columns, mask=mask, other=0.0)

    residual_output = residual + x  # One rounding in the model dtype like LlamaDecoderLayer
    tl.store(
        residual_output_ptr + row * residual_output_stride + columns, residual_output, mask=mask
    )

    hidden_states = residual_output.to(tl.float32)  # Upcast rounded value like LlamaRMSNorm

    variance = tl.sum(hidden_states * hidden_states, axis=0) / number_of_columns
    inverse_rms = tl.rsqrt(variance + variance_epsilon)

    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0)
    normalized_output = (hidden_states * inverse_rms).to(
        normalized_output_ptr.dtype.element_ty
    ) * weight
    tl.store(
        normalized_output_ptr + row * normalized_output_stride + columns,
        normalized_output,
        mask=mask,
    )
