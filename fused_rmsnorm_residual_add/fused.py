import functools
import os

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

        grid_function = self._grid_function_from(x, number_of_rows)

        fused_rmsnorm_residual_add_kernel[grid_function](
            x_2d,
            residual_2d,
            self._weight,
            normalized_output_2d,
            residual_output_2d,
            x_2d.stride(0),
            residual_2d.stride(0),
            normalized_output_2d.stride(0),
            residual_output_2d.stride(0),
            number_of_rows,
            number_of_columns,
            self._variance_epsilon,
            block_size=tl.constexpr(block_size),
        )

        normalized_output = normalized_output_2d.view(original_shape)
        residual_output = residual_output_2d.view(original_shape)

        return RMSNormResidualAddOutput(normalized_output, residual_output)

    def _grid_function_from(self, x, number_of_rows):
        number_of_sms = self._number_of_sms(x.device)

        def grid_function(meta):
            if meta["programs_per_sm"] == 0:
                return (number_of_rows,)
            return (min(number_of_rows, number_of_sms * meta["programs_per_sm"]),)

        return grid_function

    @staticmethod
    @functools.cache
    def _number_of_sms(device):
        if device.type == "cpu":
            return 4
        return torch.cuda.get_device_properties(device).multi_processor_count

    @classmethod
    def supported_data_types_on(cls, device):
        base_supported_data_types = super().supported_data_types_on(device)

        if device.type == "cpu":
            return base_supported_data_types - {torch.bfloat16}
        return base_supported_data_types

    @classmethod
    def clear_kernel_cache(cls):
        fused_rmsnorm_residual_add_kernel.cache.clear()

    @staticmethod
    def _assert_last_dimensions_are_contiguous(tensors):
        for tensor in tensors:
            if tensor is not None and tensor.stride(-1) != 1:
                raise ValueError("Tensor must have a contiguous last dimension.")

    @staticmethod
    def _assert_data_types_match(weight, x):
        if weight.dtype != x.dtype:
            raise ValueError("Weight must have the same datatype as input x.")

    @classmethod
    def _block_size_for(cls, number_of_columns):
        block_size = triton.next_power_of_2(number_of_columns)
        if block_size > cls._block_size_limit():
            message = f"A row of {number_of_columns} columns is too wide for a single-block kernel."
            raise ValueError(message)
        return block_size

    @classmethod
    def _block_size_limit(cls):
        return 65536


_PROGRAMS_PER_SM_CHOICES = (0, 1, 2, 4, 8, 16)  # Non-persistent loops use 0 (one program per row)
_WARP_CHOICES = (1, 2, 4, 8, 16, 32)


def _autotune_configurations():
    if os.environ.get("TRITON_INTERPRET") == "1":
        return [triton.Config({"programs_per_sm": 1}, num_warps=4, num_stages=1)]

    return [
        triton.Config({"programs_per_sm": p}, num_warps=w, num_stages=1)
        for w in _WARP_CHOICES
        for p in _PROGRAMS_PER_SM_CHOICES
    ]


def _prune_by_elements_per_thread(configs, named_args, **kwargs):
    block_size = FusedImplementation._block_size_for(named_args["number_of_columns"])

    properties = torch.cuda.get_device_properties(named_args["x_ptr"].device)
    max_warps_per_sm = properties.max_threads_per_multi_processor // 32

    kept = [
        configuration
        for configuration in configs
        if _is_worth_benchmarking(configuration, block_size, max_warps_per_sm)
    ]
    if len(kept) == 0:
        return configs
    return kept


def _is_worth_benchmarking(configuration, block_size, max_warps_per_sm):
    has_a_sane_working_set = 1 <= block_size // (32 * configuration.num_warps) <= 64
    fits_on_one_streaming_multiprocessor = (
        configuration.kwargs["programs_per_sm"] == 0
        or configuration.num_warps * configuration.kwargs["programs_per_sm"] <= max_warps_per_sm
    )

    return has_a_sane_working_set and fits_on_one_streaming_multiprocessor


@triton.autotune(
    configs=_autotune_configurations(),
    key=["number_of_columns"],
    prune_configs_by={"early_config_prune": _prune_by_elements_per_thread},
    restore_value=["residual_ptr"],  # In-place aliasing messes with autotune's multiple runs
)
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
    number_of_rows,
    number_of_columns,
    variance_epsilon,
    block_size: tl.constexpr,
    programs_per_sm: tl.constexpr,  # Not used here, needed to autotune the grid size
):
    columns = tl.arange(0, block_size)
    mask = columns < number_of_columns
    weight = tl.load(weight_ptr + columns, mask=mask, other=0.0)

    # Using a for loop causes a NumPy 2 incompatibility
    row = tl.program_id(0).to(tl.int64)
    row_step = tl.num_programs(0)
    while row < number_of_rows:
        x = tl.load(x_ptr + row * x_stride + columns, mask=mask, other=0.0)
        residual = tl.load(residual_ptr + row * residual_stride + columns, mask=mask, other=0.0)

        residual_output = residual + x  # One rounding in the model dtype like LlamaDecoderLayer
        tl.store(
            residual_output_ptr + row * residual_output_stride + columns, residual_output, mask=mask
        )

        hidden_states = residual_output.to(tl.float32)  # Upcast rounded value like LlamaRMSNorm

        variance = tl.sum(hidden_states * hidden_states, axis=0) / number_of_columns
        inverse_rms = tl.rsqrt(variance + variance_epsilon)

        normalized_output = (hidden_states * inverse_rms).to(
            normalized_output_ptr.dtype.element_ty
        ) * weight
        tl.store(
            normalized_output_ptr + row * normalized_output_stride + columns,
            normalized_output,
            mask=mask,
        )

        row += row_step
