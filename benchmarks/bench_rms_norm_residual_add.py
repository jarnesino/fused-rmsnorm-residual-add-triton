from pathlib import Path

import torch
import triton
import triton.testing

from fused_rmsnorm_residual_add.fused import FusedImplementation
from fused_rmsnorm_residual_add.reference import (
    CompiledLlamaStyleImplementation,
    FunctionalOnEagerTorchImplementation,
    NaiveLlamaStyleImplementation,
)

VARIANCE_EPSILON = 1e-6


class OutOfPlaceCallingPolicy:
    """Each call allocates its outputs."""

    @staticmethod
    def call(operation, x, residual):
        return operation.forward(x, residual)


class InPlaceCallingPolicy:
    """The residual tensor is overwritten and one output is reused."""

    def __init__(self):
        self._normalized_output = None

    def call(self, operation, x, residual):
        if self._normalized_output is None:
            self._normalized_output = torch.empty_like(x)
        return operation.forward(
            x, residual, normalized_output=self._normalized_output, residual_output=residual
        )


PROVIDERS = {
    "fused": (FusedImplementation, OutOfPlaceCallingPolicy),
    "fused_in_place": (FusedImplementation, InPlaceCallingPolicy),
    "naive_llama_style": (NaiveLlamaStyleImplementation, OutOfPlaceCallingPolicy),
    "compiled_llama_style": (CompiledLlamaStyleImplementation, OutOfPlaceCallingPolicy),
    "torch_functional": (FunctionalOnEagerTorchImplementation, OutOfPlaceCallingPolicy),
}

DATA_TYPES = (torch.bfloat16, torch.float16, torch.float32)

COLUMN_SWEEP = (
    "number_of_columns",
    [768, 1024, 2048, 3072, 4096, 5120, 8192, 12288, 16384],
    {"number_of_rows": 4096},
    False,
)
ROW_SWEEP = ("number_of_rows", [64, 256, 1024, 4096, 16384], {"number_of_columns": 4096}, True)


def _benchmark_for(swept_name, swept_values, fixed_args, x_log, data_type):
    short_name = str(data_type).removeprefix("torch.")
    axis = "N" if swept_name == "number_of_columns" else "M"

    return triton.testing.Benchmark(
        x_names=[swept_name],
        x_vals=swept_values,
        x_log=x_log,
        line_arg="provider",
        line_vals=list(PROVIDERS),
        line_names=[
            "Fused",
            "Fused (in place)",
            "Naive Llama style",
            "Compiled Llama style",
            "torch.rms_norm",
        ],
        styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("blue", "--"), ("red", "-")],
        ylabel="GB/s (median)",
        plot_name=f"forward_{axis}_{short_name}",
        args={**fixed_args, "data_type": data_type},
    )


triton_benchmarks = [
    _benchmark_for(*sweep, data_type)
    for sweep in (COLUMN_SWEEP, ROW_SWEEP)
    for data_type in DATA_TYPES
]


@triton.testing.perf_report(triton_benchmarks)
def benchmark(number_of_rows, number_of_columns, data_type, provider):
    device = triton.runtime.driver.active.get_active_torch_device()
    tensor_options = dict(device=device, dtype=data_type)
    x = torch.randn(number_of_rows, number_of_columns, **tensor_options)
    residual = torch.randn(number_of_rows, number_of_columns, **tensor_options)
    weight = torch.randn(number_of_columns, **tensor_options)

    implementation_class, calling_policy_class = PROVIDERS[provider]
    operation = implementation_class(weight, VARIANCE_EPSILON)
    if data_type not in operation.supported_data_types_on(x.device):
        return float("nan"), float("nan"), float("nan")

    calling_policy = calling_policy_class()

    if provider == "compiled_llama_style":
        torch._dynamo.reset()
    calling_policy.call(operation, x, residual)  # For autotuning and compiling beforehand
    torch.cuda.synchronize()

    ms, ms_p20, ms_p80 = triton.testing.do_bench(
        lambda: calling_policy.call(operation, x, residual),
        warmup=25,
        rep=200,
        quantiles=[0.5, 0.2, 0.8],
    )
    bytes_moved = 4 * x.numel() * x.element_size()

    return (
        _bandwidth_in_gbps(bytes_moved, ms),
        _bandwidth_in_gbps(bytes_moved, ms_p80),
        _bandwidth_in_gbps(bytes_moved, ms_p20),
    )


def _bandwidth_in_gbps(bytes_moved, time_in_milliseconds):
    return bytes_moved * 1e-9 / (time_in_milliseconds * 1e-3)


if __name__ == "__main__":
    results = Path("benchmarks/results/rmsnorm_residual_add")
    results.mkdir(parents=True, exist_ok=True)
    benchmark.run(print_data=True, save_path=str(results))
