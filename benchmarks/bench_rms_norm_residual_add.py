from pathlib import Path

import torch
import triton
import triton.testing

from fused_rmsnorm_residual_add.fused import FusedImplementation
from fused_rmsnorm_residual_add.reference import (
    FunctionalOnEagerTorchImplementation,
    HuggingFaceLlamaStyleImplementation,
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
    "hf_llama_style": (HuggingFaceLlamaStyleImplementation, OutOfPlaceCallingPolicy),
    "torch_functional": (FunctionalOnEagerTorchImplementation, OutOfPlaceCallingPolicy),
}

triton_benchmarks = [
    triton.testing.Benchmark(
        x_names=["number_of_columns"],
        x_vals=[768, 1024, 2048, 3072, 4096, 5120, 8192, 12288, 16384],
        line_arg="provider",
        line_vals=list(PROVIDERS),
        line_names=["Fused", "Fused (in place)", "Llama style", "torch.rms_norm"],
        styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("red", "-")],
        ylabel="GB/s",
        plot_name=f"rmsnorm-residual-add-bandwidth-{str(data_type).removeprefix('torch.')}",
        args={"number_of_rows": 4096, "data_type": data_type},
    )
    for data_type in (torch.bfloat16, torch.float16)
]


@triton.testing.perf_report(triton_benchmarks)
def benchmark(number_of_rows, number_of_columns, data_type, provider):
    tensor_options = dict(device="cuda", dtype=data_type)
    x = torch.randn(number_of_rows, number_of_columns, **tensor_options)
    residual = torch.randn(number_of_rows, number_of_columns, **tensor_options)
    weight = torch.randn(number_of_columns, **tensor_options)

    implementation, calling_policy_class = PROVIDERS[provider]
    operation = implementation(weight, VARIANCE_EPSILON)
    calling_policy = calling_policy_class()
    milliseconds = triton.testing.do_bench(lambda: calling_policy.call(operation, x, residual))

    # Two reads (x, residual), two writes (normalized_output and residual_output), weight is ignored
    bytes_moved = 4 * x.numel() * x.element_size()

    gigabytes_moved = bytes_moved * 1e-9
    seconds_elapsed = milliseconds * 1e-3
    return gigabytes_moved / seconds_elapsed


if __name__ == "__main__":
    results = Path("benchmarks/results/rmsnorm_residual_add")
    results.mkdir(parents=True, exist_ok=True)
    benchmark.run(print_data=True, save_path=str(results))
