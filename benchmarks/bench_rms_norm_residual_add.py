import argparse
import json
import subprocess
import sys
import time
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pandas
import torch
import triton
import triton.testing

from fused_rmsnorm_residual_add.fused import FusedImplementation, fused_rmsnorm_residual_add_kernel
from fused_rmsnorm_residual_add.reference import (
    CompiledLlamaStyleImplementation,
    FunctionalOnEagerTorchImplementation,
    LigerImplementation,
    NaiveLlamaStyleImplementation,
)

BENCH_PARAMS = {"warmup": 25, "rep": 200, "quantiles": [0.5, 0.2, 0.8]}

_measurements = []


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
    "liger": (LigerImplementation, OutOfPlaceCallingPolicy),
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
            "Liger",
            "Naive Llama style",
            "Compiled Llama style",
            "torch.rms_norm",
        ],
        styles=[
            ("green", "-"),
            ("green", "--"),
            ("orange", "-"),
            ("blue", "-"),
            ("blue", "--"),
            ("red", "-"),
        ],
        ylabel="GB/s (median)",
        plot_name=f"forward_{axis}_{short_name}",
        args={**fixed_args, "data_type": data_type},
    )


triton_benchmarks = [
    _benchmark_for(*sweep, data_type)
    for sweep in (COLUMN_SWEEP, ROW_SWEEP)
    for data_type in DATA_TYPES
]

VARIANCE_EPSILON = 1e-6


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
        lambda: calling_policy.call(operation, x, residual), **BENCH_PARAMS
    )

    bytes_moved = 4 * x.numel() * x.element_size()

    gbps_median = _bandwidth_in_gbps(bytes_moved, ms)
    gbps_p20 = _bandwidth_in_gbps(bytes_moved, ms_p80)
    gbps_p80 = _bandwidth_in_gbps(bytes_moved, ms_p20)

    _measurements.append(
        {
            "data_type": str(data_type).removeprefix("torch."),
            "number_of_rows": number_of_rows,
            "number_of_columns": number_of_columns,
            "provider": provider,
            "milliseconds_median": ms,
            "milliseconds_p20": ms_p20,
            "milliseconds_p80": ms_p80,
            "bytes_moved": bytes_moved,
            "gbps_median": gbps_median,
            "gbps_p20": gbps_p20,
            "gbps_p80": gbps_p80,
        }
    )

    return gbps_median, gbps_p20, gbps_p80


def _bandwidth_in_gbps(bytes_moved, time_in_milliseconds):
    return bytes_moved * 1e-9 / (time_in_milliseconds * 1e-3)


def _shell(*command):
    result = subprocess.run(command, capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else None


def _liger_version():
    try:
        return version("liger-kernel")
    except PackageNotFoundError:
        return None


def _gpu_slug(device):
    return torch.cuda.get_device_name(device).lower().replace(" ", "-").replace("/", "-")


def _metadata(device, clocks_locked, run_index):
    return {
        "gpu": torch.cuda.get_device_name(device),
        "capability": _capability(device),
        "driver": _driver(),
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "liger": _liger_version(),
        "python": sys.version.split()[0],
        "git_sha": _shell("git", "rev-parse", "HEAD"),
        "git_is_dirty": _git_is_dirty(),
        "do_bench": BENCH_PARAMS,
        "clocks_locked": clocks_locked,
        "autotuned_configurations": {
            f"N={key[0]} {key[1]}": {
                "num_warps": configuration.num_warps,
                "programs_per_sm": configuration.kwargs["programs_per_sm"],
            }
            for key, configuration in fused_rmsnorm_residual_add_kernel.cache.items()
        },
        "seed": _torch_seed(),
        "run_index": run_index,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }


def _capability(cuda_device):
    return list(torch.cuda.get_device_capability(cuda_device))


def _driver():
    return _shell("nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader")


def _git_is_dirty():
    return bool(_shell("git", "status", "--porcelain", "--", ".", ":!benchmarks/results"))


def _write_run(directory, device, clocks_locked, run_index):
    metadata = _metadata(device, clocks_locked, run_index)
    (directory / "meta.json").write_text(json.dumps(metadata, indent=2))
    (directory / "env.txt").write_text(_shell(sys.executable, "-m", "pip", "freeze") or "")
    pandas.DataFrame(_measurements).to_csv(directory / "measurements.csv", index=False)


def _arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--clocks-locked", action="store_true")
    parser.add_argument("--results", type=Path, default=Path(__file__).parent / "results")

    return parser.parse_args()


def _device():
    return triton.runtime.driver.active.get_active_torch_device()


def _torch_seed():
    return 0


if __name__ == "__main__":
    arguments = _arguments()
    device = _device()

    base_directory = arguments.results / _gpu_slug(device)

    for run_index in range(arguments.runs):
        FusedImplementation.clear_kernel_cache()
        torch.manual_seed(_torch_seed())
        _measurements.clear()

        run_directory = base_directory / time.strftime("%Y%m%dT%H%M%S")
        run_directory.mkdir(parents=True, exist_ok=True)

        benchmark.run(print_data=True, save_path=str(run_directory))
        _write_run(run_directory, device, arguments.clocks_locked, run_index)
