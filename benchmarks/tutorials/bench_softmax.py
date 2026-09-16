from pathlib import Path

import torch
import triton
import triton.testing

from tutorials.fused_softmax import (
    cpu_softmax,
    gpu_softmax_with_one_program_per_row,
    gpu_softmax_with_persistent_grid,
    torch_softmax,
)

PROVIDERS = {
    "triton_one_program_per_row": gpu_softmax_with_one_program_per_row,
    "triton_persistent_grid": gpu_softmax_with_persistent_grid,
    "torch": torch_softmax,
    "naive": cpu_softmax,
}

triton_benchmark = triton.testing.Benchmark(
    x_names=["number_of_columns"],
    x_vals=[128 * i for i in range(2, 100)],
    line_arg="provider",
    line_vals=list(PROVIDERS),
    line_names=["Triton (one program per row)", "Triton (persistent grid)", "Torch", "Naive"],
    styles=[("green", "-"), ("green", "--"), ("blue", "-"), ("red", "-")],
    ylabel="GB/s",
    plot_name="softmax-bandwidth",
    args={"number_of_rows": 4096},
)


@triton.testing.perf_report([triton_benchmark])
def benchmark(number_of_rows, number_of_columns, provider):
    x = torch.randn(number_of_rows, number_of_columns, device="cuda", dtype=torch.float32)
    milliseconds = triton.testing.do_bench(lambda: PROVIDERS[provider](x))

    bytes_moved = 2 * x.numel() * x.element_size()  # one read of x, one write of the result
    return bytes_moved * 1e-9 / (milliseconds * 1e-3)


if __name__ == "__main__":
    results = Path("benchmarks/results/softmax")
    results.mkdir(parents=True, exist_ok=True)
    benchmark.run(print_data=True, save_path=str(results))
