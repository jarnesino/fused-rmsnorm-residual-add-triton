import torch
import triton
import triton.language as tl

from tutorials.device import DeviceProperties


def cpu_softmax(x):
    # Read MN elements and write M elements
    maximum_value = x.max(dim=1)[0]
    # Read MN + M elements and write MN elements
    adjusted_x = x - maximum_value[:, None]  # Avoids overflows (softmax is invariant to this)

    # Read MN elements and write MN elements
    numerator = torch.exp(adjusted_x)
    # Read MN elements and write M elements
    denominator = numerator.sum(dim=1)

    # Read MN + M elements and write MN elements
    ret = numerator / denominator[:, None]
    # Total: read 5MN + 2M elements and wrote 3MN + 2M elements
    return ret


def gpu_softmax(input):
    number_of_rows, number_of_columns = input.shape
    block_size = triton.next_power_of_2(number_of_columns)
    result = torch.empty_like(input)

    kernel_arguments = (
        result,
        input,
        result.stride(0),
        input.stride(0),
        number_of_rows,
        number_of_columns,
        tl.constexpr(block_size),
    )

    kernel, grid, number_of_stages = compile_persistent_kernel(
        softmax_kernel_with_one_row_per_block,
        kernel_arguments,
        input.device,
        number_of_work_items=number_of_rows,
    )

    kernel[grid](*kernel_arguments, number_of_stages)

    return result


def compile_persistent_kernel(kernel_function, kernel_arguments, device, number_of_work_items):
    """
    Compiles kernel_function for these arguments and size a persistent grid for it.
    Unnecessary to write, but kept for tutorial purposes. Autotune does this same job.
    """

    device_properties = DeviceProperties.new_for_device(device)

    # Heuristics
    number_of_stages = 4 if device_properties.max_shared_memory_bytes() > 200_000 else 2
    number_of_warps = 8

    kernel = kernel_function.warmup(
        *kernel_arguments, num_stages=number_of_stages, num_warps=number_of_warps, grid=(1,)
    )  # Pre-compile kernel to infer register usage and later calculate thread occupancy

    kernel._init_handles()  # Load kernel onto device
    registers_per_thread = kernel.n_regs
    shared_memory_bytes_per_program = kernel.metadata.shared

    registers_per_program = registers_per_thread * device_properties.warp_size() * number_of_warps
    shared_memory_divisor = max(shared_memory_bytes_per_program, 1)  # To avoid zero division
    programs_per_multiprocessor = min(
        device_properties.max_registers_per_multiprocessor() // registers_per_program,
        device_properties.max_shared_memory_bytes() // shared_memory_divisor,
    )  # Occupancy

    resident_program_capacity = (
        device_properties.streaming_multiprocessor_count() * programs_per_multiprocessor
    )  # How many programs the GPU can hold at once
    number_of_programs = min(resident_program_capacity, number_of_work_items)  # Grid size

    grid = (number_of_programs, 1, 1)

    return kernel, grid, number_of_stages


@triton.jit
def softmax_kernel_with_one_row_per_block(
    output_ptr,
    input_ptr,
    output_row_stride,
    input_row_stride,
    n_rows,
    n_cols,
    block_size,
    num_stages: tl.constexpr,
):
    starting_row_index = tl.program_id(0)
    row_step = tl.num_programs(0)

    for row_index in tl.range(starting_row_index, n_rows, row_step, num_stages=num_stages):
        row_start_ptr = input_ptr + row_index * input_row_stride
        column_offsets = tl.arange(0, block_size)
        input_ptrs = row_start_ptr + column_offsets

        mask = column_offsets < n_cols  # Just in case the row does not fill the whole column
        negative_infinity = -float("inf")
        row = tl.load(input_ptrs, mask=mask, other=negative_infinity)  # Load row into SRAM

        # Avoids overflows (softmax is invariant to this)
        adjusted_row = row - tl.max(row, axis=0)

        numerator = tl.exp(adjusted_row)  # Fast but approximate
        denominator = tl.sum(numerator, axis=0)

        softmax_output = numerator / denominator

        output_row_start_ptr = output_ptr + row_index * output_row_stride
        output_ptrs = output_row_start_ptr + column_offsets
        tl.store(output_ptrs, softmax_output, mask=mask)  # Write output to DRAM
