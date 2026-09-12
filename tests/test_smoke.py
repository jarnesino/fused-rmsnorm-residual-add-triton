"""Environment smoke test: Triton compiles and runs a trivial kernel on this device.

Under TRITON_INTERPRET=1 this exercises the CPU interpreter — if it passes,
the container is good and every non-gpu test in the repo can run here.
"""

import torch
import triton
import triton.language as tl

def _add_kernel(X, Y, Z, N, grid, block_size):
    _add_kernel_using_constant_block_size[grid](X, Y, Z, N, tl.constexpr(block_size))

@triton.jit
def _add_kernel_using_constant_block_size(x_ptr, y_ptr, output_ptr, number_of_elements, block_size: tl.constexpr):
    pid = tl.program_id(0)
    offset = pid * block_size + tl.arange(0, block_size)
    mask = offset < number_of_elements

    x = tl.load(x_ptr + offset, mask=mask, other=0.0)
    y = tl.load(y_ptr + offset, mask=mask, other=0.0)

    tl.store(output_ptr + offset, x + y, mask=mask)


def test_triton_runs(device):
    block_size = 256
    n = 1000  # Not divisible by the block size (needs the mask)
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    z = torch.empty_like(x)
    grid = (triton.cdiv(n, block_size),)

    _add_kernel(x, y, z, n, grid, block_size)

    torch.testing.assert_close(z, x + y)

