"""Environment smoke test: Triton compiles and runs a trivial kernel on this device.

Under TRITON_INTERPRET=1 this exercises the CPU interpreter — if it passes,
the container is good and every non-gpu test in the repo can run here.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _add_kernel(X, Y, Z, N, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < N
    x = tl.load(X + offs, mask=mask, other=0.0)
    y = tl.load(Y + offs, mask=mask, other=0.0)
    tl.store(Z + offs, x + y, mask=mask)


def test_triton_runs(device):
    n = 1000  # deliberately not a power of two: exercises the mask
    x = torch.randn(n, device=device)
    y = torch.randn(n, device=device)
    z = torch.empty_like(x)
    grid = (triton.cdiv(n, 256),)
    _add_kernel[grid](x, y, z, n, BLOCK=256)
    torch.testing.assert_close(z, x + y)


def test_versions_print():
    print(
        f"\ntorch {torch.__version__}"
        f"  triton {triton.__version__}"
        f"  cuda={torch.cuda.is_available()}"
    )
