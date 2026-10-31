"""The VISTA3D 48-feature biased projections with ARM CPU accumulation order."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from medmlx_core.typing import MetalKernel

from mlx_segment_ct._types import CoreRuntime

if TYPE_CHECKING:
    from mlx.core import array

_PROJECT = """uint i = thread_position_in_grid.x;
if (i >= ROWS * 48) return;
uint row = i / 48, channel = i % 48;
if (LINEAR && ROWS < 12) {
    float4 total = float4(0);
    for (uint k = 0; k < 48; k += 4) {
        float4 x = float4(X[row * 48 + k], X[row * 48 + k + 1],
                          X[row * 48 + k + 2], X[row * 48 + k + 3]);
        float4 w = float4(W[channel * 48 + k], W[channel * 48 + k + 1],
                          W[channel * 48 + k + 2], W[channel * 48 + k + 3]);
        total = fma(x, w, total);
    }
    Y[i] = ((total.x + total.y) + (total.z + total.w)) + B[channel];
    return;
}
// Accelerate seeds bias in full 32-column tiles. Its transpose-layout
// pointwise GEMM also seeds the interior 64-row tiles; edge tiles add it last.
uint interior_end = 16 + ((ROWS - 16) / 64) * 64;
bool seed_bias = channel < 32 || (!LINEAR && row >= 16 && row < interior_end);
float total = seed_bias ? B[channel] : 0.0f;
for (uint k = 0; k < 48; ++k) {
    uint offset = LINEAR ? row * 48 + k : k * ROWS + row;
    total = fma(X[offset], W[channel * 48 + k], total);
}
uint output = LINEAR ? i : channel * ROWS + row;
Y[output] = seed_bias ? total : total + B[channel];
"""


class Projections:
    """Class MLP and encoder head; the admitted encoder has at least 8192 rows.

    Both are 48-in, 48-out projections with bias. They run as a custom Metal kernel
    that reproduces the ARM CPU reference's accumulation order. With ``linear`` and
    fewer than 12 rows the kernel uses a four-lane fma reduction, matching the CPU's
    small-GEMM path; otherwise it accumulates sequentially over the 48 inputs and
    places the bias as the reference's tiling does.
    """

    def __init__(self, mx: CoreRuntime) -> None:
        """Initialize the runtime and weights for projection layers."""
        self.mx = mx
        self.kernel = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="vista_arm_projection",
                input_names=["X", "W", "B"],
                output_names=["Y"],
                source=_PROJECT,
                header="#pragma clang fp contract(off)\n",
            ),
        )

    def __call__(self, values: array, weight: array, bias: array, *, linear: bool) -> array:
        """Apply a 48-to-48 projection in float32.

        With ``linear`` true, ``values`` is ``(rows, 48)`` (class MLP) and ``weight`` is
        ``(48, 48)``. Otherwise ``values`` is channel-first ``(1, 48, D, H, W)`` (the
        encoder head, a 1x1x1 convolution) and ``weight`` is ``(48, 48, 1, 1, 1)``.
        ``bias`` is ``(48,)``. The result has the shape of ``values``. Shapes
        and dtypes are caller requirements and are not checked before dispatch.
        """
        return self.kernel(
            inputs=[values, weight, bias],
            template=[("ROWS", values.size // 48), ("LINEAR", linear)],
            grid=(values.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[tuple(values.shape)],
            output_dtypes=[self.mx.float32],
        )[0]
