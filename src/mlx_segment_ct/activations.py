"""GELU with PyTorch 2.14.1's ARM float32 erf and exponential arithmetic."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from medmlx_core.typing import MetalKernel

from mlx_segment_ct._types import CoreRuntime

if TYPE_CHECKING:
    from mlx.core import array

_GELU = """uint i = thread_position_in_grid.x;
if (i >= SIZE) return;
float x = X[i], z = x * 0.70710678118654752440f;
float t = 1.0f / fma(0.3275911f, abs(z), 1.0f);
float r = fma(1.061405429f, t, -1.453152027f);
r = fma(r, t, 1.421413741f);
r = fma(r, t, -0.284496736f);
r = fma(r, t, 0.254829592f);
float exponent = -(z * z), exponential;
if (exponent < -0x1.5d5e2ap+6f) {
    exponential = exp(exponent);
} else {
    // PyTorch's NEON exp_u20 range reduction and polynomial.
    float n = round(exponent * 0x1.715476p+0f);
    float reduced = fma(-n, 0x1.62e4p-1f, exponent);
    reduced = fma(-n, 0x1.7f7d1cp-20f, reduced);
    float scale = as_type<float>((uint(int(n)) << 23) + 0x3f800000u);
    float square = reduced * reduced;
    float p = fma(reduced, 0x1.0e4020p-7f, 0x1.573e2ep-5f);
    float q = fma(reduced, 0x1.555e66p-3f, 0x1.fffdb6p-2f);
    q = fma(p, square, q);
    float polynomial = fma(q, square, 0x1.ffffecp-1f * reduced);
    exponential = fma(polynomial, scale, scale);
}
float erf_value = fma(-t * exponential, r, 1.0f);
erf_value = as_type<float>(as_type<uint>(erf_value) ^ (as_type<uint>(z) & 0x80000000u));
Y[i] = (x * 0.5f) * (1.0f + erf_value);
"""


class Gelu:
    """Source-ordered ARM GELU on Metal; no reference-runtime dependency.

    The kernel reproduces PyTorch's vectorized erf (a polynomial fit) and exponential
    in the reference's operation order, with floating-point contraction disabled
    except for the explicit ``fma`` calls.
    """

    def __init__(self, mx: CoreRuntime) -> None:
        """Store the native runtime used for the GELU activation."""
        self.mx = mx
        self.kernel = cast(
            MetalKernel,
            mx.fast.metal_kernel(
                name="vista_arm_gelu",
                input_names=["X"],
                output_names=["Y"],
                source=_GELU,
                header="#pragma clang fp contract(off)\n",
            ),
        )

    def __call__(self, values: array) -> array:
        """Evaluate erf-form GELU using the kernel's float32 approximation.

        The result has the input shape; this polynomial implementation is not an
        exact mathematical evaluation of the error function.
        """
        return self.kernel(
            inputs=[values],
            template=[("SIZE", values.size)],
            grid=(values.size, 1, 1),
            threadgroup=(256, 1, 1),
            output_shapes=[tuple(values.shape)],
            output_dtypes=[self.mx.float32],
        )[0]
