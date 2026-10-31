"""FP32 upsampling for the VISTA3D automatic decoder."""

from __future__ import annotations

from itertools import product
from typing import TYPE_CHECKING

from mlx_segment_ct._types import CoreRuntime

if TYPE_CHECKING:
    from mlx.core import array

from medmlx_core.precision import Float32Operators


class AutomaticUpsample:
    """Kernel 3, stride 2, padding 1, output padding 1, without bias.

    Compute the eight output parities at input resolution. This avoids the
    full-resolution convolution workspace used by MLX's generic deconvolution.

    ``weight`` is the Torch ``ConvTranspose3d`` layout ``(in, out, 3, 3, 3)``. Calling
    the module maps float32 ``(B, in, D, H, W)`` to ``(B, out, 2D, 2H, 2W)``.
    """

    def __init__(self, weight: array, *, mx: CoreRuntime) -> None:
        """Store the runtime and weights for automatic upsampling."""
        self.mx = mx
        self.phases = tuple(product(range(2), repeat=3))
        self.operators = Float32Operators(mx, 1, 1e-5)
        self.weights: list[list[tuple[tuple[int, ...], array]]] = []
        for phase in self.phases:
            contributions: list[tuple[tuple[int, ...], array]] = []
            # Torch contracts input channels first, then col2vol adds kernel
            # offsets in z/y/x order. Combining these reductions changes rounding.
            # Along each axis an even output position uses only the center tap (1);
            # an odd one uses taps 0 and 2, fed by the next and the same input.
            for kernel in product(*[(0, 2) if odd else (1,) for odd in phase]):
                packed = weight[:, :, kernel[0], kernel[1], kernel[2]].T
                shifts = tuple(int(offset == 0) for offset in kernel)
                contributions.append((shifts, mx.contiguous(packed[:, :, None, None, None])))
            self.weights.append(contributions)
        mx.eval([packed for phase in self.weights for _, packed in phase])

    def __call__(self, x: array) -> array:
        """Run the automatic decoder upsampling and convolution path."""
        mx = self.mx
        batch, _, depth, height, width = x.shape
        values = mx.pad(x, [(0, 0), (0, 0), (0, 1), (0, 1), (0, 1)])
        outputs: list[array] = []

        def convolve(shifts: tuple[int, ...], weight: array) -> array:
            """Compute one shifted convolution contribution."""
            z, y, offset = shifts
            return self.operators.conv(
                values[:, :, z : z + depth, y : y + height, offset : offset + width],
                weight,
                None,
                padding=0,
                stride=1,
            )

        for contributions in self.weights:
            shifts, weight = contributions[0]
            result = convolve(shifts, weight)
            for shifts, weight in contributions[1:]:
                result = result + convolve(shifts, weight)
            outputs.append(result)
        channels = outputs[0].shape[1]
        interleaved = mx.reshape(
            mx.stack(outputs, axis=-1), (batch, channels, depth, height, width, 2, 2, 2)
        )
        return mx.reshape(
            mx.transpose(interleaved, (0, 1, 2, 5, 3, 6, 4, 7)),
            (batch, channels, 2 * depth, 2 * height, 2 * width),
        )
