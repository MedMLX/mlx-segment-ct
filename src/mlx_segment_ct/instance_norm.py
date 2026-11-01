"""MONAI's contiguous CPU InstanceNorm arithmetic, evaluated on Metal."""

from __future__ import annotations

import struct
from itertools import count
from math import prod
from typing import TYPE_CHECKING, Literal, cast

from medmlx_core.typing import MetalKernel

from mlx_segment_ct._types import CoreRuntime

if TYPE_CHECKING:
    from mlx.core import array

_NAMES = count()
_HEADER = """#pragma clang fp contract(off)
// Two float32 components retain the CPU accumulator's low-order terms.
float2 accumulate(float2 total, float value) {
    float sum = total.x + value, recovered = sum - total.x;
    float error = (total.x - (sum - recovered)) + (value - recovered) + total.y;
    float high = sum + error;
    return float2(high, error - (high - sum));
}
"""
_SOURCES = {
    "sum": (
        ["X", "MEAN"],
        ["TOTAL"],
        """
        uint i = thread_position_in_grid.x;
        if (i >= GROUPS * TILES) return;
        uint g = i / TILES, start = (i % TILES) * 256;
        float2 total = float2(0);
        for (uint j = start; j < min(start + 256, uint(N)); ++j) {
            float x = X[g * N + j];
            if (VARIANCE) { float delta = x - MEAN[g]; x = delta * delta; }
            total = accumulate(total, x);
        }
        TOTAL[2 * i] = total.x; TOTAL[2 * i + 1] = total.y;
        """,
    ),
    "mean": (
        ["TOTAL"],
        ["MEAN"],
        """
        uint g = thread_position_in_grid.x;
        if (g >= GROUPS) return;
        float2 total = float2(0);
        for (uint t = 0; t < TILES; ++t) {
            total = accumulate(total, TOTAL[2 * (g * TILES + t)]);
            total = accumulate(total, TOTAL[2 * (g * TILES + t) + 1]);
        }
        float mean = total.x / float(N);
        float residual = fma(-mean, float(N), total.x) + total.y;
        MEAN[g] = mean + residual / float(N);
        """,
    ),
    "coeff": (
        ["TOTAL", "MEAN", "W", "B", "EPS"],
        ["SCALE", "BIAS"],
        """
        uint g = thread_position_in_grid.x;
        if (g >= GROUPS) return;
        float2 total = float2(0);
        for (uint t = 0; t < TILES; ++t) {
            total = accumulate(total, TOTAL[2 * (g * TILES + t)]);
            total = accumulate(total, TOTAL[2 * (g * TILES + t) + 1]);
        }
        // CPU BatchNorm stores its variance sum in float32 before dividing.
        float variance = (total.x + total.y) / float(N);
        float hi = variance + EPS[0], recovered = hi - variance;
        float lo = (variance - (hi - recovered)) + (EPS[0] - recovered) + EPS[1];
        float r = rsqrt(hi), q = hi * r, qe = fma(hi, r, -q);
        float error = fma(-q, r, 1.0f) - qe * r - lo * r * r;
        r = fma(0.5f * r, error, r);
        float scale = r * W[g % CHANNELS];
        SCALE[g] = scale; BIAS[g] = fma(-MEAN[g], scale, B[g % CHANNELS]);
        """,
    ),
    "affine": (
        ["X", "SCALE", "BIAS", "RESIDUAL"],
        ["Y"],
        """
        uint i = thread_position_in_grid.x;
        if (i >= GROUPS * N) return;
        uint g = i / N;
        // Torch's vectorized affine rounds the multiplication before addition.
        float value = X[i] * SCALE[g] + BIAS[g];
        // Keep residual addition after the rounded affine, before activation.
        if (ADD_RESIDUAL) value = value + RESIDUAL[i];
        if (ACTIVATION == 1) value = isnan(value) ? value : max(value, 0.0f);
        if (ACTIVATION == 2) value = value >= 0.0f ? value : 0.01f * value;
        Y[i] = value;
        """,
    ),
}


class InstanceNorm:
    """Two-pass moments and source-rounded affine for NCDHW InstanceNorm."""

    def __init__(self, mx: CoreRuntime, eps: float = 1e-5) -> None:
        """Initialize the native instance-normalization runtime."""
        self.mx = mx
        # eps as a float32 high part plus its residual; the coefficient kernel adds both.
        high = struct.unpack("f", struct.pack("f", eps))[0]
        self.eps = mx.array([high, eps - high], dtype=mx.float32)
        namespace = next(_NAMES)
        self.kernels = {
            name: cast(
                MetalKernel,
                mx.fast.metal_kernel(
                    name=f"vista_instance_norm_{namespace}_{name}",
                    input_names=inputs,
                    output_names=outputs,
                    source=source,
                    header=_HEADER,
                ),
            )
            for name, (inputs, outputs, source) in _SOURCES.items()
        }

    def __call__(
        self,
        values: array,
        weight: array,
        bias: array,
        *,
        activation: Literal["none", "relu", "leaky_relu"] = "none",
        residual: array | None = None,
    ) -> array:
        """Normalize each (sample, channel) over its spatial voxels, then scale and shift.

        ``values`` is float32 ``(B, C, D, H, W)``; ``weight`` and ``bias`` are ``(C,)``.
        An optional float32 residual of the same shape is added after the affine,
        before the activation (leaky ReLU uses slope 0.01). The two-pass moments
        and each float32 rounding step are unchanged. Returns the input shape.
        These shapes and dtypes are caller requirements, not independently
        validated before the Metal kernels are dispatched.
        """
        batch, channels = values.shape[:2]
        size = prod(values.shape[2:])
        groups = batch * channels
        tiles = (size + 255) // 256
        constants = {"GROUPS": groups, "CHANNELS": channels, "TILES": tiles, "N": size}

        def run(
            name: str, inputs: list[array], shape: tuple[int, ...], *, variance: bool = False
        ) -> list[array]:
            """Dispatch one instance-normalization kernel for the selected layout."""
            template: list[tuple[str, int | bool]] = [
                *constants.items(),
                ("VARIANCE", variance),
            ]
            if name == "affine":
                template.extend(
                    [
                        ("ACTIVATION", ("none", "relu", "leaky_relu").index(activation)),
                        ("ADD_RESIDUAL", residual is not None),
                    ]
                )
            return self.kernels[name](
                inputs=inputs,
                template=template,
                grid=(groups * tiles if name == "sum" else prod(shape), 1, 1),
                threadgroup=(256, 1, 1),
                output_shapes=[shape] * len(_SOURCES[name][1]),
                output_dtypes=[self.mx.float32] * len(_SOURCES[name][1]),
            )

        # The first pass sums raw values and ignores its MEAN input (VARIANCE is false);
        # ``weight`` only fills that input slot.
        totals = run("sum", [values, weight], (groups, tiles, 2))[0]
        mean = run("mean", [totals], (groups,))[0]
        totals = run("sum", [values, mean], (groups, tiles, 2), variance=True)[0]
        scale, offset = run("coeff", [totals, mean, weight, bias, self.eps], (groups,))
        # The disabled residual slot uses an already contiguous small coefficient
        # buffer, avoiding a second contiguous copy when values is a strided view.
        return run(
            "affine",
            [values, scale, offset, scale if residual is None else residual],
            tuple(values.shape),
        )[0]
