"""Seeded inputs shared by the upstream-reference recorder and parity tests."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_segment_ct._types import (
        ArrayValue,
        Float32Cast,
        Float32Dtype,
        NumericArrayArithmetic,
        StandInRuntime,
    )
    from scripts.oracle_types import HostArray

import numpy as np


def random_weights(shapes: Mapping[str, tuple[int, ...]], seed: int) -> dict[str, HostArray]:
    """Create reproducible synthetic weights matching the requested tensor shapes."""
    rng = np.random.default_rng(seed)
    weights: dict[str, HostArray] = {}
    for name, shape in shapes.items():
        value = rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)
        if ".norm" in name and name.endswith(".weight"):
            value = value + np.float32(1)
        weights[name] = value
    return weights


class AutomaticStandIn[ArrayT: ArrayValue, DtypeT: Float32Dtype]:
    """Cheap seeded MLX network shared by both real window paths, without mocks."""

    def __init__(self, seed: int, *, mx: StandInRuntime[ArrayT, DtypeT]) -> None:
        """Store the array runtime for synthetic automatic scores."""
        self.mx = mx
        rng = np.random.default_rng(seed)
        self.scale = mx.array(rng.standard_normal((1, 3, 1, 1, 1), dtype=np.float32))
        self.bias = mx.array(rng.standard_normal((1, 3, 1, 1, 1), dtype=np.float32))

    def __call__(self, image: ArrayT) -> ArrayT:
        # A local spatial term makes overlapping windows disagree, exercising
        # blend/count/order logic rather than an identity-stitching tautology.
        """Produce deterministic synthetic automatic scores on the input grid."""
        grid = np.indices(image.shape[2:], dtype=np.float32)
        ramp = (
            grid[0] * np.float32(0.125)
            + grid[1] * np.float32(0.0625)
            + grid[2] * np.float32(0.03125)
        )
        return cast("ArrayT", image * self.scale + self.bias + self.mx.array(ramp)[None, None])


class PointStandIn[ArrayT: ArrayValue, DtypeT: Float32Dtype]:
    """Point-model-shaped MLX callable that depends on normalized clicks and codes."""

    def __init__(self, seed: int, *, mx: StandInRuntime[ArrayT, DtypeT]) -> None:
        """Store the array runtime for synthetic point scores."""
        self.mx = mx
        rng = np.random.default_rng(seed)
        self.scale = mx.array(rng.uniform(0.5, 1.5), dtype=mx.float32)
        self.bias = mx.array(rng.uniform(-0.25, 0.25), dtype=mx.float32)

    def __call__(self, image: ArrayT, positions: ArrayT, codes: ArrayT, target: int) -> ArrayT:
        """Produce deterministic synthetic point-conditioned scores."""
        mx = self.mx
        return cast(
            "ArrayT",
            (
                cast(
                    "NumericArrayArithmetic[ArrayT]",
                    cast("NumericArrayArithmetic[ArrayT]", image * self.scale + self.bias)
                    + mx.sum(positions) * 0.125,
                )
                + mx.sum(cast("Float32Cast[ArrayT, DtypeT]", codes).astype(mx.float32)) * 0.03125
                + target * 0.00390625
            ),
        )


class AutomaticModelStand[ArrayT: ArrayValue, DtypeT: Float32Dtype]:
    """``AutomaticModel``-compatible stand-in for runner fixtures: (weights, class_ids, *, mx).

    Scales are powers of two and offsets are dyadic, so products are exact and any
    fused multiply-add rounds like the unfused sum. A patch-local ramp makes classes
    win in different regions and overlapping windows disagree; low scores stay
    background.
    """

    def __init__(
        self, weights: object, class_ids: Sequence[int], *, mx: StandInRuntime[ArrayT, DtypeT]
    ) -> None:
        """Initialize synthetic automatic-model parameters for the selected classes."""
        self.mx = mx
        ids = np.asarray(class_ids, dtype=np.int64)
        shape = (1, len(ids), 1, 1, 1)
        self.scale = mx.array((2.0 ** -(ids % 3) / 4).astype(np.float32).reshape(shape))
        self.bias = mx.array(((ids % 5) / 16 - 0.25).astype(np.float32).reshape(shape))
        self.slope = mx.array((((ids % 4) - 1.5) / 4).astype(np.float32).reshape(shape))

    def __call__(self, image: ArrayT) -> ArrayT:
        """Produce synthetic per-class scores for a model window."""
        mx = self.mx
        depth, height, width = image.shape[2:]
        ramp = (
            np.arange(depth, dtype=np.float32)[:, None, None] / np.float32(2)
            + np.arange(height, dtype=np.float32)[None, :, None]
            - np.arange(width, dtype=np.float32)[None, None, :]
            - np.float32(depth // 4)
        ) / np.float32(4)
        return cast(
            "ArrayT", image * self.scale + mx.array(ramp)[None, None] * self.slope + self.bias
        )
