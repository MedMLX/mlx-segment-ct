"""Exact parity for fused normalization epilogues on NCDHW inputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

import numpy as np
import pytest

from mlx_segment_ct._types import ArrayBoundary, CoreRuntime
from mlx_segment_ct.instance_norm import InstanceNorm

if TYPE_CHECKING:
    from mlx.core import array


@pytest.mark.parametrize("shape", [(2, 7, 3, 5, 19), (1, 48, 16, 16, 16)])
@pytest.mark.parametrize("activation", ["none", "relu", "leaky_relu"])
@pytest.mark.parametrize("with_residual", [False, True])
def test_fused_epilogue_preserves_float32_rounding(
    mx: CoreRuntime,
    shape: tuple[int, ...],
    activation: Literal["none", "relu", "leaky_relu"],
    with_residual: bool,
) -> None:
    """Compare fused output against separately materialized affine/add/activation."""
    rng = np.random.default_rng(591)
    boundary = cast("ArrayBoundary[array]", mx)
    host = rng.standard_normal(shape, dtype=np.float32)
    # Include a constant channel and tiny variance around a large channel mean.
    host[:, 0] = 3.0
    host[:, 1] = 1024.0 + host[:, 1] * np.float32(0.001)
    values = boundary.array(host)
    weight = boundary.array(rng.standard_normal((shape[1],), dtype=np.float32))
    bias = boundary.array(rng.standard_normal((shape[1],), dtype=np.float32))
    residual = boundary.array(rng.standard_normal(shape, dtype=np.float32))
    norm = InstanceNorm(mx)
    expected = norm(values, weight, bias)
    mx.eval(expected)
    if with_residual:
        expected = expected + residual
        mx.eval(expected)
    if activation == "relu":
        expected = mx.maximum(expected, 0)
    elif activation == "leaky_relu":
        expected = mx.where(expected >= 0, expected, 0.01 * expected)
    actual = norm(
        values,
        weight,
        bias,
        activation=activation,
        residual=residual if with_residual else None,
    )
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32)
    )


@pytest.mark.parametrize("activation", ["relu", "leaky_relu"])
def test_fused_activation_preserves_special_values(
    mx: CoreRuntime, activation: Literal["relu", "leaky_relu"]
) -> None:
    """Keep signed-zero, infinity and NaN behavior of the separate activation."""
    boundary = cast("ArrayBoundary[array]", mx)
    values = boundary.array(np.full((1, 5, 1, 1, 4), -0.0, dtype=np.float32))
    weight = mx.ones((5,))
    bias = boundary.array(np.array([-0.0, 0.0, np.inf, -np.inf, np.nan], dtype=np.float32))
    norm = InstanceNorm(mx)
    affine = norm(values, weight, bias)
    mx.eval(affine)
    expected = (
        mx.maximum(affine, 0)
        if activation == "relu"
        else mx.where(affine >= 0, affine, 0.01 * affine)
    )
    actual = norm(values, weight, bias, activation=activation)
    mx.eval(actual, expected)
    np.testing.assert_array_equal(
        np.asarray(actual).view(np.uint32), np.asarray(expected).view(np.uint32)
    )
