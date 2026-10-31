# Copyright (c) MONAI Consortium
# Portions derived from MONAI, licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""Host VISTA3D click-centered correction inference, without bundle or I/O coupling."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx.core import array

    from mlx_segment_ct._types import ArrayBoundary

import numpy as np

from mlx_segment_ct._types import (
    ArrayBoundary,
    ArrayValue,
    HostArray,
)


def _pad_previous_mask(inputs: HostArray, roi: Sequence[int]) -> tuple[HostArray, list[int]]:
    """Pad spatial dimensions to the ROI and return padding in reverse-axis order."""
    pairs: list[tuple[int, int]] = []
    for width, size in zip(roi, inputs.shape[2:], strict=True):
        extra = max(width - size, 0)
        pairs.append((extra // 2, extra - extra // 2))
    pad = [value for pair in reversed(pairs) for value in pair]
    return np.pad(inputs, [(0, 0), (0, 0), *pairs]), pad


def _get_window_idx(
    p: float,
    roi: int,
    size: int,
    center_only: bool,
    margin: int,
) -> tuple[list[float], list[float]]:
    """Choose centered or margin-offset window bounds around a click."""
    if p - roi // 2 < 0:
        left, right = 0, roi
    elif p + roi // 2 > size:
        left, right = size - roi, size
    else:
        left, right = int(p) - roi // 2, int(p) + roi // 2
    if center_only:
        return [left], [right]
    left_most, right_most = max(0, p - roi + margin), min(size, p + roi - margin)
    return [left_most, right_most - roi, left], [left_most + roi, right_most, right]


def point_based_window_inferer[ArrayT: ArrayValue](
    inputs: ArrayValue,
    roi_size: Sequence[int],
    predictor: Callable[..., HostArray],
    point_coords: HostArray,
    point_labels: HostArray,
    class_vector: HostArray | None = None,
    prompt_class: HostArray | None = None,
    prev_mask: HostArray | None = None,
    point_start: int = 0,
    center_only: bool = True,
    margin: int = 5,
    *,
    mx: ArrayBoundary[ArrayT] | None = None,
    **kwargs: object,
) -> HostArray:
    """Port MONAI 1.6.0's single-object point inferer in its original loop order.

    Predictor takes MLX patches and NumPy prompt/mask geometry. Result is host
    FP32 NCDHW. Previous masks and inputs are copied before any window mutation.
    Padding is constant zero, coverage is binary (overlaps sum, not average),
    and untouched voxels keep previous logits, or NaN when no previous mask.
    Only array data is returned; affine/meta restoration belongs to the caller.
    The released ROIs have even dimensions. For an odd ROI dimension, a centered
    interior window spans ``2 * (roi // 2)`` voxels, one fewer than the ROI.

    Args:
        inputs: image, ``(1, C, X, Y, Z)``.
        roi_size: window size per spatial axis; smaller images are zero-padded.
        predictor: called once per window with the MLX patch and keyword arguments
            ``point_coords``, ``point_labels``, ``class_vector``, ``prompt_class``,
            ``patch_coords`` (the window slices), ``prev_mask`` and ``kwargs``.
            Coordinates and previous logits passed to it use the padded grid.
        point_coords: ``(1, N, 3)`` click positions in voxel units of ``inputs``.
        point_labels: ``(1, N)`` click codes (see ``InteractivePrompt``).
        class_vector: Optional automatic-branch class IDs, forwarded to the predictor.
        prompt_class: Optional point-branch class IDs, forwarded to the predictor.
        prev_mask: optional previous logits on the ``inputs`` grid.
        point_start: index of the first click that gets its own windows; earlier
            clicks are still passed to the predictor as context.
        center_only: one window per click, shifted at image edges; otherwise up to
            three windows per axis (27 per click): centered and edge-inset windows.
        margin: Edge inset used only when ``center_only`` is false.
        mx: Array boundary for creating patches; defaults to the shared MLX runtime.
        **kwargs: Extra predictor keyword arguments, forwarded unchanged.

    Raises:
        ValueError: more than one object, a non-5D input, or no window selected.
    """
    inputs, point_coords = np.asarray(inputs), np.asarray(point_coords)
    if point_coords.shape[0] != 1:
        raise ValueError("Only supports single object point click.")
    if inputs.ndim != 5:
        raise ValueError("Input image should be 5D.")
    if mx is None:
        from medmlx_core.runtime import import_mlx

        mx = cast("ArrayBoundary[ArrayT]", import_mlx())
    # ``pad`` is in torch ``F.pad`` order (last axis first), as in MONAI: [z, y, x] pairs.
    image, pad = _pad_previous_mask(inputs.copy(), roi_size)
    points = point_coords + np.array([pad[-2], pad[-4], pad[-6]], dtype=point_coords.dtype)
    previous = (
        _pad_previous_mask(np.asarray(prev_mask).copy(), roi_size)[0]
        if prev_mask is not None
        else None
    )
    stitched_output = None
    stitched_mask: HostArray | None = None
    for p in points[0][point_start:]:
        lx_, rx_ = _get_window_idx(p[0], roi_size[0], image.shape[-3], center_only, margin)
        ly_, ry_ = _get_window_idx(p[1], roi_size[1], image.shape[-2], center_only, margin)
        lz_, rz_ = _get_window_idx(p[2], roi_size[2], image.shape[-1], center_only, margin)
        for i in range(len(lx_)):
            for j in range(len(ly_)):
                for k in range(len(lz_)):
                    window = (
                        slice(None),
                        slice(None),
                        slice(int(lx_[i]), int(rx_[i])),
                        slice(int(ly_[j]), int(ry_[j])),
                        slice(int(lz_[k]), int(rz_[k])),
                    )
                    output = np.asarray(
                        predictor(
                            cast("ArrayBoundary[array]", mx).array(image[window]),
                            point_coords=points,
                            point_labels=point_labels,
                            class_vector=class_vector,
                            prompt_class=prompt_class,
                            patch_coords=[window],
                            prev_mask=previous,
                            **kwargs,
                        )
                    )
                    if stitched_output is None:
                        stitched_output = np.zeros(
                            (1, output.shape[1], *image.shape[2:]), dtype=np.float32
                        )
                        stitched_mask = np.zeros_like(stitched_output)
                    stitched_output[window] += output
                    cast("HostArray", stitched_mask)[window] = 1
    if stitched_output is None:
        raise ValueError("Input configuration resulted in no prediction windows being selected.")
    # Every initialized output is paired with its coverage allocation above.
    stitched_mask = cast("HostArray", stitched_mask)
    with np.errstate(divide="ignore", invalid="ignore"):
        stitched_output /= stitched_mask
    crop = (
        slice(None),
        slice(None),
        slice(pad[4], image.shape[-3] - pad[5]),
        slice(pad[2], image.shape[-2] - pad[3]),
        slice(pad[0], image.shape[-1] - pad[1]),
    )
    stitched_output, stitched_mask = stitched_output[crop], stitched_mask[crop]
    if previous is not None:
        previous = previous[crop]
        stitched_output[stitched_mask < 1] = previous[stitched_mask < 1]
    return stitched_output
