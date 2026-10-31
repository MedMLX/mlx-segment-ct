# Copyright (c) MONAI Consortium
# Portions derived from MONAI, licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""MLX point windows with NumPy/SciPy VISTA3D component combination."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_segment_ct._types import ArrayBoundary

import numpy as np
from scipy import ndimage
from scipy.special import expit

from mlx_segment_ct._types import (
    ArrayBoundary,
    ArrayValue,
    BoolArray,
    HostArray,
    Metrics,
    PatchCoordinates,
    PointPredictor,
)
from mlx_segment_ct.correction import normalized_point_positions


def update_point_to_patch(
    patch_coords: Sequence[slice],
    point_coords: HostArray,
    point_labels: HostArray,
) -> tuple[HostArray | None, HostArray | None]:
    """Exclude both patch faces and remove padded click columns as VISTA3D does.

    ``patch_coords`` ends with the window's three spatial slices. ``point_coords`` is
    ``(1, N, 3)`` in image voxel units and ``point_labels`` ``(1, N)``. A click counts
    only if it is strictly inside the window on every axis. Returns window-relative
    coordinates and codes of the clicks inside (outside clicks and padding clicks are
    dropped), or ``(None, None)`` if no click is inside.
    """
    starts = np.array([part.start for part in patch_coords[-3:]], dtype=point_coords.dtype)[
        None, None
    ]
    ends = np.array([part.stop for part in patch_coords[-3:]], dtype=point_coords.dtype)[None, None]
    indices = ((point_coords - starts) > 0).all(2) & ((ends - point_coords) > 0).all(2)
    points, codes = point_coords.copy() - starts, point_labels.copy()
    if indices.any():
        codes[~indices] = -1
        points[~indices] = 0
        keep = (codes != -1).any(0)
        return points[:, keep], codes[:, keep]
    return None, None


def keep_merge_components_with_points(
    img_pos: HostArray,
    img_neg: HostArray,
    point_coords: HostArray,
    point_labels: HostArray,
    pos_val: Sequence[int] = (1, 3),
    neg_val: Sequence[int] = (0, 2),
    margins: int = 3,
) -> BoolArray:
    """Port MONAI's boolean scikit-image label path and click-margin selection.

    ``img_pos`` and ``img_neg`` are boolean ``(B, 1, X, Y, Z)`` masks. Positive-code
    clicks (``pos_val``) select a component of ``img_pos`` and negative-code clicks
    (``neg_val``) a component of ``img_neg``: the one with the largest label ID in the
    first non-empty cube around the rounded click, with half-widths 0 to ``margins - 1``
    voxels. Returns the union of the selected components, shaped like ``img_pos``.

    scikit-image's boolean implementation calls SciPy ``ndimage.label`` with
    this exact rank/connectivity. Preserve component numbering: a margin that
    touches several components selects the largest label ID, not their union.
    MONAI casts Torch click coordinates to FP32 before this rounded lookup.
    """
    structure = ndimage.generate_binary_structure(img_pos.ndim, 3)
    features_pos = ndimage.label(img_pos, structure=structure)[0]
    features_neg = ndimage.label(img_neg, structure=structure)[0]
    outs = np.zeros_like(img_pos)
    for bs in range(point_coords.shape[0]):
        for i, p in enumerate(point_coords[bs]):
            if point_labels[bs, i] in pos_val:
                features = features_pos
            elif point_labels[bs, i] in neg_val:
                features = features_neg
            else:
                continue
            for margin in range(margins):
                x, y, z = cast(
                    "tuple[int, int, int]",
                    np.round(cast("HostArray", p).astype(np.float32)).astype(int).tolist(),
                )
                left, right = max(x - margin, 0), min(x + margin + 1, features.shape[-3])
                top, down = max(y - margin, 0), min(y + margin + 1, features.shape[-2])
                front, back = max(z - margin, 0), min(z + margin + 1, features.shape[-1])
                region = features[bs, 0, left:right, top:down, front:back]
                if (region > 0).any():
                    index = region.max()
                    outs[[bs]] += np.isin(features[[bs]], index)
                    break
    outs[outs > 1] = 1
    return cast("BoolArray", outs)


def connected_components_combine(
    logits: HostArray,
    point_logits: HostArray,
    point_coords: HostArray,
    point_labels: HostArray,
    mapping_index: BoolArray,
    thred: float = 0.5,
) -> HostArray:
    """Apply VISTA3D's original in-place component update, including NaN arithmetic.

    ``logits`` (existing automatic or previous scores) and ``point_logits`` are
    ``(1, 1, X, Y, Z)``. Components of disagreement between the two that contain a
    click of the matching sign take the point logits; other voxels keep ``logits``.
    NaN (uncovered) voxels of ``logits`` stay NaN, because the original arithmetic
    propagates them. ``logits`` is modified in place and returned; ``point_logits``
    is also modified in place.
    """
    selected = logits[mapping_index]
    inside: list[int] = []
    for i in range(selected.shape[0]):
        coords = point_coords[i].round().astype(int)
        inside.append(int(np.any([selected[i, 0, p[0], p[1], p[2]] > 0 for p in coords])))
    inside_array = np.array(inside).reshape(-1, 1, 1, 1, 1)
    nan_mask = np.isnan(selected)
    # expit retains Torch's FP32 sigmoid decisions just above zero. NumPy's
    # exp rounds differently there and can change entire selected components.
    selected = expit(np.nan_to_num(selected, nan=-9999))
    pos_region = expit(point_logits) > thred
    diff_pos = ((selected <= thred) | inside_array.astype(bool)) & pos_region
    diff_neg = (selected > thred) & ~pos_region
    cc = keep_merge_components_with_points(diff_pos, diff_neg, point_coords, point_labels)
    fill_mask = nan_mask & (pos_region & ~cc)
    if fill_mask.any():
        point_logits[fill_mask] = -1
    cc = (nan_mask | cc).astype(logits.dtype)
    logits[mapping_index] *= 1 - cc
    logits[mapping_index] += cc * point_logits
    return logits


class PointWindowPredictor[ArrayT: ArrayValue]:
    """VISTA3D.forward: optional automatic branch, point head, then component fusion.

    One call handles one window. It returns host float32 logits ``(1, 1, x, y, z)``
    for that window: automatic scores only if no click is inside it and the
    automatic branch is active, ``NINF_VALUE`` (-9999) if neither applies, otherwise
    the point logits, fused with the automatic or previous scores where available.
    A request without a class prompt queries the point head with class ID 132.
    Per-window seconds are appended to ``metrics["patch_seconds"]``.
    """

    NINF_VALUE = -9999

    def __init__(
        self,
        model: PointPredictor[ArrayT],
        *,
        mx: ArrayBoundary[ArrayT],
        metrics: Metrics,
        automatic_model: Callable[[ArrayT], ArrayT] | None = None,
    ) -> None:
        """Store the native point model, array runtime, and timing metrics."""
        self.model, self.mx, self.metrics = model, mx, metrics
        self.automatic_model = automatic_model

    def __call__(
        self,
        image: ArrayT,
        *,
        point_coords: HostArray,
        point_labels: HostArray,
        class_vector: HostArray | None,
        prompt_class: HostArray | None,
        patch_coords: PatchCoordinates,
        prev_mask: HostArray | None,
        transpose: bool,
        **kwargs: object,
    ) -> HostArray:
        # These bundle inferer settings are ignored by upstream VISTA3D.forward.
        """Predict a point-conditioned window using its patch coordinates and prompts."""
        ignored = {"sw_batch_size", "with_coord", "mode", "sw_device", "device", "overlap"}
        unsupported = kwargs.keys() - ignored
        if unsupported:
            raise ValueError(
                f"Unsupported VISTA3D point prediction settings: {sorted(unsupported)}"
            )
        if not transpose:
            raise ValueError("Interactive inference requires transpose=True")
        points, codes = update_point_to_patch(patch_coords[0], point_coords, point_labels)
        tick = time.perf_counter()
        automatic = None
        if class_vector is not None:
            if self.automatic_model is None:
                raise ValueError(
                    "Supported-class interactive inference requires an automatic model"
                )
            automatic = self.automatic_model(image)
            self.mx.eval(automatic)
            automatic = np.array(automatic)
            if not np.isfinite(automatic).all():
                raise ValueError("MLX automatic decoder returned non-finite logits")
        if points is None or codes is None or not (codes != -1).any():
            self.metrics["patch_seconds"].append(time.perf_counter() - tick)
            return (
                automatic
                if automatic is not None
                else np.full((1, 1, *image.shape[2:]), self.NINF_VALUE, np.float32)
            )
        target = 132 if prompt_class is None else int(prompt_class.ravel()[0])
        logits = self.point_logits(image, points, codes, target)
        if automatic is not None:
            logits = connected_components_combine(
                automatic.swapaxes(0, 1), logits, points, codes, np.array([True])
            )
        elif prev_mask is not None:
            logits = connected_components_combine(
                prev_mask[cast("tuple[slice, ...]", patch_coords[0])].swapaxes(0, 1),
                logits,
                points,
                codes,
                np.array([True]),
            )
        self.metrics["patch_seconds"][-1] = time.perf_counter() - tick
        return logits.swapaxes(0, 1)

    def point_logits(
        self,
        image: ArrayT,
        points: HostArray,
        codes: HostArray,
        target: int,
    ) -> HostArray:
        """The shared point forward, with host normalization before the FP32 cast."""
        mx = self.mx
        mx.synchronize()
        tick = time.perf_counter()
        logits = self.model(
            mx.array(image),
            mx.array(normalized_point_positions(points, tuple(image.shape[2:]))),
            mx.array(codes),
            target,
        )
        mx.eval(logits)
        mx.synchronize()
        host = np.array(logits)
        if not np.isfinite(host).all():
            raise ValueError("MLX point decoder returned non-finite logits")
        self.metrics["patch_seconds"].append(time.perf_counter() - tick)
        return host
