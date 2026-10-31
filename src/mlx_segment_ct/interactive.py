# Copyright (c) MONAI Consortium
# Portions derived from MONAI, licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""VISTA3D bundle inference for a single object's clicks and optional previous logits."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import numpy as np
from medmlx_core.errors import InvalidInputError
from scipy import ndimage

from mlx_segment_ct._types import (
    ArrayBoundary,
    ArrayValue,
    HostArray,
    IntArray,
    Metrics,
    PointPredictor,
)
from mlx_segment_ct.bundle_transforms import Preprocessed
from mlx_segment_ct.correction_inferer import point_based_window_inferer
from mlx_segment_ct.labels import InteractivePrompt
from mlx_segment_ct.mlx_point_window import PointWindowPredictor


def transform_points(prepared: Preprocessed, prompt: InteractivePrompt) -> InteractivePrompt:
    """Bundle evaluator: inv(current affine) @ original affine @ homogeneous clicks.

    ``prompt`` holds (x, y, z) voxel indices on the original image grid; the returned
    prompt holds float voxel coordinates on ``prepared.image``'s grid (after 1.5 mm
    resampling, foreground crop and RAS orientation). Raises ``InvalidInputError``
    for a click outside that grid, which can happen when it lies in the cropped-away
    background.
    """
    points, _ = prompt.arrays()
    transform = np.linalg.inv(prepared.affine) @ prepared.original_affine
    homogeneous = np.concatenate((points[0], np.ones((len(prompt.points), 1))), axis=1)
    coords = (transform @ homogeneous.T).T[:, :3]
    if ((coords < 0) | (coords >= np.asarray(prepared.image.shape[1:]))).any():
        raise InvalidInputError(
            "Interactive click lies outside the preprocessed CT foreground crop"
        )
    return InteractivePrompt(
        tuple(tuple(p) for p in coords), prompt.point_labels, prompt.class_label, prompt.rules
    )


def interactive_logits[ArrayT: ArrayValue](
    image: HostArray,
    prompt: InteractivePrompt,
    point_model: PointPredictor[ArrayT],
    *,
    mx: ArrayBoundary[ArrayT],
    roi_size: Sequence[int],
    automatic_model: Callable[[ArrayT], ArrayT] | None = None,
    metrics: Metrics | None = None,
    previous_logits: HostArray | None = None,
    point_start: int = 0,
) -> HostArray:
    """Point-centered bundle windows; FP32 NCDHW logits retain NaN outside coverage.

    ``prompt`` coordinates are on the preprocessed image grid. With active clicks,
    class IDs through 132 require ``automatic_model`` even with previous logits;
    unlabelled objects and IDs above 132 use only the point branch for either checkpoint.
    All-padding class prompts retain the automatic branch, including IDs above 132.
    The bundle uses center_only=True. Its overlap/gaussian arguments do not affect
    the point inferer: overlapping patches are summed with a binary coverage map.

    Args:
        image: preprocessed image, float32 ``(1, 1, X, Y, Z)``.
        prompt: clicks and optional class (see ``InteractivePrompt``).
        point_model: the point branch, called per window.
        mx: array boundary used to move patches to the device.
        roi_size: three positive window sizes; the bundle uses 128^3, or
            192x192x128 for NV-Segment-CTMR.
        automatic_model: required exactly when ``prompt.uses_automatic``.
        metrics: receives per-window seconds in ``patch_seconds``.
        previous_logits: raw logits from an earlier call, same shape as ``image``.
        point_start: index of the first new click; earlier clicks stay as context.

    Returns:
        Host float32 logits ``(1, 1, X, Y, Z)``; voxels no window covered keep
        ``previous_logits``, or are NaN without it.

    Raises:
        InvalidInputError: malformed image, ROI, clicks, ``point_start``, previous
            logits, or a missing/unexpected automatic model.
    """
    if image.ndim != 5 or image.shape[:2] != (1, 1) or image.dtype != np.float32:
        raise InvalidInputError("Interactive image must be one FP32 NCDHW volume")
    if len(roi_size) != 3 or any(type(n) is not int or n <= 0 for n in roi_size):
        raise InvalidInputError("Interactive roi_size must contain three positive integers")
    points, codes = prompt.arrays()
    if ((points < 0) | (points >= np.asarray(image.shape[2:]))).any():
        raise InvalidInputError("Interactive click lies outside the preprocessed CT grid")
    if type(point_start) is not int or not 0 <= point_start < len(prompt.points):
        raise InvalidInputError("point_start must identify at least one new click")
    if previous_logits is not None and (
        previous_logits.shape != image.shape
        or previous_logits.dtype != np.float32
        or np.isinf(previous_logits).any()
    ):
        raise InvalidInputError("Previous logits must match the FP32 preprocessed image grid")
    use_automatic = prompt.uses_automatic
    if use_automatic != (automatic_model is not None):
        raise InvalidInputError(
            "Provide an automatic model exactly when the bundle enables the automatic branch"
        )
    metrics = {"patch_seconds": []} if metrics is None else metrics
    target = None if prompt.class_label is None else np.array([prompt.class_label])
    predictor = PointWindowPredictor(
        point_model, automatic_model=automatic_model, mx=mx, metrics=metrics
    )
    # transpose, mode, overlap, with_coord and sw_batch_size mirror the bundle's inferer
    # call. The point inferer forwards them to PointWindowPredictor, which requires
    # transpose=True and ignores the rest.
    return point_based_window_inferer(
        image,
        roi_size,
        predictor,
        points,
        codes,
        class_vector=target if use_automatic else None,
        prompt_class=target,
        prev_mask=previous_logits,
        point_start=point_start,
        center_only=True,
        transpose=True,
        mode="gaussian",
        overlap=0.3,
        with_coord=True,
        sw_batch_size=1,
        mx=mx,
    )


def interactive_labels(logits: HostArray, prompt: InteractivePrompt) -> HostArray:
    """Bundle VistaPostTransformd before inversion, preserving NaN and mean-fill behavior.

    Returns float32 ``(1, X, Y, Z)`` labels on the preprocessed grid: 0 for
    background, 1 (no class) or ``prompt.class_label`` for the object, and NaN for
    voxels no window covered.
    """
    pred = interactive_scores(logits, prompt)
    pred[pred < 0] = 0
    pred[pred > 0] = 1
    if prompt.class_label is not None:
        pred += np.float32(0.5)
        pred[pred == 1.5] = prompt.class_label
        pred[pred == 0.5] = 0
    return pred


def interactive_scores(logits: HostArray, prompt: InteractivePrompt) -> HostArray:
    """VistaPostTransformd's (1, X, Y, Z) scores before thresholding.

    Without a class, positive regions that contain no positive click take the mean
    of the unselected scores, as upstream does. NaN marks uncovered voxels.

    Raises:
        InvalidInputError: logits are not float32 ``(1, 1, X, Y, Z)``, contain
            infinity, or a positive click rounds outside the grid.
    """
    if logits.ndim != 5 or logits.shape[:2] != (1, 1) or logits.dtype != np.float32:
        raise InvalidInputError("Interactive logits must have shape (1, 1, X, Y, Z), float32")
    if np.isinf(logits).any():
        raise InvalidInputError("Interactive logits contain infinity")
    pred = logits[0].copy()
    if prompt.class_label is None:
        positive = [
            p for p, code in zip(prompt.points, prompt.point_labels, strict=True) if code in (1, 3)
        ]
        if positive:
            field = pred[0]
            features = ndimage.label(field > 0, ndimage.generate_binary_structure(3, 3))[0]
            coords = np.asarray([[round(v) for v in point] for point in positive])
            if ((coords < 0) | (coords >= np.asarray(field.shape))).any():
                raise InvalidInputError(
                    "Rounded positive click lies outside the preprocessed CT grid"
                )
            indices = np.unique(features[tuple(cast("IntArray", coords.T))])
            selected = np.isin(features, indices[indices != 0])
            unselected = ~selected & ~np.isnan(field)
            fill = field[unselected].mean() if unselected.any() else np.float32(np.nan)
            field[(field > 0) & ~selected] = fill
    return pred
