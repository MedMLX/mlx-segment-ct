"""Bounded FP32 automatic stitching in MONAI patch/reduction order."""

from __future__ import annotations

from collections.abc import Buffer, Callable, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import groupby, product
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_segment_ct._types import ArrayBoundary

import numpy as np

from mlx_segment_ct._types import (
    ArrayBoundary,
    ArrayValue,
    CoordinatePredictor,
    HostArray,
    Metrics,
    RollingMetricsBuilder,
)


def label_dtype(class_ids: Sequence[int]) -> type[np.uint8] | type[np.uint16]:
    """The narrowest label workspace: bytes for NV-Segment-CT, 16 bits past ID 255."""
    return np.uint8 if max(class_ids, default=0) <= 255 else np.uint16


def _validate_class_ids(class_ids: Sequence[int]) -> None:
    """Admit ordered, unique automatic foreground IDs."""
    if (
        not class_ids
        or len(set(class_ids)) != len(class_ids)
        or any(value <= 0 or value >= 512 for value in class_ids)
    ):
        raise ValueError("Automatic class IDs must be unique foreground IDs below 512")


def _finite_plane(scores: HostArray, rows_per_tile: int) -> bool:
    """Check all classes, including padded voxels, with bounded boolean temporaries."""
    return all(
        np.isfinite(scores[:, start : start + rows_per_tile]).all()
        for start in range(0, scores.shape[1], rows_per_tile)
    )


class LabelReducer:
    """Tile first-index max and global-ID lookup without full-volume temporaries.

    ``class_ids`` are the global label IDs of the score channels, in channel order;
    ``shape`` is the ``(Y, Z)`` size of one plane. A voxel takes the ID of its first
    maximum channel when that score is positive, otherwise background (0). Any
    non-finite score sets ``invalid``.
    """

    def __init__(self, class_ids: Sequence[int], shape: Sequence[int]):
        """Validate class IDs and initialize the streamed label reducer."""
        _validate_class_ids(class_ids)
        dtype = label_dtype(class_ids)
        self.ids = np.array(class_ids, dtype=dtype)
        self.invalid = False
        self.rows_per_tile = min(shape[0], max(1, 32768 // shape[1]))
        tile_shape = (self.rows_per_tile, shape[1])
        self.values = np.empty(tile_shape, dtype=np.float32)
        self.indices = np.empty(tile_shape, dtype=np.int64)
        self.labels = np.empty(tile_shape, dtype=dtype)
        self.background = np.empty(tile_shape, dtype=bool)

    def __call__(self, scores: HostArray, output: HostArray) -> None:
        """Reduce one ``(classes, Y, Z)`` score plane into the ``(Y, Z)`` ``output`` labels."""
        for start in range(0, scores.shape[1], self.rows_per_tile):
            end = min(scores.shape[1], start + self.rows_per_tile)
            tile = scores[:, start:end, :]
            count = end - start
            values, indices = self.values[:count], self.indices[:count]
            labels, background = self.labels[:count], self.background[:count]
            self.invalid |= not np.isfinite(tile).all()
            np.max(tile, axis=0, out=values)
            np.argmax(tile, axis=0, out=indices)
            np.take(self.ids, indices, out=labels)
            np.less_equal(values, 0, out=background)
            labels[background] = 0
            output[start:end] = labels


def bounded_labels(scores: ArrayValue, class_ids: Sequence[int]) -> HostArray:
    """Reduce one FP32 score volume, returning bundle FP32 output labels.

    ``scores`` is ``(1, len(class_ids), X, Y, Z)`` in class-ID order; the result is
    float32 ``(1, 1, X, Y, Z)`` global labels. Raises ``ValueError`` for a wrong
    shape or dtype, or non-finite scores.
    """
    scores = np.asarray(scores)
    if scores.ndim != 5 or scores.shape[:2] != (1, len(class_ids)) or scores.dtype != np.float32:
        raise ValueError("Automatic reduction requires one FP32 score tensor in class-ID order")
    reducer = LabelReducer(class_ids, scores.shape[-2:])
    output = np.empty((1, 1, *scores.shape[2:]), dtype=label_dtype(class_ids))
    for plane in range(scores.shape[2]):
        reducer(scores[0, :, plane], output[0, 0, plane])
    if reducer.invalid:
        raise ValueError("NV automatic scores contain NaN or infinity")
    return output.astype(scores.dtype)


def centered_padding(shape: Sequence[int], roi: Sequence[int]) -> list[tuple[int, int]]:
    """Per-axis (before, after) padding that centers an image smaller than its ROI."""
    return [
        (max(patch - size, 0) // 2, (max(patch - size, 0) + 1) // 2)
        for size, patch in zip(shape, roi, strict=True)
    ]


def ring_slices(start: int, length: int, depth: int) -> list[tuple[slice, slice]]:
    """Pair patch-relative and ring slices, splitting only at wraparound."""
    slot = start % depth
    first = min(length, depth - slot)
    parts = [(slice(0, first), slice(slot, slot + first))]
    if first < length:
        parts.append((slice(first, length), slice(0, length - first)))
    return parts


def rolling_logits[ArrayT: ArrayValue](
    inputs: ArrayValue,
    predictor: CoordinatePredictor[ArrayT],
    class_ids: Sequence[int],
    roi_size: Sequence[int],
    *,
    overlap: float = 0.25,
    weight: ArrayValue | None = None,
    finalized: Callable[[int, HostArray, HostArray], None] | None = None,
    metrics: Metrics | RollingMetricsBuilder | None = None,
    mx: ArrayBoundary[ArrayT] | None = None,
) -> HostArray:
    """Predict once per patch; return FP32 labels using an axis-0 score ring.

    Only the MLX predictor runs on the device. Host scores occupy
    ``classes * roi[0] * padded[1] * padded[2] * 4`` bytes, independent of
    volume depth; one more finished prediction can wait for the host. Host
    accumulation and ``finalized`` run on one worker thread, in plane order,
    while the next patch is predicted; label reduction of each finished plane
    runs on a second thread beside ``finalized``. ``finalized`` receives borrowed
    score/count plane views; copy them inside the callback if needed after
    they are recycled.
    """
    labels = stream_scores(
        inputs,
        predictor,
        class_ids,
        roi_size,
        overlap=overlap,
        weight=weight,
        finalized=finalized,
        metrics=metrics,
        mx=mx,
        reduce_labels=True,
    )
    assert labels is not None
    return labels


def stream_scores[ArrayT: ArrayValue](
    inputs: ArrayValue,
    predictor: CoordinatePredictor[ArrayT],
    class_ids: Sequence[int],
    roi_size: Sequence[int],
    *,
    overlap: float = 0.25,
    weight: ArrayValue | None = None,
    finalized: Callable[[int, HostArray, HostArray], None] | None = None,
    metrics: Metrics | RollingMetricsBuilder | None = None,
    mx: ArrayBoundary[ArrayT] | None = None,
    reduce_labels: bool = False,
) -> HostArray | None:
    """Stream normalized score/count planes in axis-0 order, optionally reducing labels.

    Score-only restoration replaces label reduction with a bounded finite-value
    scan, running beside ``finalized`` on the same worker as label reduction.
    Both routes reject nonfinite scores and missing coverage after all finalized
    callbacks, before a runner can publish output. Callback views are borrowed
    until the callback returns; the slot is recycled after both consumers finish.

    ``inputs`` is float32 ``(1, 1, X, Y, Z)``, edge-padded to at least ``roi_size``.
    Windows are placed with the given ``overlap`` fraction; scores are divided by
    the accumulated weight (``weight`` defaults to ones, i.e. constant blending).
    With ``reduce_labels`` the result is float32 ``(1, 1, X, Y, Z)`` global labels
    cropped to the input size; otherwise ``None``. Raises ``ValueError`` for
    malformed inputs, a patch output of the wrong shape, non-finite scores, or
    voxels no window covered.
    """
    if mx is None:
        from medmlx_core.runtime import import_mlx

        mx = cast("ArrayBoundary[ArrayT]", import_mlx())
    inputs = np.asarray(inputs)
    if inputs.ndim != 5 or inputs.shape[:2] != (1, 1) or inputs.dtype != np.float32:
        raise ValueError("Rolling NV inference requires one FP32 3D image")
    roi = tuple(int(v) for v in roi_size)
    if len(roi) != 3 or any(v < 1 for v in roi) or any(v < 1 for v in inputs.shape[2:]):
        raise ValueError("Rolling NV inference requires positive 3D image/ROI sizes")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be >= 0 and < 1")
    original = tuple(inputs.shape[2:])
    padded = tuple(max(size, patch) for size, patch in zip(original, roi, strict=True))
    pairs = centered_padding(original, roi)
    padding = [value for pair in reversed(pairs) for value in pair]
    raw = np.pad(inputs, [(0, 0), (0, 0), *pairs], mode="edge") if any(padding) else inputs
    interval = tuple(
        patch if patch == size else max(int(patch * (1 - overlap)), 1)
        for size, patch in zip(padded, roi, strict=True)
    )
    starts = [
        [min(i * step, size - patch) for i in range((size - patch + step - 1) // step + 1)]
        for size, patch, step in zip(padded, roi, interval, strict=True)
    ]
    windows = [
        tuple(slice(i, i + width) for i, width in zip(pos, roi, strict=True))
        for pos in product(*starts)
    ]
    depth, channels = roi[0], len(class_ids)
    scores = np.zeros((channels, depth, *padded[1:]), dtype=np.float32)
    counts = np.zeros((depth, *padded[1:]), dtype=np.float32)
    reducer = LabelReducer(class_ids, padded[1:]) if reduce_labels else None
    if reducer is None:
        _validate_class_ids(class_ids)
    labels = np.empty(padded, dtype=label_dtype(class_ids)) if reduce_labels else None
    rows_per_tile = min(padded[1], max(1, 32768 // padded[2]))
    invalid = False
    weights = np.ones(roi, dtype=np.float32) if weight is None else np.asarray(weight)
    if weights.shape != roi or weights.dtype != np.float32:
        raise ValueError("Rolling weights must be an FP32 ROI map")
    groups = [
        (int(origin), list(patches)) for origin, patches in groupby(windows, lambda s: s[0].start)
    ]
    if metrics is not None:
        metrics.update(
            padded_shape=list(padded),
            padding=padding,
            streaming_axis=0,
            ring_depth=depth,
            predicted_accumulator_bytes=scores.nbytes,
            count_buffer_bytes=counts.nbytes,
            label_buffer_bytes=0 if labels is None else labels.nbytes,
            accumulator_dtype="float32",
        )

    def accumulate(origin: int, spatial: tuple[slice, ...], prediction: HostArray) -> None:
        """Accumulate a patch of scores and optional blending weights."""
        if weight is not None:
            # MONAI weights in the predictor dtype before FP32 accumulation.
            prediction = np.array(prediction)
            prediction *= weights
        # The default all-ones map changes no value, so its scores come straight
        # from the device buffer without a copy or multiply.
        for source, target in ring_slices(origin, depth, depth):
            transverse = spatial[1:]
            scores[(slice(None), target, *transverse)] += prediction[0, :, source]
            counts[(target, *transverse)] += weights[source]

    def finish(origin: int, end: int) -> None:
        """Finalize completed score planes and track invalid values."""
        nonlocal invalid
        for plane in range(origin, end):
            slot = plane % depth
            count = counts[slot]
            invalid |= bool((count <= 0).any())
            with np.errstate(divide="ignore", invalid="ignore"):
                scores[:, slot] /= count
            reduced = None
            finite = None
            if reducer is not None and labels is not None:
                # Reduction owns its finite-value state; this host worker owns
                # coverage state. Wait below before either reuses the plane.
                reduced = reduction.submit(reducer, scores[:, slot], labels[plane])
            else:
                # Preserve overlap with restoration without computing its unused
                # low-resolution labels. The host owns the returned invalid state.
                finite = reduction.submit(_finite_plane, scores[:, slot], rows_per_tile)
            if finalized is not None:
                finalized(plane, scores[:, slot], count)
            if reduced is not None:
                reduced.result()
            if finite is not None:
                invalid |= not finite.result()
            scores[:, slot] = 0
            count[:] = 0

    # One host thread runs every accumulation and finished plane in submission
    # order, which is the serial order, while the device predicts the next
    # patch; MLX releases the GIL while it waits. Draining before each new
    # patch keeps at most one finished prediction queued on the host.
    queued: list[Future[None]] = []

    def drain() -> None:
        """Finish queued asynchronous work before reusing its storage."""
        for task in queued:
            task.result()
        queued.clear()

    with ThreadPoolExecutor(max_workers=1) as host, ThreadPoolExecutor(max_workers=1) as reduction:
        for index, (origin, patches) in enumerate(groups):
            for spatial in patches:
                coords = [slice(0, 1), slice(None), *spatial]
                prediction = predictor(mx.array(raw[tuple(coords)]), [coords])
                mx.eval(prediction)
                if tuple(prediction.shape) != (1, channels, *roi):
                    raise ValueError(
                        "NV patch output does not match its admitted channel/spatial shape"
                    )
                # A read-only use of the evaluated buffer; the view keeps it alive.
                values = np.asarray(memoryview(cast("Buffer", prediction)))
                drain()
                queued.append(host.submit(accumulate, origin, spatial, values))
            end = groups[index + 1][0] if index + 1 < len(groups) else padded[0]
            queued.append(host.submit(finish, origin, end))
        drain()
    if invalid or (reducer is not None and reducer.invalid):
        raise ValueError("NV rolling logits contain nonfinite scores or missing coverage")
    if labels is None:
        return None
    crop = tuple(
        slice(before, before + size) for (before, _), size in zip(pairs, original, strict=True)
    )
    return labels[crop][None, None].astype(inputs.dtype)
