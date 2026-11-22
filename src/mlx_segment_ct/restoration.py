"""Score-space restoration of NV-Segment-CT outputs to the source image grid.

The bundle picks labels on the 1.5 mm preprocessed grid and restores them with
``Invertd(nearest_interp=True)``, so on a finer source grid each decision repeats
over several voxels and boundaries step. ``ScoreRestorer`` instead interpolates
the scores trilinearly at every source voxel, at the preprocessed position that
nearest restoration rounds, and decides the label there with the bundle's rule:
the first maximum class ID when positive, otherwise background.

Scores arrive one preprocessed plane at a time along axis 0, as ``rolling_logits``
finalizes them, so only two score planes are held. Source voxels are bucketed by
the pair of planes that encloses them. A voxel whose eight neighbors share one
label keeps it, because the interpolated scores preserve each neighbor's first
maximum and its sign; only classes positive somewhere in the pair can win
elsewhere. Positions beyond the preprocessed grid (the bundle's foreground crop)
clamp to its edge and uncovered (non-finite) neighbors are left out, renormalizing
the remaining weights. As with nearest restoration, a voxel whose nearest neighbor
is outside the grid is background and one whose nearest neighbor is uncovered is NaN.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np

from mlx_segment_ct._types import BoolArray, DoubleArray, FloatArray, HostArray, IntArray
from mlx_segment_ct.bundle_transforms import RestorationGrid

_CHUNK_VOXELS = 1 << 20
_GATHER_VALUES = 1 << 22  # bounds each (voxels, classes) score gather


@dataclass(frozen=True, slots=True)
class _Plane:
    values: FloatArray  # (Y, Z, classes), contiguous per voxel; uncovered voxels zeroed
    labels: FloatArray  # (Y, Z); NaN where uncovered
    covered: BoolArray  # (Y, Z)
    positive: BoolArray  # (classes,): positive somewhere in the plane


@dataclass(frozen=True, slots=True)
class _AxisBuckets:
    axes: tuple[IntArray, IntArray, IntArray]  # source indices in preprocessed axis order
    positions: tuple[HostArray, HostArray, HostArray]  # corresponding preprocessed positions
    source_axes: tuple[int, ...]  # source axis driving each preprocessed axis
    plane_buckets: IntArray  # floor(p0) + 1 in axes[0] order
    offsets: IntArray


class ScoreRestorer:
    """Restore labels from scores fed plane by plane in preprocessed axis-0 order.

    ``class_ids`` name the score channels in order; one channel restores one object.
    """

    def __init__(self, grid: RestorationGrid, class_ids: Sequence[int]) -> None:
        """Initialize score-plane restoration for the grid and class IDs."""
        if not class_ids:
            raise ValueError("Restoration requires at least one class ID")
        self._grid = grid
        self._ids = np.asarray(class_ids, dtype=np.float32)
        self._planes = grid.preprocessed_shape[0]
        self._output = np.zeros(grid.source_shape, dtype=np.float32)
        self._axis_buckets = self._separable_buckets()
        self._order: HostArray | None = None
        if self._axis_buckets is None:
            self._order, self._offsets = self._buckets()
        else:
            self._offsets = self._axis_buckets.offsets
        self._previous: _Plane | None = None
        self._next = 0

    def add_plane(self, index: int, scores: HostArray) -> None:
        """Accept preprocessed plane ``index`` as (classes, Y, Z) scores; copies the view."""
        if index != self._next:
            raise ValueError(f"Restoration expected preprocessed plane {self._next}, not {index}")
        shape = self._grid.preprocessed_shape[1:]
        if scores.shape != (self._ids.size, *shape):
            raise ValueError("Restoration planes must be (classes, Y, Z) on the preprocessed grid")
        # A contiguous channels-last copy: the caller's (possibly borrowed) view is not written.
        channels_last = np.array(np.moveaxis(scores, 0, -1), dtype=np.float32, order="C")
        covered = np.isfinite(channels_last).all(axis=-1)
        values = channels_last
        if not covered.all():
            values[~covered] = 0
        labels = _decide(values.reshape(-1, self._ids.size), self._ids).reshape(shape)
        labels[~covered] = np.nan
        current = _Plane(values, labels, covered, np.max(values, axis=(0, 1), initial=0) > 0)
        self._restore(index, self._previous, current)
        self._previous = current
        self._next += 1

    def finish(self) -> tuple[FloatArray, DoubleArray]:
        """The (1, X, Y, Z) float32 restored labels and their affine."""
        if self._next != self._planes:
            raise ValueError(
                f"Restoration received {self._next} of {self._planes} preprocessed planes"
            )
        self._restore(self._planes, self._previous, None)
        self._next += 1
        return self._output[None], self._grid.affine

    def _separable_buckets(self) -> _AxisBuckets | None:
        """Keep one-dimensional maps when source axes stay independent through reorientation."""
        grid = self._grid
        if grid.axis_indices is None and grid.theta is not None:
            return None
        linear = np.abs(grid.to_preprocessed[:3, :3])
        if (
            not all(grid.source_shape)
            or not np.isfinite(grid.to_preprocessed[:3]).all()
            or not ((linear == 0) | (linear == 1)).all()
            or not (linear.sum(axis=0) == 1).all()
            or not (linear.sum(axis=1) == 1).all()
            or (
                grid.axis_indices is not None
                and not all(np.isfinite(axis).all() for axis in grid.axis_indices)
            )
        ):
            return None
        source_axes = tuple(int(axis) for axis in np.argmax(linear, axis=1))
        axes: list[IntArray] = []
        positions: list[HostArray] = []
        for row, source_axis in enumerate(source_axes):
            varying = np.arange(grid.source_shape[source_axis], dtype=np.int64)
            indices = [np.zeros(varying.size, dtype=np.int64) for _ in range(3)]
            indices[source_axis] = varying
            # Use the grid's Torch-compatible arithmetic, including axis-map rounding.
            position = grid.coordinates(*indices)[row]
            nearest = np.rint(position)
            inside = (nearest >= 0) & (nearest < grid.preprocessed_shape[row])
            axes.append(varying[inside])
            positions.append(position[inside])
        plane_buckets = np.floor(positions[0]).astype(np.int64) + 1
        counts = np.bincount(plane_buckets, minlength=self._planes + 2)
        counts *= axes[1].size * axes[2].size
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        return _AxisBuckets(
            (axes[0], axes[1], axes[2]),
            (positions[0], positions[1], positions[2]),
            source_axes,
            plane_buckets,
            offsets,
        )

    def _buckets(self) -> tuple[HostArray, IntArray]:
        """Source voxel indices sorted by bucket ``floor(p0) + 1``, and bucket offsets.

        Voxels whose nearest preprocessed voxel is outside the grid stay background.
        """
        total = int(np.prod(self._grid.source_shape))
        outside = self._planes + 1
        dtype = np.uint16 if outside < np.iinfo(np.uint16).max else np.uint32
        buckets = np.empty(total, dtype=dtype)
        for start in range(0, total, _CHUNK_VOXELS):
            flat = np.arange(start, min(total, start + _CHUNK_VOXELS))
            coords = self._grid.coordinates(*_unravel(flat, self._grid.source_shape))
            chunk = (np.floor(coords[0]) + 1).astype(np.int64)
            chunk[~_inside(tuple(np.rint(c) for c in coords), self._grid.preprocessed_shape)] = (
                outside
            )
            buckets[start : start + flat.size] = chunk
        counts = np.bincount(buckets, minlength=outside + 1)
        offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
        order = np.empty(
            int(offsets[outside]), dtype=np.int32 if total < np.iinfo(np.int32).max else np.int64
        )
        cursor = offsets[:outside].copy()
        for start in range(0, total, _CHUNK_VOXELS):
            chunk = buckets[start : start + _CHUNK_VOXELS]
            local = np.flatnonzero(chunk != outside)
            local = local[np.argsort(chunk[local], kind="stable")]
            sorted_buckets = chunk[local].astype(np.int64)
            chunk_counts = np.bincount(sorted_buckets, minlength=outside)
            first = np.cumsum(chunk_counts) - chunk_counts
            rank = np.arange(local.size) - first[sorted_buckets]
            order[cursor[sorted_buckets] + rank] = start + local
            cursor += chunk_counts
        return order, offsets

    def _restore(self, bucket: int, lower: _Plane | None, upper: _Plane | None) -> None:
        """Restore the voxels between preprocessed planes ``bucket - 1`` and ``bucket``."""
        if self._offsets[bucket] == self._offsets[bucket + 1]:
            return
        edge = lower if lower is not None else upper
        assert edge is not None
        pair = (edge, upper if upper is not None else edge)  # clamp to the grid's edge plane
        # Cells (padded by one so clamped corners repeat the edge) whose eight corners agree.
        low, high = (np.pad(plane.labels, 1, mode="edge") for plane in pair)
        first = low[:-1, :-1]
        rows, columns = first.shape
        uniform = np.ones(first.shape, dtype=bool)
        for labels in (low, high):
            for dy in (0, 1):
                for dz in (0, 1):
                    uniform &= labels[dy : dy + rows, dz : dz + columns] == first
        active = np.flatnonzero(pair[0].positive | pair[1].positive)
        values = tuple(
            plane.values if active.size == self._ids.size else plane.values[..., active]
            for plane in pair
        )
        step = max(1, min(_CHUNK_VOXELS, _GATHER_VALUES // max(1, active.size)))

        def interpolate(p0: HostArray, p1: HostArray, p2: HostArray) -> FloatArray:
            return self._mixed(
                bucket - 1,
                (p0, p1, p2),
                (pair[0].covered, pair[1].covered),
                (values[0], values[1]),
                self._ids[active],
            )

        if self._axis_buckets is not None:
            self._restore_separable(bucket, self._axis_buckets, uniform, first, step, interpolate)
            return

        assert self._order is not None
        indices = self._order[self._offsets[bucket] : self._offsets[bucket + 1]]
        output = self._output.reshape(-1)
        for start in range(0, indices.size, step):
            flat = indices[start : start + step].astype(np.int64, copy=False)
            p0, p1, p2 = self._grid.coordinates(*_unravel(flat, self._grid.source_shape))
            cell = (np.floor(p1).astype(np.int64) + 1, np.floor(p2).astype(np.int64) + 1)
            result = np.where(uniform[cell], first[cell], np.float32(0)).astype(np.float32)
            mixed = np.flatnonzero(~uniform[cell])
            if mixed.size:
                result[mixed] = interpolate(p0[mixed], p1[mixed], p2[mixed])
            output[flat] = result

    def _restore_separable(
        self,
        bucket: int,
        axes: _AxisBuckets,
        uniform: BoolArray,
        first: FloatArray,
        step: int,
        interpolate: Callable[[HostArray, HostArray, HostArray], FloatArray],
    ) -> None:
        """Map each transverse cell once for all source planes enclosed by this bucket."""
        selected = axes.plane_buckets == bucket
        px, py, pz = axes.positions
        dx, dy, dz = axes.axes
        jy = np.floor(py).astype(np.int64) + 1
        jz = np.floor(pz).astype(np.int64) + 1
        output = self._output.transpose(axes.source_axes)
        # Bound the 2D map on wide source grids independently of the score gathers.
        width = min(jz.size, _CHUNK_VOXELS)
        height = _CHUNK_VOXELS // width
        for y_start in range(0, jy.size, height):
            y = slice(y_start, y_start + height)
            for z_start in range(0, jz.size, width):
                z = slice(z_start, z_start + width)
                cell = jy[y, None], jz[None, z]
                result = first[cell]
                mixed_y, mixed_z = np.nonzero(~uniform[cell])
                mixed_py, mixed_pz = py[y][mixed_y], pz[z][mixed_z]
                for dest, position in zip(dx[selected], px[selected], strict=True):
                    for start in range(0, mixed_y.size, step):
                        subset = slice(start, start + step)
                        mixed_px = np.full(mixed_py[subset].size, position, dtype=px.dtype)
                        result[mixed_y[subset], mixed_z[subset]] = interpolate(
                            mixed_px, mixed_py[subset], mixed_pz[subset]
                        )
                    output[dest, dy[y, None], dz[None, z]] = result

    def _mixed(
        self,
        lower_index: int,
        coords: tuple[HostArray, HostArray, HostArray],
        covered: tuple[BoolArray, BoolArray],
        values: tuple[FloatArray, FloatArray],
        ids: FloatArray,
    ) -> FloatArray:
        """Interpolate covered neighbor scores, renormalized; NaN if the nearest is uncovered."""
        size_y, size_z = self._grid.preprocessed_shape[1:]
        p0, p1, p2 = coords
        y0, z0 = np.floor(p1).astype(np.int64), np.floor(p2).astype(np.int64)
        fx, fy, fz = p0 - lower_index, p1 - y0, p2 - z0
        # Round absolute coordinates to even, independently of the interpolation.
        nearest_y = _clip(np.rint(p1).astype(np.int64), size_y)
        nearest_z = _clip(np.rint(p2).astype(np.int64), size_z)
        nearest_covered = np.where(
            np.rint(p0) == lower_index,
            covered[0][nearest_y, nearest_z],
            covered[1][nearest_y, nearest_z],
        )
        ys = (_clip(y0, size_y), _clip(y0 + 1, size_y))
        zs = (_clip(z0, size_z), _clip(z0 + 1, size_z))
        total = np.zeros((p0.size, ids.size), dtype=np.float32)
        weight_sum = np.zeros(p0.size, dtype=np.float32)
        for dx, wx in ((0, 1 - fx), (1, fx)):
            for dy, wy in ((0, 1 - fy), (1, fy)):
                for dz, wz in ((0, 1 - fz), (1, fz)):
                    y, z = ys[dy], zs[dz]
                    usable = covered[dx][y, z]
                    weight = np.where(usable, wx * wy * wz, 0).astype(np.float32)
                    # Advanced indexing owns this gather, so weighting can reuse it.
                    weighted = values[dx][y, z]
                    weighted *= weight[:, None]
                    total += weighted
                    del weighted
                    weight_sum += weight
        with np.errstate(divide="ignore", invalid="ignore"):
            total /= weight_sum[:, None]
        total[~nearest_covered] = 0
        labels = _decide(total, ids)
        return np.where(nearest_covered, labels, np.float32(np.nan)).astype(np.float32)


def _decide(scores: FloatArray, ids: FloatArray) -> FloatArray:
    """``LabelReducer``'s rule on (voxels, classes): first maximum, background unless positive."""
    if not ids.size:
        return np.zeros(scores.shape[0], dtype=np.float32)
    winners = np.argmax(scores, axis=-1)
    labels = ids[winners]
    labels[np.take_along_axis(scores, winners[:, None], axis=-1)[:, 0] <= 0] = 0
    return labels


def _unravel(flat: IntArray, shape: tuple[int, int, int]) -> tuple[IntArray, IntArray, IntArray]:
    """Convert flat source voxel indices into three coordinate arrays."""
    d, h, w = np.unravel_index(flat, shape)
    return (
        d.astype(np.int64, copy=False),
        h.astype(np.int64, copy=False),
        w.astype(np.int64, copy=False),
    )


def _inside(coords: tuple[HostArray, ...], shape: tuple[int, int, int]) -> BoolArray:
    """Return a mask for coordinates within the specified spatial shape."""
    inside = np.ones(coords[0].shape, dtype=bool)
    for value, size in zip(coords, shape, strict=True):
        inside &= (value >= 0) & (value < size)
    return inside


def _clip(index: IntArray, size: int) -> IntArray:
    """Clamp voxel indices to a valid axis range."""
    return np.clip(index, 0, size - 1)
