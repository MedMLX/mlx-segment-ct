# Copyright (c) MONAI Consortium
# Portions derived from MONAI, licensed under Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""NumPy port of the pinned NVIDIA VISTA3D bundle transforms.

NV-Segment-CTMR f9f5f51 configs/inference.json uses MONAI 1.6.0:

* ``LoadImaged`` (NibabelReader, float32) and ``EnsureChannelFirstd``;
* ``Spacingd`` to 1.5 mm, bilinear (``nearest`` for a correction mask), border
  padding, ``align_corners=False``, float64 resampling;
* ``CropForegroundd`` on voxels above zero, margin 10, ``allow_smaller=True``;
* ``ScaleIntensityRanged`` from [-963.82, 1053.68] to [0, 1] with clipping, or for
  NV-Segment-CTMR ``ScaleIntensityRangePercentilesd`` from the 1st/99th percentiles to
  [0, 1] with clipping;
* ``Orientationd`` to RAS and ``CastToTyped`` float32;
* ``Invertd`` with nearest interpolation, then ``SaveImaged`` (float32 NIfTI).

Torch's CPU arithmetic is reproduced in float64 NumPy: ``torch.linspace`` with its
fused multiply-add, ``affine_grid`` and ``grid_sample`` (5D, border padding) in
their source operation order, and the 4x4 affine products. MONAI's NumPy
helpers are called exactly as MONAI calls them. For axis-aligned affines (no
rotation or shear) the results are bitwise identical to MONAI on the fixture
host, because every product then has a single nonzero term. Oblique affines go
through the same algorithm but their multi-term sums can round differently from
a particular BLAS by float64 rounding.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
from typing import Literal, cast

import nibabel as nib
import numpy as np
from medmlx_core.errors import InvalidInputError
from nibabel import orientations
from nibabel.filebasedimages import ImageFileError
from nibabel.spatialimages import HeaderDataError

from mlx_segment_ct._types import (
    BoolArray,
    DoubleArray,
    HostArray,
    IntArray,
    NiftiHeader,
    NiftiImage,
    NiftiRuntime,
    OrientationRuntime,
)

nifti_api = cast("NiftiRuntime", nib)
orientation_api = cast("OrientationRuntime", orientations)

AFFINE_TOL = 1e-3
RESAMPLE_SPACING = (1.5, 1.5, 1.5)
INTENSITY_RANGE = (-963.8247715525971, 1053.678477684517)
INTENSITY_PERCENTILES = (1, 99)
# MONAI ``percentile`` uses np.percentile above this many voxels (pytorch#64947),
# torch.quantile otherwise.
NUMPY_PERCENTILE_VOXELS = 1_000_000
type Intensity = Literal["range", "percentiles"]
CROP_MARGIN = 10
_INT_MAX, _INT_MIN = 2**31 - 1, -(2**31)
_VELTKAMP_SPLIT = 134217729.0  # 2**27 + 1
_CHUNK_VOXELS = 1 << 21


@dataclass(frozen=True, slots=True)
class _Resampled:
    """One ``SpatialResample`` call: its source affine and spatial shape."""

    source_affine: DoubleArray
    source_shape: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class _Cropped:
    """One ``CropForeground`` call: the box in the resampled grid and that grid's shape."""

    start: tuple[int, int, int]
    end: tuple[int, int, int]
    source_shape: tuple[int, int, int]


@dataclass(frozen=True, slots=True)
class Preprocessed:
    """A preprocessed volume with the trace that ``Invertd`` replays."""

    image: HostArray  # (1, X, Y, Z) float32 in the preprocessed grid
    affine: DoubleArray  # float64 4x4 after Orientationd
    original_affine: DoubleArray  # float64 4x4 as loaded
    resampled: _Resampled
    cropped: _Cropped
    oriented_from: DoubleArray  # the affine Orientationd received
    extra: dict[str, HostArray]  # further keys (correction mask), same grid

    def invert(self, volume: HostArray) -> tuple[HostArray, DoubleArray]:
        """``Invertd(nearest_interp=True)``: restore a (C, X, Y, Z) field to the source grid.

        Returns the float32 field and the affine MONAI records for it.
        """
        data, affine = _orient(volume, self.affine, nifti_api.aff2axcodes(self.oriented_from))
        # CropForeground.inverse: the zero padding is undone by a full-extent crop
        # (identity translation), then Crop.inverse pads back what was cropped.
        start, end, shape = self.cropped.start, self.cropped.end, self.cropped.source_shape
        pads = [(begin, size - stop) for begin, stop, size in zip(start, end, shape, strict=True)]
        if any(value for pair in pads for value in pair):
            data = np.pad(data, [(0, 0), *pads])
            affine = affine @ _translate([-begin for begin, _ in pads])
        return _spatial_resample(
            data, affine, self.resampled.source_affine, self.resampled.source_shape, "nearest"
        )

    def restoration_grid(self) -> RestorationGrid:
        """The source-voxel to preprocessed-voxel map that ``invert`` samples through."""
        # Orientation is a signed axis permutation with an integer offset.
        to_preprocessed = np.rint(np.linalg.solve(self.affine, self.oriented_from))
        start = _translate([-value for value in self.cropped.start])
        # The affine ``invert`` resamples from: oriented back, then uncropped.
        _, resampled_affine = _orientation(
            self.affine, _shape(self.image), nifti_api.aff2axcodes(self.oriented_from)
        )
        if any(self.cropped.start) or self.cropped.end != self.cropped.source_shape:
            resampled_affine = resampled_affine @ start
        theta, affine = _resample_plan(
            resampled_affine,
            self.resampled.source_affine,
            self.cropped.source_shape,
            self.resampled.source_shape,
        )
        return RestorationGrid(
            source_shape=self.resampled.source_shape,
            affine=affine,
            preprocessed_shape=_shape(self.image),
            resampled_shape=self.cropped.source_shape,
            theta=theta,
            to_preprocessed=to_preprocessed @ start,
            axis_indices=_axis_indices(
                theta, self.cropped.source_shape, self.resampled.source_shape
            ),
        )


@dataclass(frozen=True, slots=True)
class RestorationGrid:
    """Where ``Invertd`` samples the preprocessed grid for each source voxel.

    ``coordinates`` returns the continuous preprocessed indices that nearest
    restoration rounds, through the same Torch ``affine_grid`` and border-clipped
    ``grid_sample`` source indices.
    """

    source_shape: tuple[int, int, int]
    affine: DoubleArray  # the restored label map's affine, as ``invert`` returns it
    preprocessed_shape: tuple[int, int, int]
    resampled_shape: tuple[int, int, int]
    theta: DoubleArray | None  # None when MONAI skips resampling (identity grid)
    to_preprocessed: DoubleArray  # resampled index -> preprocessed index
    # Resampled (iz, iy, ix) per source d, h, w index when each depends on one axis.
    axis_indices: tuple[HostArray, HostArray, HostArray] | None = None

    def coordinates(
        self, d: IntArray, h: IntArray, w: IntArray
    ) -> tuple[HostArray, HostArray, HostArray]:
        """Preprocessed-grid indices for source voxels at array indices ``(d, h, w)``.

        ``d``, ``h`` and ``w`` index the source array's first, second and third
        spatial axes. Returns float indices along the preprocessed array's three
        axes; the caller rounds them (half to even) for nearest restoration.
        """
        resampled: tuple[HostArray, HostArray, HostArray]
        if self.axis_indices is not None:
            iz, iy, ix = self.axis_indices
            resampled = (iz[d], iy[h], ix[w])
        elif self.theta is None:
            resampled = (d.astype(np.float64), h.astype(np.float64), w.astype(np.float64))
        else:
            depth, height, width = self.source_shape
            bases = (base_axis(width)[w], base_axis(height)[h], base_axis(depth)[d])
            ix, iy, iz = _grid_coordinates(self.theta, self.resampled_shape, bases)
            resampled = (iz, iy, ix)
        m = self.to_preprocessed
        p0, p1, p2 = (
            resampled[0] * m[row, 0] + resampled[1] * m[row, 1] + resampled[2] * m[row, 2]
            + m[row, 3]
            for row in range(3)
        )
        return p0, p1, p2


def load_volume(path: Path) -> tuple[HostArray, DoubleArray]:
    """``LoadImaged(image_only=True)`` and ``EnsureChannelFirstd`` via NibabelReader.

    Accepts a 3D NIfTI-1/2 image or a 4D image with one trailing channel.
    Returns the float32 (1, X, Y, Z) array and the float64 affine. Header
    spatial units must be millimeters or unknown, with unknown assumed to mean
    millimeters. Explicit meter or micron units are rejected rather than rescaled.
    The affine is checked for finiteness and invertibility, but image voxel values
    are not checked for finiteness here.
    """
    try:
        image = nifti_api.load(str(path))
    except (OSError, EOFError, ValueError, ImageFileError, HeaderDataError) as exc:
        raise InvalidInputError(f"NV-Segment-CT cannot read NIfTI {path}: {exc}") from exc
    if not isinstance(image, nib.Nifti1Image | nib.Nifti2Image):
        raise InvalidInputError(f"NV-Segment-CT requires a NIfTI image: {path}")
    image = _correct_nifti_header_if_necessary(cast("NiftiImage", image))
    try:
        spatial_units, _ = image.header.get_xyzt_units()
    except KeyError as exc:
        raise InvalidInputError(f"NV-Segment-CT has an invalid NIfTI unit code: {path}") from exc
    if spatial_units not in ("mm", "unknown"):
        raise InvalidInputError(
            f"NV-Segment-CT spatial units are {spatial_units!r}; convert the affine "
            "and header to millimeters before inference"
        )
    affine = np.array(image.affine, copy=True)
    header = image.header.as_byteswapped("<") if _swappable(image.header) else image.header
    dim = header["dim"]
    spatial_shape = list(dim[1:])[: max(min(int(dim[0]), 3), 1)]
    data = np.asanyarray(image.dataobj).astype(np.float32)
    if len(spatial_shape) != 3 or data.ndim not in (3, 4):
        raise InvalidInputError(f"NV-Segment-CT requires a 3D CT volume: {path} {data.shape}")
    # EnsureChannelFirst: no channel axis for 3D data, channel-last otherwise.
    data = data[None] if data.ndim == len(spatial_shape) else np.moveaxis(data, -1, 0)
    if data.shape[0] != 1:
        raise InvalidInputError(f"NV-Segment-CT requires one image channel: {path} {data.shape}")
    if not np.isfinite(affine).all() or np.linalg.det(affine[:3, :3]) == 0:
        raise InvalidInputError(f"NV-Segment-CT requires an invertible finite affine: {path}")
    return data, affine


def preprocess(
    image: HostArray,
    affine: DoubleArray,
    extra: dict[str, tuple[HostArray, DoubleArray]] | None = None,
    *,
    intensity: Intensity = "range",
) -> Preprocessed:
    """Run the pinned preprocessing on the image and optional nearest-resampled keys.

    ``extra`` maps keys to (volume, affine) pairs for nearest resampling, followed by
    the image's crop and RAS orientation. ``intensity`` selects
    NV-Segment-CT's fixed range (``"range"``) or NV-Segment-CTMR's percentiles.

    ``image`` is float32 ``(1, X, Y, Z)`` as ``load_volume`` returns it, and ``affine``
    its voxel-to-world matrix (NIfTI RAS, assumed millimeters). Voxel values must
    be finite. They are scaled to [0, 1] in the usual nondegenerate case (see
    ``scale_intensity_percentiles`` for the equal-percentile case). The result is
    on the RAS-oriented 1.5 mm foreground crop.

    Raises:
        InvalidInputError: image voxels are non-finite, no voxel is above zero after
            resampling, or an ``extra`` volume does not resample onto the image grid.
        ValueError: unknown ``intensity``.
    """
    if not np.isfinite(image).all():
        raise InvalidInputError("NV-Segment-CT image contains non-finite voxel intensities")
    resampled_image, image_affine = _spacing(image, affine, "bilinear")
    resampled = _Resampled(affine, _shape(image))
    start, end = _foreground_box(resampled_image)
    if any(stop <= begin for begin, stop in zip(start, end, strict=True)):
        raise InvalidInputError(
            "NV-Segment-CT found no voxel above 0 after resampling; the CT foreground crop is empty"
        )
    cropped = _Cropped(start, end, _shape(resampled_image))
    window = (slice(None), *(slice(begin, stop) for begin, stop in zip(start, end, strict=True)))
    data = resampled_image[window]
    image_affine = image_affine @ _translate(start)
    if intensity == "range":
        data = scale_intensity_range(data)
    elif intensity == "percentiles":
        data = scale_intensity_percentiles(data)
    else:
        raise ValueError(f"Unknown NV intensity scaling: {intensity!r}")
    oriented_from = image_affine
    data, image_affine = _orient(data, image_affine, ("R", "A", "S"))
    outputs: dict[str, HostArray] = {}
    for key, (volume, volume_affine) in (extra or {}).items():
        values, values_affine = _spacing(volume, volume_affine, "nearest")
        if _shape(values) != cropped.source_shape:
            raise InvalidInputError(f"NV-Segment-CT {key} resamples to a different grid")
        values = values[window]
        values_affine = values_affine @ _translate(start)
        outputs[key] = _orient(values, values_affine, ("R", "A", "S"))[0].astype(np.float32)
    return Preprocessed(
        image=data.astype(np.float32),
        affine=image_affine,
        original_affine=np.array(affine, dtype=np.float64, copy=True),
        resampled=resampled,
        cropped=cropped,
        oriented_from=oriented_from,
        extra=outputs,
    )


def scale_intensity_range(image: HostArray) -> HostArray:
    """``ScaleIntensityRanged`` on float32 data, in Torch's scalar operation order."""
    a_min, a_max = INTENSITY_RANGE
    values = (image - np.float32(a_min)) / np.float32(a_max - a_min)
    values = values * np.float32(1) + np.float32(0)
    return np.clip(values, np.float32(0), np.float32(1)).astype(np.float32)


def scale_intensity_percentiles(image: HostArray) -> HostArray:
    """``ScaleIntensityRangePercentilesd(lower=1, upper=99, b_min=0, b_max=1, clip=True)``.

    Percentiles of the whole float32 tensor follow MONAI's ``percentile``; scaling
    then repeats ``ScaleIntensityRange`` with float32 bounds, including its unclipped
    ``img - a_min + b_min`` result when both percentiles are equal.
    """
    values = np.asarray(image, dtype=np.float32)
    lower, upper = INTENSITY_PERCENTILES
    a_min, a_max = monai_percentile(values, lower), monai_percentile(values, upper)
    span = np.float32(a_max - a_min)
    if span == 0:
        return (values - a_min + np.float32(0)).astype(np.float32)
    scaled = (values - a_min) / span
    scaled = scaled * np.float32(1) + np.float32(0)
    return np.clip(scaled, np.float32(0), np.float32(1)).astype(np.float32)


def monai_percentile(values: HostArray, q: int) -> np.float32:
    """MONAI 1.6.0 ``percentile`` of a float32 Torch tensor, as a float32 value.

    Above ``NUMPY_PERCENTILE_VOXELS`` MONAI calls ``np.percentile`` and casts the
    float64 result back to float32. Smaller tensors use ``torch.quantile``: the float32
    ``q / 100`` times ``n - 1`` as a float64 rank, a float32 weight, then ATen's linear
    lerp, whose CPU kernel rounds ``weight * (end - start) + base`` once (a fused
    multiply-add).
    """
    flat = np.asarray(values, dtype=np.float32).ravel()
    if flat.size > NUMPY_PERCENTILE_VOXELS:
        return np.float32(np.percentile(flat, q))
    ordered = np.sort(flat)
    if np.isnan(ordered[-1]):
        return np.float32(np.nan)
    rank = float(np.float32(q / 100.0)) * (ordered.size - 1)
    below, above = int(rank), math.ceil(rank)
    weight = np.float32(rank - below)
    start, end = np.float32(ordered[below]), np.float32(ordered[above])
    delta = np.float32(end - start)
    if abs(weight) < 0.5:
        return _fused_multiply_add(weight, delta, start)
    return _fused_multiply_add(np.float32(weight - np.float32(1)), delta, end)


def _fused_multiply_add(a: np.float32, b: np.float32, c: np.float32) -> np.float32:
    """Float32 ``a * b + c`` rounded once to nearest, ties to even."""
    if not np.isfinite([a, b, c]).all():
        return np.float32(a * b + c)
    exact = Fraction(float(a)) * Fraction(float(b)) + Fraction(float(c))
    nearest = np.float32(float(exact))  # at most one float32 step from the exact result
    candidates = (
        np.nextafter(nearest, np.float32(-np.inf)),
        nearest,
        np.nextafter(nearest, np.float32(np.inf)),
    )
    return min(
        candidates,
        key=lambda value: (abs(Fraction(float(value)) - exact), int(value.view(np.int32)) & 1),
    )


def save_label_map(path: Path, labels: HostArray, affine: DoubleArray) -> None:
    """Save float32 labels and the supplied affine in a new NIfTI-1 header.

    The source header, including its unit code, is not copied. The sform uses
    the supplied affine and qform uses NiBabel's shear-stripped representation;
    both have code 1.
    """
    data = np.asarray(labels)
    data = data[0] if data.ndim == 4 and data.shape[0] == 1 else data
    image = nifti_api.Nifti1Image(data.astype(np.float32, copy=False), affine)
    image.set_sform(affine, code=1)
    image.set_qform(affine, code=1)
    nifti_api.save(image, str(path))


def _spacing(volume: HostArray, affine: DoubleArray, mode: str) -> tuple[HostArray, DoubleArray]:
    """``Spacing(pixdim=1.5, diagonal=False)``: target grid, then ``SpatialResample``."""
    affine_ = _to_affine_nd(affine)
    out_d = np.array(RESAMPLE_SPACING, dtype=np.float64)
    orig_d = _affine_to_spacing(affine_)
    for index, spacing in enumerate(orig_d):
        target = out_d[index]
        out_d[index] = spacing if target - AFFINE_TOL <= spacing <= target + AFFINE_TOL else target
    new_affine = _zoom_affine(affine_, out_d)
    output_shape, offset = _compute_shape_offset(_shape(volume), affine_, new_affine)
    new_affine[:3, -1] = offset[:3]
    return _spatial_resample(volume, affine, new_affine, tuple(output_shape.tolist()), mode)


def _spatial_resample(
    volume: HostArray,
    src_affine: DoubleArray,
    dst_affine: DoubleArray,
    spatial_size: tuple[int, ...],
    mode: str,
) -> tuple[HostArray, DoubleArray]:
    """MONAI ``spatial_resample`` through ``AffineTransform`` (Torch grid sampling)."""
    theta, new_affine = _resample_plan(src_affine, dst_affine, _shape(volume), spatial_size)
    if theta is None:
        return volume.astype(np.float32), new_affine
    return _grid_sample(volume, theta, spatial_size, mode), new_affine


def _resample_plan(
    src_affine: DoubleArray,
    dst_affine: DoubleArray,
    in_shape: tuple[int, ...],
    spatial_size: tuple[int, ...],
) -> tuple[DoubleArray | None, DoubleArray]:
    """The (3, 4) Torch ``theta`` and output affine; no ``theta`` when MONAI skips."""
    src = _to_affine_nd(src_affine)
    dst = _to_affine_nd(dst_affine)
    try:
        xform = _to_affine_nd(np.linalg.solve(src, dst))
    except np.linalg.LinAlgError as exc:
        raise InvalidInputError(f"NV-Segment-CT source affine is not invertible: {src}") from exc
    same_size = np.allclose(spatial_size, in_shape)
    if (np.allclose(src, dst, atol=AFFINE_TOL) and same_size) or (
        np.allclose(xform, np.eye(4), atol=AFFINE_TOL) and same_size
    ):
        return None, np.array(src_affine, dtype=np.float64)
    new_affine = _torch_matmul(src, xform)
    theta = _to_norm_affine(xform, in_shape, spatial_size)
    reverse = [2, 1, 0]
    theta[:3] = theta[reverse]
    theta[:, :3] = theta[:, reverse]
    return theta[:3], new_affine


def _to_norm_affine(
    affine: DoubleArray, src_size: tuple[int, ...], dst_size: tuple[int, ...]
) -> DoubleArray:
    """``to_norm_affine(align_corners=False)``; Torch batched 4x4 products."""
    src_xform = _normalize_transform(src_size)
    dst_xform = _normalize_transform(dst_size)
    inverse = np.linalg.inv(dst_xform[None])[0]
    return _torch_matmul(_torch_matmul(src_xform, affine), inverse)


def _normalize_transform(shape: tuple[int, ...]) -> DoubleArray:
    """Build the align-corners-false voxel-to-normalized coordinate transform."""
    values = np.asarray(shape, dtype=np.float64)
    norm = values.copy()
    norm[norm <= 0.0] = 2.0
    norm = (1.0 / norm) * 2.0  # Torch evaluates ``2.0 / t`` as reciprocal(t) * 2.0
    result = np.diag(np.concatenate((norm, np.ones(1))))
    result[:-1, -1] = (1.0 / values) * 1.0 - 1.0
    return result


def _torch_matmul(a: DoubleArray, b: DoubleArray) -> DoubleArray:
    """Torch CPU 4x4 float64 product: sequential sums without fused multiply-add."""
    result = np.zeros((a.shape[0], b.shape[1]))
    for k in range(a.shape[1]):
        result = result + a[:, k : k + 1] * b[k : k + 1, :]
    return result


@lru_cache(maxsize=64)
def base_axis(size: int) -> HostArray:
    """``affine_grid`` base coordinates: ``torch.linspace(-1, 1)`` scaled for align_corners=False.

    Torch computes element i as fma(step, i, -1) below the midpoint and
    fma(-step, n - 1 - i, 1) above it; Fractions give that single rounding exactly.
    """
    if size <= 1:
        return np.zeros(1)
    step = (1.0 - -1.0) / (size - 1)
    half = size // 2
    values = [
        float(Fraction(step) * i + Fraction(-1.0))
        if i < half
        else float(Fraction(-step) * (size - i - 1) + Fraction(1.0))
        for i in range(size)
    ]
    result = np.array(values, dtype=np.float64) * (size - 1) / size
    result.setflags(write=False)
    return result


def _source_index(coordinate: HostArray, size: int) -> HostArray:
    """``grid_sampler_compute_source_index``: unnormalize, border clip, int-range guard.

    Clang contracts the unnormalization ``(coordinate + 1) * size - 1`` into one
    fused multiply-add; a separately rounded product moves near-integer indices.
    """
    shifted = np.asarray(coordinate + 1.0, dtype=np.float64)
    value = _fused_multiply_add_float64(shifted, float(size), -1.0) / 2.0
    value = np.where(value < 0.0, 0.0, value)  # std::max(value, 0)
    limit = float(size - 1)
    value = np.where(value < limit, value, limit)  # std::min(size - 1, value)
    valid = np.isfinite(value) & (value <= _INT_MAX - 1) & (value >= _INT_MIN)
    return np.where(valid, value, -100.0)


def _fused_multiply_add_float64(a: DoubleArray, b: float, c: float) -> DoubleArray:
    """Float64 ``a * b + c`` rounded once, without ``math.fma`` (Python 3.13+).

    Dekker's exact product and Knuth's TwoSum carry the exact value; Boldo and
    Melquiond's round-to-odd tail (IEEE Trans. Computers, 2008) then makes the
    final addition round correctly.
    """
    product = a * b
    a_high, a_low = _split(a)
    b_high, b_low = _split(np.asarray(b, dtype=np.float64))
    product_error = ((a_high * b_high - product) + a_high * b_low + a_low * b_high) + a_low * b_low
    total, total_error = _two_sum(np.full_like(product, c), product)
    tail, tail_error = _two_sum(total_error, product_error)
    even = (tail.view(np.int64) & 1) == 0
    toward = np.where(tail_error > 0, np.inf, -np.inf)
    odd = np.where((tail_error != 0) & even, np.nextafter(tail, toward), tail)
    fused = total + odd
    return np.where(np.isfinite(fused), fused, product + c)


def _split(value: DoubleArray) -> tuple[DoubleArray, DoubleArray]:
    """Split float64 values into high and low parts for error-free products."""
    scaled = _VELTKAMP_SPLIT * value
    high = scaled - (scaled - value)
    return high, value - high


def _two_sum(a: DoubleArray, b: DoubleArray) -> tuple[DoubleArray, DoubleArray]:
    """Return a float64 sum and its rounding error."""
    total = a + b
    b_virtual = total - a
    return total, (a - (total - b_virtual)) + (b - b_virtual)


def _grid_sample(
    volume: HostArray, theta: HostArray, out_shape: tuple[int, ...], mode: str
) -> HostArray:
    """``grid_sample(affine_grid(theta))`` for 5D input, border padding, align_corners=False.

    Output voxels are processed in slabs along the first axis; each voxel follows
    Torch's CPU kernel exactly, including corner order and bounds checks.
    """
    sizes = _shape(volume)  # Torch (D, H, W)
    depth, height, width = (int(value) for value in out_shape)
    base_w = base_axis(width)[None, None, :]
    base_h = base_axis(height)[None, :, None]
    base_d = base_axis(depth)[:, None, None]
    source = volume.astype(np.float64) if mode == "bilinear" else volume
    output = np.empty((volume.shape[0], depth, height, width), dtype=np.float32)
    rows = max(1, _CHUNK_VOXELS // max(1, height * width))
    for first in range(0, depth, rows):
        last = min(depth, first + rows)
        ix, iy, iz = _grid_coordinates(theta, sizes, (base_w, base_h, base_d[first:last]))
        block = (slice(None), slice(first, last))
        if mode == "nearest":
            output[block] = _nearest(source, ix, iy, iz)
        elif mode == "bilinear":
            output[block] = _trilinear(source, ix, iy, iz)
        else:
            raise ValueError(f"Unsupported resampling mode: {mode}")
    return output


def _grid_coordinates(
    theta: HostArray,
    sizes: tuple[int, ...],
    bases: tuple[HostArray, HostArray, HostArray],
) -> tuple[HostArray, HostArray, HostArray]:
    """Border-clipped ``(ix, iy, iz)`` source indices for broadcast ``(w, h, d)`` bases."""
    base_w, base_h, base_d = bases
    separable = all(theta[j, k] == 0 for j in range(3) for k in range(3) if j != k)
    coordinates: list[HostArray] = []
    for j in range(3):
        t = theta[j]
        if separable:
            grid = bases[j] * t[j] + t[3]
        else:
            grid = (base_w * t[0] + base_d * t[2]) + (base_h * t[1] + t[3])
        coordinates.append(grid)
    return (
        _source_index(coordinates[0], sizes[2]),
        _source_index(coordinates[1], sizes[1]),
        _source_index(coordinates[2], sizes[0]),
    )


def _axis_indices(
    theta: HostArray | None, sizes: tuple[int, ...], out_shape: tuple[int, int, int]
) -> tuple[HostArray, HostArray, HostArray] | None:
    """``_grid_coordinates`` evaluated once per output axis, when the grid is separable."""
    depth, height, width = out_shape
    if theta is None:
        return (
            np.arange(depth, dtype=np.float64),
            np.arange(height, dtype=np.float64),
            np.arange(width, dtype=np.float64),
        )
    if any(theta[j, k] != 0 for j in range(3) for k in range(3) if j != k):
        return None
    ix, iy, iz = _grid_coordinates(
        theta, sizes, (base_axis(width), base_axis(height), base_axis(depth))
    )
    return iz, iy, ix


def _gather(
    source: HostArray, z: IntArray, y: IntArray, x: IntArray
) -> tuple[HostArray, BoolArray]:
    """Gather clipped voxels and report which indices are inside the source."""
    sizes = _shape(source)
    inside = (z >= 0) & (z < sizes[0]) & (y >= 0) & (y < sizes[1]) & (x >= 0) & (x < sizes[2])
    values = source[
        :, np.clip(z, 0, sizes[0] - 1), np.clip(y, 0, sizes[1] - 1), np.clip(x, 0, sizes[2] - 1)
    ]
    return values, inside


def _nearest(source: HostArray, ix: HostArray, iy: HostArray, iz: HostArray) -> HostArray:
    """Sample nearest voxels and set out-of-grid values to zero."""
    x, y, z = (np.rint(value).astype(np.int64) for value in (ix, iy, iz))  # std::nearbyint
    values, inside = _gather(source, z, y, x)
    return np.where(inside, values, 0)


def _trilinear(source: HostArray, ix: HostArray, iy: HostArray, iz: HostArray) -> HostArray:
    # Sample ``source`` (C, Z, Y, X) at continuous voxel indices; float64 result (C, ...).
    # Corners outside the grid contribute nothing, so edge samples fade toward zero.
    """Interpolate the eight neighboring voxels with zero outside the grid."""
    x0, y0, z0 = (np.floor(value).astype(np.int64) for value in (ix, iy, iz))
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1
    east, west = ix - x0, x1 - ix
    south, north = iy - y0, y1 - iy
    bottom, top = iz - z0, z1 - iz
    corners = (
        (z0, y0, x0, west * north * top),
        (z0, y0, x1, east * north * top),
        (z0, y1, x0, west * south * top),
        (z0, y1, x1, east * south * top),
        (z1, y0, x0, west * north * bottom),
        (z1, y0, x1, east * north * bottom),
        (z1, y1, x0, west * south * bottom),
        (z1, y1, x1, east * south * bottom),
    )
    shape = np.broadcast_shapes(ix.shape, iy.shape, iz.shape)
    result = np.zeros((source.shape[0], *shape), dtype=np.float64)
    for z, y, x, weight in corners:
        values, inside = _gather(source, z, y, x)
        result = np.where(inside, result + values * weight, result)
    return result


def _foreground_box(image: HostArray) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """``generate_spatial_bounding_box`` for voxels above zero, margin 10, allow_smaller."""
    data = (image > 0).any(0)
    spatial = data.shape
    start, end = [0, 0, 0], [0, 0, 0]
    for axis, reduced in enumerate(itertools.combinations(reversed(range(3)), 2)):
        profile = data.any(axis=reduced)
        if not profile.any():
            return (0, 0, 0), (0, 0, 0)
        hits = np.flatnonzero(profile)
        start[axis] = max(int(hits[0]) - CROP_MARGIN, 0)
        end[axis] = min(int(hits[-1]) + CROP_MARGIN + 1, spatial[axis])
    return (start[0], start[1], start[2]), (end[0], end[1], end[2])


def _orient(
    volume: HostArray, affine: DoubleArray, axcodes: tuple[str | None, ...]
) -> tuple[HostArray, DoubleArray]:
    """``Orientation(axcodes)``: flips, then the axis permutation; NumPy affine product."""
    ornt, new_affine = _orientation(affine, _shape(volume), axcodes)
    flips = [axis + 1 for axis, flip in enumerate(ornt[:, 1]) if flip == -1]
    data = np.flip(volume, axis=tuple(flips)) if flips else volume
    order = [0, *(int(axis) + 1 for axis in np.argsort(ornt[:, 0]))]
    if order != [0, 1, 2, 3]:
        data = np.transpose(data, order)
    return np.ascontiguousarray(data), new_affine


def _orientation(
    affine: DoubleArray, shape: tuple[int, int, int], axcodes: tuple[str | None, ...]
) -> tuple[HostArray, DoubleArray]:
    # Returns the nibabel orientation transform (per axis: source axis, flip) and the affine
    # of the reoriented volume.
    """Compute the voxel-axis reorientation and its updated affine."""
    affine_ = _to_affine_nd(affine)
    source = nifti_api.io_orientation(affine_)
    target = orientation_api.axcodes2ornt(axcodes[:3], labels=None)
    ornt = orientation_api.ornt_transform(source, target)
    xform = orientation_api.inv_ornt_aff(ornt, shape)
    return ornt, np.asarray(affine, dtype=np.float64) @ xform


def _translate(shift: list[int] | tuple[int, ...]) -> DoubleArray:
    """Build a homogeneous affine for a voxel translation."""
    result = np.eye(4)
    result[:3, 3] = shift
    return result


def _shape(volume: HostArray) -> tuple[int, int, int]:
    """Return the three spatial dimensions of a channels-first volume."""
    return (int(volume.shape[1]), int(volume.shape[2]), int(volume.shape[3]))


def _to_affine_nd(affine: DoubleArray) -> DoubleArray:
    # Embed the leading rotation/zoom block and translation in a 4x4 identity.
    """Embed spatial rotation and translation in a 4x4 affine."""
    matrix = np.array(affine, dtype=np.float64, copy=True)
    result = np.eye(4, dtype=np.float64)
    size = max(min(3, len(matrix) - 1), 1)
    result[:size, :size] = matrix[:size, :size]
    if size > 1:
        result[:size, -1] = matrix[:size, -1]
    return result


def _affine_to_spacing(affine: DoubleArray, rank: int = 3) -> DoubleArray:
    """Compute per-axis voxel spacing from an affine."""
    block = np.asarray(affine[:rank, :rank], dtype=float)
    spacing = np.sqrt(np.sum(block * block, axis=0))
    spacing[spacing == 0] = 1.0
    return spacing


def _zoom_affine(affine: DoubleArray, scale: HostArray) -> DoubleArray:
    """``zoom_affine(diagonal=False)``: keep orientation and rotation, set voxel sizes."""
    affine = np.array(affine, dtype=float, copy=True)
    norm = _affine_to_spacing(affine, len(affine) - 1)
    scale_np = np.array(scale, dtype=float, copy=True)[: len(affine) - 1]
    scale_np = np.asarray([s if s and s > 0 else n for s, n in zip(scale_np, norm, strict=True)])
    scale_np[scale_np == 0] = 1.0
    rzs = affine[:-1, :-1]
    zs = np.linalg.cholesky(rzs.T @ rzs).T
    rotation = rzs @ np.linalg.inv(zs)
    s = np.sign(np.diag(zs)) * np.abs(scale_np)
    new_affine = np.eye(len(affine))
    new_affine[:-1, :-1] = rotation @ np.diag(s)
    return new_affine


def _compute_shape_offset(
    spatial_shape: tuple[int, ...], in_affine: DoubleArray, out_affine: DoubleArray
) -> tuple[HostArray, DoubleArray]:
    """``compute_shape_offset(scale_extent=False)``."""
    shape = np.array(tuple(spatial_shape), copy=True, dtype=float)
    in_coords = [(0.0, dim - 1.0) for dim in shape]
    corners = cast("DoubleArray", np.asarray(np.meshgrid(*in_coords, indexing="ij"))).reshape(
        (len(shape), -1)
    )
    corners = np.concatenate((corners, np.ones_like(corners[:1])))
    try:
        corners_out = np.linalg.solve(out_affine, in_affine) @ corners
    except np.linalg.LinAlgError as exc:
        raise InvalidInputError(f"Affine {out_affine} is not invertible") from exc
    corners = in_affine @ corners
    all_dist = corners_out[:-1].copy()
    corners_out = corners_out[:-1] / corners_out[-1]
    out_shape = np.round(cast("DoubleArray", np.ptp(corners_out, axis=1)) + 1.0)
    offset = None
    for i in range(corners.shape[1]):
        min_corner = np.min(all_dist - all_dist[:, i : i + 1], 1)
        if np.allclose(min_corner, 0.0, rtol=AFFINE_TOL):
            offset = corners[:-1, i]
            break
    if offset is None:
        offset = (
            in_affine[:-1, :-1] @ (shape / 2.0)
            + in_affine[:-1, -1]
            - out_affine[:-1, :-1] @ (out_shape / 2.0)
        )
    return out_shape.astype(int, copy=False), offset


def _swappable(header: NiftiHeader) -> bool:
    """Return whether the NIfTI header can be converted to little endian."""
    try:
        header.as_byteswapped("<")
    except ValueError:
        return False
    return True


def _correct_nifti_header_if_necessary(image: NiftiImage) -> NiftiImage:
    """MONAI's reader fix-up: make pixdim agree with the affine, preferring a matching form."""
    if image.header.get("dim") is None:
        return image
    dim = image.header["dim"][0]
    if dim >= 5:
        return image
    pixdim = np.asarray(image.header.get_zooms())[:dim]
    if np.allclose(pixdim, _affine_to_spacing(image.affine, dim)):
        return image
    return _rectify_header_sform_qform(image)


def _rectify_header_sform_qform(image: NiftiImage) -> NiftiImage:
    # Keep the sform (then qform) if its voxel sizes match pixdim; else copy the other form
    # over it when that one matches; otherwise reset pixdim from the affine.
    """Make header forms and spacing consistent, preferring matching forms."""
    d = image.header["dim"][0]
    pixdim = np.asarray(image.header.get_zooms())[:d]
    sform, qform = image.get_sform(), image.get_qform()
    sform_mismatch = not np.allclose(_affine_to_spacing(sform, d), pixdim)
    qform_mismatch = not np.allclose(_affine_to_spacing(qform, d), pixdim)
    if image.header["sform_code"] != 0:
        if not sform_mismatch:
            return image
        if not qform_mismatch:
            image.set_sform(image.get_qform())
            return image
    if image.header["qform_code"] != 0:
        if not qform_mismatch:
            return image
        if not sform_mismatch:
            image.set_qform(image.get_sform())
            return image
    image.header.set_zooms(_affine_to_spacing(image.affine, d))
    return image
