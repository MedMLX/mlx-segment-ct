"""Load recorded upstream CPU outputs without importing the reference framework."""

from __future__ import annotations

import json
import platform
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from mlx_segment_ct._types import ArrayValue, NiftiImage, NiftiRuntime, NumericInput
    from scripts.oracle_types import FixtureArrays, FixtureMetadata, HostArray, StringArray

import nibabel as nib
import numpy as np
from scipy import ndimage

FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE_KEY = f"{platform.system().lower()}-{platform.machine()}-cpu"


def load_fixture(name: str) -> tuple[FixtureArrays, FixtureMetadata]:
    """Load recorded arrays and metadata for the current reference platform."""
    path = FIXTURES / REFERENCE_KEY / f"{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing upstream oracle {path}; regenerate as described in README"
        )
    with np.load(path, allow_pickle=False) as archive:
        arrays = cast("FixtureArrays", dict(archive))
    info = cast(
        "FixtureMetadata", json.loads(str(cast("dict[str, StringArray]", arrays).pop("metadata")))
    )
    assert info["schema_version"] == 1
    assert info["monai_version"] == "1.6.0"
    assert info["bundle_revision"] == "f9f5f51b589e5dc9c23c453cf5138398e4084056"
    assert info["reference_backend"] == REFERENCE_KEY
    if "output_parts" in info:
        arrays["output"] = cast(
            "HostArray",
            np.concatenate(
                [load_fixture(f"{name}_{i}")[0]["output"] for i in range(info["output_parts"])],
                axis=1,
            ),
        )
    return arrays, info


def assert_bitwise(actual: ArrayValue, expected: HostArray, name: str) -> None:
    """Compare host values with the reference at the bit level."""
    host = np.asarray(actual)
    assert host.dtype == expected.dtype, f"{name}: dtype changed"
    assert np.array_equal(host, expected, equal_nan=True), f"{name}: upstream output differs"


class HostArrays:
    """An injected NumPy array boundary for host orchestration tests."""

    @staticmethod
    def array(value: NumericInput, dtype: type[np.float32] | None = None) -> HostArray:
        """Convert fixture values to a host array."""
        return cast("HostArray", np.asarray(value, dtype=dtype))

    float32 = np.float32
    int32 = np.int32
    sum = staticmethod(np.sum)
    abs = staticmethod(np.abs)
    arange = staticmethod(np.arange)

    @staticmethod
    def default_device() -> str:
        """Identify the recorded host array boundary."""
        return "recorded CPU array boundary"

    @staticmethod
    def eval(*args: object) -> None:
        """Accept evaluation requests for already materialized host arrays."""
        pass

    @staticmethod
    def synchronize() -> None:
        """Accept synchronization requests for synchronous host arrays."""
        pass

    @staticmethod
    def reset_peak_memory() -> None:
        """Accept peak-memory reset requests on the host stand-in."""
        pass

    @staticmethod
    def get_peak_memory() -> int:
        """Report no tracked native memory for the host stand-in."""
        return 0


def load_nifti(path: str | Path) -> NiftiImage:
    """Load and type-check a recorded NIfTI image."""
    image = cast("NiftiRuntime", nib).load(str(path))
    assert isinstance(image, nib.Nifti1Image | nib.Nifti2Image)
    return cast("NiftiImage", image)


def empty_weights(*args: object) -> dict[str, HostArray]:
    """Return an empty parameter mapping for a synthetic model."""
    return {}


NEAR_TIE = 1e-4


def dense_restoration(
    scores: HostArray,
    preprocessed_affine: HostArray,
    source_affine: HostArray,
    source_shape: tuple[int, ...],
    class_ids: list[int],
) -> tuple[HostArray, HostArray, HostArray]:
    """Score restoration by SciPy trilinear interpolation of the whole score field.

    Each source voxel maps through the two affines to the preprocessed grid. Scores
    are interpolated over finite neighbors only (normalized by their weight), the
    first maximum class ID wins when positive, an uncovered nearest neighbor gives
    NaN and a nearest neighbor outside the grid gives background. Returns the
    labels, the voxels whose position needs no clamping, and each decision's
    margin (to the runner-up score and to zero; zero on a half-voxel tie).
    """
    index = np.indices(source_shape).reshape(3, -1).astype(np.float64)
    transform = np.linalg.solve(preprocessed_affine, source_affine)
    position = transform[:3, :3] @ index + transform[:3, 3:]
    size = np.asarray(scores.shape[1:])[:, None]
    nearest = np.rint(position).astype(np.int64)
    inside = ((nearest >= 0) & (nearest < size)).all(axis=0)
    interior = ((position >= 0) & (position <= size - 1)).all(axis=0)
    finite = np.isfinite(scores).all(axis=0)
    weight = ndimage.map_coordinates(finite.astype(np.float64), position, order=1, mode="nearest")
    filled = np.where(finite, scores, 0).astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        values = np.stack(
            [ndimage.map_coordinates(c, position, order=1, mode="nearest") for c in filled]
        ) / weight
    clipped = tuple(np.clip(nearest[axis], 0, size[axis, 0] - 1) for axis in range(3))
    covered = finite[clipped]
    ids = np.asarray(class_ids, dtype=np.float32)
    labels = ids[np.argmax(values, axis=0)]
    labels[values.max(axis=0) <= 0] = 0
    labels = np.where(covered, labels, np.float32(np.nan))
    labels = np.where(inside, labels, np.float32(0))
    ordered = np.sort(values, axis=0)
    margin = np.abs(ordered[-1])
    if len(class_ids) > 1:
        margin = np.minimum(margin, ordered[-1] - ordered[-2])
    margin = np.where(covered & inside, margin, np.inf)
    # A position on a half-voxel tie has two nearest neighbors; rounding decides which.
    tie = (np.abs(position - np.floor(position) - 0.5) < 1e-9).any(axis=0)
    margin = np.where(tie, 0.0, margin)
    return (
        cast("HostArray", labels.reshape(source_shape)),
        cast("HostArray", interior.reshape(source_shape)),
        cast("HostArray", margin.reshape(source_shape)),
    )


def assert_restored(
    actual: HostArray, expected: tuple[HostArray, HostArray, HostArray], name: str
) -> None:
    """Equal on unclamped voxels, except decisions closer than ``NEAR_TIE``."""
    labels, interior, margin = expected
    same = (actual == labels) | (np.isnan(actual) & np.isnan(labels))
    differs = interior & ~same
    assert not (differs & (margin >= NEAR_TIE)).any(), f"{name}: restored labels differ"
    assert interior.mean() > 0.5, f"{name}: too few unclamped voxels compared"
