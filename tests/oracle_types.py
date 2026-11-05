"""Recorded fixture metadata schemas and the nibabel/runner interfaces tests consume.

These declarations are independent of mlx_segment_ct at runtime. Numerical
algorithms, seeds and receipts belong exclusively to the upstream producers.
Interfaces are deliberately partial.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, NotRequired, Protocol, TypedDict

import numpy as np
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
    from mlx_segment_ct._types import RunResult

HostArray = NDArray[
    np.bool_
    | np.int8
    | np.int16
    | np.int32
    | np.int64
    | np.uint8
    | np.uint16
    | np.uint32
    | np.uint64
    | np.float16
    | np.float32
    | np.float64
]
type NumericFixtures = dict[str, HostArray]


class CaseMetadata(TypedDict, total=False):
    """Describe one recorded reference case in a fixture's ``cases`` map."""

    target: int | None
    roi: list[int]
    point_start: int
    class_ids: list[int]
    label_prompt: list[int] | None
    axis_aligned: bool
    padded_shape: list[int]
    patch_count: int
    preprocessed_shape: list[int]


class FixtureMetadata(TypedDict):
    """Describe the metadata header written with every recorded reference fixture."""

    schema_version: int
    monai_version: str
    bundle_revision: str
    reference_backend: str
    output_parts: NotRequired[int]
    weight_shapes: NotRequired[dict[str, list[int]]]
    weight_seed: NotRequired[int]
    cases: NotRequired[dict[str, CaseMetadata]]
    requests: NotRequired[list[list[str] | list[int]]]
    sizes: NotRequired[list[int]]


class GraphMetadata(TypedDict):
    """Describe a graph recording's seeded weight shapes and optional seed."""

    weight_shapes: dict[str, list[int]]
    weight_seed: NotRequired[int]


class RunnerMetadata(TypedDict):
    """Describe one recorded interactive runner case and its point start."""

    target: int | None
    roi: list[int]
    point_start: int


class SegmentMetadata(TypedDict):
    """Describe one recorded automatic-segmentation runner case."""

    class_ids: list[int]
    label_prompt: list[int] | None
    padded_shape: list[int]
    patch_count: int
    preprocessed_shape: list[int]


class CtmrSegmentMetadata(SegmentMetadata):
    """Describe one recorded NV-Segment-CTMR segmentation case."""

    roi: list[int]


class CtmrMetadata(TypedDict):
    """Describe the recorded NV-Segment-CTMR prompt requests and modalities."""

    requests: list[list[int] | list[str]]
    modalities: list[str]


class PredictionKeywordArguments(TypedDict):
    """Describe recorded keywords for one point-window prediction."""

    point_coords: HostArray
    point_labels: HostArray
    class_vector: HostArray | None
    prompt_class: HostArray | None
    patch_coords: Sequence[Sequence[slice]]
    prev_mask: HostArray | None
    transpose: bool


class RawRunner(Protocol):
    """Call a reference runner with file-based arguments."""

    def __call__(self, **kwargs: object) -> RunResult:
        """Execute the reference runner with supplied arguments."""
        ...


class OracleNiftiHeader(Protocol):
    """Expose the NIfTI header fields used by fixture recording."""

    @property
    def binaryblock(self) -> bytes:
        """Return binary image-header bytes."""
        ...
    def set_zooms(self, zooms: Sequence[float]) -> None:
        """Set image-header voxel spacing."""
        ...
    def __getitem__(self, key: str) -> HostArray:
        """Read a named header field."""
        ...
    def get_zooms(self) -> tuple[float, ...]:
        """Return image-header voxel spacing."""
        ...


class OracleNiftiArrayProxy(Protocol):
    """File-backed NIfTI data supports conversion without ndarray methods."""

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray[np.generic]:
        """Convert lazy NIfTI data to a host array."""
        ...


class OracleNiftiImage(Protocol):
    """Expose the NIfTI image surface used by fixture recording.

    Affines map voxel indices to world coordinates in millimeters.
    """

    @property
    def affine(self) -> HostArray:
        """Return the image voxel-to-world affine."""
        ...
    @property
    def header(self) -> OracleNiftiHeader:
        """Return the NIfTI header."""
        ...
    @property
    def dataobj(self) -> HostArray | OracleNiftiArrayProxy:
        """Return the image data array or proxy."""
        ...
    def set_sform(self, affine: HostArray | None, code: int | None = None) -> None:
        """Set the image sform affine and optional code."""
        ...
    def set_qform(self, affine: HostArray, code: int | None = None) -> None:
        """Set the image qform affine and optional code."""
        ...
    def to_bytes(self) -> bytes:
        """Serialize the NIfTI image to bytes."""
        ...


class OracleNiftiConstructor(Protocol):
    """Construct a NIfTI-1 image from data and a voxel-to-world affine."""

    def __call__(self, dataobj: HostArray, affine: HostArray) -> OracleNiftiImage:
        """Construct a NIfTI image from data and affine."""
        ...


class OracleNiftiRuntime(Protocol):
    """Expose the nibabel calls used by fixture recording."""

    @property
    def Nifti1Image(self) -> OracleNiftiConstructor:
        """Return the NIfTI-1 image constructor."""
        ...
    def load(self, filename: str) -> object:
        """Load an image from a path."""
        ...


class RequestsMetadata(TypedDict):
    """Describe a recording that stores prompt requests in its metadata."""

    requests: list[list[int] | list[str]]


class GridMetadata(TypedDict):
    """Describe a recording that stores normalized grid axis sizes."""

    sizes: list[int]


class CasesMetadata(TypedDict):
    """Describe a recording whose metadata maps case names to their details."""

    cases: dict[str, CaseMetadata]


class GeometryMetadata(TypedDict):
    """Describe the class ids recorded for one geometry case."""

    class_ids: list[int]


class SeededGraphMetadata(TypedDict):
    """Describe a recording that stores seeded weight shapes and their seed."""

    weight_shapes: dict[str, list[int]]
    weight_seed: int
