"""Consumed native MLX and host-orchestration contracts.

The native graph uses real MLX array/Dtype identities. The orchestration boundary
is generic because the existing CPU oracle tests inject NumPy arrays there.
These partial interfaces describe the MLX and medmlx-core versions pinned in uv.lock;
they do not replace either runtime or claim complete upstream API coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from os import PathLike
from typing import TYPE_CHECKING, Literal, NotRequired, Protocol, Required, TypedDict

import numpy as np
from medmlx_core.typing import MlxRuntime
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
    from mlx.core import Dtype, array

type NumericScalar = (
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
)
HostArray = NDArray[NumericScalar]
FloatArray = NDArray[np.float32]
DoubleArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
BoolArray = NDArray[np.bool_]
type ArrayValue = array | HostArray
type NumericInput = bool | int | float | ArrayValue | Sequence[NumericInput]
type ArrayTree = array | Sequence[ArrayTree] | Mapping[str, ArrayTree]
type PatchCoordinates = Sequence[Sequence[slice]]


class ArrayBoundary[ArrayT](Protocol):
    @property
    def array(self) -> Callable[..., ArrayT]:
        """Return the array constructor for the injected runtime."""
        ...
    def eval(self, *args: ArrayT | Sequence[ArrayT] | Mapping[str, ArrayT]) -> None:
        """Evaluate arrays and collections on the injected runtime."""
        ...
    def synchronize(self) -> None:
        """Wait for outstanding work on the injected runtime."""
        ...


class CoreRuntime(MlxRuntime, Protocol):
    def eval(self, *args: array | Mapping[str, array] | Sequence[array]) -> None:
        """Evaluate native arrays and their containers."""
        ...
    def synchronize(self) -> None:
        """Wait for outstanding native work to complete."""
        ...
    def async_eval(self, *args: ArrayTree) -> None:
        """Schedule evaluation of native array trees."""
        ...
    def ones(self, shape: Sequence[int], dtype: Dtype | None = None) -> array:
        """Create a native array filled with ones."""
        ...
    def sin(self, a: array) -> array:
        """Compute elementwise sine on a native array."""
        ...
    def cos(self, a: array) -> array:
        """Compute elementwise cosine on a native array."""
        ...
    def abs(self, a: array) -> array:
        """Compute elementwise absolute values."""
        ...
    def sum(
        self, a: array, axis: int | Sequence[int] | None = None, keepdims: bool = False
    ) -> array:
        """Sum array elements over selected axes."""
        ...
    def softmax(
        self, a: array, axis: int | Sequence[int] | None = None, *, precise: bool = False
    ) -> array:
        """Normalize scores along the selected axis."""
        ...
    def addmm(
        self, c: array, a: array, b: array, alpha: float = 1.0, beta: float = 1.0
    ) -> array:
        """Multiply matrices and add the supplied bias."""
        ...
    def meshgrid(
        self, *arrays: array, indexing: Literal["xy", "ij"] = "xy", sparse: bool = False
    ) -> tuple[array, ...]:
        """Create coordinate grids from native axes."""
        ...
    def contiguous(self, a: array, /, *, allow_col_major: bool = False) -> array:
        """Return a contiguous array in an accepted layout."""
        ...
    def load(self, file: str, /, format: str | None = None) -> array | dict[str, array]:
        """Load an array or named tensor mapping from a file."""
        ...
    def save_safetensors(
        self, file: str, arrays: Mapping[str, array], metadata: Mapping[str, str] | None = None
    ) -> None:
        """Write named native arrays to a safetensors file."""
        ...


class PointPredictor[ArrayT](Protocol):
    def __call__(self, image: ArrayT, positions: ArrayT, codes: ArrayT, target: int) -> ArrayT:
        """Run point-conditioned prediction for an image and target class."""
        ...


class CoordinatePredictor[ArrayT](Protocol):
    def __call__(self, image: ArrayT, coordinates: PatchCoordinates, /) -> ArrayT:
        """Predict a window using its coordinates in the full image."""
        ...


class PointOptions(TypedDict, total=False):
    transpose: Required[bool]
    sw_batch_size: int
    with_coord: bool
    mode: str
    sw_device: object
    device: object
    overlap: float


class PointWindowCall[ArrayT](Protocol):
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
        """Run point-conditioned window inference and return host scores."""
        ...


class Metrics(TypedDict):
    patch_seconds: list[float]
    padded_shape: NotRequired[list[int]]
    padding: NotRequired[list[int]]
    streaming_axis: NotRequired[int]
    ring_depth: NotRequired[int]
    predicted_accumulator_bytes: NotRequired[int]
    count_buffer_bytes: NotRequired[int]
    label_buffer_bytes: NotRequired[int]
    accumulator_dtype: NotRequired[str]


class OutputLabel(TypedDict):
    value: int
    name: str | None


class InteractiveParameters(TypedDict):
    roi_size: list[int]
    center_only: bool
    padding_mode: str
    overlap_reduction: str
    threshold: int
    uncovered_label: int
    positive_components: bool


class Details(TypedDict, total=False):
    weights_revision: str | None
    mode: Required[str]
    label_prompt: list[int]
    label_names: list[str]
    automatic_branch: bool
    point_start: int
    output_labels: Required[list[OutputLabel]]
    points: list[list[float]]
    point_labels: list[int]
    effective_point_labels: list[int]
    preprocessed_points: list[list[float]]
    preprocessed_affine: list[list[float]]
    preprocessed_shape: Required[list[int]]
    output_shape: Required[list[int]]
    output_dtype: Required[str]
    algorithm: Required[str]
    parameters: InteractiveParameters
    patch_count: Required[int]
    load_preprocess_seconds: Required[float]
    inference_stitch_seconds: Required[float]
    restore_save_seconds: Required[float]
    class_ids: list[int]
    roi_size: list[int]
    overlap: float
    blend_mode: str
    padding_mode: str
    padded_shape: list[int]
    modality: str
    checkpoint: str
    restoration: str
    backend: str
    mlx_version: str
    mlx_default_device: str
    precision: str
    source_revision: str
    mlx_peak_memory_bytes: int | None
    elapsed_seconds: float
    warning: str


class PublishedDetails(TypedDict, total=False):
    weights_revision: str | None
    mode: Required[str]
    label_prompt: list[int]
    label_names: list[str]
    automatic_branch: bool
    point_start: int
    output_labels: Required[list[OutputLabel]]
    points: list[list[float]]
    point_labels: list[int]
    effective_point_labels: list[int]
    preprocessed_points: list[list[float]]
    preprocessed_affine: list[list[float]]
    preprocessed_shape: Required[list[int]]
    output_shape: Required[list[int]]
    output_dtype: Required[str]
    algorithm: Required[str]
    parameters: InteractiveParameters
    patch_count: Required[int]
    load_preprocess_seconds: Required[float]
    inference_stitch_seconds: Required[float]
    restore_save_seconds: Required[float]
    class_ids: list[int]
    roi_size: list[int]
    overlap: float
    blend_mode: str
    padding_mode: str
    padded_shape: list[int]
    modality: str
    checkpoint: Required[str]
    restoration: Required[str]
    backend: Required[str]
    mlx_version: Required[str]
    mlx_default_device: Required[str]
    precision: Required[str]
    source_revision: Required[str]
    mlx_peak_memory_bytes: Required[int | None]
    elapsed_seconds: Required[float]
    warning: Required[str]


class Outputs(TypedDict):
    mask: str


class RunResult(TypedDict):
    outputs: Outputs
    details: PublishedDetails


class RunArguments(TypedDict, total=False):
    image_path: str | PathLike[str]
    weights_path: str | PathLike[str]
    output_dir: str | PathLike[str]
    segment_all: bool
    prompt: str | Sequence[str] | None
    label_prompt: Sequence[int] | None
    previous_logits_path: str | PathLike[str] | None
    points: Sequence[Sequence[float]] | None
    point_labels: Sequence[int] | None
    point_start: int
    interactive: bool
    overwrite: bool
    checkpoint: str
    modality: str | None
    restoration: str


class NiftiHeader(Protocol):
    @property
    def binaryblock(self) -> bytes:
        """Return the binary image header."""
        ...
    def __getitem__(self, key: str) -> HostArray:
        """Read a named image header field."""
        ...
    def get(self, key: str) -> HostArray | None:
        """Return a named header field if present."""
        ...
    def get_zooms(self) -> tuple[float, ...]:
        """Return voxel spacing from the image header."""
        ...
    def set_zooms(self, zooms: Sequence[float] | HostArray) -> None:
        """Set voxel spacing in the image header."""
        ...
    def get_xyzt_units(self) -> tuple[str, str]:
        """Return the declared spatial and temporal units."""
        ...
    def as_byteswapped(self, endianness: str | None = None) -> NiftiHeader:
        """Return a header with the requested byte order."""
        ...


class NiftiArrayProxy(Protocol):
    """File-backed NIfTI data supports conversion without ndarray methods."""

    def __array__(self, dtype: DTypeLike | None = None) -> NDArray[np.generic]:
        """Convert the lazy image data proxy to a host array."""
        ...


class NiftiImage(Protocol):
    @property
    def shape(self) -> tuple[int, ...]:
        """Return the image dimensions."""
        ...
    def get_data_dtype(self) -> np.dtype[NumericScalar]:
        """Return the stored image data dtype."""
        ...
    @property
    def header(self) -> NiftiHeader:
        """Return the image header."""
        ...
    @property
    def affine(self) -> DoubleArray:
        """Return the voxel-to-world affine."""
        ...
    @property
    def dataobj(self) -> HostArray | NiftiArrayProxy:
        """Return the image array or lazy data proxy."""
        ...
    def get_sform(self) -> DoubleArray:
        """Return the stored sform affine."""
        ...
    def get_qform(self) -> DoubleArray:
        """Return the stored qform affine."""
        ...
    def set_sform(self, affine: HostArray, code: int | None = None) -> None:
        """Set the sform affine and optional code."""
        ...
    def set_qform(self, affine: HostArray, code: int | None = None) -> None:
        """Set the qform affine and optional code."""
        ...


class NiftiConstructor(Protocol):
    def __call__(self, dataobj: HostArray, affine: HostArray) -> NiftiImage:
        """Construct a NIfTI image from data and affine."""
        ...


class NiftiRuntime(Protocol):
    @property
    def Nifti1Image(self) -> NiftiConstructor:
        """Return the NIfTI-1 image constructor."""
        ...
    def load(self, filename: str) -> object:
        """Load an image from a path."""
        ...
    def save(self, img: NiftiImage, filename: str) -> None:
        """Write an image to a path."""
        ...
    def aff2axcodes(self, aff: HostArray) -> tuple[str | None, ...]:
        """Determine spatial axis codes from an affine."""
        ...
    def io_orientation(self, affine: HostArray) -> DoubleArray:
        """Determine voxel-axis orientation from an affine."""
        ...


class OrientationRuntime(Protocol):
    def axcodes2ornt(self, axcodes: tuple[str | None, ...], labels: None = None) -> DoubleArray:
        """Convert axis codes to an orientation transform."""
        ...
    def ornt_transform(self, start_ornt: DoubleArray, end_ornt: DoubleArray) -> DoubleArray:
        """Compute the transform between orientations."""
        ...
    def inv_ornt_aff(self, ornt: DoubleArray, shape: Sequence[int]) -> DoubleArray:
        """Return the inverse voxel affine for an orientation and shape."""
        ...


class RollingMetrics(TypedDict):
    patch_seconds: list[float]
    padded_shape: list[int]
    padding: list[int]
    streaming_axis: int
    ring_depth: int
    predicted_accumulator_bytes: int
    count_buffer_bytes: int
    label_buffer_bytes: int
    accumulator_dtype: str


class SlidingWindowCall(Protocol):
    def __call__(
        self,
        inputs: ArrayValue,
        roi_size: Sequence[int],
        sw_batch_size: int,
        predictor: Callable[..., array],
        *,
        overlap: float,
        mode: str,
        padding_mode: str,
        roi_weight_map: ArrayValue | None,
        with_coord: bool,
        **kwargs: object,
    ) -> HostArray:
        """Run overlapping window inference and return combined host scores."""
        ...


class SlidingWindowNamespace(Protocol):
    @property
    def sliding_window_inference(self) -> SlidingWindowCall:
        """Return the sliding-window inference callable."""
        ...


class StandInRuntime[ArrayT, DtypeT](ArrayBoundary[ArrayT], Protocol):
    @property
    def float32(self) -> DtypeT:
        """Return the injected runtime float32 dtype."""
        ...
    def sum(self, a: ArrayT) -> ArrayT | NumericScalar:
        """Sum the supplied array values."""
        ...


type Float32Dtype = Dtype | type[np.float32]


class Float32Cast[ArrayT, DtypeT](Protocol):
    def astype(self, dtype: DtypeT) -> ArrayT:
        """Convert the array to the requested dtype."""
        ...


class RollingMetricsBuilder(TypedDict):
    patch_seconds: NotRequired[list[float]]
    padded_shape: NotRequired[list[int]]
    padding: NotRequired[list[int]]
    streaming_axis: NotRequired[int]
    ring_depth: NotRequired[int]
    predicted_accumulator_bytes: NotRequired[int]
    count_buffer_bytes: NotRequired[int]
    label_buffer_bytes: NotRequired[int]
    accumulator_dtype: NotRequired[str]


class AutomaticDetails(TypedDict):
    class_ids: list[int]
    padded_shape: list[int]
    label_names: list[str] | None


class InteractiveDetails(TypedDict):
    automatic_branch: bool
    effective_point_labels: list[int]
    point_start: int


class NumericArrayArithmetic[ArrayT](Protocol):
    """Native and NumPy arrays accept corresponding arrays and NumPy scalars.

    MLX 0.32.3 accepts the NumPy reduction scalars used by host-injected point
    fixtures; its bundled array operator declaration omits those scalar types.
    """

    def __add__(self, other: ArrayValue | NumericScalar | float, /) -> ArrayT:
        """Add an array or scalar to the native values."""
        ...
