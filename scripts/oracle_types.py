"""Recorder-owned host schemas and consumed pinned Torch/MONAI interfaces.

These declarations are independent of mlx_segment_ct at runtime. Native Torch
Tensor identities are retained; numerical algorithms, seeds and receipts belong
exclusively to the upstream producers. Interfaces are deliberately partial.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, NotRequired, Protocol, Required, TypedDict

import numpy as np
from numpy.typing import DTypeLike, NDArray

if TYPE_CHECKING:
    from torch import Generator, Tensor, dtype
    from typing_extensions import TypedDict as ExtTypedDict

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
StringArray = NDArray[np.str_]
type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
type FixtureArrays = dict[str, HostArray | StringArray]
type NumericFixtures = dict[str, HostArray]
type ShapeMap = dict[str, tuple[int, ...]]
type PayloadValue = (
    HostArray
    | str
    | int
    | float
    | bool
    | Sequence[int]
    | Sequence[float]
    | Sequence[str]
    | Sequence[Sequence[int]]
)

if TYPE_CHECKING:

    class NPZPayload(ExtTypedDict, extra_items=PayloadValue):
        """Type named payload arrays passed to ``numpy.savez_compressed``."""

        allow_pickle: NotRequired[bool]


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


class WindowMetadata(TypedDict):
    """Describe the window geometry of one recorded interactive case."""

    target: int | None
    roi: list[int]
    point_start: NotRequired[int]


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


class RecordingMetadata(TypedDict, total=False):
    """Describe optional metadata that a recorder can attach to a fixture."""

    weight_shapes: ShapeMap | None
    output_parts: int
    weight_seed: int
    cases: dict[str, CaseMetadata]
    requests: list[list[int] | list[str]]
    sizes: list[int]
    encoder: str


class SaveRecording(Protocol):
    """Publish recorded arrays and metadata as one reference fixture."""

    def __call__(
        self, name: str, arrays: Mapping[str, PayloadValue], **metadata: object
    ) -> None:
        """Write a reference recording using the supplied named values."""
        ...


class SourcePointHead(Protocol):
    """Expose the pinned MONAI point head used by point recordings."""

    @property
    def transformer(self) -> SourceTransformer:
        """Return the source point-head transformer."""
        ...
    @property
    def pe_layer(self) -> SourcePositionEncoding:
        """Return the source positional encoder."""
        ...
    def register_forward_pre_hook(
        self, hook: Callable[[object, tuple[Tensor, ...]], None]
    ) -> RemovedHook:
        """Register a callback before point-head inference."""
        ...
    def register_forward_hook(
        self, hook: Callable[[object, tuple[Tensor, ...], Tensor], None]
    ) -> RemovedHook:
        """Register a callback after point-head inference."""
        ...
    def __call__(
        self,
        out: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        class_vector: Tensor | None = None,
    ) -> Tensor:
        """Run the source point-conditioned head."""
        ...
    def eval(self) -> SourcePointHead:
        """Set the point head to evaluation mode."""
        ...
    def state_dict(self) -> dict[str, Tensor]:
        """Return the point-head parameter mapping."""
        ...
    def load_state_dict(
        self, state: Mapping[str, Tensor], strict: bool = True, assign: bool = False
    ) -> object:
        """Load a point-head parameter mapping."""
        ...


class SourceNetwork(Protocol):
    """Expose the pinned MONAI VISTA3D network used by full-model recordings."""

    @property
    def image_encoder(self) -> SourceFeatures:
        """Return the source image encoder."""
        ...
    @property
    def point_head(self) -> SourcePointHead:
        """Return the source point head."""
        ...
    @property
    def class_head(self) -> SourceClassHead:
        """Return the source class head."""
        ...
    def __call__(
        self,
        image: Tensor,
        *,
        point_coords: Tensor | None = None,
        point_labels: Tensor | None = None,
        class_vector: Tensor | None = None,
        prompt_class: Tensor | None = None,
        patch_coords: Sequence[Sequence[slice]] | None = None,
        prev_mask: Tensor | None = None,
        transpose: bool = False,
        **kwargs: object,
    ) -> Tensor:
        """Run the reference segmentation network."""
        ...
    def eval(self) -> SourceNetwork:
        """Set the source network to evaluation mode."""
        ...
    def state_dict(self) -> dict[str, Tensor]:
        """Return source network parameters."""
        ...
    def load_state_dict(
        self, state: Mapping[str, Tensor], strict: bool = True, assign: bool = False
    ) -> object:
        """Load source network parameters."""
        ...


class SourceFeatures(Protocol):
    """Expose the image encoder used by the reference network."""

    def __call__(
        self, image: Tensor, with_point: bool, with_label: bool
    ) -> tuple[Tensor | None, Tensor | None]:
        """Compute the requested source image features."""
        ...
    def eval(self) -> SourceFeatures:
        """Set the feature model to evaluation mode."""
        ...
    def state_dict(self) -> dict[str, Tensor]:
        """Return source feature parameters."""
        ...
    def load_state_dict(
        self, state: Mapping[str, Tensor], strict: bool = True, assign: bool = False
    ) -> object:
        """Load source feature parameters."""
        ...


class SourceAutomatic(Protocol):
    """Predict automatic class scores for one image patch."""

    def __call__(self, image: Tensor) -> Tensor:
        """Compute automatic segmentation scores for an image."""
        ...


class SourceWindowPredictor(Protocol):
    """Predict one window patch at its image coordinates."""

    def __call__(self, image: Tensor, coordinates: Sequence[Sequence[slice]], /) -> Tensor:
        """Predict a reference window at its image coordinates."""
        ...


class TransformData(TypedDict, total=False):
    """Describe the sample mapping exchanged between bundle transforms.

    The image entry is required; prediction and prompt entries appear as
    transforms add them.
    """

    image: Required[MetaTensorValue]
    pred: Tensor
    label_prompt: Tensor | list[int] | list[str] | None
    points: Tensor
    point_labels: Tensor | list[list[int]] | None


class SourceTransform(Protocol):
    """Apply one pinned bundle transform to a sample mapping."""

    def __call__(self, data: TransformData) -> TransformData:
        """Apply a reference transform to a data mapping."""
        ...


class MetaTensorValue(Protocol):
    """Expose the MONAI metadata-tensor surface used by fixture recording."""

    @property
    def affine(self) -> Tensor:
        """Return the image affine carried with the tensor."""
        ...
    @property
    def meta(self) -> ImageMetadata:
        """Return the tensor image metadata."""
        ...
    @property
    def applied_operations(self) -> list[dict[str, object]]:
        """Return recorded transform operations."""
        ...
    @property
    def shape(self) -> Sequence[int]:
        """Return the tensor dimensions."""
        ...
    def numpy(self) -> HostArray:
        """Convert the tensor to a host array."""
        ...
    def __getitem__(self, index: None) -> Tensor:
        """Select a tensor view by index."""
        ...


class FunctionalRuntime(Protocol):
    """Expose the Torch functional operations used by reference recording."""

    def conv_transpose3d(
        self,
        input: Tensor,
        weight: Tensor,
        bias: Tensor | None = None,
        stride: int = 1,
        padding: int = 0,
        output_padding: int = 0,
    ) -> Tensor:
        """Apply a 3D transposed convolution."""
        ...
    def linear(self, input: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
        """Apply a linear transform and optional bias."""
        ...
    def gelu(self, input: Tensor, approximate: str = "none") -> Tensor:
        """Apply the GELU activation."""
        ...
    def pad(
        self, input: Tensor, pad: Sequence[int], mode: str = "constant", value: float | None = None
    ) -> Tensor:
        """Pad a tensor using the specified padding."""
        ...
    def affine_grid(
        self, theta: Tensor, size: Sequence[int], align_corners: bool = False
    ) -> Tensor:
        """Create a normalized sampling grid from an affine."""
        ...


class TorchNNRuntime(Protocol):
    """Expose the Torch neural-network namespace used by reference recording."""

    @property
    def functional(self) -> FunctionalRuntime:
        """Return functional reference operations."""
        ...


class TorchRuntime(Protocol):
    """Expose the Torch module surface consumed by reference recording."""

    @property
    def float32(self) -> dtype:
        """Return the reference float32 dtype."""
        ...
    @property
    def float64(self) -> dtype:
        """Return the reference float64 dtype."""
        ...
    @property
    def nn(self) -> TorchNNRuntime:
        """Return reference neural-network operations."""
        ...
    def manual_seed(self, seed: int) -> Generator:
        """Seed the reference random generator."""
        ...
    def from_numpy(self, ndarry: HostArray) -> Tensor:
        """Create a reference tensor from a NumPy array."""
        ...
    def tensor(
        self, data: PayloadValue | Sequence[int], *, dtype: dtype | None = None
    ) -> Tensor:
        """Create a reference tensor from supplied values."""
        ...
    def zeros(self, *size: int) -> Tensor:
        """Create a zero-filled reference tensor."""
        ...
    def eye(self, n: int, m: int | None = None, *, dtype: dtype | None = None) -> Tensor:
        """Create a reference identity matrix."""
        ...
    def stack(self, tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
        """Stack reference tensors along an axis."""
        ...


class RemovedHook(Protocol):
    """Remove a hook registered on a pinned source module."""

    def remove(self) -> None:
        """Remove the registered source-model hook."""
        ...


class SourceTransformer(Protocol):
    """Expose the point-head transformer used by reference recording."""

    def __call__(
        self, image_embedding: Tensor, image_pe: Tensor, point_embedding: Tensor
    ) -> tuple[Tensor, Tensor]:
        """Run the source transformer over image and query tensors."""
        ...
    @property
    def final_attn_token_to_image(self) -> Callable[[Tensor, Tensor, Tensor], Tensor]:
        """Return the final token-to-image attention operation."""
        ...
    def register_forward_pre_hook(
        self, hook: Callable[[object, tuple[Tensor, ...]], None]
    ) -> RemovedHook:
        """Register a callback before transformer inference."""
        ...


class SourcePositionEncoding(Protocol):
    """Expose the point-head positional encoder."""

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        """Encode spatial coordinates for source point attention."""
        ...


class RawSample(TypedDict, total=False):
    """Describe a raw bundle sample before preprocessing.

    The image entry is a filesystem path; prompt entries are optional.
    """

    image: str
    label_prompt: list[int] | list[str] | None
    point_labels: list[list[int]] | None


class Preprocessing(Protocol):
    """Apply the pinned bundle preprocessing to a raw sample."""

    def __call__(self, data: RawSample) -> TransformData:
        """Apply source preprocessing to a raw sample."""
        ...


class TensorPointInference(Protocol):
    """Run the pinned point-conditioned window inferer over an image."""

    def __call__(
        self,
        inputs: Tensor,
        roi_size: Sequence[int],
        predictor: SourceNetwork,
        point_coords: Tensor,
        point_labels: Tensor,
        class_vector: Tensor | None = None,
        prompt_class: Tensor | None = None,
        prev_mask: Tensor | None = None,
        point_start: int = 0,
        center_only: bool = True,
        margin: int = 5,
        **kwargs: object,
    ) -> Tensor:
        """Run reference point-conditioned window inference."""
        ...


class TensorWindowInference(Protocol):
    """Run the pinned sliding-window inferer over an image."""

    def __call__(
        self,
        inputs: Tensor,
        roi_size: Sequence[int],
        sw_batch_size: int,
        predictor: Callable[..., Tensor],
        *,
        overlap: float = 0.25,
        mode: str = "constant",
        padding_mode: str = "constant",
        with_coord: bool = False,
    ) -> Tensor:
        """Run reference sliding-window inference."""
        ...


class SourceClassHead(Protocol):
    """Expose the automatic class head used by reference recording."""

    def register_forward_hook(
        self, hook: Callable[[object, tuple[Tensor, ...], tuple[Tensor, None]], None]
    ) -> RemovedHook:
        """Register a callback after the source class head."""
        ...


class PredictionKeywordArguments(TypedDict):
    """Describe recorded keywords for one point-window prediction."""

    point_coords: HostArray
    point_labels: HostArray
    class_vector: HostArray | None
    prompt_class: HostArray | None
    patch_coords: Sequence[Sequence[slice]]
    prev_mask: HostArray | None
    transpose: bool


class BoxRequest(TypedDict, total=False):
    """Describe prompt overrides for interactive request validation.

    Fields stay ``object`` so malformed values reach validation unchanged.
    """

    points: object
    point_labels: object
    prompt: object
    label_prompt: object


class RawRunner(Protocol):
    """Call a reference runner with file-based arguments."""

    def __call__(self, **kwargs: object) -> RunResult:
        """Execute the reference runner with supplied arguments."""
        ...


class WeightOrder(TypedDict):
    """Describe the seeded weight ordering stored in ``weight_order.json``."""

    schema_version: int
    automatic: list[str]
    point: list[str]


if TYPE_CHECKING:

    class BundleConfig(ExtTypedDict, extra_items=JsonValue):
        """Describe the NV-Segment-CT bundle configuration staged for recording."""

        subclass: dict[str, list[int] | str]
        bundle_root: str
        device: str
        output_dir: str
        separate_folder: bool


class ConfigParserValue(Protocol):
    """Resolve named parsed content from a MONAI bundle configuration."""

    def get_parsed_content(self, id: str) -> object:
        """Resolve a named item from the bundle configuration."""
        ...


class IntensityTransform(Protocol):
    """Apply a pinned MONAI intensity transform to an image."""

    def __call__(self, img: Tensor) -> Tensor:
        """Apply the source intensity transform."""
        ...


class SourceComponentOperations(Protocol):
    """Expose the VISTA3D point-to-patch and component-fusion operations."""

    def update_point_to_patch(
        self, patch_coords: Sequence[slice], point_coords: Tensor, point_labels: Tensor
    ) -> tuple[Tensor | None, Tensor | None]:
        """Convert valid clicks to patch-relative coordinates."""
        ...
    def connected_components_combine(
        self,
        logits: Tensor,
        point_logits: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        mapping_index: Tensor,
    ) -> Tensor:
        """Combine segmentation components selected by clicks."""
        ...


class PostTransform(Protocol):
    """Apply one step of the pinned bundle postprocessing."""

    def __call__(self, data: TransformData) -> TransformData:
        """Apply source postprocessing to a data mapping."""
        ...


class PromptTransform(Protocol):
    """Apply a pinned VISTA label or point transform to a prompt mapping."""

    def __call__(self, data: PromptData) -> PromptData:
        """Apply source prompt transformations."""
        ...


class MonoWindowNamespace(Protocol):
    """Expose MONAI's sliding-window callable from the inferer module."""

    @property
    def sliding_window_inference(self) -> TensorWindowInference:
        """Return the reference sliding-window callable."""
        ...


class PromptData(TypedDict, total=False):
    """Describe the prompt mapping consumed by VISTA label and point transforms."""

    pred: Tensor
    label_prompt: Tensor | list[int] | list[str] | None
    points: Tensor
    point_labels: Tensor | list[list[int]] | None


class PostPrediction(TypedDict):
    """Describe the single-prediction mapping returned by postprocessing."""

    pred: Tensor


class ParsedCodes(TypedDict):
    """Describe the point codes produced by a VISTA prompt transform."""

    point_labels: list[list[int]]


class ParsedLabels(TypedDict):
    """Describe the class labels produced by a VISTA prompt transform."""

    label_prompt: list[int]


class ResetOperations(Protocol):
    """Reset recorded transform operations on a metadata tensor."""

    def __call__(self, data: MetaTensorValue) -> MetaTensorValue:
        """Reset transform operation identifiers on a metadata tensor."""
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


class SourcePredictionKeywords(TypedDict, total=False):
    """Describe the prediction keywords forwarded to a source window predictor.

    ``patch_coords`` is required; the inferer supplies the remaining keys.
    """

    point_coords: Tensor
    point_labels: Tensor
    class_vector: Tensor | None
    prompt_class: Tensor | None
    patch_coords: Required[Sequence[Sequence[slice]]]
    prev_mask: Tensor | None
    transpose: bool
    sw_batch_size: int
    with_coord: bool
    mode: str
    overlap: float


class ResetNamespace(Protocol):
    """Expose the module-level transform-operation reset callable."""

    @property
    def reset_ops_id(self) -> ResetOperations:
        """Return the transform-operation reset callable."""
        ...


if TYPE_CHECKING:

    class ImageMetadata(ExtTypedDict, extra_items=object):
        """Describe the metadata-tensor fields recorded with a source image."""

        original_affine: HostArray


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


class RequestInput(TypedDict):
    """Describe the keyword arguments accepted by the interactive request resolver."""

    prompt: object
    label_prompt: object
    points: object
    point_labels: object


class FileRunPaths(TypedDict):
    """Describe the file paths of a runner invocation."""

    image_path: Path
    weights_path: Path
    output_dir: Path


class FileImageOutputPaths(TypedDict):
    """Describe the image path and output directory of a file run without weights."""

    image_path: Path
    output_dir: Path
