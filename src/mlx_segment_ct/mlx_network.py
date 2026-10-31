"""FP32 VISTA3D automatic encoder and class head in MLX, with NCDHW tensors."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import prod
from typing import TYPE_CHECKING, Literal, cast

from mlx_segment_ct._types import ArrayValue, CoreRuntime

if TYPE_CHECKING:
    from mlx.core import array

    from mlx_segment_ct._types import ArrayBoundary

from medmlx_core.precision import Float32Operators

from mlx_segment_ct.activations import Gelu
from mlx_segment_ct.instance_norm import InstanceNorm
from mlx_segment_ct.mlx_upsample import AutomaticUpsample
from mlx_segment_ct.projections import Projections

BLOCKS_DOWN = (1, 2, 2, 4, 4)
FEATURES = 48


class FeatureModel:
    """Shared SegResNetDS2 encoder with one explicitly selected decoder branch.

    Tensors are float32 NCDHW. ``branch`` is ``up_layers_auto`` for the automatic
    decoder or ``up_layers`` for the point decoder.
    """

    def __init__(self, weights: Mapping[str, array], *, mx: CoreRuntime, branch: str):
        """Initialize the native shared feature graph and its selected branch."""
        self.weights = weights
        self.mx = mx
        self.branch = branch
        self.upsamples = {
            stage: AutomaticUpsample(
                weights[f"image_encoder.{branch}.{stage}.upsample.deconv.weight"], mx=mx
            )
            for stage in range(4)
        }
        self.operators = Float32Operators(mx, 1, 1e-5)
        self.instance_norm = InstanceNorm(mx)
        self.gelu = Gelu(mx)
        self.projections = Projections(mx)

    def validate_image(self, image: ArrayValue) -> None:
        """Raise ``ValueError`` unless ``image`` is one float32 ``(1, 1, D, H, W)`` volume.

        Every spatial size must be divisible by 16 (four stride-2 stages), and the
        deepest level must hold more than one voxel because InstanceNorm needs it.
        """
        if (
            image.ndim != 5
            or image.shape[:2] != (1, 1)
            or image.dtype != self.mx.float32
            or any(size % 16 for size in image.shape[2:])
        ):
            raise ValueError("NV MLX requires one FP32 NCDHW image divisible by 16")
        if prod(size // 16 for size in image.shape[2:]) <= 1:
            raise ValueError("VISTA3D InstanceNorm requires more than one deepest spatial voxel")

    def encode(self, image: array) -> array:
        """Run encoder and selected decoder; returns ``(1, 48, D, H, W)`` features."""
        self.validate_image(image)
        x = self.conv(image, "image_encoder.encoder.conv_init")
        skips: list[array] = []
        for stage, count in enumerate(BLOCKS_DOWN):
            prefix = f"image_encoder.encoder.layers.{stage}"
            for block in range(count):
                x = self.residual(x, f"{prefix}.blocks.{block}")
                # Start evaluating each block so the lazy graph stays short.
                self.mx.async_eval(x)
            if stage < 4:
                skips.append(x)
                x = self.conv(x, f"{prefix}.downsample", stride=2)
        for stage in range(4):
            prefix = f"image_encoder.{self.branch}.{stage}"
            x = self.upsamples[stage](x)
            x = x + skips.pop()
            x = self.residual(x, f"{prefix}.blocks.0")
            self.mx.async_eval(x)
        prefix = f"image_encoder.{self.branch}.3.head"
        return self.projections(
            x, self.weights[f"{prefix}.weight"], self.weights[f"{prefix}.bias"], linear=False
        )

    def residual(self, x: array, prefix: str) -> array:
        """Pre-activation residual block: (InstanceNorm, ReLU, conv) twice, added to ``x``."""
        y = self.conv(self.norm(x, f"{prefix}.norm1", activation="relu"), f"{prefix}.conv1")
        y = self.conv(self.norm(y, f"{prefix}.norm2", activation="relu"), f"{prefix}.conv2")
        return x + y

    def image_mapping(self, x: array, prefix: str) -> array:
        """Two convs, each followed by a parameter-free InstanceNorm and leaky ReLU; ``x`` is
        added after the second norm and before its activation."""
        y = self.norm(self.conv(x, f"{prefix}.conv1.conv"), None, activation="leaky_relu")
        return self.norm(
            self.conv(y, f"{prefix}.conv2.conv"), None, activation="leaky_relu", residual=x
        )

    def norm(
        self,
        x: array,
        prefix: str | None,
        *,
        activation: Literal["none", "relu", "leaky_relu"] = "none",
        residual: array | None = None,
    ) -> array:
        """Normalize NCDHW values and fuse the caller's ordered affine epilogue."""
        channels = x.shape[1]
        weight = self.weights[f"{prefix}.weight"] if prefix else self.mx.ones((channels,))
        bias = self.weights[f"{prefix}.bias"] if prefix else self.mx.zeros((channels,))
        return self.instance_norm(x, weight, bias, activation=activation, residual=residual)

    def conv(self, x: array, prefix: str, *, padding: int = 1, stride: int = 1) -> array:
        """Apply a named convolution using the selected native operators."""
        return self.operators.conv(
            x,
            self.weights[f"{prefix}.weight"],
            self.weights.get(f"{prefix}.bias"),
            padding=padding,
            stride=stride,
        )

    def linear(self, x: array, prefix: str) -> array:
        """Apply a named dense layer and its bias."""
        return self.mx.addmm(self.weights[f"{prefix}.bias"], x, self.weights[f"{prefix}.weight"].T)


class AutomaticModel(FeatureModel):
    """The pinned SegResNetDS2 auto branch and learned class head.

    ``class_ids`` are global label IDs whose class embeddings are evaluated once at
    construction and define the output channel order.
    """

    def __init__(self, weights: Mapping[str, array], class_ids: Sequence[int], *, mx: CoreRuntime):
        """Initialize the automatic segmentation head for selected class IDs."""
        super().__init__(weights, mx=mx, branch="up_layers_auto")
        self.embedding = self.embed_classes(class_ids)
        mx.eval(self.embedding)

    def embed_classes(self, class_ids: Sequence[int]) -> array:
        """Evaluate the learned class-prompt path, including normalization and GELU."""
        embedding = self.weights["class_head.class_embeddings.weight"][
            cast("ArrayBoundary[array]", self.mx).array(class_ids)
        ]
        embedding = self.linear(embedding, "class_head.mlp.0")
        embedding = self.mx.reshape(
            self.norm(embedding[:, None, :, None, None], None), (-1, FEATURES)
        )
        embedding = self.gelu(embedding)
        return self.linear(embedding, "class_head.mlp.3")

    def linear(self, x: array, prefix: str) -> array:
        """Apply the automatic head projection for a named layer."""
        return self.projections(
            x, self.weights[f"{prefix}.weight"], self.weights[f"{prefix}.bias"], linear=True
        )

    def __call__(self, image: array, class_ids: Sequence[int] | None = None) -> array:
        """Return logits; explicit class IDs recompute prompts on this invocation.

        Omitting IDs uses the fixed prompts evaluated at construction. Comparisons
        with a runtime that accepts prompts per call must pass them here as well.

        ``image`` is float32 ``(1, 1, D, H, W)``. Class IDs index the 512-row class
        embedding table. Returns raw (pre-sigmoid) logits ``(1, len(class_ids), D, H, W)``,
        one channel per class in prompt order.
        """
        self.validate_image(image)
        features = self.encode(image)
        for index in range(2):
            features = self.image_mapping(features, f"class_head.image_post_mapping.{index}.layer")
            self.mx.async_eval(features)
        embedding = self.embedding if class_ids is None else self.embed_classes(class_ids)
        logits = embedding @ self.mx.reshape(features, (1, FEATURES, -1))
        return self.mx.reshape(logits, (1, embedding.shape[0], *image.shape[2:]))


def required_shapes() -> dict[str, tuple[int, ...]]:
    """Checkpoint keys for exactly the automatic graph, in source tensor layouts.

    Convolution weights are Torch ``(out, in, kD, kH, kW)``; transposed-convolution
    weights are Torch ``(in, out, kD, kH, kW)``.
    """
    shapes: dict[str, tuple[int, ...]] = {
        "image_encoder.encoder.conv_init.weight": (FEATURES, 1, 3, 3, 3)
    }
    for stage, count in enumerate(BLOCKS_DOWN):
        channels = FEATURES * 2**stage
        prefix = f"image_encoder.encoder.layers.{stage}"
        for block in range(count):
            shapes.update(_residual_shapes(f"{prefix}.blocks.{block}", channels))
        if stage < 4:
            shapes[f"{prefix}.downsample.weight"] = (channels * 2, channels, 3, 3, 3)
    for stage in range(4):
        channels = FEATURES * 2 ** (3 - stage)
        prefix = f"image_encoder.up_layers_auto.{stage}"
        shapes[f"{prefix}.upsample.deconv.weight"] = (channels * 2, channels, 3, 3, 3)
        shapes.update(_residual_shapes(f"{prefix}.blocks.0", channels))
    shapes["image_encoder.up_layers_auto.3.head.weight"] = (FEATURES, FEATURES, 1, 1, 1)
    shapes["image_encoder.up_layers_auto.3.head.bias"] = (FEATURES,)
    for index in range(2):
        for conv in (1, 2):
            shapes[f"class_head.image_post_mapping.{index}.layer.conv{conv}.conv.weight"] = (
                FEATURES,
                FEATURES,
                3,
                3,
                3,
            )
    shapes["class_head.class_embeddings.weight"] = (512, FEATURES)
    for index in (0, 3):
        shapes[f"class_head.mlp.{index}.weight"] = (FEATURES, FEATURES)
        shapes[f"class_head.mlp.{index}.bias"] = (FEATURES,)
    return shapes


def _residual_shapes(prefix: str, channels: int) -> dict[str, tuple[int, ...]]:
    """Return the required tensor shapes for one residual block."""
    return {
        f"{prefix}.{name}.{kind}": shape
        for index in (1, 2)
        for name, kind, shape in (
            (f"conv{index}", "weight", (channels, channels, 3, 3, 3)),
            (f"norm{index}", "weight", (channels,)),
            (f"norm{index}", "bias", (channels,)),
        )
    }
