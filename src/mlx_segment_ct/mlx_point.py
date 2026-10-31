"""Pinned VISTA3D PointMappingSAM and two-way decoder, entirely FP32 MLX."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mlx_segment_ct._types import CoreRuntime

if TYPE_CHECKING:
    from mlx.core import array

from mlx_segment_ct.mlx_network import FEATURES, FeatureModel, required_shapes
from mlx_segment_ct.mlx_upsample import AutomaticUpsample


class PointModel(FeatureModel):
    """One target's point branch; previous-mask combination belongs to the inferer.

    The point decoder shares the encoder with the automatic model but uses its own
    ``up_layers`` weights. Output logits are for a single object.
    """

    def __init__(self, weights: dict[str, array], *, mx: CoreRuntime):
        """Initialize the native point-conditioned correction network."""
        super().__init__(weights, mx=mx, branch="up_layers")
        self.output_upsample = AutomaticUpsample(
            weights["point_head.output_upscaling.0.weight"], mx=mx
        )

    def __call__(self, image: array, positions: array, codes: array, target: int) -> array:
        """Positions are source-normalized FP32 voxel centers, not raw coordinates.

        Args:
            image: float32 ``(1, 1, D, H, W)`` window.
            positions: ``(1, N, 3)`` clicks as ``normalized_point_positions`` returns
                them: window voxel index plus 0.5, divided by the window size.
            codes: ``(1, N)`` click codes: -1 padding, 0/1 exclude/include,
                2/3 special exclude/include.
            target: global class ID; IDs up to 132 use the supported-class embedding
                and larger IDs the zero-shot embedding.

        Returns:
            Raw logits ``(1, 1, D, H, W)``.
        """
        self.validate_image(image)
        return self.decode(self.encode(image), positions, codes, target)

    def positional_encoding(self, normalized: array) -> array:
        """Random-Fourier embedding of coordinates in (0, 1): ``(..., 3)`` to ``(..., 48)``."""
        mx = self.mx
        frequencies = (2 * normalized - 1) @ self.weights[
            "point_head.pe_layer.positional_encoding_gaussian_matrix"
        ]
        frequencies = (2 * 3.141592653589793) * frequencies
        return mx.concatenate([mx.sin(frequencies), mx.cos(frequencies)], axis=-1)

    def tokens(self, positions: array, codes: array, target: int) -> array:
        """Token sequence ``(1, N + 2, 48)``: mask token, one token per click, class token."""
        mx, weights = self.mx, self.weights
        embedding = self.positional_encoding(positions)
        embedding = mx.where(
            (codes == -1)[..., None], weights["point_head.not_a_point_embed.weight"], embedding
        )
        for code in range(4):
            addition = weights[f"point_head.point_embeddings.{code % 2}.weight"]
            if code >= 2:
                addition = addition + weights["point_head.special_class_embed.weight"]
            embedding = embedding + mx.where((codes == code)[..., None], addition, 0)
        kind = "supported_embed" if target <= 132 else "zeroshot_embed"
        return mx.concatenate(
            [
                weights["point_head.mask_tokens.weight"][None],
                embedding,
                weights[f"point_head.{kind}.weight"][None],
            ],
            axis=1,
        )

    def layer_norm(self, values: array, prefix: str) -> array:
        """LayerNorm over the last axis with the weights under ``prefix`` (eps 1e-5)."""
        return self.mx.fast.layer_norm(
            values,
            self.weights[f"{prefix}.weight"],
            self.weights[f"{prefix}.bias"],
            1e-5,
        )

    def attention(self, q: array, k: array, v: array, prefix: str) -> array:
        """Four-head attention on ``(1, tokens, 48)`` inputs; returns ``(1, len(q), 48)``."""
        mx = self.mx

        def heads(value: array, projection: str) -> array:
            """Project attention values and split them into attention heads."""
            value = self.linear(value, f"{prefix}.{projection}_proj")
            return mx.transpose(
                mx.reshape(value, (value.shape[0], value.shape[1], 4, -1)), (0, 2, 1, 3)
            )

        q, k, v = heads(q, "q"), heads(k, "k"), heads(v, "v")
        probabilities = mx.softmax(
            (q @ mx.transpose(k, (0, 1, 3, 2))) / q.shape[-1] ** 0.5, axis=-1, precise=True
        )
        if v.shape[-2] >= 4096:
            # A single long FP32 contraction accumulates observable point-mask drift.
            # Bound its accumulation depth, then reduce the partial dot products.
            batch, num_heads, queries, spatial = probabilities.shape
            tiles = spatial // 256
            partial = mx.reshape(probabilities, (batch, num_heads, queries, tiles, 256))
            values = mx.reshape(v, (batch, num_heads, tiles, 256, v.shape[-1]))
            result = mx.sum(mx.transpose(partial, (0, 1, 3, 2, 4)) @ values, axis=2)
        else:
            result = probabilities @ v
        result = mx.transpose(result, (0, 2, 1, 3))
        return self.linear(
            mx.reshape(result, (result.shape[0], result.shape[1], -1)), f"{prefix}.out_proj"
        )

    def transformer(self, keys: array, key_pe: array, tokens: array) -> tuple[array, array]:
        """Two-way transformer: ``tokens`` and image ``keys`` (with positional encoding
        ``key_pe``) attend to each other. Returns the updated ``(queries, keys)``."""
        queries = tokens
        for index in range(2):
            prefix = f"point_head.transformer.layers.{index}"
            q = queries if index == 0 else queries + tokens
            attended = self.attention(q, q, queries, f"{prefix}.self_attn")
            queries = self.layer_norm(
                attended if index == 0 else queries + attended, f"{prefix}.norm1"
            )
            attended = self.attention(
                queries + tokens, keys + key_pe, keys, f"{prefix}.cross_attn_token_to_image"
            )
            queries = self.layer_norm(queries + attended, f"{prefix}.norm2")
            mlp = self.linear(
                self.mx.maximum(self.linear(queries, f"{prefix}.mlp.linear1"), 0),
                f"{prefix}.mlp.linear2",
            )
            queries = self.layer_norm(queries + mlp, f"{prefix}.norm3")
            attended = self.attention(
                keys + key_pe, queries + tokens, queries, f"{prefix}.cross_attn_image_to_token"
            )
            keys = self.layer_norm(keys + attended, f"{prefix}.norm4")
            self.mx.async_eval(queries, keys)
        prefix = "point_head.transformer"
        attended = self.attention(
            queries + tokens, keys + key_pe, keys, f"{prefix}.final_attn_token_to_image"
        )
        return self.layer_norm(queries + attended, f"{prefix}.norm_final_attn"), keys

    def decode(self, features: array, positions: array, codes: array, target: int) -> array:
        """Decode encoder ``features`` ``(1, 48, D, H, W)`` into ``(1, 1, D, H, W)`` logits."""
        mx = self.mx
        tokens = self.tokens(positions, codes, target)
        low = self.gelu(
            self.norm(self.conv(features, "point_head.feat_downsample.0", stride=2), None)
        )
        low = self.norm(self.conv(low, "point_head.feat_downsample.3"), None)
        shape = tuple(low.shape[2:])
        grid = mx.stack(
            list(mx.meshgrid(*[(mx.arange(size) + 0.5) / size for size in shape], indexing="ij")),
            axis=-1,
        )
        key_pe = mx.reshape(self.positional_encoding(grid), (1, -1, FEATURES))
        queries, keys = self.transformer(
            mx.transpose(mx.reshape(low, (1, FEATURES, -1)), (0, 2, 1)), key_pe, tokens
        )
        hyper = queries[:, :1]
        for index in range(3):
            hyper = self.linear(hyper, f"point_head.output_hypernetworks_mlps.layers.{index}")
            if index < 2:
                hyper = mx.maximum(hyper, 0)
        up = self.output_upsample(mx.reshape(mx.transpose(keys, (0, 2, 1)), (1, FEATURES, *shape)))
        up = up + self.weights["point_head.output_upscaling.0.bias"][None, :, None, None, None]
        up = self.conv(self.gelu(self.norm(up, None)), "point_head.output_upscaling.3")
        return mx.reshape(hyper @ mx.reshape(up, (1, FEATURES, -1)), (1, 1, *up.shape[2:]))


def correction_shapes() -> dict[str, tuple[int, ...]]:
    """Exact executed tensors; unused mask/class embeddings are not converted.

    Covers the point-branch encoder (``up_layers``) and ``point_head`` weights, in
    source tensor layouts.
    """
    shapes = {
        name.replace("up_layers_auto", "up_layers"): shape
        for name, shape in required_shapes().items()
        if name.startswith("image_encoder.")
    }
    for name in (
        "feat_downsample.0",
        "feat_downsample.3",
        "output_upscaling.0",
        "output_upscaling.3",
    ):
        shapes[f"point_head.{name}.weight"] = (FEATURES, FEATURES, 3, 3, 3)
        shapes[f"point_head.{name}.bias"] = (FEATURES,)
    shapes["point_head.pe_layer.positional_encoding_gaussian_matrix"] = (3, FEATURES // 2)
    for name in (
        "point_embeddings.0",
        "point_embeddings.1",
        "not_a_point_embed",
        "special_class_embed",
        "mask_tokens",
        "zeroshot_embed",
        "supported_embed",
    ):
        shapes[f"point_head.{name}.weight"] = (1, FEATURES)
    for index in range(3):
        shapes[f"point_head.output_hypernetworks_mlps.layers.{index}.weight"] = (FEATURES, FEATURES)
        shapes[f"point_head.output_hypernetworks_mlps.layers.{index}.bias"] = (FEATURES,)
    for index in range(2):
        prefix = f"point_head.transformer.layers.{index}"
        for name in ("self_attn", "cross_attn_token_to_image", "cross_attn_image_to_token"):
            shapes.update(
                _attention_shapes(
                    f"{prefix}.{name}", FEATURES if name == "self_attn" else FEATURES // 2
                )
            )
        for norm in range(1, 5):
            shapes[f"{prefix}.norm{norm}.weight"] = (FEATURES,)
            shapes[f"{prefix}.norm{norm}.bias"] = (FEATURES,)
        for layer, shape in (("linear1", (512, FEATURES)), ("linear2", (FEATURES, 512))):
            shapes[f"{prefix}.mlp.{layer}.weight"] = shape
            shapes[f"{prefix}.mlp.{layer}.bias"] = (shape[0],)
    shapes.update(
        _attention_shapes("point_head.transformer.final_attn_token_to_image", FEATURES // 2)
    )
    for kind in ("weight", "bias"):
        shapes[f"point_head.transformer.norm_final_attn.{kind}"] = (FEATURES,)
    return shapes


def additional_shapes() -> dict[str, tuple[int, ...]]:
    """Reuse the admitted automatic cache's encoder without duplicating its weights."""
    automatic = required_shapes()
    return {name: shape for name, shape in correction_shapes().items() if name not in automatic}


def _attention_shapes(prefix: str, internal: int) -> dict[str, tuple[int, ...]]:
    """Return the required tensor shapes for an attention block."""
    shapes: dict[str, tuple[int, ...]] = {}
    for name in ("q", "k", "v", "out"):
        shape = (FEATURES, internal) if name == "out" else (internal, FEATURES)
        shapes[f"{prefix}.{name}_proj.weight"] = shape
        shapes[f"{prefix}.{name}_proj.bias"] = (shape[0],)
    return shapes
