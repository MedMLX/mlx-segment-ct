"""Seeded graph parity against recorded MONAI 1.6.0 CPU outputs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from reference_support import assert_bitwise, load_fixture, native, seeded_weights

from mlx_segment_ct import (
    AutomaticModel,
    AutomaticUpsample,
    FeatureModel,
    PointModel,
    additional_shapes,
    correction_shapes,
    normalized_point_positions,
    required_shapes,
)
from mlx_segment_ct._types import ArrayValue, CoreRuntime, HostArray
from mlx_segment_ct.mlx_point import SharedPointAutomatic

if TYPE_CHECKING:
    from mlx.core import array
    from oracle_types import GraphMetadata, NumericFixtures


def assert_numerical(actual: ArrayValue, expected: HostArray) -> None:
    """Compare native results with recorded values at fixed absolute tolerance."""
    # Torch CPU and Metal reductions/transcendentals are platform dependent, so compare
    # with the 1e-6 absolute bound of the seeded point decoder suite (the Metal run is
    # recorded in tests/fixtures/numerical-parity-20261013.json).
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=1e-6)


def fixture_weights(name: str, mx: CoreRuntime) -> tuple[NumericFixtures, dict[str, array]]:
    """Load a graph recording and its seeded native weights."""
    data, info = load_fixture(name)
    shapes = cast("GraphMetadata", info)["weight_shapes"]
    return data, seeded_weights(shapes, data["weight_seed"], mx)


def test_upsample_spatial_parities(mx: CoreRuntime) -> None:
    """Compare upsampling across spatial parity cases with the source recording."""
    data, _ = load_fixture("upsample")
    rng = np.random.default_rng(int(data["seed"]))
    image = rng.standard_normal((1, 4, 3, 5, 7), dtype=np.float32)
    weight = rng.standard_normal((4, 3, 3, 3, 3), dtype=np.float32) * np.float32(0.02)
    np.testing.assert_array_equal(image, data["image"])
    module = AutomaticUpsample(native(mx, weight), mx=mx)
    assert_numerical(module(native(mx, data["image"])), data["output"])


def test_feature_linear(mx: CoreRuntime) -> None:
    """Compare feature projections with the source recording."""
    data, weights = fixture_weights("feature_linear", mx)
    model = FeatureModel(weights, mx=mx, branch="up_layers_auto")
    assert_numerical(model.linear(native(mx, data["values"]), "class_head.mlp.0"), data["output"])


def test_point_tokens_transformer_and_long_attention(mx: CoreRuntime) -> None:
    """Compare point tokens, the two-way transformer and tiled long attention."""
    data, weights = fixture_weights("point_transformer", mx)
    model = PointModel(weights, mx=mx)
    positions, codes = native(mx, data["positions"]), native(mx, data["codes"])
    assert_numerical(model.positional_encoding(positions), data["positional_encoding"])
    for target, name in ((132, "supported_tokens"), (133, "zeroshot_tokens")):
        assert_numerical(model.tokens(positions, codes, target), data[name])
    tokens = model.tokens(positions, codes, 132)
    queries, keys = model.transformer(native(mx, data["keys"]), native(mx, data["key_pe"]), tokens)
    assert_numerical(queries, data["queries"])
    assert_numerical(keys, data["transformed_keys"])
    assert_numerical(model.gelu(native(mx, data["keys"])), data["gelu"])
    rng = np.random.default_rng(int(data["long_seed"]))
    long_keys = rng.standard_normal((1, 4096, 48), dtype=np.float32)
    long_values = rng.standard_normal((1, 4096, 48), dtype=np.float32)
    result = model.attention(
        tokens,
        native(mx, long_keys),
        native(mx, long_values),
        "point_head.transformer.final_attn_token_to_image",
    )
    assert_numerical(result, data["long_attention"])


def test_point_normalization_matches_upstream() -> None:
    """Compare point normalization with the recorded source result."""
    data, _ = load_fixture("point_normalization")
    points = data["points"].copy()
    normalized = normalized_point_positions(points, tuple(data["shape"]))
    assert_bitwise(normalized, data["normalized"], "normalized positions")
    assert np.array_equal(points, data["points"])


def test_checkpoint_key_and_shape_mapping() -> None:
    """Verify required checkpoint names and shapes against the source."""
    data, _ = load_fixture("key_shapes")
    for name, mapping in (
        ("automatic", required_shapes()),
        ("correction", correction_shapes()),
        ("additional", additional_shapes()),
    ):
        expected = {key: tuple(value) for key, value in json.loads(str(data[name])).items()}
        assert mapping == expected, f"{name} graph keys/shapes differ from MONAI"


@pytest.mark.parametrize("kind", ["feature", "automatic", "point"])
def test_full_model_equivalence_on_metal(kind: str, mx: CoreRuntime) -> None:
    """Compare seeded full-network native output with the recorded source."""
    data, weights = fixture_weights(f"full_{kind}", mx)
    image = native(mx, data["image"])
    if kind == "feature":
        actual = FeatureModel(weights, mx=mx, branch="up_layers_auto").encode(image)
    elif kind == "automatic":
        actual = AutomaticModel(weights, data["class_ids"].tolist(), mx=mx)(image)
    else:
        actual = PointModel(weights, mx=mx)(
            image, native(mx, data["positions"]), native(mx, data["codes"]), int(data["target"])
        )
    assert_numerical(actual, data["output"])
    if kind != "feature":
        np.testing.assert_array_equal(np.asarray(actual) > 0, data["output"] > 0)


def test_shared_encoder_preserves_both_decoder_logits(mx: CoreRuntime) -> None:
    """Reuse one encoder while preserving recorded point scores and separate auto scores."""
    data, weights = fixture_weights("full_point", mx)
    auto_only = {name: shape for name, shape in required_shapes().items() if name not in weights}
    weights.update(seeded_weights(auto_only, 402, mx))
    point = PointModel(weights, mx=mx)
    automatic = AutomaticModel(weights, (26,), mx=mx)
    image = native(mx, data["image"])
    separate_auto = automatic(image)
    mx.eval(separate_auto)
    auto, clicked = SharedPointAutomatic(point, automatic)(
        image, native(mx, data["positions"]), native(mx, data["codes"]), int(data["target"])
    )
    mx.eval(auto, clicked)
    np.testing.assert_array_equal(
        cast("HostArray", np.asarray(auto)), cast("HostArray", np.asarray(separate_auto))
    )
    assert_numerical(clicked, data["output"])
    np.testing.assert_array_equal(np.asarray(clicked) > 0, data["output"] > 0)

    # A second allocation is not an admitted shared checkpoint, even if its values match.
    different = dict(weights)
    name = "image_encoder.encoder.conv_init.weight"
    different[name] = native(mx, weights[name])
    automatic.weights = different
    with pytest.raises(ValueError, match="identical encoder arrays"):
        SharedPointAutomatic(point, automatic)
