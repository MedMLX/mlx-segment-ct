"""Seeded graph parity against recorded MONAI 1.6.0 CPU outputs."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from mlx_segment_ct._types import (
    ArrayBoundary,
    ArrayValue,
    CoreRuntime,
    HostArray,
)
from scripts.oracle_types import FixtureMetadata, GraphMetadata

if TYPE_CHECKING:
    from mlx.core import array

    from mlx_segment_ct._types import ArrayBoundary
    from scripts.oracle_types import FixtureArrays

import json
import subprocess
import sys

import numpy as np
import pytest
from reference_support import assert_bitwise, load_fixture

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
from scripts.fixture_inputs import random_weights


def weights_from_fixture(data: FixtureArrays, info: FixtureMetadata, mx: CoreRuntime):
    """Construct named native weights from a recorded fixture."""
    shapes = {
        key: tuple(shape) for key, shape in cast("GraphMetadata", info)["weight_shapes"].items()
    }
    return {
        key: cast("ArrayBoundary[array]", mx).array(value)
        for key, value in random_weights(
            shapes, int(cast("HostArray", data["weight_seed"]))
        ).items()
    }


def assert_numerical(actual: ArrayValue, expected: HostArray) -> None:
    # Torch CPU and Metal reductions/transcendentals are platform dependent, so compare
    # with the 1e-6 absolute bound of the seeded point decoder suite (the Metal run is
    # recorded in docs/validation/numerical-parity-20261013.json).
    """Compare native results with recorded values at fixed absolute tolerance."""
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=1e-6)


def test_upsample_spatial_parities(mx: CoreRuntime) -> None:
    """Compare upsampling across spatial parity cases with the source recording."""
    data, _info = load_fixture("upsample")
    rng = np.random.default_rng(int(cast("HostArray", data["seed"])))
    image = rng.standard_normal((1, 4, 3, 5, 7), dtype=np.float32)
    weight = rng.standard_normal((4, 3, 3, 3, 3), dtype=np.float32) * np.float32(0.02)
    np.testing.assert_array_equal(image, data["image"])
    module = AutomaticUpsample(
        cast("ArrayBoundary[array]", mx).array(weight), mx=mx
    )
    assert_numerical(
        module(cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["image"]))),
        cast("HostArray", data["output"]),
    )


def test_feature_linear(mx: CoreRuntime) -> None:
    """Compare feature projections with the source recording."""
    data, info = load_fixture("feature_linear")
    model = FeatureModel(weights_from_fixture(data, info, mx), mx=mx, branch="up_layers_auto")
    assert_numerical(
        model.linear(
            cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["values"])),
            "class_head.mlp.0",
        ),
        cast("HostArray", data["output"]),
    )


@pytest.fixture(scope="module")
def point_reference(mx: CoreRuntime):
    """Provide the point model and recorded transformer inputs."""
    data, info = load_fixture("point_transformer")
    return PointModel(weights_from_fixture(data, info, mx), mx=mx), data, mx


def test_point_tokens_and_positional_encoding(
    point_reference: tuple[PointModel, FixtureArrays, CoreRuntime],
) -> None:
    """Compare point tokens and positional encoding with the source."""
    model, data, mx = point_reference
    positions, codes = (
        cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["positions"])),
        cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["codes"])),
    )
    assert_numerical(
        model.positional_encoding(positions), cast("HostArray", data["positional_encoding"])
    )
    for target, name in ((132, "supported_tokens"), (133, "zeroshot_tokens")):
        assert_numerical(model.tokens(positions, codes, target), cast("HostArray", data[name]))


def test_point_two_way_transformer(
    point_reference: tuple[PointModel, FixtureArrays, CoreRuntime],
) -> None:
    """Compare the point transformer against the source recording."""
    model, data, mx = point_reference
    tokens = model.tokens(
        cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["positions"])),
        cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["codes"])),
        132,
    )
    queries, keys = model.transformer(
        cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["keys"])),
        cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["key_pe"])),
        tokens,
    )
    assert_numerical(queries, cast("HostArray", data["queries"]))
    assert_numerical(keys, cast("HostArray", data["transformed_keys"]))
    assert_numerical(
        model.gelu(cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["keys"]))),
        cast("HostArray", data["gelu"]),
    )


def test_long_attention_tiled_reduction(
    point_reference: tuple[PointModel, FixtureArrays, CoreRuntime],
) -> None:
    """Compare tiled long-attention reduction with the reference."""
    model, data, mx = point_reference
    rng = np.random.default_rng(int(cast("HostArray", data["long_seed"])))
    keys = rng.standard_normal((1, 4096, 48), dtype=np.float32)
    values = rng.standard_normal((1, 4096, 48), dtype=np.float32)
    tokens = model.tokens(
        cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["positions"])),
        cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["codes"])),
        132,
    )
    result = model.attention(
        tokens,
        cast("ArrayBoundary[array]", mx).array(keys),
        cast("ArrayBoundary[array]", mx).array(values),
        "point_head.transformer.final_attn_token_to_image",
    )
    assert_numerical(result, cast("HostArray", data["long_attention"]))


def test_point_normalization_matches_upstream() -> None:
    """Compare point normalization with the recorded source result."""
    data, _ = load_fixture("point_normalization")
    points = cast("HostArray", data["points"]).copy()
    normalized = normalized_point_positions(points, tuple(cast("HostArray", data["shape"])))
    assert_bitwise(normalized, cast("HostArray", data["normalized"]), "normalized positions")
    assert np.array_equal(points, cast("HostArray", data["points"]))


def test_checkpoint_key_and_shape_mapping() -> None:
    """Verify required checkpoint names and shapes against the source."""
    data, _info = load_fixture("key_shapes")
    for name, mapping in (
        ("automatic", required_shapes()),
        ("correction", correction_shapes()),
        ("additional", additional_shapes()),
    ):
        expected = {key: tuple(value) for key, value in json.loads(str(data[name])).items()}
        assert mapping == expected, f"{name} graph keys/shapes differ from MONAI"


def test_runtime_modules_do_not_import_reference_frameworks() -> None:
    # A fresh interpreter catches transitive imports even when Torch was already
    # loaded by another test. Import every runtime module with reference
    # frameworks blocked.
    """Verify runtime imports do not load source-only frameworks."""
    code = """
import importlib
import importlib.abc
import pkgutil
import sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'monai'}:
            raise AssertionError('Forbidden runtime import: ' + fullname)
sys.meta_path.insert(0, Block())
import mlx_segment_ct
for module in pkgutil.iter_modules(mlx_segment_ct.__path__):
    importlib.import_module('mlx_segment_ct.' + module.name)
from medmlx_core.precision import Float32Operators
from mlx_segment_ct.mlx_network import Float32Operators as ModelOperators
assert ModelOperators is Float32Operators
assert not {'torch', 'monai'} & sys.modules.keys()
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("kind", ["feature", "automatic", "point"])
def test_full_model_equivalence_on_metal(kind: str, mx: CoreRuntime) -> None:
    """Compare seeded full-network native output with the recorded source."""
    data, info = load_fixture(f"full_{kind}")
    weights = weights_from_fixture(data, info, mx)
    image = cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["image"]))
    if kind == "feature":
        actual = FeatureModel(weights, mx=mx, branch="up_layers_auto").encode(image)
    elif kind == "automatic":
        actual = AutomaticModel(weights, cast("HostArray", data["class_ids"]).tolist(), mx=mx)(
            image
        )
    else:
        actual = PointModel(weights, mx=mx)(
            image,
            cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["positions"])),
            cast("ArrayBoundary[array]", mx).array(cast("HostArray", data["codes"])),
            int(cast("HostArray", data["target"])),
        )
    assert_numerical(actual, cast("HostArray", data["output"]))
    if kind != "feature":
        np.testing.assert_array_equal(np.asarray(actual) > 0, cast("HostArray", data["output"]) > 0)


class ShapeValidationModel(FeatureModel):
    def __init__(self) -> None:
        """Construct only the shape-validation boundary without loading weights."""
        from reference_support import HostArrays

        self.mx = HostArrays


@pytest.mark.parametrize("shape", [(16, 16, 16), (0, 16, 16)])
def test_encoder_rejects_degenerate_upstream_normalization_domain(shape: tuple[int, ...]) -> None:
    """Reject a spatial shape that degenerates at the deepest normalization layer."""
    with pytest.raises(ValueError, match="deepest spatial voxel"):
        ShapeValidationModel().validate_image(np.zeros((1, 1, *shape), np.float32))


def test_encoder_accepts_smallest_valid_anisotropic_input() -> None:
    """Accept the smallest valid anisotropic input shape."""
    ShapeValidationModel().validate_image(np.zeros((1, 1, 16, 16, 32), np.float32))
