"""Seeded full-graph upstream oracles; no MONAI import at test time."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from reference_support import load_fixture, native, seeded_weights

from mlx_segment_ct import AutomaticModel, AutomaticUpsample, PointModel, PointWindowPredictor
from mlx_segment_ct._types import ArrayValue, CoreRuntime, HostArray, Metrics
from mlx_segment_ct.correction import normalized_point_positions
from mlx_segment_ct.correction_inferer import point_based_window_inferer

if TYPE_CHECKING:
    from mlx.core import array
    from oracle_types import GraphMetadata


def assert_logits(actual: ArrayValue, expected: HostArray) -> None:
    """Compare native logits with the reference at fixed absolute tolerance."""
    # Torch CPU and Metal reductions differ, so compare with the 1e-6 absolute bound
    # of the seeded decoder suite and require identical foreground decisions. The
    # Metal run at this bound is in tests/fixtures/numerical-parity-20261013.json.
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=0, atol=1e-6)
    np.testing.assert_array_equal(np.asarray(actual) > 0, expected > 0)


def test_automatic_graph_all_bundle_classes(mx: CoreRuntime) -> None:
    """Compare automatic graph outputs across bundle class prompts."""
    data, info = load_fixture("automatic_subset")
    weights = seeded_weights(cast("GraphMetadata", info)["weight_shapes"], data["weight_seed"], mx)
    classes = data["class_ids"].tolist()
    model = AutomaticModel(weights, classes, mx=mx)
    actual = model(native(mx, data["image"]))
    mx.eval(actual)
    assert_logits(actual, data["output"])
    # A per-call prompt must change the requested class ordering, even when the
    # instance has already evaluated its default fixed prompts at construction.
    dynamic = model(native(mx, data["image"]), list(reversed(classes)))
    mx.eval(dynamic)
    assert_logits(dynamic, data["output"][:, ::-1])


@pytest.mark.parametrize("channels", [96, 192, 384, 768])
def test_decoder_upsampling_all_spatial_parities(channels: int, mx: CoreRuntime) -> None:
    """Compare decoder upsampling across spatial parity cases."""
    data, _ = load_fixture(f"upsample_{channels}")
    rng = np.random.default_rng(int(data["seed"]))
    image = rng.standard_normal((1, channels, 3, 5, 7), dtype=np.float32)
    weight = rng.standard_normal((channels, channels // 2, 3, 3, 3), dtype=np.float32) * 0.02
    upsampled = AutomaticUpsample(native(mx, weight), mx=mx)(native(mx, image))
    mx.eval(upsampled)
    actual = cast("HostArray", np.asarray(upsampled))
    # Decoder-width upsampling keeps a 1e-4 absolute/relative bound (the record's
    # existing_decoder_width_atol/rtol in tests/fixtures/numerical-parity-20261013.json).
    np.testing.assert_allclose(actual, data["output"], rtol=1e-4, atol=1e-4)


@pytest.fixture(scope="module")
def point_model(mx: CoreRuntime) -> PointModel:
    """Provide the seeded point model shared by the point-target recordings."""
    data, info = load_fixture("point_target_1")
    weights = seeded_weights(cast("GraphMetadata", info)["weight_shapes"], data["weight_seed"], mx)
    return PointModel(weights, mx=mx)


@pytest.mark.parametrize("target", [1, 25, 26, 117, 254])
def test_point_graph_special_and_zero_shot_prompts(
    target: int, point_model: PointModel, mx: CoreRuntime
) -> None:
    """Compare point-graph outputs for special and zero-shot targets."""
    data, _ = load_fixture(f"point_target_{target}")
    positions = normalized_point_positions(data["points"], data["image"].shape[2:])
    actual = point_model(
        native(mx, data["image"]), native(mx, positions), native(mx, data["codes"]), target
    )
    mx.eval(actual)
    assert_logits(actual, data["output"])


def test_previous_logits_and_old_click_context(point_model: PointModel, mx: CoreRuntime) -> None:
    """Compare correction using previous logits and earlier click context."""
    data, _ = load_fixture("point_previous")
    metrics: Metrics = {"patch_seconds": []}
    actual = point_based_window_inferer(
        data["image"],
        [32] * 3,
        PointWindowPredictor(point_model, mx=mx, metrics=metrics),
        data["points"],
        data["codes"],
        class_vector=None,
        prompt_class=np.array([1]),
        prev_mask=data["previous"],
        point_start=1,
        center_only=True,
        transpose=True,
        mx=mx,
    )
    assert_logits(actual, data["output"])
    assert np.array_equal(actual[:, :, :16], data["previous"][:, :, :16])
    assert len(metrics["patch_seconds"]) == 1


@pytest.fixture(scope="module")
def released_point_model(mx: CoreRuntime) -> PointModel:
    """Load released point weights when their local folder is configured."""
    directory = os.environ.get("NV_SEGMENT_CT_WEIGHTS_DIR")
    if directory is None:
        pytest.skip("set NV_SEGMENT_CT_WEIGHTS_DIR for released-checkpoint comparisons")
    from mlx_segment_ct.mlx_point import correction_shapes

    root = Path(directory)
    automatic = cast("dict[str, array]", mx.load(str(root / "automatic.safetensors")))
    extra = cast("dict[str, array]", mx.load(str(root / "correction.safetensors")))
    weights = {
        name: extra[name] if name in extra else automatic[name] for name in correction_shapes()
    }
    return PointModel(weights, mx=mx)


@pytest.mark.parametrize(
    "case", ["native", "flipped", "oblique", "exclude", "small", "vessel", "tumor"]
)
def test_staged_real_128_cube_matches_independent_cuda_fp32(
    case: str, released_point_model: PointModel, mx: CoreRuntime
) -> None:
    """Compare the staged real point case with independently recorded CUDA FP32 output."""
    fixtures = os.environ.get("NV_SEGMENT_CT_CUDA_FIXTURES")
    if fixtures is None:
        pytest.skip("set NV_SEGMENT_CT_CUDA_FIXTURES for optional upstream AMOS captures")
    folder = Path(fixtures) / case
    assert (folder / "fp32/patch-0.npz").is_file(), f"Missing CUDA patch: {folder}"
    request = json.loads((folder / "request.json").read_text())
    with np.load(folder / "fp32/patch-0.npz", allow_pickle=False) as patch:
        logits = released_point_model(
            native(mx, patch["image"]),
            native(mx, normalized_point_positions(patch["points"], patch["image"].shape[2:])),
            native(mx, patch["codes"]),
            request["label_prompt"][0],
        )
    mx.eval(logits)
    mx.synchronize()
    actual = cast("HostArray", np.asarray(logits))
    expected = np.load(folder / "fp32/logits-0.npy", allow_pickle=False)
    # Released-checkpoint point decoder against the staged CUDA FP32 capture: 1e-4 bound.
    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(actual > 0, expected > 0)
