"""Upstream window placement, ordered stitching and point component fusion."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import pytest
from fixture_inputs import AutomaticStandIn, PointStandIn
from reference_support import HostArrays, assert_bitwise, load_fixture, seeded_weights

from mlx_segment_ct import (
    AutomaticModel,
    PointModel,
    PointWindowPredictor,
    automatic_logits,
    bounded_labels,
    point_based_window_inferer,
    rolling_logits,
)
from mlx_segment_ct._types import (
    ArrayBoundary,
    ArrayValue,
    CoreRuntime,
    HostArray,
    PatchCoordinates,
    PointPredictor,
    StandInRuntime,
)
from mlx_segment_ct.mlx_point_window import connected_components_combine, update_point_to_patch
from mlx_segment_ct.streaming import stream_scores

if TYPE_CHECKING:
    from mlx.core import array
    from oracle_types import GraphMetadata, NumericFixtures

HOST = cast("StandInRuntime[HostArray, type[np.float32]]", HostArrays)


def automatic_case[ArrayT: ArrayValue](
    data: NumericFixtures,
    model: Callable[[ArrayT], ArrayT],
    mx: ArrayBoundary[ArrayT],
    *,
    native: bool = False,
    reduce_labels: bool = True,
) -> None:
    """Run one recorded automatic-window case and compare it with the recording."""
    trace: list[list[int]] = []
    planes: list[HostArray] = []
    counts: list[HostArray] = []

    def predict(patch: ArrayT, coordinates: PatchCoordinates) -> ArrayT:
        trace.extend(
            [[int(cast("int", part.start)) for part in coord[2:]] for coord in coordinates]
        )
        return model(patch)

    def finalize(plane: int, scores: HostArray, count: HostArray) -> None:
        assert plane == len(planes)
        planes.append(scores.copy())
        counts.append(count.copy())

    image = data["image"].copy()
    stream = rolling_logits if reduce_labels else stream_scores
    labels = stream(
        image,
        predict,
        data["class_ids"].tolist(),
        cast("Sequence[int]", data["roi"]),
        weight=data["weight"],
        finalized=finalize,
        mx=mx,
    )
    if reduce_labels:
        assert labels is not None
        assert_bitwise(labels, data["labels"], "rolling labels")
    else:
        assert labels is None
    if native:
        # Torch CPU reference: ordered Metal convolution reductions differ from it, so
        # use the seeded decoder's strict 1e-6 absolute bound rather than bitwise equality.
        np.testing.assert_allclose(np.stack(planes, axis=1), data["planes"], rtol=0, atol=1e-6)
    else:
        assert_bitwise(np.stack(planes, axis=1), data["planes"], "finalized scores")
    assert_bitwise(np.stack(counts), data["counts"], "finalized counts")
    assert np.array_equal(trace, data["coordinates"])
    assert np.array_equal(image, data["image"])


@pytest.mark.parametrize("name", ["constant", "padded", "gaussian"])
@pytest.mark.parametrize("reduce_labels", [True, False], ids=["labels", "scores"])
def test_automatic_rolling_matches_monai(name: str, reduce_labels: bool) -> None:
    """Compare rolling automatic-window results with MONAI recordings."""
    data, _ = load_fixture(f"window_{name}")
    model = AutomaticStandIn(int(data["network_seed"]), mx=HOST)
    automatic_case(data, model, HostArrays, reduce_labels=reduce_labels)


@pytest.mark.parametrize("name", ["constant", "padded", "gaussian"])
def test_automatic_full_logit_stitching_matches_monai(name: str, mx: CoreRuntime) -> None:
    """Compare full-logit stitching with MONAI recordings."""
    data, _ = load_fixture(f"window_{name}")
    model = AutomaticStandIn(int(data["network_seed"]), mx=mx)

    def predict(p: array, c: PatchCoordinates) -> array:
        return model(p)

    roi = cast("Sequence[int]", data["roi"])
    logits = automatic_logits(data["image"], predict, roi, mode=str(data["mode"]), with_coord=True)
    # Torch/NumPy Gaussian exponentials and Metal stand-in products can differ in FP32;
    # Gaussian and constant blending maps share the same 1e-6 absolute bound.
    np.testing.assert_allclose(logits, data["logits"], rtol=0, atol=1e-6)
    supplied = automatic_logits(
        data["image"], predict, roi, roi_weight_map=data["weight"], with_coord=True
    )
    np.testing.assert_allclose(supplied, data["logits"], rtol=0, atol=1e-6)


def point_case[ArrayT: ArrayValue](
    data: NumericFixtures,
    model: PointPredictor[ArrayT],
    mx: ArrayBoundary[ArrayT],
    *,
    native: bool = False,
) -> None:
    """Run one recorded point-window case and compare it with the recording."""
    image, previous = data["image"].copy(), data["previous"].copy()
    points, codes = data["points"].copy(), data["codes"].copy()
    result = point_based_window_inferer(
        image,
        cast("Sequence[int]", data["roi"]),
        PointWindowPredictor(model, mx=mx, metrics={"patch_seconds": []}),
        points,
        codes,
        class_vector=None,
        prompt_class=data["prompt"],
        prev_mask=previous,
        point_start=int(data["point_start"]),
        center_only=bool(data["center_only"]),
        margin=int(data["margin"]),
        transpose=True,
        mx=mx,
    )
    if native:
        # Torch CPU reference; same 1e-6 absolute bound as the seeded decoder.
        np.testing.assert_allclose(result, data["output"], rtol=0, atol=1e-6)
        np.testing.assert_array_equal(result > 0, data["output"] > 0)
    elif not bool(data["center_only"]):
        # Torch CPU's normalized-click FP32 sum differs from NumPy's reduction. Across
        # the 27-window case the observed maximum is 3 * 2**-18, used as the absolute
        # bound; foreground decisions must be identical.
        np.testing.assert_allclose(result, data["output"], rtol=0, atol=3 * 2**-18)
        np.testing.assert_array_equal(result > 0, data["output"] > 0)
    else:
        assert_bitwise(result, data["output"], "point correction logits")
    for actual, name in ((previous, "previous"), (image, "image"), (points, "points")):
        assert np.array_equal(actual, data[name]), f"{name} was modified"
    assert np.array_equal(codes, data["codes"])


@pytest.mark.parametrize("name", ["overlap", "padded", "multi"])
def test_point_windows_match_monai(name: str) -> None:
    """Compare point-window results with MONAI recordings."""
    data, _ = load_fixture(f"point_window_{name}")
    point_case(data, PointStandIn(int(data["network_seed"]), mx=HOST), HostArrays)


def test_vista_click_filtering_and_component_merge() -> None:
    """Compare click filtering and component merging with the reference."""
    data, _ = load_fixture("components")
    coords = [slice(None), slice(None), slice(0, 7), slice(0, 7), slice(0, 7)]
    points, codes = update_point_to_patch(coords, data["points"], data["codes"])
    assert points is not None and codes is not None
    assert_bitwise(points, data["updated"], "eligible coordinates")
    assert_bitwise(codes, data["labels"], "eligible codes")
    result = connected_components_combine(
        data["previous"].copy(), data["predicted"].copy(), points, codes, np.array([True])
    )
    assert_bitwise(result, data["output"], "connected components")


def test_bounded_reduction_ties_background_and_global_id_order() -> None:
    """Verify tie handling, background, and global class order in bounded reduction."""
    data, _ = load_fixture("bounded_labels")
    reduced = bounded_labels(data["scores"], data["class_ids"].tolist())
    assert_bitwise(reduced, data["output"], "bounded reduction")
    assert np.array_equal(data["output"][0, 0, 0, 0], [0, 0, 26, 26])


@pytest.mark.parametrize("kind", ["automatic", "point"])
def test_full_network_window_wiring_on_metal(kind: str, mx: CoreRuntime) -> None:
    """Exercise native full-network window wiring using recorded fixtures."""
    name = "full_window_constant" if kind == "automatic" else "full_point_window_overlap"
    data, info = load_fixture(name)
    weights = seeded_weights(cast("GraphMetadata", info)["weight_shapes"], data["weight_seed"], mx)
    if kind == "automatic":
        model = AutomaticModel(weights, data["class_ids"].tolist(), mx=mx)
        automatic_case(data, model, mx, native=True)
    else:
        point_case(data, PointModel(weights, mx=mx), mx, native=True)
