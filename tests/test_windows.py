"""Upstream window placement, ordered stitching and point component fusion."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from mlx_segment_ct._types import (
    ArrayBoundary,
    ArrayValue,
    CoreRuntime,
    HostArray,
    PatchCoordinates,
    PointPredictor,
    RollingMetrics,
    RollingMetricsBuilder,
    StandInRuntime,
)
from scripts.oracle_types import GraphMetadata

if TYPE_CHECKING:
    from mlx.core import array

    from mlx_segment_ct._types import ArrayBoundary
    from scripts.oracle_types import FixtureArrays

import numpy as np
import pytest
from reference_support import HostArrays, assert_bitwise, load_fixture

from mlx_segment_ct import (
    AutomaticModel,
    PointModel,
    PointWindowPredictor,
    automatic_logits,
    bounded_labels,
    point_based_window_inferer,
    rolling_logits,
)
from mlx_segment_ct.mlx_point_window import connected_components_combine, update_point_to_patch
from mlx_segment_ct.streaming import stream_scores
from scripts.fixture_inputs import AutomaticStandIn, PointStandIn, random_weights


def automatic_case[ArrayT: ArrayValue](
    data: FixtureArrays,
    model: Callable[[ArrayT], ArrayT],
    mx: ArrayBoundary[ArrayT],
    *,
    native: bool = False,
    reduce_labels: bool = True,
) -> None:
    """Run one recorded automatic-window case and collect its results."""
    trace: list[list[int]] = []
    planes: list[HostArray] = []
    counts: list[HostArray] = []

    def predict(patch: ArrayT, coordinates: PatchCoordinates) -> ArrayT:
        """Record window coordinates and predict with the synthetic model."""
        trace.extend(
            [[int(cast("int", part.start)) for part in coord[2:]] for coord in coordinates]
        )
        return model(patch)

    def finalize(plane: int, scores: HostArray, count: HostArray) -> None:
        """Collect completed score planes in order."""
        assert plane == len(planes)
        planes.append(scores.copy())
        counts.append(count.copy())

    image = cast("HostArray", data["image"]).copy()
    metrics: RollingMetricsBuilder = {}
    stream = rolling_logits if reduce_labels else stream_scores
    labels = stream(
        image,
        predict,
        cast("HostArray", data["class_ids"]).tolist(),
        cast("Sequence[int]", data["roi"]),
        weight=cast("HostArray", data["weight"]),
        finalized=finalize,
        metrics=metrics,
        mx=mx,
    )
    if reduce_labels:
        assert labels is not None
        assert_bitwise(labels, cast("HostArray", data["labels"]), "rolling labels")
    else:
        assert labels is None
    if native:
        # Torch CPU reference: ordered Metal convolution reductions differ from it, so
        # use the seeded decoder's strict 1e-6 absolute bound rather than bitwise equality.
        np.testing.assert_allclose(
            np.stack(planes, axis=1), cast("HostArray", data["planes"]), rtol=0, atol=1e-6
        )
    else:
        assert_bitwise(
            np.stack(planes, axis=1), cast("HostArray", data["planes"]), "finalized scores"
        )
    assert_bitwise(np.stack(counts), cast("HostArray", data["counts"]), "finalized counts")
    assert np.array_equal(trace, cast("HostArray", data["coordinates"]))
    assert np.array_equal(image, cast("HostArray", data["image"]))
    padded = cast("RollingMetrics", metrics)["padded_shape"]
    assert cast("RollingMetrics", metrics)["predicted_accumulator_bytes"] == (
        len(cast("HostArray", data["class_ids"]))
        * cast("HostArray", data["roi"])[0]
        * padded[1]
        * padded[2]
        * 4
    )
    assert (
        cast("RollingMetrics", metrics)["count_buffer_bytes"]
        == cast("HostArray", data["roi"])[0] * padded[1] * padded[2] * 4
    )
    assert cast("RollingMetrics", metrics)["label_buffer_bytes"] == (
        np.prod(padded) if reduce_labels else 0
    )
    assert cast("RollingMetrics", metrics)["accumulator_dtype"] == "float32"


@pytest.mark.parametrize("name", ["constant", "padded", "gaussian"])
@pytest.mark.parametrize("reduce_labels", [True, False], ids=["labels", "scores"])
def test_automatic_rolling_matches_monai(name: str, reduce_labels: bool) -> None:
    """Compare rolling automatic-window results with MONAI recordings."""
    data, _ = load_fixture(f"window_{name}")
    automatic_case(
        data,
        AutomaticStandIn(
            int(cast("HostArray", data["network_seed"])),
            mx=cast("StandInRuntime[HostArray, type[np.float32]]", HostArrays),
        ),
        HostArrays,
        reduce_labels=reduce_labels,
    )


@pytest.mark.parametrize("name", ["constant", "padded", "gaussian"])
def test_automatic_full_logit_stitching_matches_monai(name: str, mx: CoreRuntime) -> None:
    """Compare full-logit stitching with MONAI recordings."""
    data, _ = load_fixture(f"window_{name}")
    model = AutomaticStandIn(int(cast("HostArray", data["network_seed"])), mx=mx)

    def predict(p: array, c: PatchCoordinates) -> array:
        """Run the synthetic predictor for one window."""
        return model(p)

    logits = automatic_logits(
        cast("HostArray", data["image"]),
        predict,
        cast("Sequence[int]", data["roi"]),
        mode=str(cast("dict[str, object]", data)["mode"]),
        with_coord=True,
    )
    # Torch/NumPy Gaussian exponentials and Metal stand-in products can differ in FP32;
    # Gaussian and constant blending maps share the same 1e-6 absolute bound.
    np.testing.assert_allclose(logits, cast("HostArray", data["logits"]), rtol=0, atol=1e-6)
    supplied = automatic_logits(
        cast("HostArray", data["image"]),
        predict,
        cast("Sequence[int]", data["roi"]),
        roi_weight_map=cast("HostArray", data["weight"]),
        with_coord=True,
    )
    np.testing.assert_allclose(supplied, cast("HostArray", data["logits"]), rtol=0, atol=1e-6)


def point_case[ArrayT: ArrayValue](
    data: FixtureArrays,
    model: PointPredictor[ArrayT],
    mx: ArrayBoundary[ArrayT],
    *,
    native: bool = False,
) -> None:
    """Run one recorded point-window case."""
    image, previous = (
        cast("HostArray", data["image"]).copy(),
        cast("HostArray", data["previous"]).copy(),
    )
    points, codes = (
        cast("HostArray", data["points"]).copy(),
        cast("HostArray", data["codes"]).copy(),
    )
    result = point_based_window_inferer(
        image,
        cast("Sequence[int]", data["roi"]),
        PointWindowPredictor(model, mx=mx, metrics={"patch_seconds": []}),
        points,
        codes,
        class_vector=None,
        prompt_class=cast("HostArray", data["prompt"]),
        prev_mask=previous,
        point_start=int(cast("HostArray", data["point_start"])),
        center_only=bool(cast("HostArray", data["center_only"])),
        margin=int(cast("HostArray", data["margin"])),
        transpose=True,
        mx=mx,
    )
    if native:
        # Torch CPU reference; same 1e-6 absolute bound as the seeded decoder.
        np.testing.assert_allclose(result, cast("HostArray", data["output"]), rtol=0, atol=1e-6)
        np.testing.assert_array_equal(result > 0, cast("HostArray", data["output"]) > 0)
    elif not bool(cast("HostArray", data["center_only"])):
        # Torch CPU's normalized-click FP32 sum differs from NumPy's reduction. Across
        # the 27-window case the observed maximum is 3 * 2**-18, used as the absolute
        # bound; foreground decisions must be identical.
        np.testing.assert_allclose(
            result, cast("HostArray", data["output"]), rtol=0, atol=3 * 2**-18
        )
        np.testing.assert_array_equal(result > 0, cast("HostArray", data["output"]) > 0)
    else:
        assert_bitwise(result, cast("HostArray", data["output"]), "point correction logits")
    for actual, expected in (
        (previous, cast("HostArray", data["previous"])),
        (image, cast("HostArray", data["image"])),
        (points, cast("HostArray", data["points"])),
        (codes, cast("HostArray", data["codes"])),
    ):
        assert np.array_equal(actual, expected)


@pytest.mark.parametrize("name", ["overlap", "padded", "multi"])
def test_point_windows_match_monai(name: str) -> None:
    """Compare point-window results with MONAI recordings."""
    data, _ = load_fixture(f"point_window_{name}")
    point_case(
        data,
        PointStandIn(
            int(cast("HostArray", data["network_seed"])),
            mx=cast("StandInRuntime[HostArray, type[np.float32]]", HostArrays),
        ),
        HostArrays,
    )


def test_vista_click_filtering_and_component_merge() -> None:
    """Compare click filtering and component merging with the reference."""
    data, _ = load_fixture("components")
    coords = [slice(None), slice(None), slice(0, 7), slice(0, 7), slice(0, 7)]
    points, codes = update_point_to_patch(
        coords, cast("HostArray", data["points"]), cast("HostArray", data["codes"])
    )
    assert points is not None and codes is not None
    assert_bitwise(points, cast("HostArray", data["updated"]), "eligible coordinates")
    assert_bitwise(codes, cast("HostArray", data["labels"]), "eligible codes")
    result = connected_components_combine(
        cast("HostArray", data["previous"]).copy(),
        cast("HostArray", data["predicted"]).copy(),
        points,
        codes,
        np.array([True]),
    )
    assert_bitwise(result, cast("HostArray", data["output"]), "connected components")


def test_bounded_reduction_ties_background_and_global_id_order() -> None:
    """Verify tie handling, background, and global class order in bounded reduction."""
    data, _ = load_fixture("bounded_labels")
    assert_bitwise(
        bounded_labels(
            cast("HostArray", data["scores"]), cast("HostArray", data["class_ids"]).tolist()
        ),
        cast("HostArray", data["output"]),
        "bounded reduction",
    )
    assert np.array_equal(cast("HostArray", data["output"])[0, 0, 0, 0], [0, 0, 26, 26])


@pytest.mark.parametrize("reduce_labels", [True, False], ids=["labels", "scores"])
def test_invalid_automatic_outputs_and_class_ids(reduce_labels: bool) -> None:
    """Reject invalid score outputs and unsupported class IDs."""
    image = np.zeros((1, 1, 2, 2, 2), dtype=np.float32)

    def identity(x: HostArray, c: PatchCoordinates) -> HostArray:
        """Return the input as a valid synthetic prediction."""
        return x

    def nonfinite(x: HostArray, c: PatchCoordinates) -> HostArray:
        """Return nonfinite predictions to exercise validation."""
        return x + np.float32(np.nan)

    stream = rolling_logits if reduce_labels else stream_scores
    for ids in ([], [3, 3], [0], [512]):
        with pytest.raises(ValueError, match="unique foreground IDs below 512"):
            stream(
                image,
                identity,
                ids,
                (2, 2, 2),
                mx=cast("StandInRuntime[HostArray, type[np.float32]]", HostArrays),
            )
    planes: list[int] = []

    def finalized(plane: int, scores: HostArray, count: HostArray) -> None:
        """Record finalized planes during validation."""
        planes.append(plane)

    for predictor, ids, weight, message in (
        (nonfinite, [3], None, "nonfinite scores"),
        (identity, [3, 1], None, "admitted channel/spatial shape"),
        (identity, [3], np.zeros((2, 2, 2), np.float32), "missing coverage"),
    ):
        planes.clear()
        with pytest.raises(ValueError, match=message):
            stream(
                image,
                predictor,
                ids,
                (2, 2, 2),
                weight=weight,
                finalized=finalized,
                mx=cast("StandInRuntime[HostArray, type[np.float32]]", HostArrays),
            )
        assert planes == ([] if len(ids) == 2 else [0, 1])


@pytest.mark.parametrize("value", [np.nan, np.inf])
@pytest.mark.parametrize("reduce_labels", [True, False], ids=["labels", "scores"])
def test_automatic_stream_checks_nonfinite_padding(value: float, reduce_labels: bool) -> None:
    """Reject nonfinite scores even in padded output regions."""
    image = np.zeros((1, 1, 1, 1, 1), dtype=np.float32)

    def predict(patch: HostArray, coordinates: PatchCoordinates) -> HostArray:
        """Place a nonfinite score in a synthetic patch."""
        result = patch.copy()
        result[0, 0, -1, -1, -1] = value
        return result

    planes: list[int] = []

    def finalized(plane: int, scores: HostArray, count: HostArray) -> None:
        """Record finalized planes for the padding case."""
        planes.append(plane)
        assert np.isfinite(scores[:, :1, :1]).all()

    stream = rolling_logits if reduce_labels else stream_scores
    with pytest.raises(ValueError, match="nonfinite scores"):
        stream(image, predict, [3], (2, 2, 2), finalized=finalized, mx=HostArrays)
    assert planes == [0, 1]


def test_boundary_click_returns_upstream_negative_sentinel() -> None:
    """Match the source sentinel for a boundary click."""
    data, _ = load_fixture("point_window_overlap")
    predictor = PointWindowPredictor(
        PointStandIn(801, mx=cast("StandInRuntime[HostArray, type[np.float32]]", HostArrays)),
        mx=HostArrays,
        metrics={"patch_seconds": []},
    )
    previous = cast("HostArray", data["previous"]).copy()
    actual = point_based_window_inferer(
        cast("HostArray", data["image"]),
        cast("Sequence[int]", data["roi"]),
        predictor,
        np.array([[[0, 0, 0]]], np.float64),
        np.array([[1]]),
        prev_mask=previous,
        prompt_class=cast("HostArray", data["prompt"]),
        transpose=True,
        mx=HostArrays,
    )
    assert (actual[:, :, :6, :6, :6] == -9999).all()
    assert np.array_equal(previous, cast("HostArray", data["previous"]))
    with pytest.raises(ValueError, match="no prediction windows"):
        point_based_window_inferer(
            cast("HostArray", data["image"]),
            cast("Sequence[int]", data["roi"]),
            predictor,
            cast("HostArray", data["points"]),
            cast("HostArray", data["codes"]),
            point_start=3,
            mx=HostArrays,
        )


@pytest.mark.parametrize("kind", ["automatic", "point"])
def test_full_network_window_wiring_on_metal(kind: str, mx: CoreRuntime) -> None:
    """Exercise native full-network window wiring using recorded fixtures."""
    name = "full_window_constant" if kind == "automatic" else "full_point_window_overlap"
    data, info = load_fixture(name)
    weights = {
        k: cast("ArrayBoundary[array]", mx).array(v)
        for k, v in random_weights(
            {k: tuple(s) for k, s in cast("GraphMetadata", info)["weight_shapes"].items()},
            int(cast("HostArray", data["weight_seed"])),
        ).items()
    }
    if kind == "automatic":
        automatic_case(
            data,
            AutomaticModel(weights, cast("HostArray", data["class_ids"]).tolist(), mx=mx),
            mx,
            native=True,
        )
    else:
        point_case(data, PointModel(weights, mx=mx), mx, native=True)
