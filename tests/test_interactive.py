"""MONAI-recorded interactive parity; no reference framework is imported here."""

from __future__ import annotations

import json
import subprocess
import sys
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Unpack, cast, overload

import nibabel as nib
import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from reference_support import (
    assert_restored,
    dense_restoration,
    empty_weights,
    load_fixture,
    load_nifti,
)

if TYPE_CHECKING:
    from mlx.core import array

    from mlx_segment_ct._types import ArrayBoundary

from mlx_segment_ct import (
    InteractivePrompt,
    PointWindowPredictor,
    interactive_labels,
    interactive_logits,
    medmlx,
    transform_points,
)
from mlx_segment_ct._types import (
    CoreRuntime,
    HostArray,
    InteractiveDetails,
    Metrics,
    NiftiRuntime,
    NumericInput,
    RunArguments,
)
from mlx_segment_ct.bundle_transforms import load_volume, preprocess
from mlx_segment_ct.interactive import interactive_scores
from mlx_segment_ct.labels import label_rules
from scripts.fixture_inputs import random_weights
from scripts.oracle_types import (
    BoxRequest,
    CasesMetadata,
    FixtureMetadata,
    NumericFixtures,
    PredictionKeywordArguments,
    RawRunner,
    RequestInput,
    RunnerMetadata,
    SeededGraphMetadata,
    WindowMetadata,
)

WINDOW_CASES = (
    "overlap",
    "padded",
    "boundary",
    "negative",
    "padding_code",
    "regular",
    "special",
    "lung_special",
    "zero_shot",
    "boundary_then_valid",
    "boundary_class",
    "zero_shot_padding",
)


@overload
def fixture(
    name: Literal["decoder"], case: None = None
) -> tuple[NumericFixtures, SeededGraphMetadata]:
    """Return the decoder arrays and seeded graph metadata."""
    ...


@overload
def fixture(name: str, case: str) -> tuple[NumericFixtures, RunnerMetadata]:
    """Return arrays and metadata for one named case."""
    ...


@overload
def fixture(
    name: Literal["components", "windows", "runner", "correction_runner"], case: None = None
) -> tuple[NumericFixtures, FixtureMetadata]:
    """Return a complete interactive fixture and its metadata."""
    ...


def fixture(
    name: str, case: str | None = None
) -> tuple[NumericFixtures, FixtureMetadata | RunnerMetadata | SeededGraphMetadata]:
    """Load an interactive reference fixture and select its named case."""
    arrays, info = load_fixture(f"interactive_{name}")
    if case is None:
        return cast("NumericFixtures", arrays), cast("FixtureMetadata | SeededGraphMetadata", info)
    return {
        k.removeprefix(f"{case}::"): cast("HostArray", v)
        for k, v in arrays.items()
        if k.startswith(f"{case}::")
    }, cast("RunnerMetadata", cast("CasesMetadata", info)["cases"][case])


class RecordedArrays:
    """Only the array boundary for host-path tests; never used by runtime code."""

    @staticmethod
    def array(value: NumericInput) -> HostArray:
        """Convert recorded inputs to host arrays."""
        return cast("HostArray", np.asarray(value))

    @staticmethod
    def synchronize() -> None:
        """Accept synchronization for synchronous recorded arrays."""
        pass

    @staticmethod
    def reset_peak_memory() -> None:
        """Keep the native memory API available for this host-only array boundary."""

    @staticmethod
    def get_peak_memory() -> int:
        """Report no Metal allocation for recorded NumPy reference arrays."""
        return 0

    @staticmethod
    def eval(*value: object) -> None:
        """Accept evaluation for already materialized recorded arrays."""
        pass

    @staticmethod
    def default_device() -> str:
        """Identify the recorded-array test boundary."""
        return "recorded reference arrays"


class RecordedPoint:
    def __init__(self, arrays: NumericFixtures):
        """Initialize point predictions from recorded arrays."""
        self.arrays = arrays
        self.calls = 0

    def __call__(
        self, image: HostArray, positions: HostArray, codes: HostArray, target: int
    ) -> HostArray:
        """Return the next recorded point-model output."""
        i = self.calls
        assert np.array_equal(positions, self.arrays[f"normalized_{i}"])
        assert np.array_equal(codes, self.arrays[f"patch_codes_{i}"])
        self.calls += 1
        return self.arrays[f"point_output_{i}"].copy()


class RecordedAutomatic:
    def __init__(self, arrays: NumericFixtures):
        """Initialize automatic predictions from recorded arrays."""
        self.arrays = arrays
        self.calls = 0

    def __call__(self, image: HostArray) -> HostArray:
        """Return the next recorded automatic-model output."""
        i = self.calls
        self.calls += 1
        return self.arrays[f"auto_output_{i}"].copy()


def prompt_for(data: NumericFixtures, info: WindowMetadata | RunnerMetadata) -> InteractivePrompt:
    """Construct an interactive prompt from recorded case metadata."""
    return InteractivePrompt(
        tuple(tuple(p) for p in data["points"][0]), tuple(data["codes"][0].tolist()), info["target"]
    )


@pytest.mark.parametrize("name", WINDOW_CASES)
def test_windows_and_labels_match_monai(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare interactive windows and labels with MONAI recordings."""
    data, info = fixture("windows", name)
    image, points, codes = (data[k].copy() for k in ("image", "points", "codes"))
    prompt = prompt_for(data, info)
    model = RecordedPoint(data)
    auto = RecordedAutomatic(data) if prompt.uses_automatic else None
    trace: list[int] = []
    metrics: Metrics = {"patch_seconds": []}

    call_method = "__call__"
    original = cast("Callable[..., HostArray]", getattr(PointWindowPredictor, call_method))

    def predict(
        self: PointWindowPredictor[HostArray],
        image: HostArray,
        **kwargs: Unpack[PredictionKeywordArguments],
    ) -> HostArray:
        """Replay a recorded window prediction and record the call."""
        index = len(trace)
        assert np.array_equal(image, data[f"patch_{index}"])
        bounds = [[s.start, s.stop] for s in kwargs["patch_coords"][0][-3:]]
        assert np.array_equal(bounds, data[f"windows_{index}"])
        trace.append(index)
        return original(self, image, **kwargs)

    monkeypatch.setattr(PointWindowPredictor, "__call__", predict)
    output = interactive_logits(
        image,
        prompt,
        model,
        mx=RecordedArrays,
        automatic_model=auto,
        roi_size=info["roi"],
        metrics=metrics,
    )
    assert output.dtype == data["logits"].dtype == np.float32
    assert np.array_equal(output, data["logits"], equal_nan=True)
    assert np.array_equal(interactive_labels(output, prompt), data["labels"], equal_nan=True)
    assert np.array_equal(prompt.arrays()[1], data["effective_codes"])
    assert len(metrics["patch_seconds"]) == sum(
        k.startswith("patch_") and k[6:].isdigit() for k in data
    )
    assert np.array_equal(image, data["image"])
    assert np.array_equal(points, data["points"])
    assert np.array_equal(codes, data["codes"])
    assert model.calls == sum(k.startswith("point_output_") for k in data)


@pytest.mark.parametrize(
    "name", ("disconnected", "both", "negative_only", "miss", "positive_mean", "empty_mean", "nan")
)
def test_component_postprocessing_matches_monai_including_mean_fill(name: str) -> None:
    """Compare component merging and mean-fill behavior with MONAI."""
    data, _ = fixture("components", name)
    prompt = InteractivePrompt(
        tuple(tuple(p) for p in data["points"][0]), tuple(data["codes"][0].tolist())
    )
    before = data["logits"].copy()
    output = interactive_labels(before, prompt)
    assert np.array_equal(output, data["labels"], equal_nan=True)
    assert np.array_equal(before, data["logits"], equal_nan=True)


def test_interactive_scores_rejects_malformed_and_nonfinite_logits() -> None:
    """Reject malformed and nonfinite interactive logits."""
    prompt = InteractivePrompt(((0.0, 0.0, 0.0),), (1,))
    logits = np.zeros((1, 1, 2, 2, 2), np.float32)
    with pytest.raises(InvalidInputError, match="shape"):
        interactive_scores(logits[0], prompt)
    infinite = logits.copy()
    infinite[0, 0, 0, 0, 0] = np.inf
    with pytest.raises(InvalidInputError, match="infinity"):
        interactive_scores(infinite, prompt)


@pytest.mark.parametrize(
    ("checkpoint", "name"),
    [
        *(("NV-Segment-CT", name) for name in ("ras_ties", "lps", "oblique")),
        # NV-Segment-CTMR: percentile-scaled MR, a non-cubic window and target 300.
        *(("NV-Segment-CTMR", name) for name in ("ras_ties", "lps", "oblique")),
    ],
)
def test_runner_matches_bundle_nifti_geometry_and_labels(
    checkpoint: str, name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare runner output geometry and labels with the bundle recording."""
    ctmr = checkpoint == "NV-Segment-CTMR"
    data, info = fixture("ctmr_runner" if ctmr else "runner", name)
    source = tmp_path / "ct.nii"
    source.write_bytes(data["nifti"].tobytes())
    volume, affine = load_volume(source)
    prepared = preprocess(volume, affine, intensity="percentiles" if ctmr else "range")
    original = InteractivePrompt(
        tuple(tuple(p) for p in data["source_points"][0]),
        tuple(data["codes"][0].tolist()),
        info["target"],
        label_rules(checkpoint),
    )
    transformed = transform_points(prepared, original)
    assert np.array_equal(prepared.image[None], data["image"])
    assert np.allclose(transformed.arrays()[0], data["points"], rtol=0, atol=1e-12)
    point, auto = RecordedPoint(data), RecordedAutomatic(data)

    def point_factory(*args: object, **kwargs: object) -> RecordedPoint:
        """Provide the recorded point-model stand-in."""
        return point

    def auto_factory(*args: object, **kwargs: object) -> RecordedAutomatic:
        """Provide the recorded automatic-model stand-in."""
        return auto

    monkeypatch.setattr("medmlx_core.runtime.import_mlx", lambda: RecordedArrays)
    monkeypatch.setattr(medmlx, "_load_point_weights", empty_weights)
    monkeypatch.setattr(medmlx, "load_weights", empty_weights)
    monkeypatch.setattr(medmlx, "CTMR_ROI_SIZE" if ctmr else "ROI_SIZE", tuple(info["roi"]))
    monkeypatch.setattr("mlx_segment_ct.mlx_point.PointModel", point_factory)
    monkeypatch.setattr("mlx_segment_ct.mlx_network.AutomaticModel", auto_factory)
    weights = tmp_path / "weights"
    weights.mkdir()
    for file in (medmlx.AUTOMATIC_WEIGHTS, medmlx.CORRECTION_WEIGHTS):
        (weights / file).touch()
    kwargs = RunArguments(
        image_path=source,
        weights_path=weights,
        output_dir=tmp_path / "out",
        interactive=True,
        points=data["source_points"][0].tolist(),
        point_labels=data["codes"][0].tolist(),
        label_prompt=None if info["target"] is None else [info["target"]],
        checkpoint=checkpoint,
        restoration="bundle",
    )
    result = medmlx.run(**kwargs)
    json.dumps(result)
    saved = load_nifti(result["outputs"]["mask"])
    assert np.array_equal(np.asanyarray(saved.dataobj), data["mask"])
    assert np.array_equal(saved.affine, data["mask_affine"])
    assert saved.shape == load_nifti(source).shape
    assert saved.get_data_dtype() == np.float32
    assert result["details"]["mode"] == "interactive"
    assert result["details"]["checkpoint"] == checkpoint
    assert (
        cast("InteractiveDetails", result["details"])["automatic_branch"] == original.uses_automatic
    )
    assert result["details"]["patch_count"] == sum(
        k.startswith("patch_") and k[6:].isdigit() for k in data
    )
    assert (
        cast("InteractiveDetails", result["details"])["effective_point_labels"]
        == data["effective_codes"][0].tolist()
    )
    assert source.read_bytes() == data["nifti"].tobytes()
    before = Path(result["outputs"]["mask"]).read_bytes()
    with pytest.raises(InvalidInputError, match="overwrite"):
        medmlx.run(**kwargs)
    assert Path(result["outputs"]["mask"]).read_bytes() == before

    # By default the post-processed scores are restored at source resolution.
    point.calls = auto.calls = 0
    kwargs["restoration"], kwargs["overwrite"] = "scores", True
    scored = medmlx.run(**kwargs)
    assert scored["details"]["restoration"] == "scores"
    field = interactive_scores(data["logits"], transformed)
    labels, interior, margin = dense_restoration(
        field, prepared.affine, affine, volume.shape[1:], [original.output_label]
    )
    actual = cast("HostArray", np.asanyarray(load_nifti(scored["outputs"]["mask"]).dataobj))
    assert_restored(actual, (np.nan_to_num(labels, nan=255.0), interior, margin), name)
    # Uncovered voxels are those whose nearest preprocessed voxel is uncovered, as upstream.
    assert np.array_equal(actual == 255, data["mask"] == 255)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"points": []},
        {"point_labels": []},
        {"point_labels": [True]},
        {"point_labels": [1.1]},
        {"label_prompt": [True]},
        {"label_prompt": [1.5]},
        {"label_prompt": [255]},
        {"label_prompt": [2]},
        {"label_prompt": [1, 3]},
        {"points": [[1, 2]]},
        {"points": [[1, 2, float("nan")]]},
        {"points": [[10**400, 2, 3]]},
    ],
)
def test_interactive_request_rejects_invalid_prompt(kwargs: BoxRequest) -> None:
    """Reject malformed interactive request combinations."""
    values = RequestInput(prompt=None, label_prompt=None, points=[[1, 2, 3]], point_labels=[1])
    values.update(kwargs)
    with pytest.raises(InvalidInputError):
        medmlx.resolve_interactive_request(**values)


@given(
    st.lists(
        st.tuples(
            st.floats(-100, 100, allow_nan=False, allow_infinity=False),
            st.floats(-100, 100, allow_nan=False, allow_infinity=False),
            st.floats(-100, 100, allow_nan=False, allow_infinity=False),
        ),
        min_size=1,
        max_size=8,
    ),
    st.sampled_from([-1, 0, 1, 2, 3]),
)
def test_point_request_round_trip(points: list[tuple[float, float, float]], code: int) -> None:
    """Preserve valid click positions and codes through request parsing."""
    prompt = medmlx.resolve_interactive_request(None, None, points, [code] * len(points))
    coords, codes = prompt.arrays()
    rebuilt = medmlx.resolve_interactive_request(None, None, coords[0].tolist(), codes[0].tolist())
    assert rebuilt == prompt


@given(
    st.one_of(
        st.none(),
        st.integers(),
        st.text(),
        st.lists(st.one_of(st.none(), st.integers(), st.text()), max_size=5),
    )
)
def test_point_parser_no_crash_on_malformed_backing_data(points: object) -> None:
    """Exercise malformed point backing data without an unrelated crash."""
    with suppress(InvalidInputError):
        medmlx.resolve_interactive_request(None, None, points, [1])


def test_new_imports_load_neither_mlx_nor_reference_frameworks() -> None:
    """Verify lightweight imports avoid native and source-only frameworks."""
    code = """
import sys
import mlx_segment_ct.medmlx
from mlx_segment_ct import InteractivePrompt, interactive_logits, interactive_labels
assert not {'mlx', 'monai', 'torch'} & {n.split('.')[0] for n in sys.modules}
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_explicit_mode_rejects_conflicts_before_loading_weights(tmp_path: Path) -> None:
    """Reject conflicting interactive modes before weight loading."""
    path = tmp_path / "ct.nii"
    cast("NiftiRuntime", nib).save(
        cast("NiftiRuntime", nib).Nifti1Image(np.ones((4, 4, 4), np.float32), np.eye(4)), str(path)
    )
    base: dict[str, object] = dict(
        image_path=path,
        weights_path=tmp_path,
        output_dir=tmp_path / "out",
        points=[[1, 1, 1]],
        point_labels=[1],
        interactive=True,
    )
    for change in (
        {"previous_logits_path": path},
        {"segment_all": True},
        {"point_start": 1},
        {"interactive": "yes"},
        {"point_start": 0.0},
    ):
        request: dict[str, object] = dict(base)
        request.update(change)
        with pytest.raises(InvalidInputError):
            cast("RawRunner", medmlx.run)(**request)
    with pytest.raises(AssetNotReadyError, match=r"automatic\.safetensors"):
        cast("RawRunner", medmlx.run)(**base)
    (tmp_path / medmlx.AUTOMATIC_WEIGHTS).touch()
    with pytest.raises(AssetNotReadyError, match=r"correction\.safetensors"):
        cast("RawRunner", medmlx.run)(**base)
    assert not (tmp_path / "out").exists()


def test_network_failure_does_not_publish(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid publishing an output when network inference fails."""
    data, info = fixture("runner", "ras_ties")
    path = tmp_path / "ct.nii"
    path.write_bytes(data["nifti"].tobytes())
    for file in (medmlx.AUTOMATIC_WEIGHTS, medmlx.CORRECTION_WEIGHTS):
        (tmp_path / file).touch()
    monkeypatch.setattr("medmlx_core.runtime.import_mlx", lambda: RecordedArrays)
    monkeypatch.setattr(medmlx, "_load_point_weights", empty_weights)
    monkeypatch.setattr(medmlx, "load_weights", empty_weights)
    monkeypatch.setattr(medmlx, "ROI_SIZE", tuple(info["roi"]))

    def broken(*args: object, **kwargs: object) -> None:
        """Raise a synthetic decoder failure."""
        raise RuntimeError("decoder failed")

    monkeypatch.setattr("mlx_segment_ct.mlx_point.PointModel", broken)
    with pytest.raises(ModelExecutionError, match="decoder failed"):
        medmlx.run(
            image_path=path,
            weights_path=tmp_path,
            output_dir=tmp_path / "out",
            interactive=True,
            points=data["source_points"][0].tolist(),
            point_labels=[1],
        )
    assert not (tmp_path / "out" / medmlx.MASK_FILENAME).exists()


@pytest.fixture(scope="module")
def metal() -> CoreRuntime:
    """Provide the native runtime when a subprocess Metal probe succeeds."""
    probe = subprocess.run(
        [sys.executable, "-c", "import mlx.core as mx; assert mx.metal.is_available()"],
        capture_output=True,
    )
    if probe.returncode:
        pytest.skip("Seeded MLX decoder parity requires accessible Apple Silicon Metal")
    from medmlx_core.runtime import import_mlx

    return cast("CoreRuntime", import_mlx())


@pytest.mark.parametrize("target", (None, 26, 133))
def test_seeded_mlx_point_decoder_against_monai(target: int | None, metal: CoreRuntime) -> None:
    """Compare a seeded native point decoder with the MONAI reference."""
    from mlx_segment_ct.correction import normalized_point_positions
    from mlx_segment_ct.mlx_point import PointModel, correction_shapes

    data, info = fixture("decoder")
    shapes = {k: tuple(v) for k, v in info["weight_shapes"].items()}
    weights = {
        k: cast("ArrayBoundary[array]", metal).array(v)
        for k, v in random_weights(shapes, info["weight_seed"]).items()
    }
    # Only decoder math is exercised; constructor's unused encoder upsamplers need shapes.
    for name, shape in correction_shapes().items():
        if ".upsample.deconv.weight" in name:
            weights[name] = metal.zeros(shape)
    model = PointModel(weights, mx=metal)
    positions = normalized_point_positions(data["points"], tuple(data["features"].shape[2:]))
    result = model.decode(
        cast("ArrayBoundary[array]", metal).array(data["features"]),
        cast("ArrayBoundary[array]", metal).array(positions),
        cast("ArrayBoundary[array]", metal).array(data[f"codes_{target}"]),
        132 if target is None else target,
    )
    metal.eval(result)
    assert cast("HostArray", np.asarray(result)).dtype == np.float32
    np.testing.assert_allclose(
        cast("HostArray", np.asarray(result)), data[f"output_{target}"], rtol=0, atol=1e-6
    )


@pytest.mark.parametrize("name", ("ras_ties", "lps", "sagittal", "long"))
def test_previous_logits_runner_matches_upstream(
    name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Compare correction with previous logits against the source runner."""
    data, info = fixture("correction_runner", name)
    source = tmp_path / "ct.nii"
    source.write_bytes(data["nifti"].tobytes())
    previous = tmp_path / "previous.npy"
    np.save(previous, data["previous"])
    before = previous.read_bytes()
    model = RecordedPoint(data)
    auto = RecordedAutomatic(data)

    def point_factory(*args: object, **kwargs: object) -> RecordedPoint:
        """Provide the recorded correction-model stand-in."""
        return model

    def auto_factory(*args: object, **kwargs: object) -> RecordedAutomatic:
        """Provide the recorded automatic-model stand-in for correction."""
        return auto

    monkeypatch.setattr("medmlx_core.runtime.import_mlx", lambda: RecordedArrays)
    monkeypatch.setattr(medmlx, "_load_point_weights", empty_weights)
    monkeypatch.setattr(medmlx, "load_weights", empty_weights)
    monkeypatch.setattr("mlx_segment_ct.mlx_network.AutomaticModel", auto_factory)
    monkeypatch.setattr(medmlx, "ROI_SIZE", tuple(info["roi"]))
    monkeypatch.setattr("mlx_segment_ct.mlx_point.PointModel", point_factory)
    for file in (medmlx.AUTOMATIC_WEIGHTS, medmlx.CORRECTION_WEIGHTS):
        (tmp_path / file).touch()
    result = medmlx.run(
        image_path=source,
        weights_path=tmp_path,
        output_dir=tmp_path / "out",
        previous_logits_path=previous,
        points=data["source_points"][0].tolist(),
        point_labels=data["codes"][0].tolist(),
        label_prompt=[cast("int", info["target"])],
        point_start=info["point_start"],
        restoration="bundle",
    )
    saved = load_nifti(result["outputs"]["mask"])
    assert np.array_equal(np.asanyarray(saved.dataobj), data["mask"])
    assert np.array_equal(saved.affine, data["mask_affine"])
    assert saved.get_data_dtype() == np.float32
    assert result["details"]["mode"] == "correction"
    assert cast("InteractiveDetails", result["details"])["automatic_branch"] is True
    assert cast("InteractiveDetails", result["details"])["point_start"] == info["point_start"]
    assert previous.read_bytes() == before
    assert source.read_bytes() == data["nifti"].tobytes()


@pytest.mark.parametrize("change", ["shape", "dtype", "inf"])
def test_previous_logits_rejects_incompatible_grid(change: object, tmp_path: Path) -> None:
    """Reject previous logits on an incompatible preprocessed grid."""
    values = np.zeros((1, 1, 4, 4, 4), np.float32)
    if change == "shape":
        values = values[..., :3]
    elif change == "dtype":
        values = values.astype(np.float64)
    else:
        values[0, 0, 0, 0, 0] = np.inf
    path = tmp_path / "previous.npy"
    np.save(path, values)
    with pytest.raises(InvalidInputError):
        medmlx.previous_logits(path, (1, 4, 4, 4))
