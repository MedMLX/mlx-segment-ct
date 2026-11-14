"""MONAI-recorded interactive parity; no reference framework is imported here."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Unpack, cast

import numpy as np
import pytest
from reference_support import (
    HostArrays,
    assert_restored,
    dense_restoration,
    empty_weights,
    load_fixture,
    load_nifti,
    native,
    seeded_weights,
)

from mlx_segment_ct import (
    InteractivePrompt,
    PointWindowPredictor,
    interactive_labels,
    interactive_logits,
    medmlx,
    transform_points,
)
from mlx_segment_ct._types import (
    Checkpoint,
    CoreRuntime,
    HostArray,
    JointPredictor,
    Metrics,
    PointPredictor,
    RunArguments,
)
from mlx_segment_ct.bundle_transforms import load_volume, preprocess
from mlx_segment_ct.interactive import interactive_scores
from mlx_segment_ct.labels import label_rules

if TYPE_CHECKING:
    from oracle_types import (
        CasesMetadata,
        NumericFixtures,
        PredictionKeywordArguments,
        RunnerMetadata,
        SeededGraphMetadata,
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


def fixture(name: str, case: str) -> tuple[NumericFixtures, RunnerMetadata]:
    """Select one named case from an interactive reference fixture."""
    arrays, info = load_fixture(f"interactive_{name}")
    prefix = f"{case}::"
    data = {k.removeprefix(prefix): v for k, v in arrays.items() if k.startswith(prefix)}
    return data, cast("RunnerMetadata", cast("CasesMetadata", info)["cases"][case])


class RecordedPoint:
    def __init__(self, arrays: NumericFixtures):
        """Replay recorded point-model outputs, checking each call's inputs."""
        self.arrays = arrays
        self.calls = 0

    def __call__(
        self, image: HostArray, positions: HostArray, codes: HostArray, target: int
    ) -> HostArray:
        i = self.calls
        assert np.array_equal(positions, self.arrays[f"normalized_{i}"])
        assert np.array_equal(codes, self.arrays[f"patch_codes_{i}"])
        self.calls += 1
        return self.arrays[f"point_output_{i}"].copy()


class RecordedAutomatic:
    def __init__(self, arrays: NumericFixtures):
        """Replay recorded automatic-model outputs."""
        self.arrays = arrays
        self.calls = 0

    def __call__(self, image: HostArray) -> HostArray:
        self.calls += 1
        return self.arrays[f"auto_output_{self.calls - 1}"].copy()


def replay_models(
    monkeypatch: pytest.MonkeyPatch,
    point: Callable[..., object],
    automatic: Callable[..., object] | None = None,
) -> None:
    """Route the runner's native boundaries to host arrays and recorded models."""
    monkeypatch.setattr("medmlx_core.runtime.import_mlx", lambda: HostArrays)
    monkeypatch.setattr(medmlx, "_load_point_weights", empty_weights)
    monkeypatch.setattr(medmlx, "load_weights", empty_weights)
    monkeypatch.setattr("mlx_segment_ct.mlx_point.PointModel", point)
    def joint(
        point: PointPredictor[HostArray], automatic: Callable[[HostArray], HostArray]
    ) -> JointPredictor[HostArray]:
        """Replay both captured native decoder boundaries on their original inputs."""
        def predict(
            image: HostArray, positions: HostArray, codes: HostArray, target: int
        ) -> tuple[HostArray, HostArray]:
            return automatic(image), point(image, positions, codes, target)
        return predict
    monkeypatch.setattr("mlx_segment_ct.mlx_point.SharedPointAutomatic", joint)
    if automatic is not None:
        monkeypatch.setattr("mlx_segment_ct.mlx_network.AutomaticModel", automatic)


def touch_weights(directory: Path) -> Path:
    """Create empty files with the expected weight names."""
    directory.mkdir(exist_ok=True)
    for file in (medmlx.AUTOMATIC_WEIGHTS, medmlx.CORRECTION_WEIGHTS):
        (directory / file).touch()
    return directory


@pytest.mark.parametrize("name", WINDOW_CASES)
def test_windows_and_labels_match_monai(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Compare interactive windows and labels with MONAI recordings."""
    data, info = fixture("windows", name)
    image, points, codes = (data[k].copy() for k in ("image", "points", "codes"))
    prompt = InteractivePrompt(
        tuple(tuple(p) for p in data["points"][0]), tuple(data["codes"][0].tolist()), info["target"]
    )
    model = RecordedPoint(data)
    auto = RecordedAutomatic(data) if prompt.uses_automatic else None
    trace: list[int] = []
    call = "__call__"  # a getattr keeps the generic method's unknown type out of strict checks
    original = cast("Callable[..., HostArray]", getattr(PointWindowPredictor, call))

    def predict(
        self: PointWindowPredictor[HostArray],
        image: HostArray,
        **kwargs: Unpack[PredictionKeywordArguments],
    ) -> HostArray:
        index = len(trace)
        assert np.array_equal(image, data[f"patch_{index}"])
        bounds = [[s.start, s.stop] for s in kwargs["patch_coords"][0][-3:]]
        assert np.array_equal(bounds, data[f"windows_{index}"])
        trace.append(index)
        return original(self, image, **kwargs)

    monkeypatch.setattr(PointWindowPredictor, "__call__", predict)
    metrics: Metrics = {"patch_seconds": []}
    output = interactive_logits(
        image,
        prompt,
        model,
        mx=HostArrays,
        automatic_model=auto,
        roi_size=info["roi"],
        metrics=metrics,
    )
    assert output.dtype == data["logits"].dtype == np.float32
    assert np.array_equal(output, data["logits"], equal_nan=True)
    assert np.array_equal(interactive_labels(output, prompt), data["labels"], equal_nan=True)
    assert np.array_equal(prompt.arrays()[1], data["effective_codes"])
    assert len(trace) == sum(k.startswith("patch_") and k[6:].isdigit() for k in data)
    assert model.calls == sum(k.startswith("point_output_") for k in data)
    for actual, key in ((image, "image"), (points, "points"), (codes, "codes")):
        assert np.array_equal(actual, data[key]), f"{key} was modified"


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
    replay_models(monkeypatch, lambda *a, **k: point, lambda *a, **k: auto)
    monkeypatch.setattr(medmlx, "CTMR_ROI_SIZE" if ctmr else "ROI_SIZE", tuple(info["roi"]))
    kwargs = RunArguments(
        image_path=source,
        weights_path=touch_weights(tmp_path / "weights"),
        output_dir=tmp_path / "out",
        interactive=True,
        points=data["source_points"][0].tolist(),
        point_labels=data["codes"][0].tolist(),
        label_prompt=None if info["target"] is None else [info["target"]],
        checkpoint=cast(Checkpoint, checkpoint),
        restoration="bundle",
    )
    result = medmlx.run(**kwargs)
    saved = load_nifti(result["outputs"]["mask"])
    assert np.array_equal(np.asanyarray(saved.dataobj), data["mask"])
    assert np.array_equal(saved.affine, data["mask_affine"])
    assert saved.get_data_dtype() == np.float32

    # By default the post-processed scores are restored at source resolution.
    point.calls = auto.calls = 0
    kwargs["restoration"], kwargs["overwrite"] = "scores", True
    scored = medmlx.run(**kwargs)
    field = interactive_scores(data["logits"], transformed)
    labels, interior, margin = dense_restoration(
        field, prepared.affine, affine, volume.shape[1:], [original.output_label]
    )
    actual = cast("HostArray", np.asanyarray(load_nifti(scored["outputs"]["mask"]).dataobj))
    assert_restored(actual, (np.nan_to_num(labels, nan=255.0), interior, margin), name)
    # Uncovered voxels are those whose nearest preprocessed voxel is uncovered, as upstream.
    assert np.array_equal(actual == 255, data["mask"] == 255)


@pytest.mark.parametrize("target", (None, 26, 133))
def test_seeded_mlx_point_decoder_against_monai(target: int | None, mx: CoreRuntime) -> None:
    """Compare a seeded native point decoder with the MONAI reference."""
    from mlx_segment_ct.correction import normalized_point_positions
    from mlx_segment_ct.mlx_point import PointModel, correction_shapes

    data, info = load_fixture("interactive_decoder")
    graph = cast("SeededGraphMetadata", info)
    weights = seeded_weights(graph["weight_shapes"], graph["weight_seed"], mx)
    # Only decoder math is exercised; constructor's unused encoder upsamplers need shapes.
    for name, shape in correction_shapes().items():
        if ".upsample.deconv.weight" in name:
            weights[name] = mx.zeros(shape)
    positions = normalized_point_positions(data["points"], tuple(data["features"].shape[2:]))
    result = PointModel(weights, mx=mx).decode(
        native(mx, data["features"]),
        native(mx, positions),
        native(mx, data[f"codes_{target}"]),
        132 if target is None else target,
    )
    mx.eval(result)
    host = cast("HostArray", np.asarray(result))
    assert host.dtype == np.float32
    np.testing.assert_allclose(host, data[f"output_{target}"], rtol=0, atol=1e-6)


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
    model, auto = RecordedPoint(data), RecordedAutomatic(data)
    replay_models(monkeypatch, lambda *a, **k: model, lambda *a, **k: auto)
    monkeypatch.setattr(medmlx, "ROI_SIZE", tuple(info["roi"]))
    result = medmlx.run(
        image_path=source,
        weights_path=touch_weights(tmp_path),
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
