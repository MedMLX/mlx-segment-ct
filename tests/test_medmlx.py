"""MedMLX runner contracts and recorded upstream bundle parity."""

from __future__ import annotations

import gzip
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, NoReturn, cast

import medmlx_core.weights as core_weights
import nibabel as nib
import numpy as np
import pytest
from fixture_inputs import AutomaticModelStand
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from medmlx_core.outputs import validate_outputs
from reference_support import (
    HostArrays,
    assert_restored,
    dense_restoration,
    empty_weights,
    load_fixture,
    load_nifti,
    seeded_weights,
)
from scipy import ndimage

from mlx_segment_ct import additional_shapes, medmlx, required_shapes
from mlx_segment_ct._types import (
    ArrayValue,
    CoordinatePredictor,
    CoreRuntime,
    HostArray,
    Metrics,
    NiftiImage,
    NiftiRuntime,
    RollingMetricsBuilder,
)
from mlx_segment_ct.bundle_transforms import (
    base_axis,
    load_volume,
    preprocess,
    save_label_map,
    scale_intensity_percentiles,
)
from mlx_segment_ct.labels import (
    CTMR_LABELS,
    CTMR_RULES,
    LABELS,
    class_ids_for_prompt,
    effective_point_labels,
    resolve_label_name,
)
from mlx_segment_ct.restoration import ScoreRestorer

if TYPE_CHECKING:
    from oracle_types import (
        CasesMetadata,
        CtmrMetadata,
        CtmrSegmentMetadata,
        GeometryMetadata,
        GridMetadata,
        NumericFixtures,
        RawRunner,
        RequestsMetadata,
        SegmentMetadata,
    )

GEOMETRIES = (
    "ras_ties",
    "lps",
    "sagittal",
    "iso15",
    "near15",
    "qform_only",
    "pixdim_mismatch",
    "oblique",
    "long",
)


def write_nifti(path: Path, raw: HostArray) -> Path:
    """Write compressed NIfTI bytes for a recorded fixture."""
    path.write_bytes(gzip.compress(raw.tobytes(), mtime=0))
    return path


def save_volume(path: Path, data: HostArray, affine: HostArray, *, coded: bool = True) -> Path:
    """Create and save an image with the supplied data and affine."""
    image = cast("NiftiImage", nib.Nifti1Image(data, affine))
    image.set_qform(affine, code=1 if coded else 0)
    image.set_sform(affine, code=1 if coded else 0)
    cast("NiftiRuntime", nib).save(image, str(path))
    return path


def assert_saved(path: Path, data: NumericFixtures, prefix: str) -> None:
    """Compare saved output data and geometry with recorded values."""
    saved = load_nifti(str(path))
    values = np.asanyarray(saved.dataobj)
    expected = data[f"{prefix}_data"]
    assert values.dtype == expected.dtype, f"{prefix}: dtype changed"
    assert np.array_equal(values, expected), f"{prefix}: mask differs from MONAI"
    header = np.frombuffer(saved.header.binaryblock, dtype=np.uint8)
    assert np.array_equal(header, data[f"{prefix}_header"]), f"{prefix}: NIfTI header differs"


@pytest.fixture
def weights_dir(tmp_path: Path) -> Path:
    """Create a fixture directory containing the expected weight filenames."""
    directory = tmp_path / "weights"
    directory.mkdir()
    for name in (medmlx.AUTOMATIC_WEIGHTS, medmlx.CORRECTION_WEIGHTS):
        (directory / name).touch()
    return directory


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace model boundaries with deterministic host stand-ins."""
    monkeypatch.setattr("medmlx_core.runtime.import_mlx", lambda: HostArrays)
    monkeypatch.setattr(medmlx, "load_weights", empty_weights)
    monkeypatch.setattr("mlx_segment_ct.mlx_network.AutomaticModel", AutomaticModelStand)
    from mlx_segment_ct.streaming import rolling_logits

    def host_rolling(
        inputs: ArrayValue,
        predictor: CoordinatePredictor[HostArray],
        class_ids: list[int],
        roi_size: tuple[int, ...],
        *,
        overlap: float = 0.25,
        finalized: Callable[[int, HostArray, HostArray], None] | None = None,
        metrics: Metrics | RollingMetricsBuilder | None = None,
    ) -> HostArray:
        return rolling_logits(
            inputs,
            predictor,
            class_ids,
            roi_size,
            overlap=overlap,
            finalized=finalized,
            metrics=metrics,
            mx=HostArrays,
        )

    monkeypatch.setattr("mlx_segment_ct.streaming.rolling_logits", host_rolling)


def test_prompt_resolution_matches_upstream() -> None:
    """Compare the label vocabulary and request prompt resolution with source recordings."""
    assert [value for value, _ in LABELS] == list(range(255))
    data, info = load_fixture("medmlx_requests")
    for i, request in enumerate(cast("RequestsMetadata", info)["requests"]):
        expected = tuple(dict.fromkeys(data[f"labels_{i}"].tolist()))
        if any(isinstance(value, str) for value in request):
            if resolve_label_name(cast("str", request[0])) > 132:
                plan = medmlx.resolve_interactive_request(request, None, [[1, 2, 3]], [1])
                assert plan.class_label == expected[0]
            else:
                plan = medmlx.resolve_request(segment_all=False, prompt=request, label_prompt=None)
                assert expected == class_ids_for_prompt(plan.label_ids)
        elif cast("list[int]", request)[0] <= 132:
            assert class_ids_for_prompt(cast("list[int]", request)) == expected
        if len(request) == 1 and isinstance(request[0], int):
            codes = effective_point_labels(request[0], [0, 1, -1, 2, 3])
            assert codes == data[f"codes_{i}"][0].tolist()


def test_ctmr_vocabulary_requests_and_modalities_match_pinned_bundle() -> None:
    """Compare CTMR labels, requests, and modality mapping with the bundle."""
    assert [value for value, _ in CTMR_LABELS] == list(range(346))
    data, info = load_fixture("medmlx_ctmr")
    metadata = cast("CtmrMetadata", info)
    for modality in metadata["modalities"]:
        expected = tuple(data[f"everything::{modality}"].tolist())
        assert class_ids_for_prompt(None, CTMR_RULES, modality) == expected
    assert CTMR_RULES.everything_labels() == CTMR_RULES.everything_labels("MRI_BODY")
    for i, request in enumerate(metadata["requests"]):
        expected = tuple(data[f"request::{i}"].tolist())
        if isinstance(request[0], str):
            assert (resolve_label_name(request[0], CTMR_RULES),) == expected
        else:
            assert class_ids_for_prompt(cast("list[int]", request), CTMR_RULES) == expected


@pytest.mark.parametrize("name", ["normal", "integers", "constant", "pair", "skewed", "large"])
def test_ctmr_percentile_scaling_matches_monai(name: str) -> None:
    """Compare CTMR percentile scaling with the MONAI recording."""
    data, _ = load_fixture("medmlx_ctmr")
    if name == "large":
        # Above one million voxels MONAI's percentile switches to np.percentile.
        scaled = scale_intensity_percentiles(data["percentile::large::input"])
        assert list(scaled.shape) == data["percentile::large::shape"].tolist()
        assert np.array_equal(scaled, data["percentile::large::output"])
        return
    scaled = scale_intensity_percentiles(data[f"percentile::{name}::input"])
    expected = data[f"percentile::{name}::output"]
    assert scaled.dtype == expected.dtype
    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize("name", ["lps", "sagittal", "oblique", "large"])
def test_ctmr_preprocessing_matches_upstream(name: str, tmp_path: Path) -> None:
    """Compare CTMR preprocessing with source recordings."""
    data, _ = load_fixture("medmlx_ctmr")
    raw = data[f"{name}::nifti"]
    volume, affine = load_volume(write_nifti(tmp_path / "ct.nii.gz", raw))
    prepared = preprocess(volume, affine, intensity="percentiles")
    assert prepared.image.dtype == np.float32
    if name == "large":
        assert list(prepared.image.shape) == data["large::shape"].tolist()
    assert np.array_equal(prepared.image, data[f"{name}::image"])
    assert np.array_equal(prepared.affine, data[f"{name}::affine"])


def test_affine_grid_base_matches_torch() -> None:
    """Compare normalized base grid coordinates with Torch recordings."""
    # Torch's linspace rounds each element once (a fused multiply-add); a plain
    # NumPy linspace differs by one ulp at some elements.
    data, info = load_fixture("medmlx_grid")
    for size in cast("GridMetadata", info)["sizes"]:
        expected = data[f"base_{size}"]
        assert np.array_equal(base_axis(size), expected), size
        assert base_axis(size).dtype == expected.dtype == np.float64


@pytest.mark.parametrize("name", GEOMETRIES)
def test_preprocessing_and_restoration_match_upstream(name: str, tmp_path: Path) -> None:
    """Compare geometry preprocessing and restoration with source recordings."""
    data, info = load_fixture("medmlx_geometry")
    case = cast("GeometryMetadata", cast("CasesMetadata", info)["cases"][name])
    source = write_nifti(tmp_path / f"{name}.nii.gz", data[f"{name}::nifti"])
    volume, affine = load_volume(source)
    prepared = preprocess(volume, affine)
    image = data[f"{name}::image"]
    assert prepared.image.dtype == image.dtype
    assert np.array_equal(prepared.image, image), "preprocessed image differs from MONAI"
    # All nine recorded upstream CPU affines agree bitwise on this host,
    # including oblique geometry; no BLAS tolerance is needed for these inputs.
    assert np.array_equal(prepared.affine, data[f"{name}::affine"])
    labels = data[f"{name}::labels"]
    original = labels.copy()
    restored, restored_affine = prepared.invert(labels)
    assert np.array_equal(labels, original), "Labels must be preserved"
    restored = np.nan_to_num(restored, nan=255.0)
    medmlx.validate_labels(restored, case["class_ids"])
    saved = tmp_path / "restored.nii.gz"
    save_label_map(saved, restored, restored_affine)
    assert_saved(saved, data, f"{name}::restored")
    assert load_nifti(str(saved)).shape == load_nifti(str(source)).shape


@pytest.mark.parametrize("name", GEOMETRIES)
def test_score_restoration_interpolates_scores_at_source_voxels(name: str, tmp_path: Path) -> None:
    """Verify score interpolation at original source-grid voxels."""
    data, _ = load_fixture("medmlx_geometry")
    source = write_nifti(tmp_path / f"{name}.nii.gz", data[f"{name}::nifti"])
    volume, affine = load_volume(source)
    prepared = preprocess(volume, affine)
    class_ids = [3, 1, 26]
    noise = np.random.default_rng(7).normal(size=(len(class_ids), *prepared.image.shape[1:]))
    scores = (ndimage.gaussian_filter(noise, (0, 1.5, 1.5, 1.5)) * 6).astype(np.float32)
    restorer = ScoreRestorer(prepared.restoration_grid(), class_ids)
    for index in range(scores.shape[1]):
        restorer.add_plane(index, scores[:, index])
    restored, restored_affine = restorer.finish()

    labels = np.asarray(class_ids, dtype=np.float32)[np.argmax(scores, axis=0)]
    bundle, bundle_affine = prepared.invert((labels * (scores.max(axis=0) > 0))[None])
    assert np.array_equal(restored_affine, bundle_affine)
    expected = dense_restoration(scores, prepared.affine, affine, volume.shape[1:], class_ids)
    assert_restored(restored[0], expected, name)
    # Voxels that nearest restoration places outside the foreground crop stay background.
    crop, _ = prepared.invert(np.ones((1, *scores.shape[1:]), dtype=np.float32))
    assert not restored[crop == 0].any()
    if name == "iso15":  # a 1.5 mm source grid is not resampled
        assert np.array_equal(restored, bundle)


@pytest.mark.parametrize("name", ["ras_ties", "sagittal", "oblique", "long"])
def test_segment_matches_upstream_worker(
    name: str, tmp_path: Path, weights_dir: Path, stand_ins: None
) -> None:
    """Compare automatic runner output with the source worker recording."""
    data, info = load_fixture("medmlx_segment")
    case = cast("SegmentMetadata", cast("CasesMetadata", info)["cases"][name])
    source = write_nifti(tmp_path / f"{name}.nii.gz", data[f"{name}::nifti"])
    result = medmlx.segment(
        model="mlx-segment-ct",
        image_path=str(source),
        weights_path=str(weights_dir),
        output_dir=str(tmp_path / "out"),
        classes=case["label_prompt"],
        restoration="bundle",
    )
    assert_saved(Path(result["outputs"]["mask"]), data, f"{name}::mask")
    labels = validate_outputs(result).artifacts["mask"].get("labels")
    assert labels is not None and labels[0] == {"value": 0, "name": "background"}
    class_ids = result["details"].get("class_ids")
    assert class_ids is not None
    assert {item["value"] for item in labels} == {0, *class_ids}


@pytest.mark.parametrize("name", ["lps", "oblique", "sagittal"])
def test_segment_ctmr_matches_upstream_worker(
    name: str,
    tmp_path: Path,
    weights_dir: Path,
    stand_ins: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare CTMR runner output with the source worker recording."""
    data, info = load_fixture("medmlx_ctmr_segment")
    case = cast("CtmrSegmentMetadata", cast("CasesMetadata", info)["cases"][name])
    # The sagittal case runs MRI_BODY's 50 classes in a smaller window.
    monkeypatch.setattr(medmlx, "CTMR_ROI_SIZE", tuple(case["roi"]))
    source = write_nifti(tmp_path / f"{name}.nii.gz", data[f"{name}::nifti"])
    result = medmlx.segment(
        model="mlx-segment-ctmr",
        image_path=str(source),
        weights_path=str(weights_dir),
        output_dir=str(tmp_path / "out"),
        classes=case["label_prompt"],
        restoration="bundle",
    )
    assert_saved(Path(result["outputs"]["mask"]), data, f"{name}::mask")


def ct_volume(shape: tuple[int, ...] = (24, 20, 12)) -> HostArray:
    """Crop the HCC-TACE-Seg contrast CT for runner tests."""
    source = os.environ.get(
        "MEDMLX_TEST_CT", str(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz")
    )
    image = load_nifti(source)
    data = np.asarray(image.dataobj, dtype=np.float32)
    crop = data[200 : 200 + shape[0], 200 : 200 + shape[1], 40 : 40 + shape[2]]
    if crop.shape != shape:
        raise ValueError(f"CT crop {crop.shape} does not match requested shape {shape}")
    return crop


REJECTIONS = {
    "bad-suffix": {"image_path": "not_nifti"},
    "out-of-range": {"classes": [512]},
    "non-boolean": {"overwrite": "yes"},
    "existing-output": {"output_dir": "existing"},
    "before-download": {"weights_path": None, "classes": [512]},
}


@pytest.mark.parametrize(
    ("model", "change"),
    [
        pytest.param(model, change, id=f"{model}-{name}")
        for model in medmlx.CHECKPOINTS
        for name, change in REJECTIONS.items()
    ],
)
def test_segment_rejects_invalid_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, model: str, change: dict[str, object]
) -> None:
    """Reject each category of malformed input before weights download or load."""

    def download(model_id: str, **options: object) -> NoReturn:
        raise AssertionError("arguments must be checked before any weight download")

    monkeypatch.setattr(core_weights, "weights_dir", download)
    image = save_volume(tmp_path / "ct.nii.gz", ct_volume(), np.eye(4))
    (tmp_path / "not_nifti").write_bytes(b"not nifti")
    (tmp_path / "existing").mkdir()
    (tmp_path / "existing" / medmlx.MASK_FILENAME).write_bytes(b"existing")
    kwargs: dict[str, object] = {
        "model": model,
        "image_path": image,
        "weights_path": tmp_path / "no-weights",
        "output_dir": tmp_path / "out",
    }
    for key, value in change.items():
        if key.endswith(("_path", "_dir")) and isinstance(value, str):
            value = tmp_path / value
        kwargs[key] = value
    with pytest.raises(InvalidInputError):
        cast("RawRunner", medmlx.segment)(**kwargs)


def test_loading_rejects_mismatched_and_corrupt_weights(tmp_path: Path, mx: CoreRuntime) -> None:
    """Reject missing, misshapen, wrongly typed, unexpected and corrupt checkpoint tensors."""
    weights = tmp_path / "automatic.safetensors"
    mx.save_safetensors(
        str(weights), {"image_encoder.encoder.conv_init.weight": mx.zeros((48, 1, 3, 3, 3))}
    )
    with pytest.raises(AssetNotReadyError):
        medmlx.load_weights(mx, weights, required_shapes())
    mx.save_safetensors(str(weights), {"class_head.mlp.0.bias": mx.zeros((47,))})
    with pytest.raises(AssetNotReadyError):
        medmlx.load_weights(mx, weights, {"class_head.mlp.0.bias": (48,)})
    mx.save_safetensors(str(weights), {"class_head.mlp.0.bias": mx.zeros((48,), dtype=mx.float16)})
    with pytest.raises(AssetNotReadyError):
        medmlx.load_weights(mx, weights, {"class_head.mlp.0.bias": (48,)})
    with pytest.raises(AssetNotReadyError):
        medmlx.load_weights(mx, weights, {})
    weights.write_bytes(b"corrupt")
    with pytest.raises(AssetNotReadyError):
        medmlx.load_weights(mx, weights, {})


def test_segment_full_networks_end_to_end_on_metal(
    tmp_path: Path, mx: CoreRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run both registered runners through the full native networks on Metal."""
    weights = tmp_path / "weights"
    weights.mkdir()
    for name, shapes, seed in (
        (medmlx.AUTOMATIC_WEIGHTS, required_shapes(), 501),
        (medmlx.CORRECTION_WEIGHTS, additional_shapes(), 502),
    ):
        mx.save_safetensors(str(weights / name), seeded_weights(shapes, seed, mx))
    monkeypatch.setattr(medmlx, "ROI_SIZE", (32, 32, 32))
    monkeypatch.setattr(medmlx, "CTMR_ROI_SIZE", (32, 32, 32))
    affine = np.diag([0.9, 0.9, 2.5, 1.0])
    source = save_volume(tmp_path / "ct.nii.gz", ct_volume((40, 36, 16)), affine)
    interactive = {"classes": [1], "points": [[20, 18, 8]], "point_labels": [1]}
    for index, (model, request, labels) in enumerate(
        (
            ("mlx-segment-ct", {"classes": [3, 1]}, {0, 1, 3}),
            ("mlx-segment-ct", interactive, {0, 1, 255}),
            ("mlx-segment-ctmr", {"classes": [300, 3]}, {0, 3, 300}),
        )
    ):
        result = cast("RawRunner", medmlx.segment)(
            model=model,
            image_path=source, weights_path=weights, output_dir=tmp_path / f"out{index}", **request
        )
        paths, artifacts = validate_outputs(result)
        mask = artifacts["mask"]
        assert (mask["kind"], mask.get("grid_of")) == ("label_map", "image_path")
        mask_labels = mask.get("labels")
        assert mask_labels is not None
        assert {item["value"] for item in mask_labels} == labels
        assert all(item["name"] for item in mask_labels)
        saved = load_nifti(paths["mask"])
        assert saved.shape == (40, 36, 16)
        assert np.allclose(saved.affine, affine, rtol=0, atol=1e-5)
        assert saved.get_data_dtype() == np.float32
        assert set(np.unique(np.asanyarray(saved.dataobj)).tolist()) <= labels
