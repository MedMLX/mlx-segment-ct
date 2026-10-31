"""MedMLX runner contracts and recorded upstream bundle parity."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast

from mlx_segment_ct._types import (
    ArrayValue,
    AutomaticDetails,
    CoordinatePredictor,
    CoreRuntime,
    HostArray,
    Metrics,
    NiftiImage,
    NiftiRuntime,
    RollingMetricsBuilder,
)

if TYPE_CHECKING:
    from mlx.core import array

    from mlx_segment_ct._types import ArrayBoundary
    from scripts.oracle_types import (
        CasesMetadata,
        CtmrMetadata,
        CtmrSegmentMetadata,
        FileImageOutputPaths,
        FileRunPaths,
        FixtureArrays,
        GeometryMetadata,
        GridMetadata,
        RawRunner,
        RequestsMetadata,
        SegmentMetadata,
    )

import gzip
import inspect
import json
import re
import subprocess
import sys
import tomllib
from importlib import import_module

import nibabel as nib
import numpy as np
import pytest
from medmlx_core.errors import AssetNotReadyError, InvalidInputError
from reference_support import (
    HostArrays,
    assert_restored,
    dense_restoration,
    empty_weights,
    load_fixture,
    load_nifti,
)
from scipy import ndimage

from mlx_segment_ct import additional_shapes, medmlx, required_shapes
from mlx_segment_ct.bundle_transforms import (
    Intensity,
    RestorationGrid,
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
    resolve_label_name,
)
from mlx_segment_ct.restoration import ScoreRestorer
from scripts.fixture_inputs import (
    AutomaticModelStand,
    random_weights,
)
from scripts.reference_bundle import large_mr, large_percentile_input, nifti_bytes

REPO = Path(__file__).resolve().parents[1]
FIXTURES = REPO / "tests" / "fixtures"
CTMR = "NV-Segment-CTMR"
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


class SpatialUnitHeader(Protocol):
    """Typed boundary for setting spatial units on a synthetic NIfTI header."""

    def set_xyzt_units(self, xyz: str) -> None:
        """Set the spatial unit code without changing temporal units."""
        ...


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


def assert_saved(path: Path, data: FixtureArrays, prefix: str) -> None:
    """Compare saved output data and geometry with recorded values."""
    saved = load_nifti(str(path))
    values = np.asanyarray(saved.dataobj)
    expected = cast("HostArray", data[f"{prefix}_data"])
    assert values.dtype == expected.dtype, f"{prefix}: dtype changed"
    assert np.array_equal(values, expected), f"{prefix}: mask differs from MONAI"
    header = np.frombuffer(saved.header.binaryblock, dtype=np.uint8)
    assert np.array_equal(header, cast("HostArray", data[f"{prefix}_header"])), (
        f"{prefix}: NIfTI header differs"
    )


@pytest.fixture
def weights_dir(tmp_path: Path):
    """Create a fixture directory containing the expected weight filenames."""
    directory = tmp_path / "weights"
    directory.mkdir()
    for name in (medmlx.AUTOMATIC_WEIGHTS, medmlx.CORRECTION_WEIGHTS):
        (directory / name).touch()
    return directory


@pytest.fixture
def stand_ins(monkeypatch: pytest.MonkeyPatch):
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
        """Run rolling score inference through the host array boundary."""
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


def test_pyproject_entry_point_and_model_declaration() -> None:
    """Verify the package entry point and registered model declaration."""
    pyproject = tomllib.loads((REPO / "pyproject.toml").read_text())
    entry_points = pyproject["project"]["entry-points"]["medmlx.models"]
    assert entry_points == {
        "mlx-segment-ct": "mlx_segment_ct.medmlx:MODEL",
        "mlx-segment-ctmr": "mlx_segment_ct.medmlx:CTMR_MODEL",
    }
    assert pyproject["project"]["name"] == "mlx-segment-ct"
    assert any(dep.startswith("nibabel") for dep in pyproject["project"]["dependencies"])
    model = medmlx.MODEL
    assert {"id", "label", "task", "run"} <= model.keys()
    assert re.fullmatch(r"[a-z0-9._-]+", model["id"])
    assert model["task"] == "segmentation"
    assert model["run"] == "mlx_segment_ct.medmlx:run"
    module, function = model["run"].split(":")
    assert getattr(import_module(module), function) is medmlx.run
    parameters = inspect.signature(medmlx.run).parameters
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in parameters.values())
    assert parameters["overwrite"].default is False
    assert {"image_path", "weights_path", "output_dir", "previous_logits_path"} <= parameters.keys()


def test_declaration_import_loads_no_mlx_or_reference_stack() -> None:
    """Verify model declaration imports do not eagerly load runtime frameworks."""
    code = (
        "import sys\n"
        "import mlx_segment_ct.medmlx as m\n"
        "assert m.MODEL['id'] == 'mlx-segment-ct'\n"
        "loaded = {name.split('.')[0] for name in sys.modules}\n"
        "assert not loaded & {'mlx', 'torch', 'monai'}, loaded\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


def test_label_vocabulary_matches_pinned_bundle() -> None:
    """Compare the label vocabulary with the pinned bundle."""
    assert [value for value, _ in LABELS] == list(range(255))


def test_prompt_resolution_matches_upstream() -> None:
    """Compare request prompt resolution with source recordings."""
    data, info = load_fixture("medmlx_requests")
    for i, request in enumerate(cast("RequestsMetadata", info)["requests"]):
        if any(isinstance(value, str) for value in request):
            from mlx_segment_ct.labels import resolve_label_name

            expected = tuple(dict.fromkeys(cast("HostArray", data[f"labels_{i}"]).tolist()))
            if resolve_label_name(cast("str", request[0])) > 132:
                plan = medmlx.resolve_interactive_request(request, None, [[1, 2, 3]], [1])
                assert plan.class_label == expected[0]
            else:
                plan = medmlx.resolve_request(segment_all=False, prompt=request, label_prompt=None)
                assert expected == class_ids_for_prompt(plan.label_ids)
        elif cast("list[int]", request)[0] <= 132:
            assert class_ids_for_prompt(cast("list[int]", request)) == tuple(
                dict.fromkeys(cast("HostArray", data[f"labels_{i}"]).tolist())
            )
        if len(request) == 1 and isinstance(request[0], int):
            from mlx_segment_ct.labels import effective_point_labels

            assert (
                effective_point_labels(request[0], [0, 1, -1, 2, 3])
                == cast("HostArray", data[f"codes_{i}"])[0].tolist()
            )


def test_ctmr_vocabulary_requests_and_modalities_match_pinned_bundle() -> None:
    """Compare CTMR labels, requests, and modality mapping with the bundle."""
    assert [value for value, _ in CTMR_LABELS] == list(range(346))
    data, info = load_fixture("medmlx_ctmr")
    metadata = cast("CtmrMetadata", info)
    for modality in metadata["modalities"]:
        expected = tuple(cast("HostArray", data[f"everything::{modality}"]).tolist())
        assert class_ids_for_prompt(None, CTMR_RULES, modality) == expected
    assert CTMR_RULES.everything_labels() == CTMR_RULES.everything_labels("MRI_BODY")
    for i, request in enumerate(metadata["requests"]):
        expected = tuple(cast("HostArray", data[f"request::{i}"]).tolist())
        if isinstance(request[0], str):
            assert (resolve_label_name(request[0], CTMR_RULES),) == expected
        else:
            assert class_ids_for_prompt(cast("list[int]", request), CTMR_RULES) == expected


@pytest.mark.parametrize("nifti2", [False, True])
@pytest.mark.parametrize("units", ["meter", "micron", "invalid"])
def test_load_volume_rejects_nonmillimeter_spatial_units(
    tmp_path: Path, nifti2: bool, units: str
) -> None:
    """Reject declared units that would silently mis-scale the 1.5 mm resampling grid."""
    data = np.zeros((3, 3, 3), dtype=np.float32)
    affine = np.eye(4, dtype=np.float64)
    constructor = nib.Nifti2Image if nifti2 else nib.Nifti1Image
    image = constructor(data, affine)
    if units == "invalid":
        image.header["xyzt_units"] = 4
    else:
        cast("SpatialUnitHeader", image.header).set_xyzt_units(units)
    path = tmp_path / "ct.nii"
    cast("NiftiRuntime", nib).save(cast("NiftiImage", image), str(path))
    with pytest.raises(InvalidInputError, match=r"spatial units.*millimeter|invalid NIfTI unit"):
        load_volume(path)


@pytest.mark.parametrize("units", ["mm", "unknown"])
def test_load_volume_accepts_millimeters_and_documented_unknown_units(
    tmp_path: Path, units: str
) -> None:
    """Preserve existing geometry for explicit millimeters and assumed millimeters."""
    data = np.arange(27, dtype=np.float32).reshape(3, 3, 3)
    affine = np.diag([0.8, 1.2, 1.5, 1.0])
    image = nib.Nifti1Image(data, affine)
    cast("SpatialUnitHeader", image.header).set_xyzt_units(units)
    path = tmp_path / "ct.nii"
    cast("NiftiRuntime", nib).save(cast("NiftiImage", image), str(path))
    loaded, loaded_affine = load_volume(path)
    np.testing.assert_array_equal(loaded[0], data)
    np.testing.assert_allclose(loaded_affine, affine, rtol=0, atol=1e-6)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("intensity", ["range", "percentiles"])
def test_preprocess_rejects_nonfinite_image_voxels(value: float, intensity: Intensity) -> None:
    """Reject invalid source intensities before either scaling mode can mask them."""
    image = np.full((1, 8, 8, 8), 100.0, dtype=np.float32)
    image[0, 4, 4, 4] = value
    with pytest.raises(InvalidInputError, match="non-finite"):
        preprocess(image, np.diag([1.5, 1.5, 1.5, 1.0]), intensity=intensity)


@pytest.mark.parametrize("name", ["normal", "integers", "constant", "pair", "skewed", "large"])
def test_ctmr_percentile_scaling_matches_monai(name: str) -> None:
    """Compare CTMR percentile scaling with the MONAI recording."""
    data, _ = load_fixture("medmlx_ctmr")
    if name == "large":
        # Above one million voxels MONAI's percentile switches to np.percentile.
        scaled = scale_intensity_percentiles(large_percentile_input())
        assert list(scaled.shape) == cast("HostArray", data["percentile::large::shape"]).tolist()
        assert np.array_equal(scaled, cast("HostArray", data["percentile::large::output"]))
        return
    scaled = scale_intensity_percentiles(cast("HostArray", data[f"percentile::{name}::input"]))
    expected = cast("HostArray", data[f"percentile::{name}::output"])
    assert scaled.dtype == expected.dtype
    assert np.array_equal(scaled, expected)


@pytest.mark.parametrize("name", ["lps", "sagittal", "oblique", "large"])
def test_ctmr_preprocessing_matches_upstream(name: str, tmp_path: Path) -> None:
    """Compare CTMR preprocessing with source recordings."""
    data, _ = load_fixture("medmlx_ctmr")
    raw = (
        nifti_bytes(large_mr(), np.diag([1.5, 1.5, 1.5, 1.0]), "both")
        if name == "large"
        else cast("HostArray", data[f"{name}::nifti"])
    )
    volume, affine = load_volume(write_nifti(tmp_path / "mr.nii.gz", raw))
    prepared = preprocess(volume, affine, intensity="percentiles")
    assert prepared.image.dtype == np.float32
    if name == "large":
        assert list(prepared.image.shape) == cast("HostArray", data["large::shape"]).tolist()
        assert np.array_equal(prepared.image, cast("HostArray", data["large::image"]))
    else:
        assert np.array_equal(prepared.image, cast("HostArray", data[f"{name}::image"]))
    assert np.array_equal(prepared.affine, cast("HostArray", data[f"{name}::affine"]))


def test_affine_grid_base_matches_torch() -> None:
    # Torch's linspace rounds each element once (a fused multiply-add); a plain
    # NumPy linspace differs by one ulp at some elements.
    """Compare normalized base grid coordinates with Torch recordings."""
    data, info = load_fixture("medmlx_grid")
    for size in cast("GridMetadata", info)["sizes"]:
        expected = cast("HostArray", data[f"base_{size}"])
        assert np.array_equal(base_axis(size), expected), size
        assert base_axis(size).dtype == expected.dtype == np.float64


@pytest.mark.parametrize("name", GEOMETRIES)
def test_preprocessing_and_restoration_match_upstream(name: str, tmp_path: Path) -> None:
    """Compare geometry preprocessing and restoration with source recordings."""
    data, info = load_fixture("medmlx_geometry")
    case = cast("GeometryMetadata", cast("CasesMetadata", info)["cases"][name])
    source = write_nifti(tmp_path / f"{name}.nii.gz", cast("HostArray", data[f"{name}::nifti"]))
    volume, affine = load_volume(source)
    prepared = preprocess(volume, affine)
    image = cast("HostArray", data[f"{name}::image"])
    assert prepared.image.dtype == image.dtype
    assert np.array_equal(prepared.image, image), "preprocessed image differs from MONAI"
    # All nine recorded upstream CPU affines agree bitwise on this host,
    # including oblique geometry; no BLAS tolerance is needed for these inputs.
    assert np.array_equal(prepared.affine, cast("HostArray", data[f"{name}::affine"]))
    labels = cast("HostArray", data[f"{name}::labels"])
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
    source = write_nifti(tmp_path / f"{name}.nii.gz", cast("HostArray", data[f"{name}::nifti"]))
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


@pytest.mark.parametrize(
    ("permutation", "reverse", "separable"),
    [
        ((0, 1, 2), False, False),
        ((0, 1, 2), False, True),
        ((1, 2, 0), False, True),
        ((2, 0, 1), False, True),
        ((0, 1, 2), True, True),
        ((1, 2, 0), True, True),
        ((2, 0, 1), True, True),
    ],
)
def test_score_restoration_takes_coverage_from_the_half_to_even_nearest_voxel(
    permutation: tuple[int, int, int], reverse: bool, separable: bool
) -> None:
    # Every odd source index lies on a half-voxel tie of a twice-as-coarse grid.
    """Verify restoration coverage follows half-to-even nearest-voxel selection."""
    transform = np.zeros((4, 4), dtype=np.float64)
    transform[3, 3] = 1
    for row, axis in enumerate(permutation):
        sign = -1 if reverse and row in (0, 2) else 1
        transform[row, axis] = sign * (1 if separable else 0.5)
        transform[row, 3] = 3 if sign < 0 else 0
    axes = tuple(np.arange(8, dtype=np.float64) / 2 for _ in range(3))
    grid = RestorationGrid(
        (8, 8, 8),
        np.eye(4),
        (4, 4, 4),
        (4, 4, 4),
        None,
        transform,
        (axes[0], axes[1], axes[2]) if separable else None,
    )
    field = np.full((1, 4, 4, 4), 3.0, dtype=np.float32)
    field[:, 2] = np.nan
    field[:, :, 2] = np.nan
    field[:, :, :, 1] = np.nan
    restorer = ScoreRestorer(grid, [9])
    for index in range(4):
        restorer.add_plane(index, field[:, index])
    restored, _ = restorer.finish()

    expected = np.empty((8, 8, 8), dtype=np.float32)
    for source in np.ndindex(expected.shape):
        # Python's independent scalar rounding also uses absolute half-to-even ties.
        near = tuple(
            round(3 - source[axis] / 2 if reverse and row in (0, 2) else source[axis] / 2)
            for row, axis in enumerate(permutation)
        )
        if any(index < 0 or index > 3 for index in near):
            expected[source] = 0
        else:
            expected[source] = np.nan if np.isnan(field[0, *near]) else 9
    assert np.array_equal(restored[0], expected, equal_nan=True)


@pytest.mark.parametrize("name", ["ras_ties", "sagittal", "oblique", "long"])
def test_run_segmentation_matches_upstream_worker(
    name: str,
    tmp_path: Path,
    weights_dir: Path,
    stand_ins: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Compare automatic runner output with the source worker recording."""
    data, info = load_fixture("medmlx_segment")
    case = cast("SegmentMetadata", cast("CasesMetadata", info)["cases"][name])
    source = write_nifti(tmp_path / f"{name}.nii.gz", cast("HostArray", data[f"{name}::nifti"]))
    output = tmp_path / "out"
    result = medmlx.run(
        image_path=str(source),
        weights_path=str(weights_dir),
        output_dir=str(output),
        label_prompt=case["label_prompt"],
        restoration="bundle",
    )
    json.dumps(result)
    mask = Path(result["outputs"]["mask"])
    assert result["outputs"] == {"mask": (output / medmlx.MASK_FILENAME).resolve().as_posix()}
    saved, reference = load_nifti(str(mask)), load_nifti(str(source))
    assert saved.shape == reference.shape
    assert np.allclose(saved.affine, reference.affine, rtol=0, atol=1e-4)
    assert saved.get_data_dtype() == np.float32
    assert set(np.unique(np.asanyarray(saved.dataobj)).tolist()) <= {0, *case["class_ids"]}
    assert_saved(mask, data, f"{name}::mask")
    details = result["details"]
    assert details["mode"] == "class_prompt"
    assert cast("AutomaticDetails", details)["class_ids"] == case["class_ids"]
    assert cast("AutomaticDetails", details)["padded_shape"] == case["padded_shape"]
    assert details["patch_count"] == case["patch_count"]
    assert details["preprocessed_shape"] == case["preprocessed_shape"]
    assert details["output_shape"] == list(reference.shape)
    assert [label["value"] for label in details["output_labels"]] == sorted({0, *case["class_ids"]})
    assert details["restoration"] == "bundle"

    # By default the same scores are restored at source resolution.
    import mlx_segment_ct.streaming as streaming

    stream = streaming.stream_scores
    planes: list[HostArray] = []

    def recording(
        inputs: ArrayValue,
        predictor: CoordinatePredictor[HostArray],
        class_ids: list[int],
        roi_size: tuple[int, ...],
        *,
        overlap: float = 0.25,
        finalized: Callable[[int, HostArray, HostArray], None] | None = None,
        metrics: Metrics | RollingMetricsBuilder | None = None,
    ) -> HostArray | None:
        """Wrap the restorer to collect finalized score planes."""
        def record(plane: int, scores: HostArray, count: HostArray) -> None:
            """Collect one score plane and forward it to restoration."""
            planes.append(scores.copy())
            if finalized is not None:
                finalized(plane, scores, count)

        result = stream(
            inputs,
            predictor,
            class_ids,
            roi_size,
            overlap=overlap,
            finalized=record,
            metrics=metrics,
        )
        assert metrics is not None and metrics.get("label_buffer_bytes") == 0
        return result

    monkeypatch.setattr("mlx_segment_ct.streaming.stream_scores", recording)
    scored = medmlx.run(
        image_path=str(source),
        weights_path=str(weights_dir),
        output_dir=str(output),
        label_prompt=case["label_prompt"],
        overwrite=True,
    )
    assert scored["details"]["restoration"] == "scores"
    volume, affine = load_volume(source)
    prepared = preprocess(volume, affine)
    size = prepared.image.shape[1:]
    # Windows pad an image smaller than the ROI symmetrically, the extra voxel after.
    start = [(roi - n) // 2 if roi > n else 0 for n, roi in zip(size, medmlx.ROI_SIZE, strict=True)]
    scores = np.stack(planes, axis=1)[
        :, *(slice(first, first + n) for first, n in zip(start, size, strict=True))
    ]
    expected = dense_restoration(
        scores, prepared.affine, affine, volume.shape[1:], case["class_ids"]
    )
    actual = cast("HostArray", np.asanyarray(load_nifti(scored["outputs"]["mask"]).dataobj))
    assert_restored(actual, expected, f"{name}::scores")


@pytest.mark.parametrize("name", ["lps", "oblique", "sagittal"])
def test_run_ctmr_matches_upstream_worker(
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
    source = write_nifti(tmp_path / f"{name}.nii.gz", cast("HostArray", data[f"{name}::nifti"]))
    result = medmlx.run(
        image_path=str(source),
        weights_path=str(weights_dir),
        output_dir=str(tmp_path / "out"),
        label_prompt=case["label_prompt"],
        checkpoint=CTMR,
        restoration="bundle",
    )
    json.dumps(result)
    assert_saved(Path(result["outputs"]["mask"]), data, f"{name}::mask")
    details = result["details"]
    assert details["checkpoint"] == CTMR
    assert details["mode"] == ("segment_all" if case["label_prompt"] is None else "class_prompt")
    assert details.get("modality") == ("MRI_BODY" if case["label_prompt"] is None else None)
    assert details.get("roi_size") == case["roi"]
    assert cast("AutomaticDetails", details)["class_ids"] == case["class_ids"]
    assert cast("AutomaticDetails", details)["padded_shape"] == case["padded_shape"]
    assert details["patch_count"] == case["patch_count"]
    assert details["preprocessed_shape"] == case["preprocessed_shape"]


def test_run_label_names_and_overwrite(tmp_path: Path, weights_dir: Path, stand_ins: None) -> None:
    """Verify named label requests and output overwrite behavior."""
    data, info = load_fixture("medmlx_segment")
    source = write_nifti(tmp_path / "case.nii.gz", cast("HostArray", data["ras_ties::nifti"]))
    output = tmp_path / "out"
    # Canonical kidney and spleen prompts expand through the bundle subclass transform.
    kwargs: FileRunPaths = {
        "image_path": source,
        "weights_path": weights_dir,
        "output_dir": output,
    }
    result = medmlx.run(prompt=["kidney", "Spleen"], restoration="bundle", **kwargs)
    assert cast("AutomaticDetails", result["details"])["label_names"] == ["kidney", "Spleen"]
    assert (
        cast("AutomaticDetails", result["details"])["class_ids"]
        == cast("SegmentMetadata", cast("CasesMetadata", info)["cases"]["ras_ties"])["class_ids"]
    )
    assert_saved(Path(result["outputs"]["mask"]), data, "ras_ties::mask")
    before = Path(result["outputs"]["mask"]).read_bytes()
    with pytest.raises(InvalidInputError, match="overwrite"):
        medmlx.run(prompt="kidney, spleen", restoration="bundle", **kwargs)
    assert Path(result["outputs"]["mask"]).read_bytes() == before
    again = medmlx.run(label_prompt=[2, 3], overwrite=True, restoration="bundle", **kwargs)
    assert_saved(Path(again["outputs"]["mask"]), data, "ras_ties::mask")
    assert sorted(path.name for path in output.iterdir()) == [medmlx.MASK_FILENAME]


def ct_volume(shape: tuple[int, ...] = (24, 20, 12)) -> HostArray:
    """Create a synthetic CT volume with foreground."""
    data = np.full(shape, -1000, dtype=np.int16)
    data[4:-4, 4:-4, 2:-2] = 60
    return data


def failure_inputs(tmp_path: Path) -> dict[str, Path]:
    """Build synthetic files for input and asset error cases."""
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    paths = {
        "image": save_volume(tmp_path / "ct.nii.gz", ct_volume(), affine),
        "mask": save_volume(tmp_path / "mask.nii.gz", np.zeros((24, 20, 12), np.uint8), affine),
        "small_mask": save_volume(
            tmp_path / "small.nii.gz", np.zeros((24, 20, 11), np.uint8), affine
        ),
        "fraction_mask": save_volume(
            tmp_path / "fraction.nii.gz", np.full((24, 20, 12), 0.5, np.float32), affine
        ),
        "uncoded": save_volume(tmp_path / "uncoded.nii.gz", ct_volume(), affine, coded=False),
        "air": save_volume(tmp_path / "air.nii.gz", np.full((24, 20, 12), -1000, np.int16), affine),
    }
    (tmp_path / "ct.mha").write_bytes(b"not nifti")
    paths["not_nifti"] = tmp_path / "ct.mha"
    (tmp_path / "file").write_text("x")
    paths["file"] = tmp_path / "file"
    return paths


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"image_path": "missing"}, "Missing NV-Segment-CT image_path"),
        ({"image_path": "not_nifti"}, "must be a NIfTI"),
        ({"overwrite": "yes"}, "overwrite must be true or false"),
        ({"prompt": "no such organ"}, "Unknown NV-Segment-CT label"),
        ({"prompt": "ivc"}, "Unknown NV-Segment-CT label"),
        ({"label_prompt": [140]}, "zero-shot"),
        ({"label_prompt": [16]}, "automatic label"),
        ({"label_prompt": [1.5]}, "must be integers"),
        ({"label_prompt": [True]}, "must be integers"),
        ({"label_prompt": [255]}, r"in \[1, 254\]"),
        ({"output_dir": "file"}, "not a directory"),
        ({"point_labels": [1]}, "at least one point"),
        ({"points": [[1, 2, 3]]}, "provided together"),
        ({"segment_all": True, "prompt": "liver"}, "cannot be combined"),
        ({"segment_all": True, "points": [[1, 2, 3]], "point_labels": [1]}, "segment_all"),
        ({"point_start": 1}, "require point prompts"),
        ({"previous_logits_path": "missing"}, "existing float32"),
        ({"checkpoint": "NV-Segment-MR"}, "checkpoint must be one of"),
        ({"checkpoint": 1}, "checkpoint must be a string"),
        ({"restoration": "nearest"}, "restoration must be one of scores, bundle"),
        ({"modality": "MRI_BODY"}, "NV-Segment-CT modality must be one of CT_BODY"),
        ({"modality": 1}, "modality must be a string"),
        ({"checkpoint": CTMR, "modality": "MRI_SPINE"}, "modality must be one of"),
        ({"checkpoint": CTMR, "modality": "CT_BODY", "label_prompt": [3]}, "cannot be combined"),
        ({"checkpoint": CTMR, "label_prompt": [512]}, r"in \[1, 511\]"),
        ({"checkpoint": CTMR, "label_prompt": [346]}, "zero-shot"),
        ({"checkpoint": CTMR, "label_prompt": [133]}, "automatic label"),
        ({"checkpoint": CTMR, "prompt": "fibula"}, "Unknown NV-Segment-CTMR label"),
        (
            {"checkpoint": CTMR, "points": [[1, 2, 3]], "point_labels": [1], "label_prompt": [512]},
            r"in \[1, 511\]",
        ),
    ],
)
def test_run_rejects_invalid_inputs(
    tmp_path: Path, change: dict[str, object], message: str
) -> None:
    """Reject malformed image, prompt, and option inputs."""
    paths = failure_inputs(tmp_path)
    kwargs: dict[str, object] = {
        "image_path": paths["image"],
        "weights_path": tmp_path / "no-weights",
        "output_dir": tmp_path / "out",
    }
    for key, value in change.items():
        kwargs[key] = (
            paths.get(value, tmp_path / value)
            if key.endswith(("_path", "_dir")) and isinstance(value, str)
            else value
        )
    with pytest.raises(InvalidInputError, match=message):
        cast("RawRunner", medmlx.run)(**kwargs)
    assert not (tmp_path / "out").exists()


def test_run_refuses_to_replace_inputs_or_outputs(tmp_path: Path) -> None:
    """Prevent overwriting existing inputs or outputs without authorization."""
    paths = failure_inputs(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    (output / medmlx.MASK_FILENAME).write_bytes(b"existing")
    with pytest.raises(InvalidInputError, match="overwrite=true"):
        medmlx.run(image_path=paths["image"], weights_path=tmp_path, output_dir=output)
    assert (output / medmlx.MASK_FILENAME).read_bytes() == b"existing"
    target = save_volume(tmp_path / medmlx.MASK_FILENAME, ct_volume(), np.eye(4))
    with pytest.raises(InvalidInputError, match="separate from the input"):
        medmlx.run(image_path=target, weights_path=tmp_path, output_dir=tmp_path, overwrite=True)


def test_run_reports_missing_weights_without_loading_mlx(tmp_path: Path) -> None:
    """Report missing weights before initializing the native runtime."""
    paths = failure_inputs(tmp_path)
    kwargs: FileImageOutputPaths = {"image_path": paths["image"], "output_dir": tmp_path / "out"}
    with pytest.raises(AssetNotReadyError, match="weights directory not found"):
        medmlx.run(weights_path=tmp_path / "absent", **kwargs)
    weights = tmp_path / "weights"
    weights.mkdir()
    with pytest.raises(AssetNotReadyError, match=r"missing automatic\.safetensors"):
        medmlx.run(weights_path=weights, **kwargs)
    (weights / medmlx.AUTOMATIC_WEIGHTS).touch()
    with pytest.raises(AssetNotReadyError, match=r"missing correction\.safetensors"):
        medmlx.run(weights_path=weights, points=[[10, 10, 6]], point_labels=[1], **kwargs)
    assert not (tmp_path / "out").exists()


def test_loading_rejects_mismatched_and_corrupt_weights(tmp_path: Path, mx: CoreRuntime) -> None:
    """Reject checkpoint tensors with incompatible shapes or corrupt files."""
    weights = tmp_path / "automatic.safetensors"
    mx.save_safetensors(
        str(weights), {"image_encoder.encoder.conv_init.weight": mx.zeros((48, 1, 3, 3, 3))}
    )
    with pytest.raises(AssetNotReadyError, match="missing 121 tensors"):
        medmlx.load_weights(mx, weights, required_shapes())
    mx.save_safetensors(str(weights), {"class_head.mlp.0.bias": mx.zeros((47,))})
    with pytest.raises(AssetNotReadyError, match=r"class_head\.mlp\.0\.bias"):
        medmlx.load_weights(mx, weights, {"class_head.mlp.0.bias": (48,)})
    mx.save_safetensors(str(weights), {"class_head.mlp.0.bias": mx.zeros((48,), dtype=mx.float16)})
    with pytest.raises(AssetNotReadyError, match=r"expected.*float32"):
        medmlx.load_weights(mx, weights, {"class_head.mlp.0.bias": (48,)})
    with pytest.raises(AssetNotReadyError, match="outside the automatic graph"):
        medmlx.load_weights(mx, weights, {})
    weights.write_bytes(b"corrupt")
    with pytest.raises(AssetNotReadyError, match="cannot read"):
        medmlx.load_weights(mx, weights, {})


def test_run_rejects_nonfinite_ct_without_publishing(
    tmp_path: Path, weights_dir: Path, stand_ins: None
) -> None:
    """Do not publish a mask after a CT contains an infinite intensity."""
    image = np.full((8, 8, 8), 100.0, dtype=np.float32)
    image[4, 4, 4] = np.inf
    source = save_volume(tmp_path / "ct.nii.gz", image, np.diag([1.5, 1.5, 1.5, 1.0]))
    output = tmp_path / "out"
    with pytest.raises(InvalidInputError, match="non-finite"):
        medmlx.run(image_path=source, weights_path=weights_dir, output_dir=output)
    assert not (output / medmlx.MASK_FILENAME).exists()


def test_run_rejects_ct_without_foreground(
    tmp_path: Path, weights_dir: Path, stand_ins: None
) -> None:
    """Reject a CT image without foreground for preprocessing."""
    paths = failure_inputs(tmp_path)
    with pytest.raises(InvalidInputError, match="no voxel above 0"):
        medmlx.run(image_path=paths["air"], weights_path=weights_dir, output_dir=tmp_path / "out")
    assert not (tmp_path / "out" / medmlx.MASK_FILENAME).exists()


def test_run_full_networks_end_to_end_on_metal(
    tmp_path: Path, mx: CoreRuntime, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise full native network runner wiring on Metal with fixture weights."""
    weights = tmp_path / "weights"
    weights.mkdir()
    for name, shapes, seed in (
        (medmlx.AUTOMATIC_WEIGHTS, required_shapes(), 501),
        (medmlx.CORRECTION_WEIGHTS, additional_shapes(), 502),
    ):
        mx.save_safetensors(
            str(weights / name),
            {
                k: cast("ArrayBoundary[array]", mx).array(v)
                for k, v in random_weights(shapes, seed).items()
            },
        )
    monkeypatch.setattr(medmlx, "ROI_SIZE", (32, 32, 32))
    affine = np.diag([0.9, 0.9, 2.5, 1.0])
    source = save_volume(tmp_path / "ct.nii.gz", ct_volume((40, 36, 16)), affine)
    result = medmlx.run(
        image_path=source,
        weights_path=weights,
        output_dir=tmp_path / "segment",
        label_prompt=[3, 1],
    )
    saved = load_nifti(result["outputs"]["mask"])
    assert saved.shape == (40, 36, 16)
    assert np.allclose(saved.affine, affine, rtol=0, atol=1e-5)
    assert saved.get_data_dtype() == np.float32
    assert set(np.unique(np.asanyarray(saved.dataobj)).tolist()) <= {0, 1, 3}
    result = medmlx.run(
        image_path=source,
        weights_path=weights,
        output_dir=tmp_path / "interactive",
        label_prompt=[1],
        points=[[20, 18, 8]],
        point_labels=[1],
    )
    assert load_nifti(result["outputs"]["mask"]).shape == (40, 36, 16)
    assert result["details"]["mode"] == "interactive"
    monkeypatch.setattr(medmlx, "CTMR_ROI_SIZE", (32, 32, 32))
    result = medmlx.run(
        image_path=source,
        weights_path=weights,
        output_dir=tmp_path / "ctmr",
        label_prompt=[300, 3],
        checkpoint=CTMR,
    )
    saved = load_nifti(result["outputs"]["mask"])
    assert saved.shape == (40, 36, 16)
    assert set(np.unique(np.asanyarray(saved.dataobj)).tolist()) <= {0, 3, 300}
    assert result["details"]["checkpoint"] == CTMR
