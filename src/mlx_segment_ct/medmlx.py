"""MedMLX declaration and runner for the pinned NVIDIA VISTA3D bundles.

Native automatic and interactive inference use MONAI 1.6.0 as their reference.
``checkpoint`` selects the NV-Segment-CT or NV-Segment-CTMR bundle configuration
for the same vista3d132 graph. Point corrections consume previous raw logits on
the preprocessed image grid.
"""

from __future__ import annotations

import os
import time
from collections.abc import Callable, Sequence
from importlib.metadata import version
from numbers import Integral
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING, cast, overload

from mlx_segment_ct._types import (
    CoreRuntime,
    Details,
    HostArray,
    Metrics,
    PatchCoordinates,
    PublishedDetails,
    RollingMetrics,
    RunResult,
)

if TYPE_CHECKING:
    from mlx.core import array

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from medmlx_core.weights import weights_dir as model_weights_dir
from medmlx_core.weights import weights_revision

from mlx_segment_ct.labels import CT_RULES, InteractivePrompt, LabelRules, Prompt, label_rules

if TYPE_CHECKING:
    from mlx_segment_ct.bundle_transforms import Intensity
    from mlx_segment_ct.restoration import ScoreRestorer

MODEL = {
    "id": "mlx-segment-ct",
    "label": "NV-Segment-CT",
    "task": "segmentation",
    "run": "mlx_segment_ct.medmlx:run",
    "summary": (
        "NVIDIA NV-Segment-CT/CTMR (VISTA-3D) CT and MRI segmentation: all supported classes, "
        "class prompts by name or ID, interactive points, or correction of previous logits."
    ),
}


CTMR_MODEL = {
    "id": "mlx-segment-ctmr",
    "label": "NV-Segment-CTMR",
    "task": "segmentation",
    "run": "mlx_segment_ct.medmlx:run_ctmr",
    "summary": (
        "NVIDIA NV-Segment-CTMR CT/MRI segmentation; NVIDIA does not support point correction."
    ),
}

MASK_FILENAME = "mlx_segment_ct_mask.nii.gz"
RESTORATIONS = ("scores", "bundle")
AUTOMATIC_WEIGHTS = "automatic.safetensors"
CORRECTION_WEIGHTS = "correction.safetensors"
SOURCE_REVISION = "f9f5f51b589e5dc9c23c453cf5138398e4084056"
ROI_SIZE = (128, 128, 128)
CTMR_ROI_SIZE = (192, 192, 128)
WARNING = "Model output requires human review and is not ground truth."
_NETWORK_FAILURE = (
    "NV-Segment-CT MLX network failed; its ordered FP32 kernels need Apple Silicon Metal"
)
_WEIGHTS_HINT = (
    "For NV-Segment-CT, download MedMLX/vista3d-ct-mlx at revision "
    "e6f45856b4a59212ca10b44a9ade2e2d672f412b from Hugging Face. Pass its directory "
    "as weights_path: automatic.safetensors is required, plus correction.safetensors "
    "for point inference. For NV-Segment-CTMR, use MedMLX/vista3d-ctmr-mlx "
    "and set checkpoint='NV-Segment-CTMR'."
)


def run(
    *,
    image_path: str | os.PathLike[str],
    weights_path: str | os.PathLike[str] | None = None,
    output_dir: str | os.PathLike[str],
    segment_all: bool = False,
    prompt: str | Sequence[str] | None = None,
    label_prompt: Sequence[int] | None = None,
    previous_logits_path: str | os.PathLike[str] | None = None,
    points: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[int] | None = None,
    point_start: int = 0,
    interactive: bool = False,
    overwrite: bool = False,
    checkpoint: str = "NV-Segment-CT",
    modality: str | None = None,
    restoration: str = "scores",
) -> RunResult:
    """Segment a CT or MRI NIfTI with automatic classes or one object's point prompts.

    ``points`` and ``point_labels`` select interactive inference; ``interactive=True``
    explicitly requires that mode. Coordinates are source-grid voxel indices; codes
    are -1 (padding), 0/1 (exclude/include), or 2/3 (special exclude/include).
    ``prompt`` accepts canonical bundle names; ``label_prompt`` accepts numeric IDs.
    Without points, coarse kidney/lung/bone labels expand to their subclasses.

    ``previous_logits_path`` accepts a float32 .npy array of shape (1, 1, X, Y, Z)
    on the preprocessed grid, as returned by ``interactive_logits``. Corrections use
    the bundle branches and upstream component fusion; ``point_start`` selects new
    clicks while earlier clicks remain context. Inputs are preserved.

    ``checkpoint`` selects settings for the supplied weights: ``"NV-Segment-CT"`` (default;
    CT HU window, 128^3 windows, labels 0-254) or ``"NV-Segment-CTMR"``
    (1st-99th percentile scaling, 192x192x128 windows, labels 0-345, prompts below
    512). ``modality`` picks the NV-Segment-CTMR all-classes set (``"CT_BODY"``,
    ``"MRI_BODY"`` default, or ``"MRI_BRAIN"``) and applies only without prompts.

    ``restoration`` selects how labels return to the source grid. ``"scores"`` (default)
    interpolates the scores trilinearly at each source voxel and decides its label
    there, so boundaries follow the source resolution. ``"bundle"`` decides on the
    1.5 mm grid and restores labels with nearest interpolation, as the NVIDIA bundle does.
    Input voxel intensities must be finite. Spatial units must be millimeters or
    unknown (assumed millimeters); other declared units require conversion before
    inference. Output NIfTI headers are newly constructed and do not retain the
    source unit code.

    The published ``MedMLX/vista3d-ct-mlx`` bundle supplies ``automatic.safetensors``
    and ``correction.safetensors`` for ``checkpoint="NV-Segment-CT"``. The
    matching CTMR bundle is ``MedMLX/vista3d-ctmr-mlx``; NVIDIA does not support
    interactive point segmentation with that checkpoint.
    The result contains absolute output paths and JSON details. Masks preserve the
    source spatial shape and affine, with float32 global labels and 255 for uncovered voxels.
    Existing outputs require ``overwrite=True``. Invalid requests, missing weights
    and inference failures raise the corresponding medmlx-core error.

    Args:
        image_path: Single-channel CT or MRI NIfTI ``.nii`` or ``.nii.gz`` volume,
            according to the checkpoint.
        weights_path: Optional model folder; omitted downloads ``MedMLX/vista3d-ct-mlx``.
            The folder contains ``automatic.safetensors`` at its root;
            point inference also requires ``correction.safetensors`` there. Use the
            published ``MedMLX/vista3d-ct-mlx`` bundle for NV-Segment-CT.
        output_dir: Directory for ``mlx_segment_ct_mask.nii.gz``; run details are returned.
        segment_all: Request the checkpoint's all-classes set; cannot be combined
            with class or point prompts. With no prompts, all classes are selected
            even when this flag is false (the default).
        prompt: Canonical anatomy names as a list or comma-separated string, case
            insensitive; use instead of ``label_prompt``. Default: None.
        label_prompt: Numeric foreground IDs instead of names; at most one with points.
            NV-Segment-CT IDs 133-254 and NV-Segment-CTMR IDs 346-511 require points.
            Point prompts reject coarse kidney (2), lung (20) and bone (21). Default: None.
        previous_logits_path: Optional float32 ``.npy`` raw logits, shape (1, 1, X, Y, Z)
            on the preprocessed grid; requires points. NaN is allowed, infinity is rejected.
        points: Clicks as (x, y, z) voxel indices on the source image grid; requires
            matching ``point_labels``. Default: None.
        point_labels: One code per click: -1 padding, 0/1 exclude/include, or 2/3 special
            exclude/include. Default: None.
        point_start: Index of the first new click (default 0); earlier clicks remain
            context. Requires point prompts when nonzero.
        interactive: Require point-prompt mode; point inputs also select it when false
            (the default).
        overwrite: Replace an existing output mask when true (default False);
            input aliases are refused.
        checkpoint: Settings matching the supplied weights: ``"NV-Segment-CT"`` (default)
            or ``"NV-Segment-CTMR"``.
        modality: All-classes set, only without prompts. NV-Segment-CT accepts ``"CT_BODY"``;
            NV-Segment-CTMR also accepts ``"MRI_BODY"`` and ``"MRI_BRAIN"``. Default None
            selects ``"CT_BODY"`` for NV-Segment-CT or ``"MRI_BODY"`` for NV-Segment-CTMR.
        restoration: ``"scores"`` (default) interpolates scores before deciding labels;
            ``"bundle"`` restores labels with nearest interpolation.
    """
    started = time.perf_counter()
    for name, value in (
        ("overwrite", overwrite),
        ("segment_all", segment_all),
        ("interactive", interactive),
    ):
        if not isinstance(cast("object", value), bool):
            raise InvalidInputError(f"NV-Segment-CT {name} must be true or false")
    if not isinstance(cast("object", checkpoint), str):
        raise InvalidInputError("NV-Segment-CT checkpoint must be a string")
    if restoration not in RESTORATIONS:
        raise InvalidInputError(
            f"NV-Segment-CT restoration must be one of {', '.join(RESTORATIONS)}"
        )
    rules = label_rules(checkpoint)
    if weights_path is None and checkpoint != "NV-Segment-CT":
        raise AssetNotReadyError(
            "NV-Segment-CTMR requires an explicit matching weights_path folder.",
            hint="Automatic mlx-segment-ct weights belong to NV-Segment-CT; "
            "supply your converted CTMR folder for this advanced option.",
        )
    if modality is not None and not isinstance(cast("object", modality), str):
        raise InvalidInputError("NV-Segment-CT modality must be a string")
    image_file = _nifti_file(image_path, "image_path")
    previous_file = None
    if previous_logits_path is not None:
        previous_file = Path(previous_logits_path).resolve()
        if not previous_file.is_file() or previous_file.suffix != ".npy":
            raise InvalidInputError("previous_logits_path must be an existing float32 .npy file")
    point_mode = interactive or points is not None or point_labels is not None
    if point_mode:
        if segment_all:
            raise InvalidInputError("Point prompts cannot be combined with segment_all")
        plan = resolve_interactive_request(prompt, label_prompt, points, point_labels, rules)
        if type(point_start) is not int or not 0 <= point_start < len(plan.points):
            raise InvalidInputError("point_start must identify at least one new click")
    else:
        if previous_file is not None or point_start != 0:
            raise InvalidInputError("previous logits and point_start require point prompts")
        plan = resolve_request(
            segment_all=segment_all, prompt=prompt, label_prompt=label_prompt, rules=rules
        )
    if modality is not None:
        if not isinstance(plan, Prompt) or not plan.segment_all:
            raise InvalidInputError(
                "NV-Segment-CT modality selects the all-classes set; it cannot be combined "
                "with label or point prompts"
            )
        rules.everything_labels(modality)
    output = Path(output_dir).resolve()
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"NV-Segment-CT output_dir is not a directory: {output}")
    destination = output / MASK_FILENAME
    if destination.resolve() in {image_file, previous_file}:
        raise InvalidInputError("NV-Segment-CT outputs must be separate from the input CT and mask")
    if destination.exists() and not overwrite:
        raise InvalidInputError(
            f"NV-Segment-CT output exists: {destination}; pass overwrite=true to replace it"
        )
    weights_root = (
        model_weights_dir("mlx-segment-ct")
        if weights_path is None else Path(weights_path).expanduser()
    )
    weights_dir = _weights_dir(weights_root, correction=point_mode)
    return _infer_and_publish(
        weights_dir,
        image_file,
        previous_file,
        destination,
        plan,
        point_start,
        started,
        modality,
        restoration=restoration,
    )


def run_ctmr(
    *,
    image_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    weights_path: str | os.PathLike[str] | None = None,
    segment_all: bool = False,
    prompt: str | Sequence[str] | None = None,
    label_prompt: Sequence[int] | None = None,
    previous_logits_path: str | os.PathLike[str] | None = None,
    points: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[int] | None = None,
    point_start: int = 0,
    interactive: bool = False,
    overwrite: bool = False,
    modality: str | None = None,
    restoration: str = "scores",
) -> RunResult:
    """Run the CTMR variant with its own automatic model folder and fixed settings.

    Omitted weights use ``MedMLX/vista3d-ctmr-mlx``; an explicit folder contains
    ``automatic.safetensors``. CTMR preprocessing and label rules are fixed
    internally, so CT weights are never substituted. The shared point/correction
    arguments are forwarded to ``run``, but NVIDIA does not support point
    segmentation with this checkpoint. Their presence is not a qualification
    of CTMR point inference.

    Args:
        image_path: Single-channel CT or MRI NIfTI volume; declared spatial units
            must be millimeters or unknown (assumed millimeters).
        output_dir: Directory for the restored label map.
        weights_path: Optional model folder containing ``automatic.safetensors``;
            omit to reuse the automatic CTMR download.
        segment_all: Request all classes for the selected modality; no prompts also
            selects this mode.
        prompt: Anatomy names instead of numeric class IDs.
        label_prompt: Numeric foreground IDs instead of anatomy names.
        previous_logits_path: Advanced shared correction input; requires point
            prompts, which NVIDIA does not support for CTMR.
        points: Advanced shared source-grid ``(x, y, z)`` point coordinates;
            CTMR point inference is not supported by NVIDIA.
        point_labels: Include/exclude codes paired with ``points``.
        point_start: Index of the first new click for the shared point interface.
        interactive: Require the shared point-prompt mode; see its support limit above.
        overwrite: Replace an existing output; inputs remain protected from aliasing.
        modality: All-classes set: ``CT_BODY``, ``MRI_BODY`` or ``MRI_BRAIN``.
            None selects ``MRI_BODY``. Inapplicable to class or point prompts.
        restoration: ``scores`` interpolates scores before selecting labels;
            ``bundle`` restores labels with the bundle's nearest-neighbor rule.
    """
    folder = (
        model_weights_dir("mlx-segment-ctmr")
        if weights_path is None else Path(weights_path).expanduser()
    )
    return run(
        image_path=image_path, output_dir=output_dir, weights_path=folder,
        checkpoint="NV-Segment-CTMR", segment_all=segment_all, prompt=prompt,
        label_prompt=label_prompt, previous_logits_path=previous_logits_path,
        points=points, point_labels=point_labels, point_start=point_start,
        interactive=interactive, overwrite=overwrite, modality=modality,
        restoration=restoration,
    )


def bundle_settings(rules: LabelRules) -> tuple[tuple[int, int, int], Intensity]:
    """Window size and intensity scaling of the checkpoint's own inference.json.

    NV-Segment-CT uses 128^3 windows with the fixed HU range; NV-Segment-CTMR uses
    192x192x128 windows with 1st-99th percentile scaling.
    """
    if rules.checkpoint == "NV-Segment-CTMR":
        return CTMR_ROI_SIZE, "percentiles"
    return ROI_SIZE, "range"


def _infer_and_publish(
    weights_dir: Path,
    image_file: Path,
    previous_file: Path | None,
    destination: Path,
    plan: Prompt | InteractivePrompt,
    point_start: int,
    started: float,
    modality: str | None = None,
    *,
    restoration: str = "scores",
) -> RunResult:
    """Run the planned request on MLX and publish the mask; see ``run`` for the contract."""
    from medmlx_core.runtime import (
        import_mlx,
        mlx_default_device_name,
    )

    mx = cast("CoreRuntime", import_mlx())
    mx.synchronize()
    mx.reset_peak_memory()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(plan, InteractivePrompt):
        details = _interactive(
            mx,
            weights_dir,
            image_file,
            destination,
            plan,
            previous_file=previous_file,
            point_start=point_start,
            restoration=restoration,
        )
    else:
        details = _segment(
            mx, weights_dir, image_file, destination, plan, modality, restoration=restoration
        )
    mx.synchronize()
    details.update(
        weights_revision=weights_revision(weights_dir),
        checkpoint=plan.rules.checkpoint,
        restoration=restoration,
        backend="mlx",
        mlx_version=version("mlx"),
        mlx_default_device=mlx_default_device_name(mx),
        precision="float32",
        source_revision=SOURCE_REVISION,
        mlx_peak_memory_bytes=mx.get_peak_memory(),
        elapsed_seconds=round(time.perf_counter() - started, 6),
        warning=WARNING,
    )
    return {
        "outputs": {"mask": destination.as_posix()},
        "details": cast("PublishedDetails", details),
    }


def resolve_interactive_request(
    prompt: object,
    label_prompt: object,
    points: object,
    point_labels: object,
    rules: LabelRules = CT_RULES,
) -> InteractivePrompt:
    """Validate raw ``run`` arguments into an ``InteractivePrompt`` on the source grid.

    Raises ``InvalidInputError`` for names and numeric IDs given together, more than
    one class, or malformed or non-finite points and codes.
    """
    from mlx_segment_ct.labels import resolve_label_name, split_label_names

    ids = _sequence(label_prompt, "label_prompt")
    if prompt is not None:
        if ids:
            raise InvalidInputError("Pass NV-Segment-CT label names or numeric label ids, not both")
        ids = tuple(
            resolve_label_name(name, rules) for name in split_label_names(_prompt_labels(prompt))
        )
    if len(ids) > 1:
        raise InvalidInputError("Interactive segmentation accepts at most one class label")
    try:
        parsed_points = tuple(_parse_point(point) for point in _sequence(points, "points"))
    except OverflowError as exc:
        raise InvalidInputError("Interactive point coordinates must be finite numbers") from exc
    return InteractivePrompt(
        parsed_points,
        cast("tuple[int, ...]", _sequence(point_labels, "point_labels")),
        cast("int", ids[0]) if ids else None,
        rules,
    )


def _interactive(
    mx: CoreRuntime,
    weights_dir: Path,
    image_file: Path,
    destination: Path,
    prompt: InteractivePrompt,
    *,
    previous_file: Path | None = None,
    point_start: int = 0,
    restoration: str = "scores",
) -> Details:
    """Run click-conditioned inference and publish its restored segmentation."""
    from mlx_segment_ct.bundle_transforms import load_volume, preprocess, save_label_map
    from mlx_segment_ct.interactive import (
        interactive_labels,
        interactive_logits,
        interactive_scores,
        transform_points,
    )
    from mlx_segment_ct.labels import label_name
    from mlx_segment_ct.mlx_network import AutomaticModel, required_shapes
    from mlx_segment_ct.mlx_point import PointModel
    from mlx_segment_ct.restoration import ScoreRestorer

    tick = time.perf_counter()
    roi_size, intensity = bundle_settings(prompt.rules)
    volume, affine = load_volume(image_file)
    points, _ = prompt.arrays()
    if ((points < 0) | (points >= np.asarray(volume.shape[1:]))).any():
        raise InvalidInputError("Interactive click is outside the original CT voxel grid")
    prepared = preprocess(volume, affine, intensity=intensity)
    transformed = transform_points(prepared, prompt)
    previous = (
        None if previous_file is None else previous_logits(previous_file, prepared.image.shape)
    )
    weights = _load_point_weights(mx, weights_dir)
    automatic = None
    try:
        point = PointModel(weights, mx=mx)
        if prompt.uses_automatic:
            assert prompt.class_label is not None
            auto_weights = load_weights(mx, weights_dir / AUTOMATIC_WEIGHTS, required_shapes())
            automatic = AutomaticModel(auto_weights, (prompt.class_label,), mx=mx)
    except RuntimeError as exc:
        raise ModelExecutionError(f"{_NETWORK_FAILURE}: {exc}") from exc
    metrics: Metrics = {"patch_seconds": []}
    load_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    try:
        logits = interactive_logits(
            prepared.image[None],
            transformed,
            point,
            mx=mx,
            roi_size=roi_size,
            automatic_model=automatic,
            metrics=metrics,
            previous_logits=previous,
            point_start=point_start,
        )
        if restoration == "bundle":
            labels = interactive_labels(logits, transformed)
        else:
            labels = interactive_scores(logits, transformed)
    except ValueError as exc:
        raise ModelExecutionError(str(exc)) from exc
    except RuntimeError as exc:
        raise ModelExecutionError(f"{_NETWORK_FAILURE}: {exc}") from exc
    inference_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    if restoration == "bundle":
        restored, restored_affine = prepared.invert(labels)
    else:
        restorer = ScoreRestorer(prepared.restoration_grid(), [prompt.output_label])
        for index in range(labels.shape[1]):
            restorer.add_plane(index, labels[:, index])
        restored, restored_affine = restorer.finish()
    # Uncovered voxels (NaN) are written as 255, as upstream does.
    restored = np.nan_to_num(restored, nan=255.0)
    if not np.isin(restored, [0, prompt.output_label, 255]).all():
        raise ModelExecutionError("Interactive restored mask contains invalid labels")
    _publish(destination, lambda path: save_label_map(path, restored, restored_affine))
    return {
        "mode": "interactive" if previous is None else "correction",
        "label_prompt": [] if prompt.class_label is None else [prompt.class_label],
        "automatic_branch": automatic is not None,
        "point_start": point_start,
        "output_labels": [
            {"value": value, "name": name}
            for value, name in (
                (0, "background"),
                (
                    prompt.output_label,
                    label_name(prompt.output_label, prompt.rules)
                    if prompt.class_label is not None
                    else "interactive object",
                ),
                (255, "uncovered"),
            )
        ],
        "points": [list(p) for p in prompt.points],
        "point_labels": list(prompt.point_labels),
        "effective_point_labels": transformed.arrays()[1][0].tolist(),
        "preprocessed_points": [list(p) for p in transformed.points],
        "preprocessed_affine": prepared.affine.tolist(),
        "preprocessed_shape": list(prepared.image.shape),
        "output_shape": list(restored.shape[1:]),
        "output_dtype": "float32",
        "algorithm": "mlx_vista3d_interactive",
        "parameters": {
            "roi_size": list(roi_size),
            "center_only": True,
            "padding_mode": "constant",
            "overlap_reduction": "sum",
            "threshold": 0,
            "uncovered_label": 255,
            "positive_components": prompt.class_label is None,
        },
        "patch_count": len(metrics["patch_seconds"]),
        "load_preprocess_seconds": round(load_seconds, 6),
        "inference_stitch_seconds": round(inference_seconds, 6),
        "restore_save_seconds": round(time.perf_counter() - tick, 6),
    }


def _segment(
    mx: CoreRuntime,
    weights_dir: Path,
    image_file: Path,
    destination: Path,
    plan: Prompt,
    modality: str | None = None,
    *,
    restoration: str = "scores",
) -> Details:
    """Bundle automatic inference with bounded FP32 accumulation.

    Writes the restored float32 label map to ``destination`` and returns the run
    details; scores are streamed along axis 0 rather than held for the whole volume.
    """
    from mlx_segment_ct.bundle_transforms import load_volume, preprocess, save_label_map
    from mlx_segment_ct.labels import class_ids_for_prompt, label_name
    from mlx_segment_ct.mlx_network import AutomaticModel, required_shapes
    from mlx_segment_ct.restoration import ScoreRestorer
    from mlx_segment_ct.streaming import rolling_logits, stream_scores

    tick = time.perf_counter()
    rules = plan.rules
    roi_size, intensity = bundle_settings(rules)
    label_ids = plan.label_ids or None
    class_ids = class_ids_for_prompt(label_ids, rules, modality)
    weights = load_weights(mx, weights_dir / AUTOMATIC_WEIGHTS, required_shapes())
    try:
        model = AutomaticModel(weights, class_ids, mx=mx)
    except RuntimeError as exc:
        raise ModelExecutionError(f"{_NETWORK_FAILURE}: {exc}") from exc
    volume, affine = load_volume(image_file)
    prepared = preprocess(volume, affine, intensity=intensity)
    inputs = prepared.image[None]
    metrics: Metrics = {"patch_seconds": []}
    load_seconds = time.perf_counter() - tick

    def predict(patch: array, coordinates: PatchCoordinates) -> array:
        """Run and time one automatic segmentation patch on the native runtime."""
        mx.synchronize()
        start = time.perf_counter()
        logits = model(patch)
        mx.eval(logits)
        mx.synchronize()
        metrics["patch_seconds"].append(time.perf_counter() - start)
        return logits

    restorer = (
        ScoreRestorer(prepared.restoration_grid(), class_ids) if restoration == "scores" else None
    )
    finalized = None if restorer is None else _restoring(restorer, inputs.shape[2:], roi_size)
    tick = time.perf_counter()
    try:
        if restorer is None:
            labels = rolling_logits(inputs, predict, class_ids, roi_size, metrics=metrics)
        else:
            labels = stream_scores(
                inputs, predict, class_ids, roi_size, metrics=metrics, finalized=finalized
            )
    except ValueError as exc:
        raise ModelExecutionError(str(exc)) from exc
    except RuntimeError as exc:
        raise ModelExecutionError(f"{_NETWORK_FAILURE}: {exc}") from exc
    inference_seconds = time.perf_counter() - tick
    tick = time.perf_counter()
    if restorer is None:
        assert labels is not None
        restored, restored_affine = prepared.invert(labels[0])
    else:
        restored, restored_affine = restorer.finish()
    # Streaming raises on missing coverage, so NaN is not expected here. Any that remains
    # becomes 255 and fails validation unless 255 is a selected class.
    restored = np.nan_to_num(restored, nan=255.0)
    validate_labels(restored, class_ids)
    _publish(destination, lambda path: save_label_map(path, restored, restored_affine))
    values = set(class_ids) | {0}
    details: Details = {
        "mode": "segment_all" if label_ids is None else "class_prompt",
        "label_names": list(plan.label_names),
        "label_prompt": list(plan.label_ids),
        "class_ids": list(class_ids),
        "output_labels": [
            {"name": label_name(value, rules), "value": value} for value in sorted(values)
        ],
        "output_shape": list(restored.shape[1:]),
        "output_dtype": "float32",
        "preprocessed_shape": list(inputs.shape),
        "algorithm": "mlx_rolling_fp32_axis0",
        "roi_size": list(roi_size),
        "overlap": 0.25,
        "blend_mode": "constant",
        "padding_mode": "replicate",
        "padded_shape": cast("RollingMetrics", metrics)["padded_shape"],
        "patch_count": len(metrics["patch_seconds"]),
        "load_preprocess_seconds": round(load_seconds, 6),
        "inference_stitch_seconds": round(inference_seconds, 6),
        "restore_save_seconds": round(time.perf_counter() - tick, 6),
    }
    if label_ids is None:
        details["modality"] = rules.default_modality if modality is None else modality
    return details


def _restoring(
    restorer: ScoreRestorer, size: tuple[int, ...], roi_size: Sequence[int]
) -> Callable[[int, HostArray, HostArray], None]:
    """Pass ``rolling_logits``' finalized score planes, unpadded, to the restorer."""
    from mlx_segment_ct.streaming import centered_padding

    (before, _), (top, _), (left, _) = centered_padding(size, roi_size)

    def finalized(plane: int, scores: HostArray, _count: HostArray) -> None:
        """Send an unpadded finalized score plane to the source-grid restorer."""
        if before <= plane < before + size[0]:
            restorer.add_plane(
                plane - before, scores[:, top : top + size[1], left : left + size[2]]
            )

    return finalized


def previous_logits(path: Path, image_shape: tuple[int, ...]) -> HostArray:
    """Load raw previous logits from a float32 ``.npy`` of shape ``(1, *image_shape)``.

    ``image_shape`` is the preprocessed image shape ``(1, X, Y, Z)``. NaN (uncovered)
    is allowed; infinity is rejected with ``InvalidInputError``.
    """
    try:
        previous = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise InvalidInputError(f"Cannot read previous logits: {path}") from exc
    if (
        not isinstance(previous, np.ndarray)
        or cast("HostArray", previous).dtype != np.float32
        or cast("HostArray", previous).shape != (1, *image_shape)
    ):
        raise InvalidInputError(
            "Previous logits must be FP32 (1, 1, X, Y, Z) on the preprocessed grid"
        )
    previous = cast("HostArray", previous)
    if np.isinf(previous).any():
        raise InvalidInputError("Previous logits cannot contain infinity")
    return previous


def resolve_request(
    *,
    segment_all: bool,
    prompt: object,
    label_prompt: object,
    rules: LabelRules = CT_RULES,
) -> Prompt:
    """Validate raw ``run`` arguments for automatic segmentation into a ``Prompt``.

    Without names or IDs the request is the checkpoint's all-classes set. Raises
    ``InvalidInputError`` when names and IDs are both given or either is invalid.
    """
    from mlx_segment_ct.labels import resolve_label_name, split_label_names

    names = _prompt_labels(prompt) if prompt is not None else ()
    label_values = _sequence(label_prompt, "label_prompt")
    if names and label_values:
        raise InvalidInputError("Pass NV-Segment-CT label names or numeric label ids, not both")
    if segment_all and (names or label_values):
        raise InvalidInputError("segment_all cannot be combined with label prompts")
    if not (names or label_values):
        return Prompt(segment_all=True, label_names=("all_supported_classes",), rules=rules)
    label_names = split_label_names(names)
    label_ids = (
        tuple(_parse_label_id(v) for v in label_values)
        if label_values
        else tuple(resolve_label_name(name, rules) for name in label_names)
    )
    return Prompt(label_ids=label_ids, label_names=label_names, rules=rules)


def _prompt_labels(prompt: object) -> tuple[str, ...]:
    """Parse one string or a sequence of canonical anatomy names."""
    if isinstance(prompt, str):
        labels = (prompt.strip(),) if prompt.strip() else ()
    else:
        labels = tuple(
            str(value).strip() for value in _sequence(prompt, "prompt") if str(value).strip()
        )
    if not labels:
        raise InvalidInputError("At least one prompt/label is required")
    return labels


@overload
def _sequence[ValueT](value: Sequence[ValueT] | None, name: str) -> tuple[ValueT, ...]:
    """Preserve element types when converting an optional sequence."""
    ...


@overload
def _sequence(value: object, name: str) -> tuple[object, ...]:
    """Validate and convert an untyped list-like input."""
    ...


def _sequence(value: object, name: str) -> tuple[object, ...]:
    """Convert a list-like input to a tuple, or reject unsupported values."""
    if value is None:
        return ()
    if isinstance(value, str) or not isinstance(value, Sequence | np.ndarray):
        raise InvalidInputError(f"NV-Segment-CT {name} must be a list")
    return tuple(cast("Sequence[object]", value))


def _parse_point(point: object) -> tuple[float, float, float]:
    """Parse one numeric three-coordinate click or raise an input error."""
    if (
        isinstance(point, str)
        or not isinstance(point, Sequence | np.ndarray)
        or len(cast("Sequence[object]", point)) != 3
    ):
        raise InvalidInputError(
            f"NV-Segment-CT points must have three coordinates (x, y, z): {point!r}"
        )
    try:
        return (
            float(cast("float", point[0])),
            float(cast("float", point[1])),
            float(cast("float", point[2])),
        )
    except (TypeError, ValueError) as exc:
        raise InvalidInputError(
            f"NV-Segment-CT point coordinates must be numeric: {point!r}"
        ) from exc


def _parse_label_id(value: object) -> int:
    """Require an integer label prompt and return its value."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise InvalidInputError(f"NV-Segment-CT label prompts must be integers: {value!r}")
    return int(value)


def _nifti_file(value: str | os.PathLike[str], name: str) -> Path:
    """Resolve and validate a NIfTI input file path."""
    path = Path(value).resolve()
    if not path.is_file():
        raise InvalidInputError(f"Missing NV-Segment-CT {name}: {path}")
    if not path.name.endswith((".nii", ".nii.gz")):
        raise InvalidInputError(f"NV-Segment-CT {name} must be a NIfTI .nii or .nii.gz: {path}")
    return path


def _weights_dir(value: str | os.PathLike[str], *, correction: bool) -> Path:
    """Validate a model folder and the files needed for the requested mode."""
    directory = Path(value).expanduser()
    if not directory.is_dir():
        raise AssetNotReadyError(
            f"NV-Segment-CT weights directory not found: {directory}",
            reason=f"missing {directory}",
            hint=_WEIGHTS_HINT,
        )
    required = (AUTOMATIC_WEIGHTS, CORRECTION_WEIGHTS) if correction else (AUTOMATIC_WEIGHTS,)
    missing = [name for name in required if not (directory / name).is_file()]
    if missing:
        raise AssetNotReadyError(
            f"NV-Segment-CT weights directory {directory} is missing " + ", ".join(missing),
            reason="missing " + ", ".join(missing),
            hint=_WEIGHTS_HINT,
        )
    return directory


def _read_safetensors(mx: CoreRuntime, path: Path) -> dict[str, array]:
    """Load a named native checkpoint mapping or report an asset error."""
    try:
        raw = mx.load(str(path))
    except (OSError, RuntimeError, ValueError) as exc:
        raise AssetNotReadyError(
            f"NV-Segment-CT cannot read {path}: {exc}",
            reason=f"unreadable {path.name}",
            hint=_WEIGHTS_HINT,
        ) from exc
    if not isinstance(raw, dict):
        raise AssetNotReadyError(
            f"NV-Segment-CT weights must be a tensor mapping: {path}",
            reason=f"invalid {path.name}",
            hint=_WEIGHTS_HINT,
        )
    return raw


def _check_tensors(
    mx: CoreRuntime, path: Path, tensors: dict[str, array], shapes: dict[str, tuple[int, ...]]
) -> None:
    """Validate required tensor shapes and float32 dtypes in a checkpoint."""
    missing = sorted(shapes.keys() - tensors.keys())
    if missing:
        raise AssetNotReadyError(
            f"NV-Segment-CT {path.name} is missing {len(missing)} tensors, e.g. {missing[0]}",
            reason=f"mismatched {path.name}",
            hint=_WEIGHTS_HINT,
        )
    for name, shape in shapes.items():
        value = tensors[name]
        if tuple(value.shape) != shape or value.dtype != mx.float32:
            raise AssetNotReadyError(
                f"Incompatible NV-Segment-CT MLX tensor in {path.name}: {name} "
                f"{tuple(value.shape)} {value.dtype}; expected {shape} float32",
                reason=f"mismatched {path.name}",
                hint=_WEIGHTS_HINT,
            )


def load_weights(
    mx: CoreRuntime, path: Path, shapes: dict[str, tuple[int, ...]]
) -> dict[str, array]:
    """Load exactly the automatic graph, FP32.

    Raises ``AssetNotReadyError`` for an unreadable file, unexpected or missing
    tensors, or a tensor with the wrong shape or dtype.
    """
    weights = _read_safetensors(mx, path)
    unexpected = sorted(weights.keys() - shapes.keys())
    if unexpected:
        raise AssetNotReadyError(
            f"NV-Segment-CT {path.name} has tensors outside the automatic graph, "
            f"e.g. {unexpected[0]}",
            reason=f"mismatched {path.name}",
            hint=_WEIGHTS_HINT,
        )
    _check_tensors(mx, path, weights, shapes)
    mx.eval(weights)
    return weights


def _load_point_weights(mx: CoreRuntime, directory: Path) -> dict[str, array]:
    """Load the shared encoder and point decoder."""
    from mlx_segment_ct.mlx_point import additional_shapes, correction_shapes

    automatic_path = directory / AUTOMATIC_WEIGHTS
    extra_path = directory / CORRECTION_WEIGHTS
    automatic = _read_safetensors(mx, automatic_path)
    extra = _read_safetensors(mx, extra_path)
    additional = additional_shapes()
    if extra.keys() != additional.keys():
        raise AssetNotReadyError(
            "NV-Segment-CT correction.safetensors does not match the point graph",
            reason="mismatched correction.safetensors",
            hint=_WEIGHTS_HINT,
        )
    _check_tensors(mx, extra_path, extra, additional)
    shared = {name: shape for name, shape in correction_shapes().items() if name not in extra}
    _check_tensors(mx, automatic_path, automatic, shared)
    weights = {
        name: extra[name] if name in extra else automatic[name] for name in correction_shapes()
    }
    mx.eval(weights)
    return weights


def validate_labels(labels: HostArray, class_ids: Sequence[int]) -> None:
    """Require finite integral background or selected label IDs."""
    # Lookup table over global IDs 0-511; 511 is the largest NV-Segment-CTMR prompt ID.
    legal = np.zeros(512, dtype=bool)
    legal[[0, *class_ids]] = True
    invalid = not np.isfinite(labels).all() or bool(
        ((labels < 0) | (labels > 511) | (labels != np.floor(labels))).any()
    )
    if not invalid:
        invalid = bool((~legal[np.clip(labels.astype(np.int64), 0, 511)]).any())
    if invalid:
        raise ModelExecutionError("NV restored mask contains invalid or non-anatomical labels")


def _publish(destination: Path, write: Callable[[Path], None]) -> None:
    """Write next to the destination, then replace it atomically."""
    with NamedTemporaryFile(
        dir=destination.parent, prefix=".mlx-segment-", suffix=".nii.gz", delete=False
    ) as handle:
        staged = Path(handle.name)
    try:
        write(staged)
        os.replace(staged, destination)
    finally:
        staged.unlink(missing_ok=True)
