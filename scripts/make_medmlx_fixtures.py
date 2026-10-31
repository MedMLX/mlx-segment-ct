"""Record bundle preprocessing, prompt transforms and restored runner masks from MONAI."""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor

    from scripts.oracle_types import (
        CaseMetadata,
        HostArray,
        IntensityTransform,
        MonoWindowNamespace,
        ParsedCodes,
        ParsedLabels,
        PayloadValue,
        Preprocessing,
        PromptTransform,
        ResetNamespace,
        SourceTransform,
        TorchRuntime,
        TransformData,
    )


import monai.inferers.utils as window_module
import monai.transforms.utils as reset_module
import numpy as np
import torch
from monai.apps.vista3d.transforms import VistaPreTransformd
from monai.data.meta_tensor import MetaTensor
from monai.transforms.intensity.array import ScaleIntensityRangePercentiles

torch_api = cast("TorchRuntime", torch)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.reference_bundle import (  # noqa: E402
    BUNDLE_CONFIG,
    GEOMETRIES,
    GRID_SIZES,
    MODALITIES,
    axis_aligned,
    bundle_labels,
    bundle_transforms,
    ctmr_bundle_labels,
    ctmr_everything_labels,
    large_mr,
    large_percentile_input,
    nifti_bytes,
    saved_mask,
    synthetic_ct,
    synthetic_mr,
    write_case,
)
from scripts.upstream import save  # noqa: E402


def prediction(sample: TransformData, logits: Tensor) -> None:
    """Construct a deterministic reference prediction from a transformed sample."""
    image = sample["image"]
    sample["pred"] = MetaTensor(
        logits,
        affine=image.affine,
        meta=cast("dict[str, object]", copy.deepcopy(image.meta)),
        applied_operations=copy.deepcopy(image.applied_operations),
    )


def record_grid_base() -> None:
    """Record reference normalized sampling-grid axes."""
    arrays: dict[str, PayloadValue] = {}
    for size in GRID_SIZES:
        theta = torch_api.eye(3, 4, dtype=torch.float64)[None]
        arrays[f"base_{size}"] = torch_api.nn.functional.affine_grid(
            theta, [1, 1, 1, 1, size], align_corners=False
        )[0, 0, 0, :, 0].numpy()
    save("medmlx_grid", arrays, sizes=list(GRID_SIZES))


def record_requests() -> None:
    """Record source preprocessing for the requested bundle inputs."""
    transform = cast(
        "PromptTransform",
        VistaPreTransformd(
            "image",
            labels_dict=bundle_labels(),
            subclass={
                k: list(range(33, 57)) + list(range(63, 98)) + [114, 120, 122] if k == "21" else v
                for k, v in BUNDLE_CONFIG["subclass"].items()
            },
        ),
    )
    requests = [
        [2, 3],
        [21, 33, 14],
        [20, 2, 3, 3],
        ["liver"],
        ["Spleen"],
        ["Left-Hippocampus"],
        [25],
        [26],
        [28],
        [29],
        [117],
        [133],
        [254],
    ]
    arrays: dict[str, PayloadValue] = {}
    for i, request in enumerate(requests):
        result = transform({"label_prompt": request, "point_labels": [[0, 1, -1, 2, 3]]})
        arrays[f"labels_{i}"] = cast("ParsedLabels", result)["label_prompt"]
        arrays[f"codes_{i}"] = cast("ParsedCodes", result)["point_labels"]
    save("medmlx_requests", arrays, requests=requests)


def record_geometry(
    root: Path, preprocessing: Preprocessing, postprocessing: SourceTransform
) -> None:
    """Record source image geometry and restoration cases."""
    arrays: dict[str, PayloadValue] = {}
    cases: dict[str, CaseMetadata] = {}
    class_ids = [3, 5, 14]
    for index, (name, (shape, affine, form)) in enumerate(GEOMETRIES.items()):
        raw = nifti_bytes(synthetic_ct(shape, 900 + index), affine, form)
        sample = preprocessing({"image": str(write_case(root, name, raw))})
        image = cast("ResetNamespace", reset_module).reset_ops_id(sample["image"])
        sample["image"] = image
        labels = np.random.default_rng(1200 + index).choice(
            np.array([0, *class_ids], np.float32), size=tuple(image.shape)
        )
        logits = torch_api.stack(
            [
                torch_api.from_numpy(np.where(labels[0] == i, 1, -1).astype(np.float32))
                for i in class_ids
            ]
        )
        prediction(sample, logits)
        sample["label_prompt"] = torch_api.tensor(class_ids)
        postprocessing(sample)
        restored = saved_mask(root / "out" / f"{name}_trans.nii.gz")
        arrays.update(
            {
                f"{name}::nifti": raw,
                f"{name}::labels": labels,
                f"{name}::image": image.numpy(),
                f"{name}::affine": image.affine.numpy(),
                **{f"{name}::restored_{k}": v for k, v in restored.items()},
            }
        )
        cases[name] = {"axis_aligned": axis_aligned(affine), "class_ids": class_ids}
    save("medmlx_geometry", arrays, cases=cases)


def automatic_patch(image: Tensor, ids: list[int]) -> Tensor:
    """Seed-independent dyadic input function from the original runner cases."""
    values = np.asarray(ids, np.int64)
    shape = (1, len(ids), 1, 1, 1)
    scale = torch_api.from_numpy((2.0 ** -(values % 3) / 4).astype(np.float32).reshape(shape))
    bias = torch_api.from_numpy(((values % 5) / 16 - 0.25).astype(np.float32).reshape(shape))
    slope = torch_api.from_numpy((((values % 4) - 1.5) / 4).astype(np.float32).reshape(shape))
    d, h, w = image.shape[2:]
    ramp = (
        np.arange(d, dtype=np.float32)[:, None, None] / 2
        + np.arange(h, dtype=np.float32)[None, :, None]
        - np.arange(w, dtype=np.float32)[None, None, :]
        - d // 4
    ) / 4
    return image * scale + torch_api.from_numpy(ramp)[None, None] * slope + bias


def record_segment(
    root: Path, preprocessing: Preprocessing, postprocessing: SourceTransform
) -> None:
    """Record source automatic-segmentation runner outputs."""
    arrays: dict[str, PayloadValue] = {}
    cases: dict[str, CaseMetadata] = {}
    plans = {"ras_ties": [2, 3], "sagittal": [6], "oblique": [1, 3], "long": [3, 14]}
    for index, (name, labels) in enumerate(plans.items()):
        shape, affine, form = GEOMETRIES[name]
        raw = nifti_bytes(synthetic_ct(shape, 1300 + index), affine, form)
        source = write_case(root, f"segment-{name}", raw)
        sample = preprocessing({"image": str(source), "label_prompt": labels})
        sample["image"] = cast("ResetNamespace", reset_module).reset_ops_id(sample["image"])
        ids = cast("ParsedLabels", sample)["label_prompt"]
        patches: list[HostArray] = []

        def predictor(
            patch: Tensor, ids: list[int] = ids, patches: list[HostArray] = patches
        ) -> Tensor:
            """Run the synthetic automatic predictor for a source window."""
            output = automatic_patch(patch, ids)
            patches.append(output.numpy().copy())
            return output

        logits = cast("MonoWindowNamespace", window_module).sliding_window_inference(
            sample["image"][None], [128] * 3, 1, predictor, overlap=0.25, padding_mode="replicate"
        )
        prediction(sample, logits[0])
        sample["label_prompt"] = torch_api.tensor(ids)
        postprocessing(sample)
        mask = saved_mask(root / "out" / f"segment-{name}_trans.nii.gz")
        arrays.update({f"{name}::nifti": raw, **{f"{name}::mask_{k}": v for k, v in mask.items()}})
        # Runner tests evaluate the same dyadic stand-in; full graph tests are separate.
        cases[name] = {
            "label_prompt": labels,
            "class_ids": ids,
            "padded_shape": [max(128, n) for n in sample["image"].shape[1:]],
            "patch_count": len(patches),
            "preprocessed_shape": [1, *sample["image"].shape],
        }
    save("medmlx_segment", arrays, cases=cases)


CTMR_GEOMETRIES = ("lps", "sagittal", "oblique")
CTMR_REQUESTS: list[list[int] | list[str]] = [
    ["fibula (deprecated)"],
    ["Left-TTG---transverse-temporal-gyrus"],
    ["Right-AnG---angular-gyrus"],
    [2, 300],
    [21, 345],
]


def record_ctmr(root: Path) -> None:
    """NV-Segment-CTMR everything labels, names, percentile scaling and preprocessing."""
    preprocessing, _ = bundle_transforms(root, ctmr=True)
    arrays: dict[str, PayloadValue] = {}
    for modality in MODALITIES:
        arrays[f"everything::{modality}"] = np.asarray(
            ctmr_everything_labels(root, modality), np.int64
        )
    transform = cast(
        "PromptTransform",
        VistaPreTransformd(
            "image",
            labels_dict=ctmr_bundle_labels(),
            subclass={
                k: list(range(33, 57)) + list(range(63, 98)) + [114, 120, 122] if k == "21" else v
                for k, v in BUNDLE_CONFIG["subclass"].items()
            },
        ),
    )
    for i, request in enumerate(CTMR_REQUESTS):
        result = transform({"label_prompt": request, "point_labels": [[0, 1]]})
        arrays[f"request::{i}"] = np.asarray(cast("ParsedLabels", result)["label_prompt"])
    scale = cast(
        "IntensityTransform",
        ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1, clip=True),
    )
    rng = np.random.default_rng(1600)
    small = {
        "normal": rng.standard_normal((1, 9, 8, 7), dtype=np.float32) * np.float32(300),
        "integers": rng.integers(-5, 6, (1, 6, 5, 4)).astype(np.float32),
        "constant": np.full((1, 3, 3, 3), 7, np.float32),
        "pair": np.array([[[[0.0]], [[1.0]]]], np.float32),
        "skewed": rng.exponential(50, (1, 20, 17, 13)).astype(np.float32),
    }
    for name, values in small.items():
        arrays[f"percentile::{name}::input"] = values
        arrays[f"percentile::{name}::output"] = scale(torch_api.from_numpy(values.copy())).numpy()
    large = scale(torch_api.from_numpy(large_percentile_input())).numpy()
    arrays["percentile::large::output"] = large
    arrays["percentile::large::shape"] = list(large.shape)
    for index, name in enumerate(CTMR_GEOMETRIES):
        shape, affine, form = GEOMETRIES[name]
        raw = nifti_bytes(synthetic_mr(shape, 1600 + index), affine, form)
        sample = preprocessing({"image": str(write_case(root, f"ctmr-{name}", raw))})
        image = sample["image"]
        arrays.update(
            {
                f"{name}::nifti": raw,
                f"{name}::image": image.numpy(),
                f"{name}::affine": image.affine.numpy(),
            }
        )
    raw = nifti_bytes(large_mr(), np.diag([1.5, 1.5, 1.5, 1.0]), "both")
    image = preprocessing({"image": str(write_case(root, "ctmr-large", raw))})["image"]
    arrays["large::image"] = image.numpy()
    arrays["large::shape"] = list(image.shape)
    arrays["large::affine"] = image.affine.numpy()
    save("medmlx_ctmr", arrays, requests=CTMR_REQUESTS, modalities=list(MODALITIES))


def record_ctmr_segment(root: Path) -> None:
    """NV-Segment-CTMR class prompts at its 192x192x128 window, and one all-classes case."""
    preprocessing, postprocessing = bundle_transforms(root, ctmr=True)
    arrays: dict[str, PayloadValue] = {}
    cases: dict[str, CaseMetadata] = {}
    plans: dict[str, tuple[list[int] | None, list[int]]] = {
        "lps": ([300, 2], [192, 192, 128]),
        "oblique": ([3, 255], [192, 192, 128]),
        # MRI_BODY's 50 classes at the full window would need several GB of scores.
        "sagittal": (None, [32, 32, 32]),
    }
    for index, (name, (labels, roi)) in enumerate(plans.items()):
        shape, affine, form = GEOMETRIES[name]
        raw = nifti_bytes(synthetic_mr(shape, 1700 + index), affine, form)
        source = write_case(root, f"ctmr-segment-{name}", raw)
        sample = preprocessing({"image": str(source), "label_prompt": labels})
        sample["image"] = cast("ResetNamespace", reset_module).reset_ops_id(sample["image"])
        ids = (
            ctmr_everything_labels(root, "MRI_BODY")
            if labels is None
            else cast("ParsedLabels", sample)["label_prompt"]
        )
        patches: list[HostArray] = []

        def predictor(
            patch: Tensor, ids: list[int] = ids, patches: list[HostArray] = patches
        ) -> Tensor:
            """Run the synthetic CTMR automatic predictor for a source window."""
            output = automatic_patch(patch, ids)
            patches.append(output.numpy().copy())
            return output

        logits = cast("MonoWindowNamespace", window_module).sliding_window_inference(
            sample["image"][None], roi, 1, predictor, overlap=0.25, padding_mode="replicate"
        )
        prediction(sample, logits[0])
        sample["label_prompt"] = torch_api.tensor(ids)
        postprocessing(sample)
        mask = saved_mask(root / "out" / f"ctmr-segment-{name}_trans.nii.gz")
        arrays.update({f"{name}::nifti": raw, **{f"{name}::mask_{k}": v for k, v in mask.items()}})
        cases[name] = {
            "label_prompt": labels,
            "class_ids": ids,
            "roi": roi,
            "padded_shape": [
                max(r, n) for r, n in zip(roi, sample["image"].shape[1:], strict=True)
            ],
            "patch_count": len(patches),
            "preprocessed_shape": [1, *sample["image"].shape],
        }
    save("medmlx_ctmr_segment", arrays, cases=cases)


def record_ctmr_all() -> None:
    """Record the supported CTMR all-label reference case."""
    torch.set_num_threads(1)
    with TemporaryDirectory() as temporary, torch.inference_mode():
        root = Path(temporary)
        record_ctmr(root)
        record_ctmr_segment(root)


def main() -> None:
    """Record reference bundle and runner fixtures with one source CPU thread."""
    torch.set_num_threads(1)
    with TemporaryDirectory() as temporary, torch.inference_mode():
        root = Path(temporary)
        preprocessing, postprocessing = bundle_transforms(root)
        record_grid_base()
        record_requests()
        record_geometry(root, preprocessing, postprocessing)
        record_segment(root, preprocessing, postprocessing)
    record_ctmr_all()


if __name__ == "__main__":
    main()
