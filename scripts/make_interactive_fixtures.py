"""Record MONAI 1.6.0 CPU point inference on seeded weights; no checkpoint download.

Run ``python scripts/make_interactive_fixtures.py`` with the dev extra. Only the
encoder/class head are small seeded stand-ins in window fixtures. VISTA3D.forward,
PointMappingSAM, point inferer and Vista transforms are the unmodified reference.
Runner cases use the pinned bundle config already recorded in this repository.
"""

from __future__ import annotations

import copy
import sys
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Unpack, cast

if TYPE_CHECKING:
    from torch import Tensor

    from scripts.oracle_types import (
        CaseMetadata,
        FunctionalRuntime,
        HostArray,
        ParsedCodes,
        PayloadValue,
        PostPrediction,
        PromptTransform,
        ResetNamespace,
        SourceNetwork,
        SourcePointHead,
        SourcePredictionKeywords,
        TensorPointInference,
        TorchRuntime,
    )

import monai.transforms.utils as reset_module
import numpy as np
import torch
from monai.apps.vista3d.inferer import point_based_window_inferer
from monai.apps.vista3d.transforms import VistaPostTransformd, VistaPreTransformd
from monai.data.meta_tensor import MetaTensor
from monai.networks.nets.vista3d import VISTA3D, PointMappingSAM
from torch import nn

torch_api = cast("TorchRuntime", torch)
functional_api = cast("FunctionalRuntime", nn.functional)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))
from scripts.make_reference_fixtures import point_shapes, seeded_head  # noqa: E402
from scripts.reference_bundle import (  # noqa: E402
    GEOMETRIES,
    bundle_transforms,
    load_nifti,
    nifti_bytes,
    synthetic_ct,
    synthetic_mr,
)
from scripts.upstream import save  # noqa: E402

SEED = 1701
FIXTURES = REPO / "tests/fixtures"


def point_head() -> SourcePointHead:
    """Create the seeded reference point head."""
    return seeded_head(SEED)


class SeededEncoder(nn.Module):
    """Project single-channel patches to 48 features for both source heads."""

    def __init__(self) -> None:
        """Initialize the synthetic reference encoder projection."""
        super().__init__()
        self.projection = nn.Conv3d(1, 48, 1)

    def forward(
        self, image: Tensor, with_point: bool, with_label: bool
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return synthetic image features for the requested heads."""
        features = self.projection(image)
        return features if with_point else None, features if with_label else None


class SeededClassHead(nn.Module):
    """Project 48-channel features to one class-score channel."""

    def __init__(self) -> None:
        """Initialize the synthetic reference class projection."""
        super().__init__()
        self.projection = nn.Conv3d(48, 1, 1)

    def forward(self, features: Tensor, class_vector: Tensor | None) -> tuple[Tensor, None]:
        """Project reference features into class scores."""
        return self.projection(features), None


def record_decoder() -> None:
    """Record seeded point-head decoder inputs and outputs."""
    head = point_head()
    rng = np.random.default_rng(1702)
    features = rng.standard_normal((1, 48, 8, 8, 8), dtype=np.float32)
    points = np.array([[[1.25, 2.5, 3.5], [6, 4, 5], [0, 0, 0]]], np.float64)
    codes = np.array([[1, 0, -1]], np.int64)
    arrays = {"features": features, "points": points, "codes": codes}
    for target in (None, 26, 133):
        vector = None if target is None else torch_api.tensor([target])
        effective = codes if target != 26 else np.where(codes >= 0, codes + 2, codes)
        arrays[f"codes_{target}"] = effective
        arrays[f"output_{target}"] = (
            head(
                torch_api.from_numpy(features),
                torch_api.from_numpy(points),
                torch_api.from_numpy(effective),
                vector,
            )
            .detach()
            .numpy()
        )
    save("interactive_decoder", arrays, weight_seed=SEED, weight_shapes=point_shapes())


def record_windows(
    image: HostArray,
    points: HostArray,
    codes: HostArray,
    target: int | None,
    roi: tuple[int, int, int],
    *,
    previous: HostArray | None = None,
    point_start: int = 0,
) -> dict[str, HostArray]:
    """Record reference point-window inputs, outputs, and hook traces."""
    torch_api.manual_seed(1703)
    model = cast(
        "SourceNetwork",
        VISTA3D(SeededEncoder(), SeededClassHead(), cast("PointMappingSAM", point_head())).eval(),
    )
    traces: dict[str, list[HostArray]] = {
        k: []
        for k in (
            "patch",
            "normalized",
            "patch_codes",
            "point_output",
            "auto_output",
            "windows",
        )
    }

    def point_before(module: object, args: tuple[Tensor, ...]) -> None:
        """Record and adjust point-head input coordinates before inference."""
        coords = args[1].clone() + 0.5
        for axis, size in enumerate(args[0].shape[2:]):
            coords[..., axis] /= size
        traces["normalized"].append(coords.float().numpy().copy())
        traces["patch_codes"].append(args[2].numpy().copy())

    def point_after(module: object, args: tuple[Tensor, ...], output: Tensor) -> None:
        """Record the output of the reference point head."""
        traces["point_output"].append(output.detach().numpy().copy())

    def auto_after(module: object, args: tuple[Tensor, ...], output: tuple[Tensor, None]) -> None:
        """Record the output of the reference automatic head."""
        traces["auto_output"].append(output[0].detach().numpy().copy())

    model.point_head.register_forward_pre_hook(point_before)
    model.point_head.register_forward_hook(point_after)
    model.class_head.register_forward_hook(auto_after)

    class WindowRecorder(nn.Module):
        def forward(self, patch: Tensor, **kwargs: Unpack[SourcePredictionKeywords]) -> Tensor:
            """Record each window and delegate prediction to the source model."""
            traces["patch"].append(patch.numpy().copy())
            traces["windows"].append(
                np.array([[s.start, s.stop] for s in kwargs["patch_coords"][0][-3:]])
            )
            return model(patch, **kwargs)

    predictor = WindowRecorder()

    data = cast("PromptTransform", VistaPreTransformd("image"))(
        {
            "label_prompt": None if target is None else [target],
            "point_labels": codes.tolist(),
        }
    )
    effective = np.array(cast("ParsedCodes", data)["point_labels"], np.int64)
    vector = None if target is None else torch_api.tensor([target])
    logits = cast("TensorPointInference", point_based_window_inferer)(
        torch_api.from_numpy(image),
        roi,
        cast("SourceNetwork", predictor),
        torch_api.from_numpy(points),
        torch_api.from_numpy(effective),
        class_vector=vector
        if target is not None and (target <= 132 or not (effective != -1).any())
        else None,
        prompt_class=vector,
        prev_mask=None if previous is None else torch_api.from_numpy(previous.copy()),
        point_start=point_start,
        transpose=True,
        mode="gaussian",
        overlap=0.3,
        with_coord=True,
        sw_batch_size=1,
    )
    post = cast("PromptTransform", VistaPostTransformd("pred"))(
        {
            "pred": logits[0].clone(),
            "points": torch_api.from_numpy(points),
            "point_labels": torch_api.from_numpy(effective),
            "label_prompt": vector,
        }
    )
    post = cast("PostPrediction", post)["pred"]
    arrays = {
        "image": image,
        "points": points,
        "codes": codes,
        "effective_codes": effective,
        "logits": logits.numpy().copy(),
        "labels": post.numpy().copy(),
    }
    if previous is not None:
        arrays["previous"] = previous
    for kind, values in traces.items():
        for index, value in enumerate(values):
            arrays[f"{kind}_{index}"] = value
    return arrays


def window_cases() -> None:
    """Build seeded point-window fixture cases."""
    rng = np.random.default_rng(1704)
    cases = {
        "overlap": ((18, 12, 10), [[4, 5, 5], [8.5, 6, 5]], [1, 0], None),
        "padded": ((5, 6, 7), [[2.5, 2, 3]], [1], None),
        "boundary": ((8, 8, 8), [[0, 0, 0]], [1], None),
        "negative": ((8, 8, 8), [[4, 4, 4]], [0], None),
        "padding_code": ((8, 8, 8), [[4, 4, 4]], [-1], None),
        "regular": ((12, 10, 10), [[4, 4, 4], [8, 5, 5]], [1, 0], 1),
        "special": ((12, 10, 10), [[4, 4, 4], [8, 5, 5]], [1, 0], 26),
        "lung_special": ((8, 8, 8), [[4, 4, 4]], [1], 28),
        "zero_shot": ((12, 10, 10), [[4, 4, 4], [8, 5, 5]], [1, 0], 133),
        "boundary_then_valid": ((18, 12, 10), [[0, 0, 0], [14, 9, 7]], [1, 1], None),
        "zero_shot_padding": ((8, 8, 8), [[4, 4, 4]], [-1], 133),
        "boundary_class": ((8, 8, 8), [[0, 0, 0]], [1], 1),
    }
    arrays: dict[str, PayloadValue] = {}
    metadata: dict[str, CaseMetadata] = {}
    for name, (shape, points, codes, target) in cases.items():
        result = record_windows(
            rng.standard_normal((1, 1, *shape), dtype=np.float32),
            np.array([points], np.float64),
            np.array([codes]),
            target,
            (8, 8, 8),
        )
        arrays.update({f"{name}::{key}": value for key, value in result.items()})
        metadata[name] = {"target": target, "roi": [8, 8, 8]}
    save("interactive_windows", arrays, cases=metadata, encoder="seeded 1x1 conv stand-in")


def record_components() -> None:
    """Record source connected-component behavior for click cases."""
    arrays: dict[str, PayloadValue] = {}
    metadata: dict[str, CaseMetadata] = {}
    base = np.full((1, 1, 8, 8, 8), -1, np.float32)
    base[:, :, 1:3, 1:3, 1:3] = 1
    base[:, :, 5:7, 5:7, 5:7] = 2
    for name, logits, points, codes in (
        ("disconnected", base, [[1, 1, 1]], [1]),
        ("both", base, [[1.5, 1.5, 1.5], [6, 6, 6]], [1, 3]),
        ("negative_only", base, [[1, 1, 1]], [0]),
        ("miss", base, [[3, 3, 3]], [1]),
        ("positive_mean", np.maximum(base, 0), [[1, 1, 1]], [1]),
        ("empty_mean", np.ones_like(base), [[1, 1, 1]], [1]),
        ("nan", np.where(np.indices(base.shape)[2] > 6, np.nan, base), [[1, 1, 1]], [1]),
    ):
        p, c = np.array([points], np.float64), np.array([codes])
        output = cast("PromptTransform", VistaPostTransformd("pred"))(
            {
                "pred": torch_api.from_numpy(logits[0].copy()),
                "points": torch_api.from_numpy(p),
                "point_labels": torch_api.from_numpy(c),
            }
        )
        output = cast("PostPrediction", output)["pred"]
        arrays.update(
            {
                f"{name}::logits": logits,
                f"{name}::points": p,
                f"{name}::codes": c,
                f"{name}::labels": output.numpy(),
            }
        )
        metadata[name] = {}
    save("interactive_components", arrays, cases=metadata)


def record_runner() -> None:
    """Record source runner outputs for interactive cases."""
    arrays: dict[str, PayloadValue] = {}
    cases: dict[str, CaseMetadata] = {}
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        preprocessing, postprocessing = bundle_transforms(root)
        for name, target in (("ras_ties", None), ("lps", 26), ("oblique", 133)):
            index = list(GEOMETRIES).index(name)
            shape, affine, form = GEOMETRIES[name]
            raw = nifti_bytes(synthetic_ct(shape, 900 + index), affine, form)
            source = root / f"{name}.nii"
            source.write_bytes(raw.tobytes())
            sample = preprocessing({"image": str(source)})
            image = cast("ResetNamespace", reset_module).reset_ops_id(sample["image"])
            sample["image"] = image
            points = np.array(
                [[[24, 26, 7]]]
                if name == "ras_ties"
                else [[[14, 22, 10]]]
                if name == "lps"
                else [[[20, 18, 9]]],
                np.float64,
            )
            transform = np.linalg.inv(image.affine.numpy()) @ image.meta["original_affine"]
            homogeneous = np.concatenate((points[0], np.ones((1, 1))), axis=1)
            transformed = (transform @ homogeneous.T).T[None, :, :3]
            result = record_windows(
                image.numpy()[None], transformed, np.array([[1]]), target, (16, 16, 16)
            )
            prediction = MetaTensor(
                result["logits"][0].copy(),
                affine=image.affine,
                meta=cast("dict[str, object]", copy.deepcopy(image.meta)),
                applied_operations=copy.deepcopy(image.applied_operations),
            )
            sample.update(
                pred=prediction,
                points=torch_api.from_numpy(transformed),
                point_labels=torch_api.from_numpy(result["effective_codes"]),
                label_prompt=None if target is None else torch_api.tensor([target]),
            )
            # SaveImaged is the bundle's final transform; retain its NIfTI bytes too.
            postprocessing(sample)

            saved = load_nifti(str(root / "out" / f"{name}_trans.nii.gz"))
            result.update(
                nifti=raw,
                source_points=points,
                affine=image.affine.numpy(),
                mask=cast("HostArray", np.asanyarray(saved.dataobj)),
                mask_affine=saved.affine,
            )
            arrays.update({f"{name}::{key}": value for key, value in result.items()})
            cases[name] = {"target": target, "roi": [16, 16, 16]}
    save("interactive_runner", arrays, cases=cases, encoder="seeded 1x1 conv stand-in")


def record_ctmr_runner() -> None:
    """NV-Segment-CTMR clicks: percentile-scaled MR and its non-cubic point window."""
    arrays: dict[str, PayloadValue] = {}
    cases: dict[str, CaseMetadata] = {}
    roi = (16, 16, 8)
    plans = {
        "ras_ties": (None, [[24.0, 26.0, 7.0]]),
        "lps": (300, [[14.0, 22.0, 10.0]]),
        "oblique": (3, [[20.0, 18.0, 9.0]]),
    }
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        preprocessing, postprocessing = bundle_transforms(root, ctmr=True)
        for index, (name, (target, clicks)) in enumerate(plans.items()):
            shape, affine, form = GEOMETRIES[name]
            raw = nifti_bytes(synthetic_mr(shape, 1800 + index), affine, form)
            source = root / f"{name}.nii"
            source.write_bytes(raw.tobytes())
            sample = preprocessing({"image": str(source)})
            image = cast("ResetNamespace", reset_module).reset_ops_id(sample["image"])
            sample["image"] = image
            points = np.array([clicks], np.float64)
            transform = np.linalg.inv(image.affine.numpy()) @ image.meta["original_affine"]
            homogeneous = np.concatenate((points[0], np.ones((len(clicks), 1))), axis=1)
            transformed = (transform @ homogeneous.T).T[None, :, :3]
            result = record_windows(image.numpy()[None], transformed, np.array([[1]]), target, roi)
            prediction = MetaTensor(
                result["logits"][0].copy(),
                affine=image.affine,
                meta=cast("dict[str, object]", copy.deepcopy(image.meta)),
                applied_operations=copy.deepcopy(image.applied_operations),
            )
            sample.update(
                pred=prediction,
                points=torch_api.from_numpy(transformed),
                point_labels=torch_api.from_numpy(result["effective_codes"]),
                label_prompt=None if target is None else torch_api.tensor([target]),
            )
            postprocessing(sample)
            saved = load_nifti(str(root / "out" / f"{name}_trans.nii.gz"))
            result.update(
                nifti=raw,
                source_points=points,
                affine=image.affine.numpy(),
                mask=cast("HostArray", np.asanyarray(saved.dataobj)),
                mask_affine=saved.affine,
            )
            arrays.update({f"{name}::{key}": value for key, value in result.items()})
            cases[name] = {"target": target, "roi": list(roi)}
    save("interactive_ctmr_runner", arrays, cases=cases, encoder="seeded 1x1 conv stand-in")


def record_correction_runner() -> None:
    """Record source runner outputs for correction cases."""
    arrays: dict[str, PayloadValue] = {}
    cases: dict[str, CaseMetadata] = {}
    plans = {
        "ras_ties": (1, [[24.0, 26.0, 7.0], [20.0, 30.0, 8.0]], [1, 0], 0),
        "lps": (25, [[14.0, 22.0, 10.0], [20.0, 21.0, 9.5]], [1, 1], 1),
        "sagittal": (3, [[3.0, 20.0, 14.0]], [0], 0),
        "long": (1, [[100.0, 12.0, 8.0], [30.0, 10.0, 6.0]], [1, 0], 0),
    }
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        preprocessing, postprocessing = bundle_transforms(root)
        for index, (name, (target, clicks, codes, start)) in enumerate(plans.items()):
            shape, affine, form = GEOMETRIES[name]
            raw = nifti_bytes(synthetic_ct(shape, 1400 + index), affine, form)
            source = root / f"{name}.nii"
            source.write_bytes(raw.tobytes())
            sample = preprocessing({"image": str(source)})
            image = cast("ResetNamespace", reset_module).reset_ops_id(sample["image"])
            sample["image"] = image
            points = np.array([clicks], np.float64)
            transform = np.linalg.inv(image.affine.numpy()) @ image.meta["original_affine"]
            homogeneous = np.concatenate((points[0], np.ones((len(clicks), 1))), axis=1)
            coords = (transform @ homogeneous.T).T[None, :, :3]
            previous = np.where(
                np.random.default_rng(1500 + index).random((1, *image.shape)) > 0.6, 1, -1
            ).astype(np.float32)
            result = record_windows(
                image.numpy()[None],
                coords,
                np.array([codes]),
                target,
                (16, 16, 16),
                previous=previous,
                point_start=start,
            )
            sample.update(
                pred=MetaTensor(
                    result["logits"][0],
                    affine=image.affine,
                    meta=cast("dict[str, object]", copy.deepcopy(image.meta)),
                    applied_operations=copy.deepcopy(image.applied_operations),
                ),
                label_prompt=torch_api.tensor([target]),
                points=torch_api.from_numpy(coords),
                point_labels=torch_api.from_numpy(result["effective_codes"]),
            )
            postprocessing(sample)

            saved = load_nifti(root / "out" / f"{name}_trans.nii.gz")
            result.update(
                nifti=raw,
                source_points=points,
                mask=cast("HostArray", np.asanyarray(saved.dataobj)),
                mask_affine=saved.affine,
            )
            arrays.update({f"{name}::{k}": v for k, v in result.items()})
            cases[name] = {"target": target, "roi": [16, 16, 16], "point_start": start}
    save("interactive_correction_runner", arrays, cases=cases, encoder="seeded 1x1 conv stand-in")


def main() -> None:
    """Check the source version and write interactive reference fixtures."""
    if version("monai") != "1.6.0":
        raise RuntimeError("Record with the pinned MONAI 1.6.0")
    torch.set_num_threads(1)
    with torch.inference_mode():
        record_decoder()
        window_cases()
        record_components()
        record_runner()
        record_correction_runner()
        record_ctmr_runner()


if __name__ == "__main__":
    main()
