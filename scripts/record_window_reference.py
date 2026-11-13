"""Record MONAI sliding-window and point-window cases on HCC-TACE CT crops."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import nibabel as nib
import numpy as np
import scipy
import skimage
import torch
from monai import __version__ as monai_version
from monai.apps.vista3d.inferer import point_based_window_inferer
from monai.apps.vista3d.transforms import VistaPostTransformd
from monai.data.utils import compute_importance_map
from monai.inferers.utils import sliding_window_inference
from monai.networks.nets.vista3d import VISTA3D
from record_full_network_reference import seeded_model
from record_interactive_reference import Encoder


class PointHead(torch.nn.Module):
    """Mirror the seeded point stand-in used by the native window test."""

    def __init__(self, seed: int) -> None:
        super().__init__()
        rng = np.random.default_rng(seed)
        self.scale = float(np.float32(rng.uniform(0.5, 1.5)))
        self.bias = float(np.float32(rng.uniform(-0.25, 0.25)))

    def forward(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
        codes: torch.Tensor,
        class_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        positions = ((points + 0.5) / torch.tensor(image.shape[2:], dtype=points.dtype)).float()
        target = 132 if class_vector is None else int(class_vector.item())
        return (
            image * self.scale
            + self.bias
            + positions.sum() * 0.125
            + codes.float().sum() * 0.03125
            + target * 0.00390625
        )


class AutomaticRecorder:
    """Capture MONAI windows and execute the selected automatic predictor."""

    def __init__(
        self,
        roi: tuple[int, ...],
        weight: torch.Tensor,
        padded: tuple[int, ...],
        model: VISTA3D | None,
        classes: np.ndarray,
        scale: torch.Tensor,
        bias: torch.Tensor,
    ) -> None:
        self.roi, self.weight, self.model = roi, weight, model
        self.classes, self.scale, self.bias = classes, scale, bias
        self.counts = np.zeros(padded, dtype=np.float32)
        self.coordinates: list[list[int]] = []

    def __call__(self, patch: torch.Tensor, coords: list[list[slice]]) -> torch.Tensor:
        slices = tuple(coords[0][2:])
        self.coordinates.append([s.start for s in slices])
        self.counts[slices] += self.weight.numpy()
        if self.model is not None:
            return self.model(
                patch, class_vector=torch.from_numpy(self.classes)[:, None], transpose=True
            )
        x, y, z = torch.meshgrid(
            *(torch.arange(v, dtype=torch.float32) for v in self.roi), indexing="ij"
        )
        ramp = x * 0.125 + y * 0.0625 + z * 0.03125
        return patch * self.scale + self.bias + ramp[None, None]


def main() -> None:
    """Regenerate automatic and point window fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct", type=Path, default=Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"
    )
    parser.add_argument("--references", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    if monai_version != "1.6.0":
        raise RuntimeError(f"Expected MONAI 1.6.0, got {monai_version}")
    torch.set_num_threads(4)
    ct = np.asarray(nib.load(args.ct).dataobj, dtype=np.float32)
    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64-cpu"
    args.references.mkdir(parents=True, exist_ok=True)
    metadata = {
        "source": "HCC-TACE-Seg HCC_063 contrast CT",
        "bundle_revision": "f9f5f51b589e5dc9c23c453cf5138398e4084056",
        "reference_backend": f"{platform.system().lower()}-{platform.machine()}-cpu",
        "monai_version": monai_version,
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "nibabel_version": nib.__version__,
        "scipy_version": scipy.__version__,
        "scikit_image_version": skimage.__version__,
    }
    for name in (
        "window_constant",
        "window_padded",
        "window_gaussian",
        "point_window_overlap",
        "point_window_padded",
        "point_window_multi",
        "full_window_constant",
        "full_point_window_overlap",
    ):
        path = fixtures / f"{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            original = dict(archive)
        info = json.loads(str(original["metadata"]))
        full_network = name.startswith("full_")
        info.update(
            reference="MONAI 1.6.0 VISTA3D"
            if full_network
            else "MONAI 1.6.0 with deterministic model boundary",
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            nibabel_version=nib.__version__,
        )
        shape = (
            tuple(original["image_shape"].tolist())
            if "image_shape" in original
            else original["image"].shape
        )
        spatial = shape[2:]
        crop = ct[100 : 100 + spatial[0], 210 : 210 + spatial[1], 55 : 55 + spatial[2]]
        if crop.shape != spatial:
            raise ValueError(f"CT crop {crop.shape} does not match {spatial}")
        image = np.clip(
            (crop + np.float32(963.8247715525971))
            / np.float32(1053.678477684517 + 963.8247715525971),
            0,
            1,
        )[None, None]
        arrays = {"image": image, "real_metadata": np.asarray(json.dumps(metadata))}
        retained = {"metadata": np.asarray(json.dumps(info)), "image_shape": np.asarray(shape)}
        roi = tuple(original["roi"].tolist())
        model = None
        if full_network:
            retained["weight_seed"] = original["weight_seed"]
            model = seeded_model(info["weight_shapes"], int(original["weight_seed"]))
        if name in {"window_constant", "window_padded", "window_gaussian", "full_window_constant"}:
            for key in ("roi", "mode", "network_seed", "class_ids"):
                retained[key] = original[key]
            rng = np.random.default_rng(int(original["network_seed"]))
            scale = torch.from_numpy(rng.standard_normal((1, 3, 1, 1, 1), dtype=np.float32))
            bias = torch.from_numpy(rng.standard_normal((1, 3, 1, 1, 1), dtype=np.float32))
            mode = str(original["mode"])
            weight = compute_importance_map(roi, mode=mode).float()
            padded = tuple(max(a, b) for a, b in zip(spatial, roi, strict=True))
            pad = tuple(
                (max(b - a, 0) // 2, (max(b - a, 0) + 1) // 2)
                for a, b in zip(spatial, roi, strict=True)
            )
            padded_image = np.pad(image, [(0, 0), (0, 0), *pad], mode="edge")
            predictor = AutomaticRecorder(
                roi, weight, padded, model, original["class_ids"], scale, bias
            )

            with torch.no_grad():
                planes = sliding_window_inference(
                    torch.from_numpy(padded_image),
                    roi,
                    1,
                    predictor,
                    overlap=0.25,
                    mode=mode,
                    roi_weight_map=weight,
                    with_coord=True,
                )
            crop_slices = tuple(slice(a, a + n) for (a, _), n in zip(pad, spatial, strict=True))
            logits = planes[(slice(None), slice(None), *crop_slices)]
            result = VistaPostTransformd(keys="pred")(
                {"pred": logits[0].clone(), "label_prompt": torch.from_numpy(original["class_ids"])}
            )["pred"]
            arrays.update(
                weight=weight.numpy(),
                planes=planes[0].numpy(),
                counts=predictor.counts,
                coordinates=np.asarray(predictor.coordinates),
                logits=logits.numpy(),
                labels=result.numpy()[None],
            )
        else:
            keys = (
                "roi",
                "points",
                "codes",
                "prompt",
                "previous",
                "point_start",
                "center_only",
                "margin",
                "network_seed",
            )
            retained.update({key: original[key] for key in keys})
            if model is None:
                model = VISTA3D(
                    Encoder(), torch.nn.Identity(), PointHead(int(original["network_seed"]))
                ).eval()
            with torch.no_grad():
                output = point_based_window_inferer(
                    inputs=torch.from_numpy(image),
                    roi_size=roi,
                    predictor=model,
                    point_coords=torch.from_numpy(original["points"]),
                    point_labels=torch.from_numpy(original["codes"]),
                    prompt_class=torch.from_numpy(original["prompt"])[None],
                    prev_mask=torch.from_numpy(original["previous"]),
                    point_start=int(original["point_start"]),
                    center_only=bool(original["center_only"]),
                    margin=int(original["margin"]),
                    transpose=True,
                )
            arrays["output"] = output.as_tensor().numpy()
        np.savez_compressed(args.references / f"{name}.npz", **arrays)
        np.savez_compressed(path, **retained)


if __name__ == "__main__":
    main()
