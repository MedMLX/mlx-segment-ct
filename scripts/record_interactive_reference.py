"""Record interactive VISTA3D runners on the HCC-TACE geometry references.

MONAI owns point windows, class/point fusion, label postprocessing and inverse
geometry. Lightweight deterministic Torch heads replace learned model weights;
their outputs are recorded for replay by the native runner tests.
"""

from __future__ import annotations

import argparse
import gzip
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import scipy
import skimage
import torch
from monai import __version__ as monai_version
from monai.apps.vista3d.inferer import point_based_window_inferer
from monai.apps.vista3d.transforms import VistaPostTransformd, VistaPreTransformd
from monai.networks.nets.vista3d import VISTA3D
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    Invertd,
    LoadImaged,
    Orientationd,
    SaveImaged,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)


class Encoder(torch.nn.Module):
    """Pass the source patch to each requested head."""

    def forward(self, image: torch.Tensor, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        return image, image


class AutomaticHead(torch.nn.Module):
    """Record deterministic automatic logits for replay."""

    def __init__(self, records: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.records = records
        self.calls = 0

    def forward(self, image: torch.Tensor, classes: torch.Tensor) -> tuple[torch.Tensor, None]:
        output = image * 0.5 - 0.25
        self.records[f"auto_output_{self.calls}"] = output.detach().cpu().numpy().copy()
        self.calls += 1
        return output, None


class PointHead(torch.nn.Module):
    """Record point-head inputs and deterministic image-dependent logits."""

    last_supported = 132

    def __init__(self, records: dict[str, np.ndarray]) -> None:
        super().__init__()
        self.records = records
        self.calls = 0

    def forward(
        self,
        image: torch.Tensor,
        points: torch.Tensor,
        codes: torch.Tensor,
        class_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shape = torch.tensor(image.shape[2:], dtype=points.dtype)
        normalized = ((points + 0.5) / shape).float()
        output = image * 0.75 - 0.35 + (normalized.mean() - 0.5) * 0.125
        self.records[f"normalized_{self.calls}"] = normalized.detach().cpu().numpy().copy()
        self.records[f"patch_codes_{self.calls}"] = codes.detach().cpu().numpy().copy()
        self.records[f"point_output_{self.calls}"] = output.detach().cpu().numpy().copy()
        self.calls += 1
        return output


def main() -> None:
    """Record all interactive runner cases and retain only prompt data in Git."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    if monai_version != "1.6.0":
        raise RuntimeError(f"Expected MONAI 1.6.0, got {monai_version}")
    torch.set_num_threads(4)
    with np.load(args.references / "medmlx_geometry.npz", allow_pickle=False) as archive:
        geometry = dict(archive)
    fixture_root = Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64-cpu"
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
    with TemporaryDirectory(prefix="medmlx-interactive-") as temporary:
        root = Path(temporary)
        for suffix in ("runner", "ctmr_runner", "correction_runner"):
            name = f"interactive_{suffix}"
            path = fixture_root / f"{name}.npz"
            with np.load(path, allow_pickle=False) as archive:
                old = dict(archive)
            info = json.loads(str(old["metadata"]))
            info.update(
                reference="MONAI 1.6.0 VISTA3D with deterministic model boundary",
                torch_version=torch.__version__,
                numpy_version=np.__version__,
                nibabel_version=nib.__version__,
            )
            retained = {"metadata": np.asarray(json.dumps(info))}
            arrays = {"real_metadata": np.asarray(json.dumps(metadata))}
            intensity = (
                ScaleIntensityRangePercentilesd(
                    keys="image", lower=1, upper=99, b_min=0, b_max=1, clip=True
                )
                if suffix == "ctmr_runner"
                else ScaleIntensityRanged(
                    keys="image",
                    a_min=-963.8247715525971,
                    a_max=1053.678477684517,
                    b_min=0,
                    b_max=1,
                    clip=True,
                )
            )
            transform = Compose(
                [
                    LoadImaged(keys="image", image_only=True),
                    EnsureChannelFirstd(keys="image"),
                    EnsureTyped(keys="image", device="cpu", track_meta=True),
                    Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
                    CropForegroundd(
                        keys="image", source_key="image", allow_smaller=True, margin=10
                    ),
                    intensity,
                    Orientationd(keys="image", axcodes="RAS"),
                ]
            )
            for case_name, case in info["cases"].items():
                prefix = f"{case_name}::"
                source_points, codes = (old[prefix + key] for key in ("source_points", "codes"))
                retained[prefix + "source_points"] = source_points
                retained[prefix + "codes"] = codes
                raw = geometry[prefix + "nifti"]
                source = root / f"{name}-{case_name}.nii.gz"
                source.write_bytes(gzip.compress(raw.tobytes(), mtime=0))
                data = transform({"image": str(source)})
                image = data["image"]
                old_affine = nib.load(source).affine
                affine = image.affine.numpy()
                homogeneous = np.concatenate(
                    [source_points[0], np.ones((source_points.shape[1], 1))], axis=1
                )
                points = (np.linalg.inv(affine) @ old_affine @ homogeneous.T).T[None, :, :3]
                target = case["target"]
                prompt = VistaPreTransformd(keys="image", subclass=None, labels_dict=None)(
                    {
                        "label_prompt": None if target is None else [target],
                        "point_labels": codes[0].tolist(),
                    }
                )
                effective = np.asarray([prompt["point_labels"]], dtype=np.int64)
                positions = torch.from_numpy(points)
                point_codes = torch.from_numpy(effective)
                prompt_class = None if target is None else torch.tensor([[target]])
                class_vector = prompt_class if target is not None and target <= 132 else None
                previous = None
                if suffix == "correction_runner":
                    previous = image[None].as_tensor() * 2 - 1
                records: dict[str, np.ndarray] = {}
                model = VISTA3D(Encoder(), AutomaticHead(records), PointHead(records)).eval()
                with torch.no_grad():
                    logits = point_based_window_inferer(
                        inputs=image[None],
                        roi_size=case["roi"],
                        predictor=model,
                        point_coords=positions,
                        point_labels=point_codes,
                        class_vector=class_vector,
                        prompt_class=prompt_class,
                        prev_mask=previous,
                        point_start=case.get("point_start", 0),
                        transpose=True,
                    )
                arrays.update({prefix + key: value for key, value in records.items()})
                arrays.update(
                    {
                        prefix + "nifti": raw,
                        prefix + "image": image[None].as_tensor().numpy(),
                        prefix + "points": points,
                        prefix + "logits": logits.as_tensor().numpy().copy(),
                    }
                )
                if previous is not None:
                    arrays[prefix + "previous"] = previous.numpy().copy()
                data.update(
                    pred=logits[0],
                    points=positions,
                    point_labels=point_codes,
                    label_prompt=None if target is None else torch.tensor([target]),
                )
                data = VistaPostTransformd(keys="pred")(data)
                data = Invertd(
                    keys="pred", transform=transform, orig_keys="image", nearest_interp=True
                )(data)
                data["pred"] = torch.nan_to_num(data["pred"], nan=255.0)
                out = root / f"output-{name}-{case_name}"
                SaveImaged(
                    keys="pred",
                    output_dir=str(out),
                    output_postfix="mask",
                    output_ext=".nii.gz",
                    separate_folder=False,
                    resample=False,
                    output_dtype=np.float32,
                    print_log=False,
                )(data)
                saved = nib.load(out / f"{name}-{case_name}_mask.nii.gz")
                arrays[prefix + "mask"] = np.asarray(saved.dataobj)
                arrays[prefix + "mask_affine"] = saved.affine
            np.savez_compressed(args.references / f"{name}.npz", **arrays)
            np.savez_compressed(path, **retained)


if __name__ == "__main__":
    main()
