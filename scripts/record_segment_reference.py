"""Record the pinned VISTA3D automatic runner on real HCC-TACE CT crops.

The network boundary uses the deterministic stand-in from the runner tests;
MONAI performs the upstream preprocessing, sliding windows and restoration.
Run after record_geometry_reference.py with MONAI 1.6.0 and Torch on CPU.
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
import torch
from monai import __version__ as monai_version
from monai.apps.vista3d.transforms import VistaPostTransformd
from monai.inferers import SlidingWindowInfererAdapt
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


def main() -> None:
    """Run source transforms and MONAI window inference for each runner case."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--references", type=Path, default=Path("work/real-reference"))
    parser.add_argument("--ctmr", action="store_true", help="Record the CTMR bundle runner")
    args = parser.parse_args()
    if monai_version != "1.6.0":
        raise RuntimeError(f"Expected MONAI 1.6.0, got {monai_version}")
    torch.set_num_threads(4)
    name = "medmlx_ctmr_segment" if args.ctmr else "medmlx_segment"
    fixture_path = (
        Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64-cpu" / f"{name}.npz"
    )
    with np.load(fixture_path, allow_pickle=False) as archive:
        info = json.loads(str(archive["metadata"]))
    info.update(
        reference="MONAI 1.6.0 VISTA3D with deterministic model boundary",
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        nibabel_version=nib.__version__,
    )
    inputs = "medmlx_ctmr" if args.ctmr else "medmlx_geometry"
    with np.load(args.references / f"{inputs}.npz", allow_pickle=False) as archive:
        geometry = dict(archive)
    arrays = {
        "real_metadata": np.asarray(
            json.dumps(
                {
                    "source": "HCC-TACE-Seg HCC_063 contrast CT",
                    "bundle_revision": "f9f5f51b589e5dc9c23c453cf5138398e4084056",
                    "reference_backend": f"{platform.system().lower()}-{platform.machine()}-cpu",
                    "monai_version": monai_version,
                    "torch_version": torch.__version__,
                    "numpy_version": np.__version__,
                    "nibabel_version": nib.__version__,
                }
            )
        )
    }
    intensity = (
        ScaleIntensityRangePercentilesd(
            keys="image", lower=1, upper=99, b_min=0, b_max=1, clip=True
        )
        if args.ctmr
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
            CropForegroundd(keys="image", source_key="image", allow_smaller=True, margin=10),
            intensity,
            Orientationd(keys="image", axcodes="RAS"),
        ]
    )
    with TemporaryDirectory(prefix="medmlx-segment-") as temporary:
        root = Path(temporary)
        for name, case in info["cases"].items():
            roi = case["roi"] if args.ctmr else [128, 128, 128]
            inferer = SlidingWindowInfererAdapt(
                roi_size=roi, sw_batch_size=1, with_coord=True, padding_mode="replicate"
            )
            image_path = root / f"{name}.nii.gz"
            raw = geometry[f"{name}::nifti"]
            image_path.write_bytes(gzip.compress(raw.tobytes(), mtime=0))
            data = transform({"image": str(image_path)})
            ids = torch.tensor(case["class_ids"], dtype=torch.int64)
            shape = (1, len(ids), 1, 1, 1)
            scale = (2.0 ** -(ids % 3) / 4).float().reshape(shape)
            bias = ((ids % 5) / 16 - 0.25).float().reshape(shape)
            slope = (((ids % 4) - 1.5) / 4).float().reshape(shape)

            def predictor(
                image: torch.Tensor,
                coordinates: object,
                *,
                scale: torch.Tensor = scale,
                slope: torch.Tensor = slope,
                bias: torch.Tensor = bias,
            ) -> torch.Tensor:
                """Mirror the deterministic test model at the upstream window boundary."""
                del coordinates
                depth, height, width = image.shape[2:]
                ramp = (
                    torch.arange(depth, dtype=torch.float32)[:, None, None] / 2
                    + torch.arange(height, dtype=torch.float32)[None, :, None]
                    - torch.arange(width, dtype=torch.float32)[None, None, :]
                    - depth // 4
                ) / 4
                return image * scale + ramp[None, None] * slope + bias

            with torch.no_grad():
                logits = inferer(data["image"][None], predictor)
            data["pred"] = logits[0]
            data["label_prompt"] = ids
            data = VistaPostTransformd(keys="pred")(data)
            data = Invertd(
                keys="pred", transform=transform, orig_keys="image", nearest_interp=True
            )(data)
            data["pred"] = torch.nan_to_num(data["pred"], nan=255.0)
            output = root / name
            SaveImaged(
                keys="pred",
                output_dir=str(output),
                output_postfix="mask",
                output_ext=".nii.gz",
                separate_folder=False,
                resample=False,
                output_dtype=np.float32,
                print_log=False,
            )(data)
            saved = nib.load(output / f"{name}_mask.nii.gz")
            arrays.update(
                {
                    f"{name}::nifti": raw,
                    f"{name}::mask_data": np.asarray(saved.dataobj),
                    f"{name}::mask_affine": saved.affine,
                    f"{name}::mask_codes": np.asarray(
                        [saved.header["qform_code"], saved.header["sform_code"]]
                    ),
                    f"{name}::mask_header": np.frombuffer(saved.header.binaryblock, dtype=np.uint8),
                }
            )
            case["preprocessed_shape"] = list(data["image"][None].shape)
            case["padded_shape"] = [
                max(size, width) for size, width in zip(data["image"].shape[1:], roi, strict=True)
            ]
            case["patch_count"] = 2 if case["padded_shape"][0] > roi[0] else 1
    output_name = "medmlx_ctmr_segment" if args.ctmr else "medmlx_segment"
    np.savez_compressed(args.references / f"{output_name}.npz", **arrays)
    np.savez_compressed(fixture_path, metadata=np.asarray(json.dumps(info)))


if __name__ == "__main__":
    main()
