"""Record VISTA3D geometry and restoration using HCC-TACE CT and labels.

The retained test headers deliberately exercise unusual geometry. Identical
headers are applied to matching CT and segmentation crops after checking the
original HCC-TACE grids; they are controlled test geometry, not patient geometry.
"""

from __future__ import annotations

import argparse
import io
import json
import platform
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
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
    Spacingd,
)


def main() -> None:
    """Recompute MONAI preprocessing and inverse labels on real image crops."""
    cache = Path.home() / ".cache/medmlx/test-data"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ct", type=Path, default=cache / "hcc_063_ct.nii.gz")
    parser.add_argument("--liver", type=Path, default=cache / "hcc_063_liver.nii.gz")
    parser.add_argument("--tumor", type=Path, default=cache / "hcc_063_tumor.nii.gz")
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    if monai_version != "1.6.0":
        raise RuntimeError(f"Expected MONAI 1.6.0, got {monai_version}")
    torch.set_num_threads(4)
    images = [nib.load(path) for path in (args.ct, args.liver, args.tumor)]
    ct, liver, tumor = images
    for image in (liver, tumor):
        if image.shape != ct.shape or not np.allclose(image.affine, ct.affine, atol=1e-5, rtol=0):
            raise ValueError("HCC-TACE CT and segmentation grids do not match")
    ct_data = np.asarray(ct.dataobj, dtype=np.float32)
    liver_data = np.asarray(liver.dataobj) != 0
    tumor_data = np.asarray(tumor.dataobj) != 0
    if not liver_data.any() or not tumor_data.any():
        raise ValueError("The HCC-TACE liver and tumor masks must be nonempty")
    reference_path = (
        Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64-cpu/medmlx_geometry.npz"
    )
    with np.load(reference_path, allow_pickle=False) as archive:
        previous = dict(archive)
    info = json.loads(str(previous["metadata"]))
    arrays = {
        "real_metadata": np.asarray(
            json.dumps(
                {
                    "source": "HCC-TACE-Seg HCC_063 contrast CT, liver and tumor masks",
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
    header_only = {"metadata": previous["metadata"]}
    transform = Compose(
        [
            LoadImaged(keys=("image", "label"), image_only=True),
            EnsureChannelFirstd(keys=("image", "label")),
            EnsureTyped(keys=("image", "label"), device="cpu", track_meta=True),
            Spacingd(keys=("image", "label"), pixdim=(1.5, 1.5, 1.5), mode=("bilinear", "nearest")),
            CropForegroundd(
                keys=("image", "label"), source_key="image", allow_smaller=True, margin=10
            ),
            ScaleIntensityRanged(
                keys="image",
                a_min=-963.8247715525971,
                a_max=1053.678477684517,
                b_min=0,
                b_max=1,
                clip=True,
            ),
            Orientationd(keys=("image", "label"), axcodes="RAS"),
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="medmlx-geometry-") as temp:
        root = Path(temp)
        for name in info["cases"]:
            header_key = f"{name}::geometry_header"
            if header_key in previous:
                header_bytes = previous[header_key].tobytes()
            else:
                old = nib.Nifti1Image.from_bytes(previous[f"{name}::nifti"].tobytes())
                header_bytes = old.header.binaryblock
            header = nib.Nifti1Header.from_fileobj(io.BytesIO(header_bytes))
            shape = header.get_data_shape()
            if len(shape) != 3:
                raise ValueError(f"Unexpected geometry shape for {name}: {shape}")
            region = tuple(
                slice(start, start + size)
                for start, size in zip((100, 210, 55), shape, strict=True)
            )
            crop = ct_data[region]
            if crop.shape != shape:
                raise ValueError(f"CT crop {crop.shape} does not match {shape}")
            labels = np.where(tumor_data[region], 5, np.where(liver_data[region], 3, 0))
            image = nib.Nifti1Image(crop, None, header=header)
            label = nib.Nifti1Image(labels.astype(np.float32), None, header=header)
            image.set_data_dtype(np.float32)
            label.set_data_dtype(np.float32)
            image_path, label_path = root / f"{name}.nii.gz", root / f"{name}-label.nii.gz"
            nib.save(image, image_path)
            nib.save(label, label_path)
            prepared = transform({"image": str(image_path), "label": str(label_path)})
            prediction = prepared["label"].clone()
            prepared["pred"] = prediction
            result = Invertd(
                keys="pred", transform=transform, orig_keys="image", nearest_interp=True
            )(prepared)
            result["pred"] = torch.nan_to_num(result["pred"], nan=255.0)
            output = root / name
            SaveImaged(
                keys="pred",
                output_dir=str(output),
                output_postfix="restored",
                output_ext=".nii.gz",
                separate_folder=False,
                resample=False,
                output_dtype=np.float32,
                print_log=False,
            )(result)
            saved = nib.load(output / f"{name}_restored.nii.gz")
            arrays.update(
                {
                    f"{name}::nifti": np.frombuffer(image.to_bytes(), dtype=np.uint8),
                    f"{name}::labels": prediction.as_tensor().numpy(),
                    f"{name}::image": prepared["image"].as_tensor().numpy(),
                    f"{name}::affine": prepared["image"].affine.numpy(),
                    f"{name}::restored_data": np.asarray(saved.dataobj),
                    f"{name}::restored_affine": saved.affine,
                    f"{name}::restored_codes": np.asarray(
                        [saved.header["qform_code"], saved.header["sform_code"]]
                    ),
                    f"{name}::restored_header": np.frombuffer(
                        saved.header.binaryblock, dtype=np.uint8
                    ),
                }
            )
            header_only[header_key] = np.frombuffer(header_bytes, dtype=np.uint8)
    np.savez_compressed(args.output / "medmlx_geometry.npz", **arrays)
    np.savez_compressed(reference_path, **header_only)


if __name__ == "__main__":
    main()
