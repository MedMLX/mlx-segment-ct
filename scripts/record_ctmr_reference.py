"""Record the pinned CTMR preprocessing transforms on HCC-TACE CT crops.

Run with MONAI 1.6.0 and its Torch CPU dependencies. The reference cases use
controlled geometry to exercise orientation and resampling with real CT pixels.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.transforms import (
    Compose,
    CropForegroundd,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    Orientationd,
    ScaleIntensityRangePercentiles,
    ScaleIntensityRangePercentilesd,
    Spacingd,
)


def main() -> None:
    """Save independent MONAI outputs outside Git and retain non-image metadata."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ct", type=Path, default=Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"
    )
    parser.add_argument("--output", type=Path, default=Path("work/real-reference"))
    args = parser.parse_args()
    if monai_version != "1.6.0":
        raise RuntimeError(f"Expected MONAI 1.6.0, got {monai_version}")
    torch.set_num_threads(4)
    source = nib.load(args.ct)
    ct = np.asarray(source.dataobj, dtype=np.float32)
    old_path = (
        Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64-cpu/medmlx_ctmr.npz"
    )
    with np.load(old_path, allow_pickle=False) as archive:
        original = dict(archive)
    arrays: dict[str, np.ndarray] = {}
    arrays["real_metadata"] = np.asarray(
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
    scaling = ScaleIntensityRangePercentiles(lower=1, upper=99, b_min=0, b_max=1, clip=True)
    crops = {
        "normal": ct[120:129, 230:238, 30:37][None],
        "integers": ct[120:126, 230:235, 30:34][None],
        "constant": ct[0:3, 0:3, 0:3][None],
        "pair": ct[120:122, 230:231, 30:31][None],
        "skewed": ct[120:140, 230:247, 30:43][None],
        "large": ct[100:204, 200:300, 0:97][None],
    }
    for name, image in crops.items():
        if image.size == 0:
            raise ValueError(f"Empty CT crop: {name}")
        arrays[f"percentile::{name}::input"] = image
        arrays[f"percentile::{name}::output"] = scaling(torch.from_numpy(image)).numpy()
        if name == "large":
            arrays[f"percentile::{name}::shape"] = np.asarray(image.shape)

    transforms = Compose(
        [
            LoadImaged(keys="image", image_only=True),
            EnsureChannelFirstd(keys="image"),
            EnsureTyped(keys="image", device="cpu", track_meta=True),
            Spacingd(keys="image", pixdim=(1.5, 1.5, 1.5), mode="bilinear"),
            CropForegroundd(keys="image", source_key="image", allow_smaller=True, margin=10),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=1, upper=99, b_min=0, b_max=1, clip=True
            ),
            Orientationd(keys="image", axcodes="RAS"),
        ]
    )
    args.output.mkdir(parents=True, exist_ok=True)
    geometries = {
        "lps": (
            (40, 44, 20),
            [[-0.8, 0, 0, 120.5], [0, -0.8, 0, 110.25], [0, 0, 1.25, -200], [0, 0, 0, 1]],
        ),
        "sagittal": (
            (30, 40, 28),
            [[0, 0, -1.2, 50], [1, 0, 0, -80], [0, -0.9, 0, 40], [0, 0, 0, 1]],
        ),
        "oblique": (
            (40, 36, 18),
            [
                [0.89780766, -0.06278083, 0, -30],
                [0.06278083, 0.89780766, 0, 20],
                [0, 0, 2, 10],
                [0, 0, 0, 1],
            ],
        ),
        "large": ((112, 104, 96), np.diag([1.5, 1.5, 1.5, 1.0])),
    }
    for name, (shape, geometry) in geometries.items():
        affine = np.asarray(geometry, dtype=np.float64)
        crop = ct[100 : 100 + shape[0], 200 : 200 + shape[1], 0 : shape[2]]
        if crop.shape != shape:
            raise ValueError(f"CT crop {crop.shape} does not match {shape}")
        image = nib.Nifti1Image(crop, affine)
        image.set_qform(affine, code=1)
        image.set_sform(affine, code=1)
        path = args.output / f"ctmr-{name}.nii.gz"
        nib.save(image, path)
        result = transforms({"image": str(path)})["image"]
        arrays[f"{name}::image"] = result.as_tensor().numpy()
        arrays[f"{name}::affine"] = result.affine.numpy()
        arrays[f"{name}::nifti"] = np.frombuffer(image.to_bytes(), dtype=np.uint8)
        if name == "large":
            arrays["large::shape"] = np.asarray(result.shape)
        path.unlink()
    np.savez_compressed(args.output / "medmlx_ctmr.npz", **arrays)
    metadata = {
        k: v
        for k, v in original.items()
        if k == "metadata" or k.startswith(("everything::", "request::"))
    }
    np.savez_compressed(old_path, **metadata)


if __name__ == "__main__":
    main()
