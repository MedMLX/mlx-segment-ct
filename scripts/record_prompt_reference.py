"""Record full MONAI VISTA3D class and point prompts on HCC-TACE CT."""

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
from record_full_network_reference import seeded_model


def main() -> None:
    """Recompute class and point outputs using upstream seeded networks."""
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
    meta = {
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
    automatic = None
    point = None
    for name in (
        "automatic_subset",
        "point_target_1",
        "point_target_25",
        "point_target_26",
        "point_target_117",
        "point_target_254",
        "point_previous",
    ):
        path = fixtures / f"{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            old = dict(archive)
        info = json.loads(str(old["metadata"]))
        info.pop("output_parts", None)
        info.update(
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            nibabel_version=nib.__version__,
        )
        shape = tuple(old["image_shape"].tolist()) if "image_shape" in old else old["image"].shape
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
        if name == "automatic_subset" and automatic is None:
            automatic = seeded_model(info["weight_shapes"], int(old["weight_seed"]))
        if name.startswith("point_") and point is None:
            point = seeded_model(info["weight_shapes"], int(old["weight_seed"]))
        with torch.no_grad():
            if name == "automatic_subset":
                assert automatic is not None
                output = automatic(
                    torch.from_numpy(image),
                    class_vector=torch.from_numpy(old["class_ids"])[:, None],
                    transpose=True,
                ).numpy()
            elif name == "point_previous":
                assert point is not None
                output = (
                    point_based_window_inferer(
                        inputs=torch.from_numpy(image),
                        roi_size=(32, 32, 32),
                        predictor=point,
                        point_coords=torch.from_numpy(old["points"]),
                        point_labels=torch.from_numpy(old["codes"]),
                        prompt_class=torch.tensor([[1]]),
                        prev_mask=torch.from_numpy(old["previous"]),
                        point_start=1,
                        center_only=True,
                        transpose=True,
                    )
                    .as_tensor()
                    .numpy()
                )
            else:
                assert point is not None
                output = point(
                    torch.from_numpy(image),
                    point_coords=torch.from_numpy(old["points"]),
                    point_labels=torch.from_numpy(old["codes"]),
                    prompt_class=torch.tensor([[int(old["target"])]]),
                    transpose=True,
                ).numpy()
        np.savez_compressed(
            args.references / f"{name}.npz",
            image=image,
            output=output,
            real_metadata=np.asarray(json.dumps(meta)),
        )
        retained = {
            key: value for key, value in old.items() if key not in {"metadata", "image", "output"}
        }
        retained["metadata"], retained["image_shape"] = (
            np.asarray(json.dumps(info)),
            np.asarray(shape),
        )
        np.savez_compressed(path, **retained)


if __name__ == "__main__":
    main()
