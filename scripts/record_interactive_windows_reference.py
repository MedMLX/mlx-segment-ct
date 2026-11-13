"""Record MONAI point-window cases using HCC-TACE CT crops.

The point windows and component operations come from the pinned MONAI VISTA3D;
deterministic Torch heads expose model-boundary inputs and outputs for replay.
"""

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
from monai.apps.vista3d.transforms import VistaPostTransformd, VistaPreTransformd
from monai.networks.nets.vista3d import VISTA3D
from record_interactive_reference import AutomaticHead, Encoder, PointHead


class RecordedVista(VISTA3D):
    """Capture image patches and bounds as MONAI sends them to the model."""

    def __init__(self, records: dict[str, np.ndarray]) -> None:
        super().__init__(Encoder(), AutomaticHead(records), PointHead(records))
        self.records = records
        self.calls = 0

    def forward(self, input_images: torch.Tensor, **kwargs: object) -> torch.Tensor:
        windows = kwargs["patch_coords"][0][-3:]
        self.records[f"patch_{self.calls}"] = input_images.detach().cpu().numpy().copy()
        self.records[f"windows_{self.calls}"] = np.asarray([[s.start, s.stop] for s in windows])
        self.calls += 1
        return super().forward(input_images, **kwargs)


def main() -> None:
    """Regenerate the interactive window archive and preserve its prompts."""
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
    path = (
        Path(__file__).resolve().parents[1]
        / "tests/fixtures/darwin-arm64-cpu/interactive_windows.npz"
    )
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
                    "scipy_version": scipy.__version__,
                    "scikit_image_version": skimage.__version__,
                }
            )
        )
    }
    for name, case in info["cases"].items():
        prefix = f"{name}::"
        shape_key = prefix + "image_shape"
        shape = tuple(old[shape_key].tolist()) if shape_key in old else old[prefix + "image"].shape
        retained[shape_key] = np.asarray(shape)
        points, codes = (old[prefix + key] for key in ("points", "codes"))
        retained[prefix + "points"], retained[prefix + "codes"] = points, codes
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
        target = case["target"]
        prompt = VistaPreTransformd(keys="image", subclass=None, labels_dict=None)(
            {
                "label_prompt": None if target is None else [target],
                "point_labels": codes[0].tolist(),
            }
        )
        effective = np.asarray([prompt["point_labels"]], dtype=np.int64)
        positions, point_codes = torch.from_numpy(points), torch.from_numpy(effective)
        prompt_class = None if target is None else torch.tensor([[target]])
        class_vector = (
            prompt_class
            if target is not None and (target <= 132 or (effective == -1).all())
            else None
        )
        records: dict[str, np.ndarray] = {}
        model = RecordedVista(records).eval()
        with torch.no_grad():
            logits = point_based_window_inferer(
                inputs=torch.from_numpy(image),
                roi_size=case["roi"],
                predictor=model,
                point_coords=positions,
                point_labels=point_codes,
                class_vector=class_vector,
                prompt_class=prompt_class,
                transpose=True,
            )
        arrays.update({prefix + key: value for key, value in records.items()})
        arrays.update(
            {
                prefix + "image": image,
                prefix + "effective_codes": effective,
                prefix + "logits": logits.as_tensor().numpy().copy(),
            }
        )
        post = {
            "pred": logits[0].clone(),
            "points": positions,
            "point_labels": point_codes,
            "label_prompt": None if target is None else torch.tensor([target]),
        }
        arrays[prefix + "labels"] = (
            VistaPostTransformd(keys="pred")(post)["pred"].as_tensor().numpy().copy()
        )
    args.references.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.references / "interactive_windows.npz", **arrays)
    np.savez_compressed(path, **retained)


if __name__ == "__main__":
    main()
