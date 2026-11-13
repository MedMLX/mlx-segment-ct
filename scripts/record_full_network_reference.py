"""Record seeded full MONAI VISTA3D graphs on an HCC-TACE CT crop."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.networks.nets.vista3d import VISTA3D, vista3d132


def seeded_model(shapes: dict[str, list[int]], seed: int) -> VISTA3D:
    """Load the same reproducible parameter sequence as the native graph tests."""
    rng = np.random.default_rng(seed)
    model = vista3d132().eval()
    upstream = model.state_dict()
    weights = {}
    for key, shape in shapes.items():
        if key not in upstream or list(upstream[key].shape) != shape:
            raise ValueError(f"Upstream parameter contract differs: {key}")
        value = rng.standard_normal(shape, dtype=np.float32) * np.float32(0.02)
        if ".norm" in key and key.endswith(".weight"):
            value += np.float32(1)
        weights[key] = torch.from_numpy(value)
    model.load_state_dict(weights, strict=False)
    return model


def main() -> None:
    """Run the upstream encoder and automatic and point heads on real CT."""
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
    crop = ct[100:116, 210:226, 55:87]
    if crop.shape != (16, 16, 32):
        raise ValueError(f"Unexpected CT crop shape {crop.shape}")
    image = np.clip(
        (crop + np.float32(963.8247715525971)) / np.float32(1053.678477684517 + 963.8247715525971),
        0,
        1,
    )[None, None]
    fixtures = Path(__file__).resolve().parents[1] / "tests/fixtures/darwin-arm64-cpu"
    args.references.mkdir(parents=True, exist_ok=True)
    for kind in ("feature", "automatic", "point"):
        name = f"full_{kind}"
        path = fixtures / f"{name}.npz"
        with np.load(path, allow_pickle=False) as archive:
            original = dict(archive)
        info = json.loads(str(original["metadata"]))
        info.pop("output_parts", None)
        info.update(
            torch_version=torch.__version__,
            numpy_version=np.__version__,
            nibabel_version=nib.__version__,
        )
        model = seeded_model(info["weight_shapes"], int(original["weight_seed"]))
        with torch.no_grad():
            if kind == "point":
                features, _ = model.image_encoder(
                    torch.from_numpy(image), with_point=True, with_label=False
                )
                positions = torch.from_numpy(original["positions"])
                points = positions * torch.tensor(image.shape[2:], dtype=torch.float32) - 0.5
                output = model.point_head(
                    features,
                    points,
                    torch.from_numpy(original["codes"]),
                    class_vector=torch.tensor([[int(original["target"])]]),
                ).numpy()
            else:
                _, features = model.image_encoder(
                    torch.from_numpy(image), with_point=False, with_label=True
                )
                output = (
                    features.numpy()
                    if kind == "feature"
                    else model.class_head(
                        features, torch.from_numpy(original["class_ids"])[:, None]
                    )[0]
                    .transpose(0, 1)
                    .numpy()
                )
        meta = {
            "source": "HCC-TACE-Seg HCC_063 contrast CT",
            "bundle_revision": "f9f5f51b589e5dc9c23c453cf5138398e4084056",
            "reference_backend": f"{platform.system().lower()}-{platform.machine()}-cpu",
            "monai_version": monai_version,
            "torch_version": torch.__version__,
            "numpy_version": np.__version__,
            "nibabel_version": nib.__version__,
        }
        np.savez_compressed(
            args.references / f"{name}.npz",
            image=image,
            output=output,
            real_metadata=np.asarray(json.dumps(meta)),
        )
        retained = {
            key: value
            for key, value in original.items()
            if key not in {"metadata", "image", "output"}
        }
        retained["metadata"] = np.asarray(json.dumps(info))
        np.savez_compressed(path, **retained)


if __name__ == "__main__":
    main()
