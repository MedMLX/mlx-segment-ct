# mlx-segment-ct

Segment CT anatomy automatically or with anatomy names and clicks, using NVIDIA VISTA-3D in native MLX.

[Docs](https://github.com/MedMLX/mlx-segment-ct/blob/main/docs/technical.md) · [CT weights](https://huggingface.co/MedMLX/vista3d-ct-mlx) · [CTMR weights](https://huggingface.co/MedMLX/vista3d-ctmr-mlx) · [MedMLX](https://github.com/MedMLX)

| Input | Output |
| :--- | :--- |
| 3D CT · NIfTI | Anatomical labels on the input grid |

**Apple Silicon · Metal · Python 3.13 / 3.14** · [uv](https://docs.astral.sh/uv/)

## Install

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "mlx-segment-ct>=0.3.4"
```

NIfTI files explicitly marked in meters or microns are rejected. Inputs with
unknown spatial units are assumed to use millimeters.

The click coordinates below are examples: replace them with voxels inside your
image and its preprocessed foreground crop. `1` includes a point; `0` excludes it.

## Run

```python
from mlx_segment_ct.medmlx import run

run(image_path="ct.nii.gz", output_dir="all")
run(image_path="ct.nii.gz", output_dir="organs", prompt="liver, spleen")
run(
    image_path="ct.nii.gz", output_dir="points",
    points=[[40, 50, 30], [44, 51, 30]], point_labels=[1, 0],
)
```

Omitting `weights_path` downloads `MedMLX/vista3d-ct-mlx` once and reuses its local
cache. An explicit model folder contains `automatic.safetensors` and
`correction.safetensors`. Point coordinates are `(x, y, z)` voxels.

The registered `mlx-segment-ctmr` model uses `mlx_segment_ct.medmlx.run_ctmr` and its
own `MedMLX/vista3d-ctmr-mlx` folder, with CTMR preprocessing and labels selected
internally. The advanced `run(checkpoint="NV-Segment-CTMR", ...)`
interface still accepts an explicitly supplied matching folder. It never substitutes
CT weights for CTMR.

The published [NV-Segment-CTMR weights](https://huggingface.co/MedMLX/vista3d-ctmr-mlx)
cover CT and MRI under noncommercial terms. NVIDIA does not support interactive
point segmentation with that checkpoint.

Downloaded commits are returned in `details.weights_revision`; explicit local
weights report `null` unless the folder carries a MedMLX download revision.

## License

[Apache-2.0](https://github.com/MedMLX/mlx-segment-ct/blob/main/LICENSE). See [third-party notices](https://github.com/MedMLX/mlx-segment-ct/blob/main/THIRD_PARTY_NOTICES.md). Model weights retain their publisher terms.

For research and education. Outputs require human review.
