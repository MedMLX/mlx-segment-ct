# mlx-segment-ct technical notes

Full details behind the [README](../README.md).

## Overview

`mlx-segment-ct` provides native FP32 MLX inference for NVIDIA NV-Segment-CT
and NV-Segment-CTMR (VISTA-3D). It implements automatic class segmentation, anatomy
prompts, and one object's positive/negative point prompts from scratch or with previous
logits. NVIDIA does not support point segmentation with the CTMR checkpoint.
Model execution uses MLX on Apple Silicon Metal; NumPy and SciPy handle host I/O,
preprocessing and postprocessing. Importing the package or its MedMLX model declaration
does not import MLX.

The numerical reference is **MONAI 1.6.0**, including `vista3d132`,
`PointMappingSAM`, the point inferer and Vista transforms. Bundle transforms and
label definitions are pinned to NVIDIA NV-Segment-CTMR revision
`f9f5f51b589e5dc9c23c453cf5138398e4084056`. The retained configuration sections are
in `scripts/reference_bundle.py`; the canonical 0–254 vocabulary, NV-Segment-CTMR's
0–345 vocabulary and its modality label sets are in `labels.py`.

## Installation and weights

```bash
uv venv --python 3.13
source .venv/bin/activate
uv pip install "mlx-segment-ct>=0.3.4"
```

Pretrained checkpoints are not included in the package. The converted
[NV-Segment-CT bundle](https://huggingface.co/MedMLX/vista3d-ct-mlx) contains
`automatic.safetensors` and `correction.safetensors` at its root:

```python
from huggingface_hub import snapshot_download

weights = snapshot_download(
    "MedMLX/vista3d-ct-mlx", revision="e6f45856b4a59212ca10b44a9ade2e2d672f412b",
)
```

Pass the returned directory as `weights_path`. Automatic segmentation requires
`automatic.safetensors`; point inference requires both files. This bundle needs no
conversion extra and uses the default `checkpoint="NV-Segment-CT"`.

The [NV-Segment-CTMR bundle](https://huggingface.co/MedMLX/vista3d-ctmr-mlx)
at revision `5965596ca734e3c62c30a93cde8d43d2edc8fb56` provides the same
files for `checkpoint="NV-Segment-CTMR"`. Its weights are licensed for
noncommercial research or evaluation under the NVIDIA OneWay Non-Commercial
License; see its model card. NVIDIA does not support interactive point
segmentation with NV-Segment-CTMR.

## MedMLX runner

The `medmlx.models` entry point is `mlx_segment_ct.medmlx:MODEL`, ID
`mlx-segment-ct`. All runner arguments are keyword-only:

```python
from mlx_segment_ct.medmlx import run

common = dict(image_path="/path/to/ct.nii.gz", weights_path=weights)
result = run(**common, output_dir="/path/to/all")  # 117 automatic classes
result = run(**common, output_dir="/path/to/organs", label_prompt=[2, 3])
result = run(
    **common,
    output_dir="/path/to/points",
    points=[[40.0, 50.0, 30.0], [44.0, 51.0, 30.0]],
    point_labels=[1, 0],
)
result = run(
    **common,
    output_dir="/path/to/tumor",
    interactive=True,
    label_prompt=[26],
    points=[[40.0, 50.0, 30.0]],
    point_labels=[1],
)
```

Class prompts accept numeric `label_prompt` or canonical bundle names in `prompt`
(a list or comma-separated string, case insensitive). Kidney (2), lung (20) and
bone (21) expand to subclasses for automatic segmentation. Point inference accepts
at most one class and rejects these coarse prompts. IDs 133–254 need points.
The runner limits names to the documented vocabulary; it does not allocate new
IDs for arbitrary anatomy names.

`checkpoint="NV-Segment-CTMR"` follows that bundle's `configs/inference.json` at the
pinned revision: CT or MRI input, 1st–99th percentile intensity scaling instead of
the HU window, ROI 192×192×128 for automatic and point windows, and labels 0–345.
IDs 346–511 are zero-shot and need points; the deprecated IDs 133 and 137–145 are not
automatic. Without prompts, `modality` selects the bundle's all-classes set:
`MRI_BODY` (default, 50 classes), `CT_BODY` (the 117 NV-Segment-CT classes) or
`MRI_BRAIN` (IDs 214–345). `MRI_BRAIN` expects skull-stripped T1 input registered as
the model card describes; that external preprocessing script is not part of the
bundle and is not ported. As upstream does, uncovered voxels are written as 255,
which is also the NV-Segment-CTMR ID of `Left-AnG---angular-gyrus`.

Point coordinates are `(x, y, z)` voxel indices on the source NIfTI grid. Each click
must be inside the source grid and the foreground crop after preprocessing; use
coordinates selected for the input image. Codes are 0/1 for exclude/include, 2/3
for special exclude/include, and -1 for padding.
The bundle's special classes 25, 26, 27, 28, 29 and 117 map 0/1 to 2/3.
With no class, output foreground is 1. With active clicks, class IDs through 132
combine automatic and point scores; IDs above 132 use the point branch for either
checkpoint. As upstream does, a class prompt with all padding codes retains the
automatic branch, including IDs above 132.

Corrections consume **previous raw logits**, before sigmoid or postprocessing:

```python
result = run(
    **common,
    output_dir="/path/to/corrected",
    label_prompt=[26],
    previous_logits_path="/path/to/previous.npy",
    points=[[40.0, 50.0, 30.0], [44.0, 51.0, 30.0]],
    point_labels=[1, 0],
    point_start=1,
)
```

The `.npy` file must be float32 `(1, 1, X, Y, Z)` on the same preprocessed grid as
this CT, as produced by `interactive_logits`. The caller owns that image/class
association. Earlier clicks remain context; `point_start` selects new windows.
Corrections that enable the automatic branch still run it, following the bundle
inferer; other corrections combine point scores with previous scores.
Uncovered voxels keep previous logits. The runner produces a single-object mask,
without categorical delta merging or protection of other objects' labels.

The result contains `outputs.mask`, an absolute path to `mlx_segment_ct_mask.nii.gz`,
and JSON `details` with prompts, window parameters, timings, source revision and
the restoration used. Every mode writes float32 global labels, preserves source
spatial shape/affine, and maps uncovered NaNs to 255. Source files are preserved;
existing outputs require `overwrite=True`. Errors use `medmlx-core`'s input, asset and
execution error types.

### Restoration to the source grid

The bundle decides labels on the 1.5 mm grid and restores them with nearest
interpolation, so on a finer CT (for example 0.7 mm in-plane) every decision covers
two to three source voxels and boundaries step. By default (`restoration="scores"`)
the runner instead interpolates the scores trilinearly at each source voxel, at the
preprocessed position nearest restoration rounds, and decides there: the first
maximum class wins when positive for automatic runs, and positive scores take the
object label for point runs, after upstream's component selection. Score planes are
consumed as the axis-0 ring finalizes them, so only two planes of scores are held.
Positions beyond the foreground crop clamp to its edge and uncovered (NaN) neighbors
are left out with the remaining weights renormalized. A voxel whose nearest
preprocessed voxel is outside the crop is background, and one whose nearest voxel is
uncovered is 255, exactly as with nearest restoration. A 1.5 mm source is not
resampled, so both restorations agree there.

`restoration="bundle"` reproduces the bundle's nearest restoration; the recorded
runner comparisons use it. Score restoration is checked against an independent
SciPy trilinear interpolation of the complete score field on the nine geometries and
the automatic and point runner recordings, excluding decisions within 1e-4 of a tie.

## Array APIs and inference behavior

```python
from pathlib import Path

from medmlx_core.runtime import import_mlx
from mlx_segment_ct import (
    AutomaticModel,
    PointModel,
    InteractivePrompt,
    interactive_logits,
    interactive_labels,
    transform_points,
)
from mlx_segment_ct.bundle_transforms import load_volume, preprocess

mx = import_mlx()
weight_dir = Path(weights)
point_weights = {
    **mx.load(str(weight_dir / "automatic.safetensors")),
    **mx.load(str(weight_dir / "correction.safetensors")),
}
volume, affine = load_volume(Path("/path/to/ct.nii.gz"))
prepared = preprocess(volume, affine)
request = InteractivePrompt(points=((40.0, 50.0, 30.0),), point_labels=(1,))
request = transform_points(prepared, request)
point = PointModel(point_weights, mx=mx)
logits = interactive_logits(prepared.image[None], request, point, mx=mx, roi_size=(128, 128, 128))
labels = interactive_labels(logits, request)
```

Images are FP32 NCDHW `(1, 1, X, Y, Z)`; full network dimensions must be divisible
by 16 and have more than one voxel at the deepest normalization level. For a
class that enables the automatic branch, supply `automatic_model=AutomaticModel(...)`
for that class. `interactive_logits` also accepts `previous_logits` and `point_start`.
`PointModel` directly accepts normalized voxel centers `(1, N, 3)` and codes
`(1, N)`; `normalized_point_positions` performs the center shift/division in host
precision before casting to FP32.

Preprocessing requires input spatial units to be millimeters, or unknown and
assumed to be millimeters. Inputs explicitly marked in meters or microns must be
converted before inference; they are rejected rather than silently rescaled.
In the current source, image voxels must be finite: NaN and infinities are rejected
before resampling or intensity scaling.
Outputs use new NIfTI headers rather than copying the input unit code.
Preprocessing reproduces the bundle's load, 1.5 mm spacing, foreground crop with
margin 10, HU or percentile scaling to [0, 1], RAS orientation and FP32 casting.
Percentiles follow MONAI's `percentile`: `torch.quantile` (float32 weight, fused
lerp) up to one million voxels and `np.percentile` above. Clicks transform
through `inv(preprocessed_affine) @ original_affine`. Bundle restoration uses nearest
interpolation and the bundle's float32 NIfTI save behavior.

Automatic inference uses ROI 128³ (192×192×128 for NV-Segment-CTMR), batch one,
overlap 0.25, constant blending and replicate padding. These are the executable
defaults of the bundle inferer's `SlidingWindowInfererAdapt` call: it does not forward
the configuration's overlap 0.3 or Gaussian mode to that automatic call.
`automatic_logits` exposes full scores;
`rolling_logits` uses an axis-0 ring, storing
`classes * roi[0] * padded[1] * padded[2] * 4` score bytes. It preserves MONAI window
order, FP32 accumulation and first-channel positive ties.

Point windows are centered per selected click, use constant-zero padding and
strict exclusion of both patch faces. An ineligible patch returns upstream's
-9999 sentinel. Overlaps sum because MONAI's coverage mask is binary.
`point_based_window_inferer` exposes centered/noncentered windows and previous
logits; `PointWindowPredictor` handles optional automatic scores and upstream
connected-component fusion. Inputs are preserved.

Without a class prompt, postprocessing selects 26-connected components at rounded
positive clicks. It reproduces the upstream mean fill of unselected positive
regions, even when that mean remains positive, and retains NaNs until restoration.
Negative-only clicks skip this component selection. Numerical parity does not
establish clinical segmentation quality.

## Verification and fixture regeneration

For development from a checkout, `make env` creates the locked environment and
`make qa` runs Ruff and strict Pyright without running models.

Recorders execute unmodified MONAI/Torch numerical code on **CPU**, never the MLX
port. They need no checkpoint, upstream source checkout or Metal access. The
current development environment uses patched MONAI 1.6.1; historical recorders
explicitly require 1.6.0 and cannot run in that environment. Do not downgrade
the project environment or describe existing fixtures as refreshed.

For historical reproduction only, create a separate temporary CPU-oracle
environment outside the checkout. It runs synthetic inputs and does not install
the MLX package. Retain the resulting framework metadata with any recordings.
Moving the maintained reference to MONAI 1.6.1 requires a separate coordinated
pin migration, fixture regeneration and numerical review.
Seeds and their original draw ordering are retained in `fixture_inputs.py` and
`weight_order.json`; graph shapes are derived from MONAI's state dict. Full graph
fixtures use 16×16×32 instead of the former 16³ input, which upstream InstanceNorm
rejects. Large outputs are split into files below 900 KB.

```bash
oracle_dir=$(mktemp -d "${TMPDIR:-/tmp}/mlx-segment-historical.XXXXXX")
uv venv --python 3.13 "$oracle_dir/venv"
oracle_python="$oracle_dir/venv/bin/python"
uv pip install --python "$oracle_python" "monai==1.6.0" "torch>=2.13,<3" \
  "numpy>=2,<3" "nibabel>=5.3,<6" "scikit-image>=0.25,<1" "einops>=0.8,<1"
export PYTHONDONTWRITEBYTECODE=1
export MPLCONFIGDIR="$oracle_dir/matplotlib"
"$oracle_python" scripts/make_reference_fixtures.py --full-models
"$oracle_python" scripts/make_medmlx_fixtures.py
"$oracle_python" scripts/make_interactive_fixtures.py
```

Fixtures live in `tests/fixtures/<system>-<machine>-cpu/`, keyed to the **reference**
backend, so the same machine's CPU oracles are used when testing MLX on Metal.
Each records schema version, MONAI/Torch/NumPy versions and NVIDIA bundle revision.
Missing recordings fail with a regeneration message. Metal tests skip when Metal
is inaccessible; optional released-checkpoint comparisons also skip when
`NV_SEGMENT_CT_WEIGHTS_DIR` or `NV_SEGMENT_CT_CUDA_FIXTURES` is unset.
No recordings from another implementation are retained.

Coverage includes nine NIfTI geometries, spacing/restoration grids, bundle prompt
expansion, special codes, conversion layouts, transposed convolution at every
encoder width, positional encoding, supported/zero-shot tokens, GELU, two-way and
4096-key attention, full encoder/automatic/point graphs, all automatic classes,
point windows, previous-logit correction and end-to-end runner publication.
Window and runner fixtures use seeded small encoder/class-head stand-ins where
stated; separate fixtures exercise complete networks. Parser properties check
round trips and malformed-input failures. Tests do not import MONAI or recorders.

Host deterministic transformations, coordinates, component decisions, labels and
conversion copies use bitwise comparisons. Network comparisons use explicit FP32
bounds beside assertions; seeded graphs retain a strict 1e-6 absolute bound
with zero relative tolerance. The existing Torch transposed-convolution
comparison at full network widths retains its 1e-4 absolute/relative bounds. The
noncentered 27-window stand-in case uses the observed maximum `3 * 2**-18`
(approximately 1.145e-5): Torch and NumPy sum normalized FP32 clicks differently,
then window accumulation amplifies that difference. Foreground decisions are bitwise.

```bash
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export HYPOTHESIS_STORAGE_DIRECTORY="$TMPDIR/mlx-segment-hypothesis"
export RUFF_CACHE_DIR="$TMPDIR/mlx-segment-ruff"
python -m pytest -q -o "cache_dir=$TMPDIR/mlx-segment-pytest"
ruff check .
make typecheck
```

The October 13, 2026 Metal run passed all 185 seeded checks on an Apple M1 Max
with MLX 0.32.3 and macOS 27.0.1. The twelve previous numerical failures were
fixed without changing fixtures, tolerances, or foreground-equality assertions;
see [the numerical parity record](validation/numerical-parity-20261013.json).
InstanceNorm uses compensated two-pass FP32 statistics and source-rounded affine
operations. Decoder upsampling contracts channels before accumulating spatial
contributions. GELU follows PyTorch 2.14.1's ARM erf/exponential approximation,
and the 48-feature biased projections retain the measured Accelerate tile orders.
All network operations execute in MLX on Metal.

The 0.2.1 release also passed all seven real-input 128³ point-network comparisons
against unmodified MONAI 1.6.0 VISTA3D on an RTX 4090, using PyTorch 2.13.0 CUDA,
the original released NV-Segment-CT Torch checkpoint at revision
`afb51518689f71e6abb367ee6301b2cd0225c66a`, float32, and disabled AMP and TF32.
Cases cover native, flipped and oblique grids, include/exclude clicks, a padded
small volume, aorta, and genuine pancreatic-tumor ground truth. Inputs and clicks
were fixed from AMOS `amos_0018` and MSD Task07 `pancreas_001` annotations before
model execution; paired image/label shapes and affines were checked first.
The dataset split memberships permit training exposure, so this is a numerical
port check, not held-out model-quality or clinical evaluation.

The existing gate compares raw `PointModel` logits for window 0 in each case with
unchanged `rtol=1e-4`, `atol=1e-4`, plus exact foreground equality. The exclude
request produced two upstream windows; both were retained, while this gate
checks window 0. It does not compare full-volume preprocessing, restoration or
automatic labels with CUDA. A separate fresh-wheel public API smoke converted the
released safetensors without Torch and segmented a real GT-selected 64³ liver
crop on Metal, preserving its shape and affine with finite float32 labels.
All 219 tests, including the seven external comparisons, passed with zero skips.

The external comparison inputs are selected with `NV_SEGMENT_CT_WEIGHTS_DIR`
(the native NV-Segment-CT bundle) and `NV_SEGMENT_CT_CUDA_FIXTURES` (independent
upstream capture directory), then checked by `make test`. Real CT crops, labels,
checkpoints and captured arrays remain private and outside Git/distributions.
Full-volume pretrained automatic segmentation, broader task quality and clinical
performance remain unqualified.

## Normalization fusion performance

On October 28, 2026, fusing the InstanceNorm affine with ReLU, or with residual
addition followed by leaky ReLU, preserved the compensated two-pass statistics
and individual FP32 rounding steps. The baseline was the original implementation
at revision `28711f931e6a60689167d7e59f81720530bf8a2b`; the candidate changed only
these normalization epilogues and their callers.

An Apple M5 Max with MLX 0.32.4 and `MLX_ENABLE_TF32=0` evaluated the retained
`vista3d_auto.runtime.npz` input with the released automatic weights: float32
`(1, 1, 128, 128, 128)`, all 117 class prompts, and
`(1, 117, 128, 128, 128)` logits. Measurements covered warm, resident
`AutomaticModel` forwards, excluding model construction, weight loading and host
output transfer. After two warmups per implementation, seven paired samples
alternated baseline/candidate order; every forward was evaluated and synchronized.
No other task-owned GPU benchmark ran concurrently.

Median latency fell from **563.246 ms to 524.966 ms** (6.80% lower, 1.073× speedup).
Baseline samples ranged from 537.504 to 589.486 ms; fused samples ranged from
505.666 to 544.930 ms. Every fused sample was faster than its paired baseline.
Raw samples in measurement order, in milliseconds:

| Pair | Baseline | Fused |
| --- | ---: | ---: |
| 1 | 554.042833 | 505.666292 |
| 2 | 537.504208 | 507.430666 |
| 3 | 546.873958 | 512.078541 |
| 4 | 589.486458 | 524.966166 |
| 5 | 565.159125 | 533.582250 |
| 6 | 581.213958 | 544.930417 |
| 7 | 563.246250 | 528.134375 |

All 245,366,784 output float32 bit patterns matched the original implementation.
Focused validation passed 17 tests: 14 normalization epilogue comparisons,
including signed zero, infinities and NaNs, and the three existing full feature,
automatic and point graph fixtures. Those seeded MONAI fixtures retained their
`rtol=0`, `atol=1e-6` gate and existing foreground checks. Separately, the retained
pretrained 128³ upstream archive differed from both original and fused outputs
by a maximum absolute 5.340576171875e-5; the fusion introduced no additional
difference. This archive comparison does not claim the seeded 1e-6 bound for the
pretrained workload. Touched source and tests passed strict Pyright and Ruff.

## Upstream and license

- [NVIDIA NV-Segment-CTMR source](https://github.com/NVIDIA-Medtech/NV-Segment-CTMR),
  revision `f9f5f51b589e5dc9c23c453cf5138398e4084056`.
- [MONAI 1.6.0 VISTA3D source](https://github.com/Project-MONAI/MONAI/blob/1.6.0/monai/networks/nets/vista3d.py).
- [NVIDIA NV-Segment-CT weights](https://huggingface.co/nvidia/NV-Segment-CT), under
  the [NVIDIA Open Model License Agreement](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).

The [pinned NVIDIA CT configuration](https://raw.githubusercontent.com/NVIDIA-Medtech/NV-Segment-CTMR/f9f5f51b589e5dc9c23c453cf5138398e4084056/NV-Segment-CT/configs/inference.json)
has retained sections in `scripts/reference_bundle.py`; network and transform
recorders use the released MONAI 1.6.0 implementation.

Package code is Apache-2.0; see [LICENSE](../LICENSE). Upstream-derived code retains
its original license obligations; see pyproject.toml and THIRD_PARTY_NOTICES.md.
