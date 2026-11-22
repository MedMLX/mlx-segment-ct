# Technical notes

## Automatic CT and MRI inference

The automatic runner projects up to 64 classes at a time. Above its resident
MLX allocations, it allows space for four full-resolution, 48-channel FP32
feature maps, or six when the classes span multiple projections. MLX can wait
for queued work to finish when it reaches this limit. The runner respects a
caller's lower limit and restores the previous setting after inference. Score
restoration keeps each class plane contiguous and accumulates the eight
neighboring scores in their original FP32 order.

On the M1 Max (64 GB), the baseline was review commit `b23cc9c`. On the
M5 Max (128 GB), the baseline in the table also included fused decoder
upsampling and class-block stitching. The M5 rows measure the additional
effects of the memory limit and score restoration.
Each value is the median of six alternating fresh-process runs per variant on
November 22, 2026, after one warm-up pair. Both Macs ran MLX 0.32.4 and NumPy
2.5.4. Wall time includes starting Python and writing the NIfTI. Peak process
footprint and peak MLX allocation use decimal GB.

| Mac | Input | Wall time (s) | Inference and stitching (s) | Peak process footprint (GB) | Peak MLX allocation (GB) |
| --- | --- | ---: | ---: | ---: | ---: |
| M1 Max | AMOS22 CT, 117 classes | 42.30 → 34.76 | 39.06 → 31.71 | 12.28 → 10.24 | 5.45 → 3.64 |
| M1 Max | HCC_063 CT, 117 classes | 19.82 → 16.65 | 18.32 → 15.21 | 9.34 → 7.36 | 5.43 → 3.62 |
| M1 Max | AMOS22 MRI, 50 classes | 18.53 → 16.30 | 17.43 → 15.27 | 12.05 → 7.37 | 10.03 → 5.22 |
| M5 Max | AMOS22 CT, 117 classes | 22.73 → 20.87 | 20.40 → 18.61 | 12.43 → 10.41 | 5.45 → 3.64 |
| M5 Max | HCC_063 CT, 117 classes | 11.39 → 10.55 | 10.18 → 9.34 | 9.50 → 7.52 | 5.43 → 3.66 |
| M5 Max | AMOS22 MRI, 50 classes | 10.07 → 10.31 | 9.18 → 9.42 | 12.14 → 7.97 | 10.03 → 5.22 |

For the M5 rows above, the 96.9% paired wall-time ratio intervals for
HCC_063 (0.895–0.983) and MRI (0.873–1.060) overlap the 2% minimum-effect
band. A separate M5 MRI comparison against review commit `b23cc9c` reduced
wall time from 11.38 to 9.60 s (paired ratio 0.817–0.856), inference and
stitching from 10.53 to 8.76 s, and peak process footprint from 12.21 to
7.60 GB. Peak MLX allocation fell from 10.03 to 5.22 GB.

Each CT produced 28 byte-identical compressed NIfTI masks across both Macs,
and the MRI produced 42, including warm-ups. The HCC_063 liver and spleen
prompt matched the review output with both score and bundle restoration.
The liver point prompt also matched byte for byte.

In a single M1 Max HCC_063 run per setting, four feature maps used 6.79 GB
and took 18.12 s. Six used 7.36 GB and took 16.81 s. On MRI, the four-map
setting saved 2.51 GB at a cost of 0.30 s in the same screen.

## Preprocessing: 1.5 mm resampling

`Spacingd` resampling gathers trilinear corners from a float64 copy of the source
volume. NiBabel returns voxels in column-major order, so the copy is made
row-major to match the gather order. Output values are unchanged.

| Input | Before | After | Method |
| --- | ---: | ---: | --- |
| 768 × 768 × 101 abdominal CT, bilinear resampling to 1.5 mm | 1251 ms | 685 ms | Median warm call, 6 alternating fresh-process rounds of 3 warm calls each; paired sign interval ×0.31–0.64 (97%) |

Measured on an Apple M5 Max (128 GB) with MLX 0.32.4 and NumPy 2.5.4 on
2026-11-18. Preprocessed images and `Invertd` restorations are bit-identical to
the previous implementation on this CT and on a 576 × 468 × 72 abdominal MRI.

## Peak memory and score restoration

The starting tree includes the preprocessing change above. Measurements used
an Apple M1 Max (64 GB) with macOS 27.2.1, MLX 0.32.4 and NumPy 2.5.4.
Perfkit interleaved fresh processes in six counted rounds per input after a
warm-up round. Wall time and peak process footprint come from
`/usr/bin/time -l`. MLX peak and inference with stitching come from the
runner. The table shows medians and uses decimal GB.

| Input and cumulative change | MLX peak (GB) | Process footprint (GB) | Wall time (s) | Inference and stitching (s) |
| --- | ---: | ---: | ---: | ---: |
| HCC-TACE-Seg HCC_063 CT, 512 × 512 × 99, all 117 classes, starting tree | 5.93 | 16.86 | 19.31 | 17.76 |
| Release patch references | 5.43 | 15.87 | 19.31 | 17.76 |
| Add a 1 GiB MLX cache limit | 5.43 | 9.35 | 20.32 | 18.81 |
| Add separable-grid restoration tables | 5.43 | 9.34 | 19.94 | 18.43 |
| AMOS22 `amos_0544` MRI, 576 × 468 × 72, all 50 CTMR MRI_BODY classes, starting tree | 10.98 | 23.89 | 18.88 | 17.72 |
| Release patch references | 10.03 | 22.94 | 18.74 | 17.64 |
| Add a 1 GiB MLX cache limit | 10.03 | 11.98 | 18.96 | 17.88 |
| Add separable-grid restoration tables | 10.03 | 11.97 | 18.69 | 17.62 |

For each input, all 28 written NIfTI files, including warm-up outputs, were
byte-identical. The host score ring occupied 2.45 GB for the 18-patch CT and
1.19 GB for the eight-patch MRI in every variant. After inference with the
patch references released, MLX retained 11.99 GB in free buffers for CT and
20.51 GB for MRI. With the cache limit these fell to 1.23 GB and 1.86 GB
respectively, reducing peak process footprint by 6.52 GB and 10.96 GB.
MLX trims cached buffers during allocation, so the recorded cache after the
last release can exceed the setting.

Paired wall-time ratios compare each change against the preceding variant
(96.9% intervals).

| Input | Release patch references | Add cache limit | Add restoration tables |
| --- | ---: | ---: | ---: |
| HCC_063 | 0.992–1.001 | 1.036–1.057 | 0.979–0.986 |
| `amos_0544` | 0.940–1.065 | 0.979–1.051 | 0.956–0.998 |

The MRI wall-time comparisons and the CT restoration result are inconclusive
at perfkit's 2% minimum-effect band.

### Score restoration on HCC_063

On the separable grid, each bucket's transverse cell indices and uniform labels
are shared by all source planes in the bucket. Mixed cells use the existing
ordered trilinear interpolation.

Perfkit spans around all 275 calls to `ScoreRestorer.add_plane` measured a
median of 4.22 s with the two memory changes and 3.84 s with the restoration
tables across six paired rounds. The 96.9% paired ratio interval was
0.902–0.911. Active MLX peak stayed at 5.43 GB and peak process footprint
was 9.34 GB before and 9.33 GB after. Written masks matched byte for byte.

### AMOS22 CT used in the README

The 768 × 768 × 101 AMOS22 CT `amos_0008` used all 117 classes and 36 patches.
The same six-round protocol ran with Python profiling enabled for each
variant. All 21 written NIfTI files were byte-identical.

| Cumulative change | MLX peak (GB) | Process footprint (GB) | Wall time (s) | Inference and stitching (s) |
| --- | ---: | ---: | ---: | ---: |
| Starting tree | 5.94 | 19.41 | 40.80 | 37.54 |
| Release patch references and add the cache limit | 5.45 | 12.31 | 42.89 | 39.71 |
| Add separable-grid restoration tables | 5.45 | 12.28 | 42.03 | 38.85 |

The paired wall-time ratio was 1.049–1.054 for the two memory changes together.
Adding the restoration tables gave a ratio of 0.975–0.985 against those
changes, which reaches the 2% minimum-effect band (96.9% intervals).

### Class and point prompts on HCC_063

The class prompt requested liver and spleen. The interactive run requested
liver with a positive click at source voxel `(184, 215, 65)`. Each check used
six paired rounds. All 14 written NIfTI files in each check were byte-identical.

| Prompt | MLX peak (GB) | Process footprint (GB) | Wall time (s) | Inference and stitching (s) |
| --- | ---: | ---: | ---: | ---: |
| Liver and spleen | 4.96 → 4.94 | 12.38 → 6.47 | 16.02 → 16.85 | 14.53 → 15.37 |
| Liver with a point | 5.61 → 5.61 | 13.86 → 6.71 | 5.10 → 4.83 | 1.40 → 1.51 |

The 96.9% paired wall-time intervals were 1.047–1.056 for the class prompt
and 0.943–0.953 for the point prompt. The point prompt's inference and
stitching interval was 1.076–1.110.
