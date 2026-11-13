# Real CT test input and CTMR references

The runner and CTMR preprocessing tests use the HCC-TACE-Seg HCC_063 contrast
CT at `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz`; `MEDMLX_TEST_CT` overrides
the runner input. The CTMR cases use real CT crops with controlled affine geometry
to exercise orientation and resampling, including the large percentile path.

The geometry cases use the corresponding liver and tumor masks, with codes 3
and 5 assigned to exercise label restoration. These codes are test values, not
the bundle's anatomical label definitions. The same controlled geometry is
applied to each CT and mask crop after their source grids are checked.

Record the image-bearing references with MONAI 1.6.0 and Torch on CPU:

```sh
python scripts/record_ctmr_reference.py --ct PATH_TO_HCC_063_CT
python scripts/record_geometry_reference.py --ct PATH_TO_HCC_063_CT \
  --liver PATH_TO_HCC_063_LIVER --tumor PATH_TO_HCC_063_TUMOR
python scripts/record_segment_reference.py
python scripts/record_segment_reference.py --ctmr
python scripts/record_interactive_reference.py
python scripts/record_interactive_windows_reference.py
python scripts/record_full_network_reference.py
python scripts/record_window_reference.py
python scripts/record_prompt_reference.py
```

References are written under `work/real-reference`, outside Git. Set
`MEDMLX_TEST_REFERENCES` to load them from another directory. The tracked CTMR
fixtures retain the label vocabulary, request metadata and controlled geometry
headers, with no CT or segmentation pixels.
The runner recordings use MONAI sliding windows and postprocessing with the
tests' deterministic model boundary, rather than the released network weights.
The interactive recorders also require SciPy and scikit-image for MONAI's
connected-component operations.
For the HCC-TACE 117-class recording, the largest observed CPU/Metal difference
was 1.20e-6 across 3,833,856 logits. Its 2e-6 absolute bound retains exact
foreground-decision agreement; the other seeded graph comparisons retain 1e-6.
