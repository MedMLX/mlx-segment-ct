"""Host point normalization for the VISTA3D positional encoder."""

from __future__ import annotations

import numpy as np

from mlx_segment_ct._types import HostArray


def normalized_point_positions(points: HostArray, shape: tuple[int, ...]) -> HostArray:
    """Keep source coordinate precision through center shift/division, then cast.

    ``points`` holds voxel indices with the last axis ordered (x, y, z) on the image
    whose spatial shape is ``shape``. Each coordinate gets +0.5 (the voxel center)
    and is divided by that axis size. In-bounds integer voxel indices yield values
    in (0, 1); this helper does not clip or validate coordinates. The result is
    float32 with the shape of ``points``. Callers must supply positive axis sizes.

    MONAI transforms clicks in float64 and casts only after normalization. MLX
    does not support float64, so this geometry calculation stays on the host.
    """
    coords = points.copy() + 0.5
    for axis, size in enumerate(shape):
        coords[..., axis] /= size
    return coords.astype(np.float32)
