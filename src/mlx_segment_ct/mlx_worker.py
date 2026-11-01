"""Automatic MLX window inference; bundle transforms and worker I/O stay with the caller."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING, cast

from mlx_segment_ct._types import ArrayValue, HostArray, SlidingWindowNamespace

if TYPE_CHECKING:
    from mlx.core import array


def automatic_logits(
    inputs: ArrayValue,
    predictor: Callable[..., array],
    roi_size: Sequence[int] = (128, 128, 128),
    *,
    overlap: float = 0.25,
    mode: str = "constant",
    roi_weight_map: ArrayValue | None = None,
    with_coord: bool = False,
    **kwargs: object,
) -> HostArray:
    """Return NCDHW logits with the admitted NV automatic/class-prompt settings.

    Window batch size stays one: the NV models accept one image at a time.
    ``predictor`` accepts/returns MLX arrays; optional coordinates have MONAI's
    slice layout. Class order belongs to the supplied ``AutomaticModel``.

    ``inputs`` is a float32 ``(1, 1, X, Y, Z)`` volume. Images smaller than
    ``roi_size`` are replicate-padded, and the result is cropped back to the input
    size.
    """
    import medmlx_core.sliding_window as sliding_window

    return cast("SlidingWindowNamespace", sliding_window).sliding_window_inference(
        inputs,
        roi_size,
        1,
        predictor,
        overlap=overlap,
        mode=mode,
        padding_mode="replicate",
        roi_weight_map=roi_weight_map,
        with_coord=with_coord,
        **kwargs,
    )
