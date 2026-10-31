"""Native MLX implementation of NV-Segment-CT (VISTA-3D).

Public names load on first access, so importing the package (or the medmlx
declaration in ``mlx_segment_ct.medmlx``) does not import MLX.
"""

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mlx_segment_ct.correction import normalized_point_positions
    from mlx_segment_ct.correction_inferer import (
        point_based_window_inferer,
    )
    from mlx_segment_ct.interactive import (
        interactive_labels,
        interactive_logits,
        transform_points,
    )
    from mlx_segment_ct.labels import InteractivePrompt
    from mlx_segment_ct.mlx_network import AutomaticModel, FeatureModel, required_shapes
    from mlx_segment_ct.mlx_point import PointModel, additional_shapes, correction_shapes
    from mlx_segment_ct.mlx_point_window import PointWindowPredictor
    from mlx_segment_ct.mlx_upsample import AutomaticUpsample
    from mlx_segment_ct.mlx_worker import automatic_logits
    from mlx_segment_ct.streaming import LabelReducer, bounded_labels, rolling_logits

__version__ = "0.3.4"

_EXPORTS = {
    "AutomaticModel": "mlx_segment_ct.mlx_network",
    "AutomaticUpsample": "mlx_segment_ct.mlx_upsample",
    "FeatureModel": "mlx_segment_ct.mlx_network",
    "InteractivePrompt": "mlx_segment_ct.labels",
    "LabelReducer": "mlx_segment_ct.streaming",
    "PointModel": "mlx_segment_ct.mlx_point",
    "PointWindowPredictor": "mlx_segment_ct.mlx_point_window",
    "additional_shapes": "mlx_segment_ct.mlx_point",
    "automatic_logits": "mlx_segment_ct.mlx_worker",
    "bounded_labels": "mlx_segment_ct.streaming",
    "correction_shapes": "mlx_segment_ct.mlx_point",
    "interactive_labels": "mlx_segment_ct.interactive",
    "interactive_logits": "mlx_segment_ct.interactive",
    "normalized_point_positions": "mlx_segment_ct.correction",
    "point_based_window_inferer": "mlx_segment_ct.correction_inferer",
    "required_shapes": "mlx_segment_ct.mlx_network",
    "rolling_logits": "mlx_segment_ct.streaming",
    "transform_points": "mlx_segment_ct.interactive",
}

__all__ = [
    "AutomaticModel",
    "AutomaticUpsample",
    "FeatureModel",
    "InteractivePrompt",
    "LabelReducer",
    "PointModel",
    "PointWindowPredictor",
    "additional_shapes",
    "automatic_logits",
    "bounded_labels",
    "correction_shapes",
    "interactive_labels",
    "interactive_logits",
    "normalized_point_positions",
    "point_based_window_inferer",
    "required_shapes",
    "rolling_logits",
    "transform_points",
]


def __getattr__(name: str) -> object:
    """Load a public export from its module on first access."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'mlx_segment_ct' has no attribute {name!r}")
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List eager and lazy package exports."""
    return sorted({*globals(), *_EXPORTS})
