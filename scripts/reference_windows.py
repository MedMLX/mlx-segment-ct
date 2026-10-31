"""Window and component oracles from MONAI's unmodified CPU inferers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor

    from scripts.oracle_types import (
        FunctionalRuntime,
        MonoWindowNamespace,
        PayloadValue,
        PostPrediction,
        PromptTransform,
        SaveRecording,
        ShapeMap,
        SourceAutomatic,
        SourceComponentOperations,
        SourceNetwork,
        TensorPointInference,
        TorchRuntime,
    )


import monai.inferers.utils as window_module
import numpy as np
import torch
from monai.apps.vista3d.inferer import point_based_window_inferer
from monai.apps.vista3d.transforms import VistaPostTransformd
from monai.data.utils import compute_importance_map
from monai.networks.nets.vista3d import VISTA3D
from torch import nn

torch_api = cast("TorchRuntime", torch)
functional_api = cast("FunctionalRuntime", nn.functional)


class AutomaticOracle(nn.Module):
    """Seed 701 and the original patch-local input function, evaluated in Torch."""

    def __init__(self) -> None:
        """Initialize the seeded synthetic source automatic predictor."""
        super().__init__()
        rng = np.random.default_rng(701)
        self.scale = torch_api.from_numpy(rng.standard_normal((1, 3, 1, 1, 1), dtype=np.float32))
        self.bias = torch_api.from_numpy(rng.standard_normal((1, 3, 1, 1, 1), dtype=np.float32))

    def forward(self, image: Tensor) -> Tensor:
        """Generate synthetic automatic scores for a source window."""
        grid = np.indices(image.shape[2:], dtype=np.float32)
        ramp = grid[0] * 0.125 + grid[1] * 0.0625 + grid[2] * 0.03125
        return image * self.scale + self.bias + torch_api.from_numpy(ramp)[None, None]


class ImageEncoder(nn.Module):
    """Return the image as features for the requested source heads."""

    def forward(
        self, image: Tensor, with_point: bool, with_label: bool
    ) -> tuple[Tensor | None, Tensor | None]:
        """Return image features for the requested source heads."""
        return image if with_point else None, image if with_label else None


class PointOracle(nn.Module):
    """Original seed 801 input function; VISTA3D owns filtering and component fusion."""

    def forward(
        self,
        image: Tensor,
        point_coords: Tensor,
        point_labels: Tensor,
        class_vector: Tensor | None = None,
    ) -> Tensor:
        """Generate seeded synthetic point-head scores."""
        rng = np.random.default_rng(801)
        scale = torch_api.tensor(rng.uniform(0.5, 1.5), dtype=torch.float32)
        bias = torch_api.tensor(rng.uniform(-0.25, 0.25), dtype=torch.float32)
        normalized = ((point_coords + 0.5) / torch_api.tensor(image.shape[2:])).float()
        target = 132 if class_vector is None else class_vector.ravel()[0]
        return (
            image * scale
            + bias
            + normalized.sum() * 0.125
            + point_labels.float().sum() * 0.03125
            + target * 0.00390625
        )


def point_oracle() -> SourceNetwork:
    """Build the reference network with synthetic point components."""
    return cast("SourceNetwork", VISTA3D(ImageEncoder(), nn.Identity(), PointOracle()).eval())


def record_automatic(
    save: SaveRecording,
    *,
    model: SourceNetwork | None = None,
    shapes: ShapeMap | None = None,
    weight_seed: int | None = None,
) -> None:
    """Record automatic sliding-window source outputs."""
    full = model is not None
    for name, shape, roi, mode in (
        ("constant", (13, 9, 11), (6, 6, 6), "constant"),
        ("padded", (3, 7, 5), (6, 6, 6), "constant"),
        ("gaussian", (13, 9, 11), (6, 6, 6), "gaussian"),
    ):
        if full:
            if name != "constant":
                continue
            shape, roi = (19, 17, 31), (16, 16, 32)
        image = np.random.default_rng(702).standard_normal((1, 1, *shape), dtype=np.float32)
        network = model if full else AutomaticOracle()
        coordinates: list[list[int]] = []

        def predict(
            patch: Tensor,
            coords: Sequence[Sequence[slice]],
            network: SourceAutomatic | SourceNetwork = network,
            coordinates: list[list[int]] = coordinates,
        ) -> Tensor:
            """Predict a window and record its source-grid coordinates."""
            coordinates.extend([[s.start for s in c[2:]] for c in coords])
            return (
                cast("SourceNetwork", network)(
                    patch, class_vector=torch_api.tensor([3, 1, 26])[:, None], transpose=True
                )
                if full
                else network(patch)
            )

        logits = cast("MonoWindowNamespace", window_module).sliding_window_inference(
            torch_api.from_numpy(image),
            roi,
            1,
            predict,
            overlap=0.25,
            mode=mode,
            padding_mode="replicate",
            with_coord=True,
        )
        assert isinstance(logits, torch.Tensor)
        labels = cast("PromptTransform", VistaPostTransformd("pred"))(
            {"pred": logits[0].clone(), "label_prompt": torch_api.tensor([3, 1, 26])}
        )
        labels = cast("PostPrediction", labels)["pred"]
        pairs = [
            (max(r - n, 0) // 2, (max(r - n, 0) + 1) // 2) for n, r in zip(shape, roi, strict=True)
        ]
        padded = functional_api.pad(
            torch_api.from_numpy(image),
            [v for pair in reversed(pairs) for v in pair],
            mode="replicate",
        )
        original_coordinates = coordinates.copy()
        padded_logits = cast("MonoWindowNamespace", window_module).sliding_window_inference(
            padded,
            roi,
            1,
            predict,
            overlap=0.25,
            mode=mode,
            padding_mode="replicate",
            with_coord=True,
        )
        assert isinstance(padded_logits, torch.Tensor)
        weights = compute_importance_map(roi, mode).numpy()
        counts = np.zeros(padded.shape[2:], np.float32)
        for position in original_coordinates:
            counts[tuple(slice(i, i + r) for i, r in zip(position, roi, strict=True))] += weights
        arrays: dict[str, PayloadValue] = dict(
            image=image,
            roi=roi,
            mode=mode,
            network_seed=701,
            class_ids=[3, 1, 26],
            weight=weights,
            logits=logits.numpy(),
            planes=padded_logits.numpy()[0],
            counts=counts,
            labels=labels.numpy()[None],
            coordinates=original_coordinates,
        )
        if full:
            arrays["weight_seed"] = cast("int", weight_seed)
        save(
            f"{'full_' if full else ''}window_{name}",
            arrays,
            **({"weight_shapes": shapes} if full else {}),
        )


def record_point(
    save: SaveRecording,
    *,
    model: SourceNetwork | None = None,
    shapes: ShapeMap | None = None,
    weight_seed: int | None = None,
) -> None:
    """Record point-conditioned source window outputs."""
    full = model is not None
    for name, shape, roi, points, start, center_only in (
        ("overlap", (13, 9, 11), (6, 6, 6), [[[2, 3.25, 3], [5.5, 4, 4], [10, 5, 7]]], 1, True),
        ("padded", (3, 7, 5), (6, 6, 6), [[[0, 1.25, 0.25], [2, 5, 4], [1.5, 3, 2]]], 0, True),
        ("multi", (13, 9, 11), (6, 6, 6), [[[4, 4, 4], [5, 4, 4], [6, 5, 5]]], 0, False),
    ):
        if full:
            if name != "overlap":
                continue
            shape, roi = (19, 17, 31), (16, 16, 32)
            points, start = [[[4, 5.25, 6], [14, 11, 10], [9, 8, 7]]], 0
        image = np.random.default_rng(802).standard_normal((1, 1, *shape), dtype=np.float32)
        previous = np.where(image > 0.25, np.float32(1), np.float32(-1))
        coords = np.array(points, dtype=np.float64)
        codes = np.array([[3, 2, 1]], dtype=np.int64)
        prompt = np.array([26], dtype=np.int64)
        network = model if full else point_oracle()
        output = cast("TensorPointInference", point_based_window_inferer)(
            torch_api.from_numpy(image),
            roi,
            network,
            torch_api.from_numpy(coords),
            torch_api.from_numpy(codes),
            class_vector=None,
            prompt_class=torch_api.from_numpy(prompt),
            prev_mask=torch_api.from_numpy(previous),
            point_start=start,
            center_only=center_only,
            margin=1,
            transpose=True,
        )
        arrays: dict[str, PayloadValue] = dict(
            image=image,
            roi=roi,
            points=coords,
            codes=codes,
            prompt=prompt,
            previous=previous,
            point_start=start,
            center_only=center_only,
            margin=1,
            network_seed=801,
            output=output.numpy(),
        )
        if full:
            arrays["weight_seed"] = cast("int", weight_seed)
        save(
            f"{'full_' if full else ''}point_window_{name}",
            arrays,
            **({"weight_shapes": shapes} if full else {}),
        )


def record_components(save: SaveRecording) -> None:
    """Record seeded source connected-component cases."""
    rng = np.random.default_rng(901)
    previous = np.where(rng.random((1, 1, 7, 7, 7)) > 0.8, 1, -1).astype(np.float32)
    predicted = rng.standard_normal(previous.shape, dtype=np.float32)
    predicted.reshape(-1)[:8] = [0, 1e-8, -1e-8, 6e-8, 9e-8, 1.2e-7, -6e-8, -9e-8]
    points = np.array([[[2.50000001, 2, 2], [4, 4, 4], [3, 3, 3], [1, 5, 1], [0, 0, 0]]])
    codes = np.array([[1, 0, 3, 2, -1]], dtype=np.int64)
    coords = [slice(None), slice(None), slice(0, 7), slice(0, 7), slice(0, 7)]
    model = point_oracle()
    updated, labels = cast("SourceComponentOperations", model).update_point_to_patch(
        coords, torch_api.from_numpy(points), torch_api.from_numpy(codes)
    )
    output = cast("SourceComponentOperations", model).connected_components_combine(
        torch_api.from_numpy(previous.copy()),
        torch_api.from_numpy(predicted.copy()),
        cast("Tensor", updated),
        cast("Tensor", labels),
        torch_api.tensor([True]),
    )
    save(
        "components",
        dict(
            previous=previous,
            predicted=predicted,
            points=points,
            codes=codes,
            updated=cast("Tensor", updated).numpy(),
            labels=cast("Tensor", labels).numpy(),
            output=output.numpy(),
        ),
    )
    scores = rng.standard_normal((1, 3, 2, 3, 4), dtype=np.float32)
    scores[:, :, 0, 0] = np.array([0, -1, 1, 2], dtype=np.float32)
    output = cast("PromptTransform", VistaPostTransformd("pred"))(
        {
            "pred": torch_api.from_numpy(scores[0].copy()),
            "label_prompt": torch_api.tensor([26, 3, 1]),
        }
    )
    output = cast("PostPrediction", output)["pred"]
    save("bounded_labels", dict(scores=scores, class_ids=[26, 3, 1], output=output.numpy()[None]))
