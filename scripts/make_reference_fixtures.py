"""Record seeded Torch/MONAI 1.6.0 oracles, including full graphs, on CPU."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from torch import Tensor

    from scripts.oracle_types import (
        FunctionalRuntime,
        HostArray,
        PayloadValue,
        ShapeMap,
        SourceNetwork,
        SourcePointHead,
        TensorPointInference,
        TorchRuntime,
    )


import numpy as np
import torch
from monai.networks.nets.vista3d import PointMappingSAM, vista3d132
from torch import nn

torch_api = cast("TorchRuntime", torch)
functional_api = cast("FunctionalRuntime", nn.functional)

_PE_ENCODING_METHOD = "_pe_encoding"

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.fixture_inputs import random_weights  # noqa: E402
from scripts.reference_windows import (  # noqa: E402
    record_automatic,
    record_components,
    record_point,
)
from scripts.upstream import graph_shapes, save  # noqa: E402


def point_shapes() -> dict[str, tuple[int, ...]]:
    """Return the reference point-head parameter shapes without allocating weights."""
    with torch.device("meta"):
        head = cast("SourcePointHead", PointMappingSAM(48))
    state = {f"point_head.{k}": v for k, v in head.state_dict().items()}
    order = json.loads((REPO / "scripts/weight_order.json").read_text())["point"]
    return {k: tuple(state[k].shape) for k in order if k.startswith("point_head.")}


def seeded_head(seed: int) -> SourcePointHead:
    """Build a reproducibly initialized reference point head."""
    torch_api.manual_seed(seed)
    head = cast("SourcePointHead", PointMappingSAM(48)).eval()
    weights = random_weights(point_shapes(), seed)
    head.load_state_dict(
        {k.removeprefix("point_head."): torch_api.from_numpy(v) for k, v in weights.items()},
        strict=False,
    )
    return head


def record_subgraphs() -> None:
    """Record seeded reference outputs for network subgraphs."""
    rng = np.random.default_rng(624)
    image = rng.standard_normal((1, 4, 3, 5, 7), dtype=np.float32)
    weight = rng.standard_normal((4, 3, 3, 3, 3), dtype=np.float32) * np.float32(0.02)
    output = functional_api.conv_transpose3d(
        torch_api.from_numpy(image),
        torch_api.from_numpy(weight),
        stride=2,
        padding=1,
        output_padding=1,
    )
    save("upsample", dict(image=image, seed=624, output=output.numpy()))
    shapes: dict[str, tuple[int, ...]] = {
        f"image_encoder.up_layers_auto.{i}.upsample.deconv.weight": (2, 2, 3, 3, 3)
        for i in range(4)
    }
    shapes.update({"class_head.mlp.0.weight": (48, 48), "class_head.mlp.0.bias": (48,)})
    weights = random_weights(shapes, 102)
    values = rng.standard_normal((1, 5, 48), dtype=np.float32)
    output = functional_api.linear(
        torch_api.from_numpy(values),
        torch_api.from_numpy(weights["class_head.mlp.0.weight"]),
        torch_api.from_numpy(weights["class_head.mlp.0.bias"]),
    )
    save(
        "feature_linear",
        dict(values=values, output=output.numpy(), weight_seed=102),
        weight_shapes=shapes,
    )
    record_transformer(rng)


def record_transformer(rng: np.random.Generator) -> None:
    """Record reference transformer inputs, weights, and outputs."""
    shapes: dict[str, tuple[int, ...]] = {
        f"image_encoder.up_layers.{i}.upsample.deconv.weight": (2, 2, 3, 3, 3) for i in range(4)
    }
    shapes.update(point_shapes())
    weights = random_weights(shapes, 207)
    head = seeded_head(207)
    head.load_state_dict(
        {
            k.removeprefix("point_head."): torch_api.from_numpy(v)
            for k, v in weights.items()
            if k.startswith("point_head.")
        },
        strict=False,
    )
    positions = np.array(
        [[[0.01, 0.4, 0.99], [0.2, 0.3, 0.4], [0.8, 0.5, 0.1], [0.6, 0.7, 0.8], [0.9, 0.2, 0.3]]],
        dtype=np.float32,
    )
    codes = np.array([[-1, 0, 1, 2, 3]], dtype=np.int32)
    keys = rng.standard_normal((1, 7, 48), dtype=np.float32)
    pe = rng.standard_normal((1, 7, 48), dtype=np.float32)
    tokens: list[HostArray] = []
    hook = head.transformer.register_forward_pre_hook(
        lambda module, args: tokens.append(args[2].numpy().copy())
    )
    for target in (132, 133):
        head(
            torch_api.zeros(1, 48, 4, 4, 4),
            torch_api.from_numpy(positions.astype(np.float64) * 4 - 0.5),
            torch_api.from_numpy(codes),
            torch_api.tensor([target]),
        )
    hook.remove()
    queries, transformed = head.transformer(
        torch_api.from_numpy(keys).transpose(1, 2),
        torch_api.from_numpy(pe).transpose(1, 2),
        torch_api.from_numpy(tokens[0]),
    )
    long_rng = np.random.default_rng(210)
    long_keys = long_rng.standard_normal((1, 4096, 48), dtype=np.float32)
    long_values = long_rng.standard_normal((1, 4096, 48), dtype=np.float32)
    attention = head.transformer.final_attn_token_to_image(
        torch_api.from_numpy(tokens[0]),
        torch_api.from_numpy(long_keys),
        torch_api.from_numpy(long_values),
    )
    save(
        "point_transformer",
        dict(
            positions=positions,
            codes=codes,
            keys=keys,
            key_pe=pe,
            weight_seed=207,
            long_seed=210,
            positional_encoding=cast(
                "Callable[[Tensor], Tensor]", getattr(head.pe_layer, _PE_ENCODING_METHOD)
            )(torch_api.from_numpy(positions)).numpy(),
            supported_tokens=tokens[0],
            zeroshot_tokens=tokens[1],
            queries=queries.numpy(),
            transformed_keys=transformed.numpy(),
            long_attention=attention.numpy(),
            gelu=functional_api.gelu(torch_api.from_numpy(keys)).numpy(),
        ),
        weight_shapes=shapes,
    )


def record_schemas() -> None:
    """Record required source-network tensor names and shapes."""
    with torch.device("meta"):
        state = vista3d132().state_dict()
    automatic, point = graph_shapes(state, "automatic"), graph_shapes(state, "point")
    additional = {k: s for k, s in point.items() if k not in automatic}
    save(
        "key_shapes",
        {
            k: json.dumps(v)
            for k, v in dict(automatic=automatic, correction=point, additional=additional).items()
        },
    )
    points = np.array([[[0, 1.25, 2.75], [12.123456789, 8.5, 6]]], dtype=np.float64)
    normalized = (
        ((torch_api.from_numpy(points) + 0.5) / torch_api.tensor([17, 19, 21])).float().numpy()
    )
    save("point_normalization", dict(points=points, shape=[17, 19, 21], normalized=normalized))


def full_model(kind: str) -> tuple[SourceNetwork, ShapeMap, int]:
    """Build a seeded reference model and report its parameter shapes."""
    seed = {"feature": 401, "automatic": 402, "point": 403}[kind]
    with torch.device("meta"):
        model = cast("SourceNetwork", vista3d132())
    shapes = graph_shapes(model.state_dict(), "point" if kind == "point" else "automatic")
    model.load_state_dict(
        {k: torch_api.from_numpy(v) for k, v in random_weights(shapes, seed).items()},
        strict=False,
        assign=True,
    )
    return model.eval(), shapes, seed


def record_full(kind: str) -> None:
    """Record full reference-model outputs for the selected branch."""
    model, shapes, seed = full_model(kind)
    # A 16-cube has a one-voxel InstanceNorm domain and is rejected by upstream.
    # Keep seed 404, using the smallest valid anisotropic domain instead.
    image = np.random.default_rng(404).standard_normal((1, 1, 16, 16, 32), dtype=np.float32)
    arrays: dict[str, PayloadValue] = dict(image=image, weight_seed=seed)
    if kind == "feature":
        output = cast(
            "Tensor",
            model.image_encoder(torch_api.from_numpy(image), with_point=False, with_label=True)[1],
        )
        for i, part in enumerate(output.numpy().reshape(1, 4, 12, 16, 16, 32)[0]):
            save(f"full_feature_{i}", dict(output=part[None]), weight_shapes=shapes)
        save("full_feature", arrays, weight_shapes=shapes, output_parts=4)
    elif kind == "automatic":
        ids = [3, 1, 26]
        output = model(
            torch_api.from_numpy(image), class_vector=torch_api.tensor(ids)[:, None], transpose=True
        )
        save(
            "full_automatic",
            dict(**arrays, output=output.numpy(), class_ids=ids),
            weight_shapes=shapes,
        )
        record_automatic(save, model=model, shapes=shapes, weight_seed=seed)
    else:
        positions = np.array([[[0.2, 0.4, 0.6], [0.8, 0.6, 0.4]]], dtype=np.float32)
        coords = positions * np.array(image.shape[2:], np.float32) - 0.5
        codes = np.array([[3, 2]], dtype=np.int32)
        output = model(
            torch_api.from_numpy(image),
            point_coords=torch_api.from_numpy(coords),
            point_labels=torch_api.from_numpy(codes),
            prompt_class=torch_api.tensor([26]),
            transpose=True,
        )
        save(
            "full_point",
            dict(**arrays, output=output.numpy(), positions=positions, codes=codes, target=26),
            weight_shapes=shapes,
        )
        record_point(save, model=model, shapes=shapes, weight_seed=seed)


def save_parts(
    name: str,
    arrays: dict[str, PayloadValue],
    output: HostArray,
    shapes: dict[str, tuple[int, ...]],
    width: int,
) -> None:
    """Split output channels and store the resulting reference arrays."""
    parts = [output[:, i : i + width] for i in range(0, output.shape[1], width)]
    for i, part in enumerate(parts):
        save(f"{name}_{i}", dict(output=part))
    save(name, arrays, weight_shapes=shapes, output_parts=len(parts))


def record_decoder_parities() -> None:
    """Record decoder subgraph cases across supported channel sizes."""
    for channels in (96, 192, 384, 768):
        rng = np.random.default_rng(624)
        image = rng.standard_normal((1, channels, 3, 5, 7), dtype=np.float32)
        weight = rng.standard_normal((channels, channels // 2, 3, 3, 3), dtype=np.float32) * 0.02
        output = functional_api.conv_transpose3d(
            torch_api.from_numpy(image),
            torch_api.from_numpy(weight),
            stride=2,
            padding=1,
            output_padding=1,
        ).numpy()
        save_parts(f"upsample_{channels}", dict(channels=channels, seed=624), output, {}, 64)


def record_graph_prompts() -> None:
    """Record reference graph outputs for class prompt cases."""
    model, shapes, seed = full_model("automatic")
    ids = sorted(
        set(range(1, 133)) - {2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132}
    )
    image = np.random.default_rng(23).random((1, 1, 32, 32, 32), dtype=np.float32)
    output = model(
        torch_api.from_numpy(image), class_vector=torch_api.tensor(ids)[:, None], transpose=True
    ).numpy()
    save_parts(
        "automatic_subset", dict(image=image, class_ids=ids, weight_seed=seed), output, shapes, 4
    )
    del model
    model, shapes, seed = full_model("point")
    image = np.random.default_rng(74).random((1, 1, 32, 32, 32), dtype=np.float32)
    coords = np.array([[[12.25, 18.5, 19], [21, 10, 13.75]]], dtype=np.float32)
    for target, codes in ((1, [1, 0]), (25, [3, 2]), (26, [3, 2]), (117, [2, 3]), (254, [1, 0])):
        labels = np.array([codes], np.int64)
        output = model(
            torch_api.from_numpy(image),
            point_coords=torch_api.from_numpy(coords),
            point_labels=torch_api.from_numpy(labels),
            prompt_class=torch_api.tensor([target]),
            transpose=True,
        ).numpy()
        save(
            f"point_target_{target}",
            dict(
                image=image,
                points=coords,
                codes=labels,
                target=target,
                output=output,
                weight_seed=seed,
            ),
            weight_shapes=shapes,
        )
    image = np.random.default_rng(42).random((1, 1, 48, 32, 32), dtype=np.float32)
    previous = np.full_like(image, -1)
    previous[:, :, 28:38, 10:20, 10:20] = 1
    coords = np.array([[[4.0, 6, 6], [33.0, 16, 16]]], np.float32)
    labels = np.array([[1, 0]], np.int64)
    from monai.apps.vista3d.inferer import point_based_window_inferer

    output = cast("TensorPointInference", point_based_window_inferer)(
        torch_api.from_numpy(image),
        [32] * 3,
        model,
        point_coords=torch_api.from_numpy(coords),
        point_labels=torch_api.from_numpy(labels),
        class_vector=None,
        prompt_class=torch_api.tensor([1]),
        prev_mask=torch_api.from_numpy(previous),
        point_start=1,
        center_only=True,
        transpose=True,
    ).numpy()
    save(
        "point_previous",
        dict(
            image=image,
            points=coords,
            codes=labels,
            previous=previous,
            output=output,
            weight_seed=seed,
        ),
        weight_shapes=shapes,
    )


def main() -> None:
    """Parse recorder options and write selected source-model fixtures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-models", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    with torch.inference_mode():
        record_subgraphs()
        record_schemas()
        record_automatic(save)
        record_point(save)
        record_components(save)
        if args.full_models:
            for kind in ("feature", "automatic", "point"):
                record_full(kind)
            record_decoder_parities()
            record_graph_prompts()


if __name__ == "__main__":
    main()
