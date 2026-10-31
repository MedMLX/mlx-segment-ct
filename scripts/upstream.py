"""CPU oracle provenance and bounded fixture publication for MONAI 1.6.0."""

from __future__ import annotations

import json
import platform
from collections.abc import Mapping
from importlib.metadata import version
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast

if TYPE_CHECKING:
    from torch import Tensor

    from scripts.oracle_types import NPZPayload, PayloadValue, WeightOrder

import numpy as np

from scripts.reference_bundle import SOURCE_REVISION

REPO = Path(__file__).resolve().parents[1]


def reference_key() -> str:
    """Identify the host platform and CPU architecture for reference recordings."""
    return f"{platform.system().lower()}-{platform.machine()}-cpu"


def save(name: str, arrays: Mapping[str, PayloadValue], **metadata: object) -> None:
    """Write reference arrays and metadata after checking the pinned MONAI version."""
    if version("monai") != "1.6.0":
        raise RuntimeError("Record with MONAI 1.6.0")
    info = {
        "schema_version": 1,
        "reference": "MONAI 1.6.0 VISTA3D",
        "bundle_revision": SOURCE_REVISION,
        "reference_backend": reference_key(),
        "monai_version": version("monai"),
        "torch_version": version("torch"),
        "numpy_version": np.__version__,
        "nibabel_version": version("nibabel"),
        **metadata,
    }
    directory = REPO / "tests/fixtures" / reference_key()
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / f"{name}.npz"
    np.savez_compressed(path, metadata=json.dumps(info), **cast("NPZPayload", arrays))
    if path.stat().st_size >= 900_000:
        raise RuntimeError(f"Fixture exceeds 900 KB: {path}")
    print(f"{path.relative_to(REPO)}: {path.stat().st_size} bytes")


def graph_shapes(
    state: Mapping[str, Tensor], kind: Literal["automatic", "point"]
) -> dict[str, tuple[int, ...]]:
    """Derive shapes from upstream; preserve the seeded input's draw order."""
    order = cast("WeightOrder", json.loads((REPO / "scripts/weight_order.json").read_text()))
    if order["schema_version"] != 1:
        raise RuntimeError("Unsupported seeded weight ordering")
    return {name: tuple(state[name].shape) for name in order[kind]}
