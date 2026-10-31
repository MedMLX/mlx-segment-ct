"""Probe Metal outside the test process; host tests collect without importing MLX."""

from __future__ import annotations

import subprocess
import sys
from typing import cast

import pytest

from mlx_segment_ct._types import ArrayBoundary, ArrayValue, CoreRuntime


@pytest.fixture(scope="session")
def mx() -> CoreRuntime:
    """Provide the Metal runtime when the subprocess probe succeeds."""
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import mlx.core as m; raise SystemExit(not m.metal.is_available())",
        ],
        capture_output=True,
        check=False,
    )
    if probe.returncode:
        pytest.skip("Native MLX execution requires accessible Apple Silicon Metal")
    from medmlx_core.runtime import import_mlx

    module = cast("CoreRuntime", import_mlx())
    module.set_default_device(module.gpu)
    return module


@pytest.fixture
def array_backend(request: pytest.FixtureRequest) -> ArrayBoundary[ArrayValue]:
    """Provide the requested host or native array backend."""
    if request.param == "host":
        from reference_support import HostArrays

        return HostArrays
    return cast("ArrayBoundary[ArrayValue]", request.getfixturevalue("mx"))
