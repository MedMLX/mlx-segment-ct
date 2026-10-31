"""Offline reuse of weights prepared through the shared download contract."""

from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, Protocol, cast

import nibabel as nib
import numpy as np
import pytest
from medmlx_core import runtime
from medmlx_core import weights as shared_weights
from numpy.typing import NDArray

from mlx_segment_ct import medmlx


class NiftiHeader(Protocol):
    def set_xyzt_units(self, xyz: str) -> None: ...


class NiftiImage(Protocol):
    header: NiftiHeader


class NiftiIO(Protocol):
    def Nifti1Image(
        self,
        dataobj: NDArray[np.float32],
        affine: NDArray[np.float64],
    ) -> NiftiImage: ...

    def save(self, image: NiftiImage, filename: str) -> None: ...


class CheckpointReached(Exception):
    """Stop at native checkpoint I/O before learned inference."""


@pytest.mark.parametrize("model_id", ["mlx-segment-ct", "mlx-segment-ctmr"])
def test_preloaded_weights_are_reused_with_the_hub_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_id: str,
) -> None:
    """Pre-download by registered ID, then reach the same checkpoint without network."""
    offline = False

    def model_info(_repo_id: str) -> SimpleNamespace:
        assert not offline, "Pre-downloaded weights must be reusable with the Hub offline"
        return SimpleNamespace(sha="a" * 40)

    def snapshot_download(_repo_id: str, *, local_dir: Path, **_options: object) -> str:
        assert not offline, "An offline run must not download another model folder"
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "automatic.safetensors").write_bytes(b"native checkpoint placeholder")
        (local_dir / "correction.safetensors").write_bytes(b"native checkpoint placeholder")
        return str(local_dir)

    hub = SimpleNamespace(
        HfApi=lambda: SimpleNamespace(model_info=model_info),
        snapshot_download=snapshot_download,
    )

    def import_hub(_name: str) -> SimpleNamespace:
        return hub

    monkeypatch.setenv("MEDMLX_ASSET_CACHE", str(tmp_path / "cache"))
    monkeypatch.setattr(shared_weights, "import_module", import_hub)

    preloaded = shared_weights.weights_dir(model_id)
    assert shared_weights.weights_revision(preloaded) == "a" * 40
    offline = True
    selected: list[Path] = []

    class NativeCheckpointIO:
        def synchronize(self) -> None:
            pass

        def reset_peak_memory(self) -> None:
            pass

        def set_cache_limit(self, _limit: int) -> int:
            return 0

        def load(self, path: str, **_options: object) -> NoReturn:
            selected.append(Path(path))
            raise CheckpointReached("native checkpoint I/O reached")

    monkeypatch.setattr(runtime, "import_mlx", NativeCheckpointIO)
    image = tmp_path / "ct.nii.gz"
    nifti = cast(NiftiIO, nib)
    volume = nifti.Nifti1Image(np.arange(64, dtype=np.float32).reshape(4, 4, 4), np.eye(4))
    volume.header.set_xyzt_units("mm")
    nifti.save(volume, str(image))
    with pytest.raises(CheckpointReached, match="native checkpoint I/O reached"):
        runner = medmlx.run if model_id == "mlx-segment-ct" else medmlx.run_ctmr
        runner(image_path=str(image), output_dir=str(tmp_path / "out"))
    assert selected == [preloaded / "automatic.safetensors"]
