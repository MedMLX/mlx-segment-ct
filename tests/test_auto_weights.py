"""Automatic and explicit weight folders use the same CT request path."""

from pathlib import Path
from typing import NoReturn

import pytest
from medmlx_core.errors import AssetNotReadyError

from mlx_segment_ct import medmlx


class InferenceReached(Exception):
    """Stop before tensor loading."""


@pytest.mark.parametrize("automatic", [True, False])
def test_model_folder_defaults_and_explicit_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, automatic: bool,
) -> None:
    """Verify the default CT model download and explicit folder override."""
    folder = tmp_path / "model"
    folder.mkdir()
    (folder / "automatic.safetensors").write_bytes(b"fixture")
    (folder / "correction.safetensors").write_bytes(b"fixture")
    image = tmp_path / "ct.nii.gz"
    image.write_bytes(b"request construction does not read this input")
    downloads: list[str] = []
    selected: list[object] = []

    def download(model_id: str) -> Path:
        """Record the requested CT model and provide a fixture folder."""
        downloads.append(model_id)
        return folder

    def infer(*args: object, **kwargs: object) -> NoReturn:
        """Record the folder passed into inference."""
        selected.append(args[0])
        raise InferenceReached

    monkeypatch.setattr(medmlx, "model_weights_dir", download)
    monkeypatch.setattr(medmlx, "_infer_and_publish", infer)
    with pytest.raises(InferenceReached):
        medmlx.run(image_path=image, output_dir=tmp_path / "out",
                   weights_path=None if automatic else folder)
    assert downloads == (["mlx-segment-ct"] if automatic else [])
    assert selected == [folder]


def test_ctmr_never_substitutes_automatic_ct_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a missing CTMR bundle without using CT weights."""
    def download(model_id: str) -> NoReturn:
        """Fail if the advanced CTMR option attempts a CT model download."""
        raise AssertionError("CTMR must not download the CT model")

    monkeypatch.setattr(medmlx, "model_weights_dir", download)
    with pytest.raises(AssetNotReadyError, match="CTMR requires an explicit"):
        medmlx.run(
            image_path="ct.nii.gz", output_dir=tmp_path / "out", checkpoint="NV-Segment-CTMR",
        )


@pytest.mark.parametrize("automatic", [True, False])
def test_ctmr_variant_uses_its_own_model_folder_and_fixed_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, automatic: bool,
) -> None:
    """Verify CTMR uses its own download and configuration."""
    folder = tmp_path / "ctmr"
    folder.mkdir()
    downloads: list[str] = []
    selected: dict[str, object] = {}

    def download(model_id: str) -> Path:
        """Record the CTMR download request."""
        downloads.append(model_id)
        return folder

    def run(**kwargs: object) -> NoReturn:
        """Record the selected CTMR runner options."""
        selected.update(kwargs)
        raise InferenceReached

    monkeypatch.setattr(medmlx, "model_weights_dir", download)
    monkeypatch.setattr(medmlx, "run", run)
    with pytest.raises(InferenceReached):
        medmlx.run_ctmr(image_path="mr.nii.gz", output_dir=tmp_path / "out",
                        weights_path=None if automatic else folder, modality="MRI_BODY")
    assert downloads == (["mlx-segment-ctmr"] if automatic else [])
    assert selected["weights_path"] == folder
    assert selected["checkpoint"] == "NV-Segment-CTMR"
    assert selected["modality"] == "MRI_BODY"
