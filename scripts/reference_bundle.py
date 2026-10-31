"""Pinned NVIDIA bundle transform config and synthetic geometry inputs."""

from __future__ import annotations

import ast
import copy
import gzip
import json
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from scripts.oracle_types import (
        BundleConfig,
        ConfigParserValue,
        HostArray,
        JsonValue,
        OracleNiftiImage,
        OracleNiftiRuntime,
        Preprocessing,
        SourceTransform,
    )

import nibabel as nib
import numpy as np

SOURCE_REVISION = "f9f5f51b589e5dc9c23c453cf5138398e4084056"
BUNDLE_CONFIG: BundleConfig = {
    "imports": ["$import numpy as np", "$import copy", "$import json", "$import pathlib"],
    "bundle_root": "./",
    "image_key": "image",
    "output_dir": "$@bundle_root + '/eval'",
    "output_ext": ".nii.gz",
    "output_dtype": "$np.float32",
    "output_postfix": "trans",
    "separate_folder": True,
    "everything_labels": (
        "$list(set([i+1 for i in range(132)]) - "
        "set([2,16,18,20,21,23,24,25,26,27,128,129,130,131,132]))"
    ),
    "metadata_path": "$@bundle_root + '/configs/metadata.json'",
    "metadata": "$json.loads(pathlib.Path(@metadata_path).read_text())",
    "labels_dict": "$@metadata['network_data_format']['outputs']['pred']['channel_def']",
    "subclass": {
        "2": [14, 5],
        "20": [28, 29, 30, 31, 32],
        "21": "$list(range(33, 57)) + list(range(63, 98)) + [114, 120, 122]",
    },
    "input_channels": 1,
    "resample_spacing": [1.5, 1.5, 1.5],
    "sw_batch_size": 1,
    "patch_size": [128, 128, 128],
    "device": "cpu",
    "preprocessing_transforms": [
        {"_target_": "LoadImaged", "keys": "@image_key", "image_only": True},
        {"_target_": "EnsureChannelFirstd", "keys": "@image_key"},
        {"_target_": "EnsureTyped", "keys": "@image_key", "device": "@device", "track_meta": True},
        {
            "_target_": "Spacingd",
            "keys": "@image_key",
            "pixdim": "@resample_spacing",
            "mode": "bilinear",
        },
        {
            "_target_": "CropForegroundd",
            "keys": "@image_key",
            "allow_smaller": True,
            "margin": 10,
            "source_key": "@image_key",
        },
        {
            "_target_": "monai.apps.vista3d.transforms.VistaPreTransformd",
            "keys": "@image_key",
            "subclass": "@subclass",
            "labels_dict": "@labels_dict",
        },
        {
            "_target_": "ScaleIntensityRanged",
            "keys": "@image_key",
            "a_min": -963.8247715525971,
            "a_max": 1053.678477684517,
            "b_min": 0,
            "b_max": 1,
            "clip": True,
        },
        {"_target_": "Orientationd", "keys": "@image_key", "axcodes": "RAS"},
        {"_target_": "CastToTyped", "keys": "@image_key", "dtype": "$torch.float32"},
    ],
    "preprocessing": {"_target_": "Compose", "transforms": "$@preprocessing_transforms "},
    "postprocessing": {
        "_target_": "Compose",
        "transforms": [
            {"_target_": "ToDeviced", "keys": "pred", "device": "cpu", "_disabled_": True},
            {"_target_": "monai.apps.vista3d.transforms.VistaPostTransformd", "keys": "pred"},
            {
                "_target_": "Invertd",
                "keys": "pred",
                "transform": "$copy.deepcopy(@preprocessing)",
                "orig_keys": "@image_key",
                "nearest_interp": True,
                "to_tensor": True,
            },
            {
                "_target_": "Lambdad",
                "func": "$lambda x: torch.nan_to_num(x, nan=255)",
                "keys": "pred",
            },
            {
                "_target_": "SaveImaged",
                "keys": "pred",
                "resample": False,
                "output_dir": "@output_dir",
                "output_ext": "@output_ext",
                "output_dtype": "@output_dtype",
                "output_postfix": "@output_postfix",
                "separate_folder": "@separate_folder",
            },
        ],
    },
}


# NV-Segment-CTMR f9f5f51 configs/label_mappings.json entries read by its inference.json,
# as [dataset label, global label] pairs.
TOTAL_SEGMENTATOR_MR: list[list[int]] = [
    [1, 9],
    [2, 8],
    [3, 6],
    [4, 104],
    [5, 105],
    [6, 22],
    [7, 91],
    [8, 92],
    [9, 62],
    [10, 13],
    [11, 11],
    [12, 93],
    [13, 94],
    [14, 10],
    [15, 98],
    [16, 99],
    [17, 100],
    [18, 101],
    [19, 102],
    [20, 103],
    [21, 115],
    [22, 95],
    [23, 96],
    [24, 87],
    [25, 88],
    [26, 58],
    [27, 59],
    [28, 60],
    [29, 61],
    [30, 106],
    [31, 107],
    [32, 7],
    [33, 134],
    [34, 14],
    [35, 5],
    [36, 1],
    [37, 135],
    [38, 136],
    [39, 4],
    [40, 17],
    [41, 118],
    [42, 97],
    [43, 89],
    [44, 90],
    [45, 19],
    [46, 121],
    [47, 3],
    [48, 12],
    [49, 15],
    [50, 146],
]
LUMIR: list[list[int]] = [
    [4, 214],
    [11, 215],
    [23, 216],
    [30, 217],
    [31, 218],
    [32, 219],
    [35, 220],
    [36, 221],
    [37, 222],
    [38, 223],
    [39, 224],
    [40, 225],
    [41, 226],
    [44, 227],
    [45, 228],
    [47, 229],
    [48, 230],
    [49, 231],
    [50, 232],
    [51, 233],
    [52, 234],
    [55, 235],
    [56, 236],
    [57, 237],
    [58, 238],
    [59, 239],
    [60, 240],
    [61, 241],
    [62, 242],
    [71, 243],
    [72, 244],
    [73, 245],
    [75, 246],
    [76, 247],
    [100, 248],
    [101, 249],
    [102, 250],
    [103, 251],
    [104, 252],
    [105, 253],
    [106, 254],
    [107, 255],
    [108, 256],
    [109, 257],
    [112, 258],
    [113, 259],
    [114, 260],
    [115, 261],
    [116, 262],
    [117, 263],
    [118, 264],
    [119, 265],
    [120, 266],
    [121, 267],
    [122, 268],
    [123, 269],
    [124, 270],
    [125, 271],
    [128, 272],
    [129, 273],
    [132, 274],
    [133, 275],
    [134, 276],
    [135, 277],
    [136, 278],
    [137, 279],
    [138, 280],
    [139, 281],
    [140, 282],
    [141, 283],
    [142, 284],
    [143, 285],
    [144, 286],
    [145, 287],
    [146, 288],
    [147, 289],
    [148, 290],
    [149, 291],
    [150, 292],
    [151, 293],
    [152, 294],
    [153, 295],
    [154, 296],
    [155, 297],
    [156, 298],
    [157, 299],
    [160, 300],
    [161, 301],
    [162, 302],
    [163, 303],
    [164, 304],
    [165, 305],
    [166, 306],
    [167, 307],
    [168, 308],
    [169, 309],
    [170, 310],
    [171, 311],
    [172, 312],
    [173, 313],
    [174, 314],
    [175, 315],
    [176, 316],
    [177, 317],
    [178, 318],
    [179, 319],
    [180, 320],
    [181, 321],
    [182, 322],
    [183, 323],
    [184, 324],
    [185, 325],
    [186, 326],
    [187, 327],
    [190, 328],
    [191, 329],
    [192, 330],
    [193, 331],
    [194, 332],
    [195, 333],
    [196, 334],
    [197, 335],
    [198, 336],
    [199, 337],
    [200, 338],
    [201, 339],
    [202, 340],
    [203, 341],
    [204, 342],
    [205, 343],
    [206, 344],
    [207, 345],
]
MODALITIES = ("CT_BODY", "MRI_BODY", "MRI_BRAIN")


def ctmr_config(modality: str = "MRI_BODY") -> BundleConfig:
    """NV-Segment-CTMR/configs/inference.json: its differences from NV-Segment-CT's."""
    config = copy.deepcopy(BUNDLE_CONFIG)
    config.update(
        {
            "label_mapping": (
                "$json.loads(pathlib.Path(@bundle_root + "
                "'/configs/label_mappings.json').read_text())"
            ),
            "ct_body": (
                "$list(set([i+1 for i in range(132)]) - "
                "set([2,16,18,20,21,23,24,25,26,27,128,129,130,131,132]))"
            ),
            "mri_body": "$[i[1] for i in @label_mapping['TotalSegmentatorMR']]",
            "mri_brain": "$[i[1] for i in @label_mapping['LUMIR']]",
            "modality": modality,
            "everything_labels": (
                "${'CT_BODY': @ct_body, 'MRI_BODY': @mri_body, 'MRI_BRAIN': @mri_brain}[@modality]"
            ),
            "patch_size": [192, 192, 128],
        }
    )
    transforms = cast("list[dict[str, JsonValue]]", config["preprocessing_transforms"])
    index = next(
        i for i, item in enumerate(transforms) if item["_target_"] == "ScaleIntensityRanged"
    )
    transforms[index] = {
        "_target_": "ScaleIntensityRangePercentilesd",
        "keys": "@image_key",
        "lower": 1,
        "upper": 99,
        "b_min": 0,
        "b_max": 1,
        "clip": True,
    }
    return config


def axis_aligned(matrix: HostArray) -> bool:
    """Return whether each voxel axis maps to exactly one world axis."""
    block = np.asarray(matrix)[:3, :3]
    return bool(((block != 0).sum(axis=0) == 1).all() and ((block != 0).sum(axis=1) == 1).all())


def oblique_affine() -> HostArray:
    """Construct a small oblique affine for geometry fixtures."""
    angle = np.deg2rad(4.0)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0.0], [np.sin(angle), np.cos(angle), 0.0], [0.0, 0.0, 1.0]]
    )
    affine = np.eye(4)
    affine[:3, :3] = rotation @ np.diag([0.9, 0.9, 2.0])
    affine[:3, 3] = [-30.0, 20.0, 10.0]
    return affine


GEOMETRIES: dict[str, tuple[tuple[int, int, int], HostArray, str]] = {
    "ras_ties": (
        (48, 52, 14),
        np.array([[0.75, 0, 0, -20.1], [0, 0.703125, 0, -30.2], [0, 0, 2.5, 15.3], [0, 0, 0, 1]]),
        "both",
    ),
    "lps": (
        (40, 44, 20),
        np.array([[-0.8, 0, 0, 120.5], [0, -0.8, 0, 110.25], [0, 0, 1.25, -200.0], [0, 0, 0, 1]]),
        "both",
    ),
    "sagittal": (
        (30, 40, 28),
        np.array([[0, 0, -1.2, 50.0], [1.0, 0, 0, -80.0], [0, -0.9, 0, 40.0], [0, 0, 0, 1]]),
        "both",
    ),
    "iso15": ((30, 28, 24), np.diag([1.5, 1.5, 1.5, 1.0]), "both"),
    "near15": ((30, 28, 12), np.diag([1.5004, 1.4996, 3.0, 1.0]), "both"),
    "qform_only": (
        (44, 44, 30),
        np.array([[-0.7, 0, 0, 90.0], [0, 0.7, 0, -100.0], [0, 0, 0.625, -50.0], [0, 0, 0, 1]]),
        "qform",
    ),
    "pixdim_mismatch": ((36, 36, 16), np.diag([0.9, 0.9, 2.5, 1.0]), "zooms"),
    "oblique": ((40, 36, 18), oblique_affine(), "both"),
    "long": (
        (200, 24, 16),
        np.array([[1.2, 0, 0, -100.0], [0, 1.5, 0, 12.0], [0, 0, 3.0, -7.5], [0, 0, 0, 1]]),
        "both",
    ),
}


def synthetic_ct(shape: tuple[int, int, int], seed: int) -> HostArray:
    """Air outside an ellipsoidal body with a brighter inner organ, int16 HU."""
    rng = np.random.default_rng(seed)
    grid = np.indices(shape, dtype=np.float64)
    centre = (np.array(shape, dtype=np.float64)[:, None, None, None] - 1) / 2
    radius = ((grid - centre) / (np.array(shape)[:, None, None, None] * 0.38)) ** 2
    body = radius.sum(axis=0) <= 1
    organ = (
        (grid - centre * np.array([0.8, 1.1, 1.0])[:, None, None, None])
        / (np.array(shape)[:, None, None, None] * 0.18)
    ) ** 2
    data = np.full(shape, -1000, dtype=np.int16)
    data[body] = 40 + rng.integers(-30, 30, int(body.sum()))
    inner = organ.sum(axis=0) <= 1
    data[inner] = 160 + rng.integers(-20, 20, int(inner.sum()))
    return data


def nifti_bytes(data: HostArray, affine: HostArray, form: str) -> HostArray:
    """Serialize an image and affine as a NIfTI byte stream."""
    image = cast("OracleNiftiRuntime", nib).Nifti1Image(data, affine)
    if form == "qform":
        image.set_sform(None, code=0)
        image.set_qform(affine, code=1)
    elif form == "zooms":
        image.header.set_zooms(tuple(float(v) * 1.1 for v in np.abs(np.diag(affine)[:3])))
    else:
        image.set_qform(affine, code=1)
        image.set_sform(affine, code=1)
    return np.frombuffer(image.to_bytes(), dtype=np.uint8)


def write_case(directory: Path, name: str, raw: HostArray) -> Path:
    """Write a named synthetic NIfTI case to the fixture directory."""
    path = directory / f"{name}.nii.gz"
    path.write_bytes(gzip.compress(raw.tobytes(), mtime=0))
    return path


def saved_mask(path: Path) -> dict[str, HostArray]:
    """Read a saved reference mask and its image metadata."""
    image = load_nifti(path)
    header = image.header
    return {
        "data": cast("HostArray", np.asanyarray(image.dataobj)),
        "affine": np.asarray(image.affine, dtype=np.float64),
        "codes": np.array([int(header["qform_code"]), int(header["sform_code"])]),
        "header": np.frombuffer(header.binaryblock, dtype=np.uint8),
    }


GRID_SIZES = (1, 2, 3, 7, 23, 97, 128, 141, 255, 333, 512)


def _labels_assignment(name: str) -> ast.expr:
    """Read a named literal assignment from the package label source."""
    source = Path(__file__).resolve().parents[1] / "src/mlx_segment_ct/labels.py"
    tree = ast.parse(source.read_text())
    node = next(
        n
        for n in tree.body
        if isinstance(n, ast.AnnAssign) and isinstance(n.target, ast.Name) and n.target.id == name
    )
    assert node.value is not None
    return node.value


def bundle_labels() -> dict[int, str]:
    """Return the package label mapping as a plain dictionary."""
    return dict(ast.literal_eval(_labels_assignment("LABELS")))


def ctmr_bundle_labels() -> dict[int, str]:
    """NV-Segment-CTMR channel_def: renamed NV-Segment-CT names plus literal extensions."""
    renamed = cast("dict[int, str]", ast.literal_eval(_labels_assignment("CTMR_RENAMED")))
    node = _labels_assignment("CTMR_LABELS")
    assert isinstance(node, ast.Tuple)
    extra = [
        cast("tuple[int, str]", ast.literal_eval(item))
        for item in node.elts
        if not isinstance(item, ast.Starred)
    ]
    labels = [(value, renamed.get(value, name)) for value, name in bundle_labels().items()]
    labels.extend(extra)
    return dict(labels)


def _stage_bundle(root: Path, *, ctmr: bool) -> None:
    """Stage the configuration files needed by the reference bundle."""
    (root / "configs").mkdir(exist_ok=True)
    labels = ctmr_bundle_labels() if ctmr else bundle_labels()
    (root / "configs/metadata.json").write_text(
        json.dumps(
            {
                "network_data_format": {"outputs": {"pred": {"channel_def": labels}}},
            }
        )
    )
    if ctmr:
        (root / "configs/label_mappings.json").write_text(
            json.dumps({"TotalSegmentatorMR": TOTAL_SEGMENTATOR_MR, "LUMIR": LUMIR})
        )


def _parser(root: Path, config: BundleConfig) -> ConfigParserValue:
    """Create the source bundle configuration parser."""
    from monai.bundle.config_parser import ConfigParser

    config.update(
        bundle_root=str(root), device="cpu", output_dir=str(root / "out"), separate_folder=False
    )
    return cast("ConfigParserValue", ConfigParser(config=config))


def bundle_transforms(root: Path, *, ctmr: bool = False) -> tuple[Preprocessing, SourceTransform]:
    """Parse the NV-Segment-CT (or NV-Segment-CTMR) pre- and postprocessing."""
    _stage_bundle(root, ctmr=ctmr)
    config = ctmr_config() if ctmr else copy.deepcopy(BUNDLE_CONFIG)
    parser = _parser(root, config)
    return (
        cast("Preprocessing", parser.get_parsed_content("preprocessing")),
        cast("SourceTransform", parser.get_parsed_content("postprocessing")),
    )


def ctmr_everything_labels(root: Path, modality: str) -> list[int]:
    """Evaluate NV-Segment-CTMR ``everything_labels`` for one modality."""
    _stage_bundle(root, ctmr=True)
    parser = _parser(root, ctmr_config(modality))
    return [
        int(value) for value in cast("list[int]", parser.get_parsed_content("everything_labels"))
    ]


def synthetic_mr(shape: tuple[int, int, int], seed: int) -> HostArray:
    """Zero background around an ellipsoidal body with a brighter organ, int16 MR-like values."""
    data = synthetic_ct(shape, seed).astype(np.int32)
    return np.where(data == -1000, 0, data * 3 + 120).astype(np.int16)


def large_percentile_input() -> HostArray:
    """A float32 tensor above MONAI's np.percentile threshold, with many tied values."""
    i, j, k = np.indices((104, 100, 97), dtype=np.int64)
    return ((i * 7 + j * 13 + k * 29) % 1009 - 200).astype(np.float32)[None]


def large_mr() -> HostArray:
    """A 1.5 mm volume whose foreground crop exceeds MONAI's np.percentile threshold."""
    i, j, k = np.indices((112, 104, 96), dtype=np.int64)
    data = (i * 3 + j * 5 + k * 7) % 500 + 1
    data[:2] = 0
    data[:, -2:] = 0
    return data.astype(np.int16)


def load_nifti(path: str | Path) -> OracleNiftiImage:
    """Load a NIfTI image through the typed reference boundary."""
    image = cast("OracleNiftiRuntime", nib).load(str(path))
    if not isinstance(image, nib.Nifti1Image | nib.Nifti2Image):
        raise TypeError(f"Reference fixture requires NIfTI: {path}")
    return cast("OracleNiftiImage", image)
