# Label vocabulary: NV-Segment-CTMR f9f5f51 NV-Segment-CT/ and NV-Segment-CTMR/
# configs/metadata.json and configs/label_mappings.json, Apache-2.0; see THIRD_PARTY_NOTICES.md.
"""NV-Segment-CT and NV-Segment-CTMR labels, prompt validation and class-prompt expansion.

The vocabularies and class-prompt rules come from the pinned repository (NV-Segment-CTMR
``f9f5f51b589e5dc9c23c453cf5138398e4084056``). NV-Segment-CT: ``configs/metadata.json``
channel names below 255, ``configs/inference.json`` ``subclass`` and ``everything_labels``,
and ``scripts/evaluator.py`` prompt checks. NV-Segment-CTMR: the same files of its own
bundle, with channel names through 345 and modality ``everything_labels`` built from
``configs/label_mappings.json``. Both use MONAI 1.6.0 ``VistaPreTransformd`` defaults.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from numbers import Real
from typing import cast

import numpy as np
from medmlx_core.errors import InvalidInputError

from mlx_segment_ct._types import HostArray

LABELS: tuple[tuple[int, str], ...] = (
    (0, "background"),
    (1, "liver"),
    (2, "kidney"),
    (3, "spleen"),
    (4, "pancreas"),
    (5, "right kidney"),
    (6, "aorta"),
    (7, "inferior vena cava"),
    (8, "right adrenal gland"),
    (9, "left adrenal gland"),
    (10, "gallbladder"),
    (11, "esophagus"),
    (12, "stomach"),
    (13, "duodenum"),
    (14, "left kidney"),
    (15, "bladder"),
    (16, "prostate or uterus (deprecated)"),
    (17, "portal vein and splenic vein"),
    (18, "rectum"),
    (19, "small bowel"),
    (20, "lung"),
    (21, "bone"),
    (22, "brain"),
    (23, "lung tumor"),
    (24, "pancreatic tumor"),
    (25, "hepatic vessel"),
    (26, "hepatic tumor"),
    (27, "colon cancer primaries"),
    (28, "left lung upper lobe"),
    (29, "left lung lower lobe"),
    (30, "right lung upper lobe"),
    (31, "right lung middle lobe"),
    (32, "right lung lower lobe"),
    (33, "vertebrae L5"),
    (34, "vertebrae L4"),
    (35, "vertebrae L3"),
    (36, "vertebrae L2"),
    (37, "vertebrae L1"),
    (38, "vertebrae T12"),
    (39, "vertebrae T11"),
    (40, "vertebrae T10"),
    (41, "vertebrae T9"),
    (42, "vertebrae T8"),
    (43, "vertebrae T7"),
    (44, "vertebrae T6"),
    (45, "vertebrae T5"),
    (46, "vertebrae T4"),
    (47, "vertebrae T3"),
    (48, "vertebrae T2"),
    (49, "vertebrae T1"),
    (50, "vertebrae C7"),
    (51, "vertebrae C6"),
    (52, "vertebrae C5"),
    (53, "vertebrae C4"),
    (54, "vertebrae C3"),
    (55, "vertebrae C2"),
    (56, "vertebrae C1"),
    (57, "trachea"),
    (58, "left iliac artery"),
    (59, "right iliac artery"),
    (60, "left iliac vena"),
    (61, "right iliac vena"),
    (62, "colon"),
    (63, "left rib 1"),
    (64, "left rib 2"),
    (65, "left rib 3"),
    (66, "left rib 4"),
    (67, "left rib 5"),
    (68, "left rib 6"),
    (69, "left rib 7"),
    (70, "left rib 8"),
    (71, "left rib 9"),
    (72, "left rib 10"),
    (73, "left rib 11"),
    (74, "left rib 12"),
    (75, "right rib 1"),
    (76, "right rib 2"),
    (77, "right rib 3"),
    (78, "right rib 4"),
    (79, "right rib 5"),
    (80, "right rib 6"),
    (81, "right rib 7"),
    (82, "right rib 8"),
    (83, "right rib 9"),
    (84, "right rib 10"),
    (85, "right rib 11"),
    (86, "right rib 12"),
    (87, "left humerus"),
    (88, "right humerus"),
    (89, "left scapula"),
    (90, "right scapula"),
    (91, "left clavicula"),
    (92, "right clavicula"),
    (93, "left femur"),
    (94, "right femur"),
    (95, "left hip"),
    (96, "right hip"),
    (97, "sacrum"),
    (98, "left gluteus maximus"),
    (99, "right gluteus maximus"),
    (100, "left gluteus medius"),
    (101, "right gluteus medius"),
    (102, "left gluteus minimus"),
    (103, "right gluteus minimus"),
    (104, "left autochthon"),
    (105, "right autochthon"),
    (106, "left iliopsoas"),
    (107, "right iliopsoas"),
    (108, "left atrial appendage"),
    (109, "brachiocephalic trunk"),
    (110, "left brachiocephalic vein"),
    (111, "right brachiocephalic vein"),
    (112, "left common carotid artery"),
    (113, "right common carotid artery"),
    (114, "costal cartilages"),
    (115, "heart"),
    (116, "left kidney cyst"),
    (117, "right kidney cyst"),
    (118, "prostate"),
    (119, "pulmonary vein"),
    (120, "skull"),
    (121, "spinal cord"),
    (122, "sternum"),
    (123, "left subclavian artery"),
    (124, "right subclavian artery"),
    (125, "superior vena cava"),
    (126, "thyroid gland"),
    (127, "vertebrae S1"),
    (128, "bone lesion"),
    (129, "kidney mass (deprecated)"),
    (130, "liver tumor (deprecated)"),
    (131, "vertebrae L6 (deprecated)"),
    (132, "airway"),
    (133, "fibula"),
    (134, "intervertebral discs"),
    (135, "left lung"),
    (136, "right lung"),
    (137, "left quadriceps femoris"),
    (138, "right quadriceps femoris"),
    (139, "left sartorius"),
    (140, "right sartorius"),
    (141, "left thigh medial compartment"),
    (142, "right thigh medial compartment"),
    (143, "left thigh posterior compartment"),
    (144, "right thigh posterior compartment"),
    (145, "tibia"),
    (146, "vertebrae"),
    (147, "prostate transitional zone"),
    (148, "prostate peripheral zone"),
    (149, "left atrium"),
    (150, "white matter hyperintensity"),
    (151, "left ventricle"),
    (152, "right ventricle"),
    (153, "right atrium"),
    (154, "left ventricle myocardium"),
    (155, "ascending aorta"),
    (156, "muscles"),
    (157, "fat"),
    (158, "abdominal tissue"),
    (159, "mediastinal tissue"),
    (160, "gonads"),
    (161, "uterocervix"),
    (162, "uterus (deprecated)"),
    (163, "breast left"),
    (164, "breast right"),
    (165, "thyroid left"),
    (166, "thyroid right"),
    (167, "thymus"),
    (168, "skin"),
    (169, "heart tissue"),
    (170, "celiac trunk"),
    (171, "pulmonary artery"),
    (172, "cheek left"),
    (173, "cheek right"),
    (174, "eyeball left"),
    (175, "eyeball right"),
    (176, "brain tumor"),
    (177, "chiasm"),
    (178, "left temporal lobe"),
    (179, "right temporal lobe"),
    (180, "left eye"),
    (181, "right eye"),
    (182, "left lens"),
    (183, "right lens"),
    (184, "left optic nerve"),
    (185, "right optic nerve"),
    (186, "left middle ear"),
    (187, "right middle ear"),
    (188, "left internal auditory canal"),
    (189, "right internal auditory canal"),
    (190, "left tympanic cavity"),
    (191, "right tympanic cavity"),
    (192, "left vestibular semicircular canals"),
    (193, "right vestibular semicircular canals"),
    (194, "left cochlea"),
    (195, "right cochlea"),
    (196, "left ethmoid bone"),
    (197, "right ethmoid bone"),
    (198, "pituitary"),
    (199, "oral cavity"),
    (200, "left mandible"),
    (201, "right mandible"),
    (202, "left submandibular"),
    (203, "right submandibular"),
    (204, "left parotid"),
    (205, "right parotid"),
    (206, "left mastoid"),
    (207, "right mastoid"),
    (208, "left temporomandibular joint"),
    (209, "right temporomandibular joint"),
    (210, "larynx"),
    (211, "larynx glottic"),
    (212, "larynx supraglot"),
    (213, "pharynxConst"),
    (214, "3rd-Ventricle"),
    (215, "4th-Ventricle"),
    (216, "Right-Accumbens-Area"),
    (217, "Left-Accumbens-Area"),
    (218, "Right-Amygdala"),
    (219, "Left-Amygdala"),
    (220, "Brain-Stem"),
    (221, "Right-Caudate"),
    (222, "Left-Caudate"),
    (223, "Right-Cerebellum-Exterior"),
    (224, "Left-Cerebellum-Exterior"),
    (225, "Right-Cerebellum-White-Matter"),
    (226, "Left-Cerebellum-White-Matter"),
    (227, "Right-Cerebral-White-Matter"),
    (228, "Left-Cerebral-White-Matter"),
    (229, "Right-Hippocampus"),
    (230, "Left-Hippocampus"),
    (231, "Right-Inf-Lat-Vent"),
    (232, "Left-Inf-Lat-Vent"),
    (233, "Right-Lateral-Ventricle"),
    (234, "Left-Lateral-Ventricle"),
    (235, "Right-Pallidum"),
    (236, "Left-Pallidum"),
    (237, "Right-Putamen"),
    (238, "Left-Putamen"),
    (239, "Right-Thalamus-Proper"),
    (240, "Left-Thalamus-Proper"),
    (241, "Right-Ventral-DC"),
    (242, "Left-Ventral-DC"),
    (243, "Cerebellar-Vermal-Lobules-I-V"),
    (244, "Cerebellar-Vermal-Lobules-VI-VII"),
    (245, "Cerebellar-Vermal-Lobules-VIII-X"),
    (246, "Left-Basal-Forebrain"),
    (247, "Right-Basal-Forebrain"),
    (248, "Right-ACgG--anterior-cingulate-gyrus"),
    (249, "Left-ACgG--anterior-cingulate-gyrus"),
    (250, "Right-AIns--anterior-insula"),
    (251, "Left-AIns--anterior-insula"),
    (252, "Right-AOrG--anterior-orbital-gyrus"),
    (253, "Left-AOrG--anterior-orbital-gyrus"),
    (254, "Right-AnG---angular-gyrus"),
)

# ``everything_labels`` in the pinned inference.json, in increasing global-ID order.
AUTOMATIC_LABEL_IDS: tuple[int, ...] = tuple(
    sorted(
        frozenset(range(1, 133)) - {2, 16, 18, 20, 21, 23, 24, 25, 26, 27, 128, 129, 130, 131, 132}
    )
)
# ``subclass`` in the pinned inference.json: coarse prompts expand to these IDs.
SUBCLASS: dict[int, tuple[int, ...]] = {
    2: (14, 5),
    20: (28, 29, 30, 31, 32),
    21: (*range(33, 57), *range(63, 98), 114, 120, 122),
}
# MONAI 1.6.0 ``VistaPreTransformd`` default ``special_index``.
SPECIAL_INDEX = (25, 26, 27, 28, 29, 117)

# NV-Segment-CTMR configs/metadata.json: the NV-Segment-CT names with these channels
# marked deprecated, extended by the LUMIR brain structures 255-345.
CTMR_RENAMED: dict[int, str] = {
    133: "fibula (deprecated)",
    137: "left quadriceps femoris (deprecated)",
    138: "right quadriceps femoris (deprecated)",
    139: "left sartorius (deprecated)",
    140: "right sartorius (deprecated)",
    141: "left thigh medial compartment (deprecated)",
    142: "right thigh medial compartment (deprecated)",
    143: "left thigh posterior compartment (deprecated)",
    144: "right thigh posterior compartment (deprecated)",
    145: "tibia (deprecated)",
    155: "ascending aorta (deprecated)",
}
CTMR_LABELS: tuple[tuple[int, str], ...] = (
    *((value, CTMR_RENAMED.get(value, name)) for value, name in LABELS),
    (255, "Left-AnG---angular-gyrus"),
    (256, "Right-Calc--calcarine-cortex"),
    (257, "Left-Calc--calcarine-cortex"),
    (258, "Right-CO----central-operculum"),
    (259, "Left-CO----central-operculum"),
    (260, "Right-Cun---cuneus"),
    (261, "Left-Cun---cuneus"),
    (262, "Right-Ent---entorhinal-area"),
    (263, "Left-Ent---entorhinal-area"),
    (264, "Right-FO----frontal-operculum"),
    (265, "Left-FO----frontal-operculum"),
    (266, "Right-FRP---frontal-pole"),
    (267, "Left-FRP---frontal-pole"),
    (268, "Right-FuG---fusiform-gyrus"),
    (269, "Left-FuG---fusiform-gyrus"),
    (270, "Right-GRe---gyrus-rectus"),
    (271, "Left-GRe---gyrus-rectus"),
    (272, "Right-IOG---inferior-occipital-gyrus"),
    (273, "Left-IOG---inferior-occipital-gyrus"),
    (274, "Right-ITG---inferior-temporal-gyrus"),
    (275, "Left-ITG---inferior-temporal-gyrus"),
    (276, "Right-LiG---lingual-gyrus"),
    (277, "Left-LiG---lingual-gyrus"),
    (278, "Right-LOrG--lateral-orbital-gyrus"),
    (279, "Left-LOrG--lateral-orbital-gyrus"),
    (280, "Right-MCgG--middle-cingulate-gyrus"),
    (281, "Left-MCgG--middle-cingulate-gyrus"),
    (282, "Right-MFC---medial-frontal-cortex"),
    (283, "Left-MFC---medial-frontal-cortex"),
    (284, "Right-MFG---middle-frontal-gyrus"),
    (285, "Left-MFG---middle-frontal-gyrus"),
    (286, "Right-MOG---middle-occipital-gyrus"),
    (287, "Left-MOG---middle-occipital-gyrus"),
    (288, "Right-MOrG--medial-orbital-gyrus"),
    (289, "Left-MOrG--medial-orbital-gyrus"),
    (290, "Right-MPoG--postcentral-gyrus"),
    (291, "Left-MPoG--postcentral-gyrus"),
    (292, "Right-MPrG--precentral-gyrus"),
    (293, "Left-MPrG--precentral-gyrus"),
    (294, "Right-MSFG--superior-frontal-gyrus"),
    (295, "Left-MSFG--superior-frontal-gyrus"),
    (296, "Right-MTG---middle-temporal-gyrus"),
    (297, "Left-MTG---middle-temporal-gyrus"),
    (298, "Right-OCP---occipital-pole"),
    (299, "Left-OCP---occipital-pole"),
    (300, "Right-OFuG--occipital-fusiform-gyrus"),
    (301, "Left-OFuG--occipital-fusiform-gyrus"),
    (302, "Right-OpIFG-opercular-part-of-the-IFG"),
    (303, "Left-OpIFG-opercular-part-of-the-IFG"),
    (304, "Right-OrIFG-orbital-part-of-the-IFG"),
    (305, "Left-OrIFG-orbital-part-of-the-IFG"),
    (306, "Right-PCgG--posterior-cingulate-gyrus"),
    (307, "Left-PCgG--posterior-cingulate-gyrus"),
    (308, "Right-PCu---precuneus"),
    (309, "Left-PCu---precuneus"),
    (310, "Right-PHG---parahippocampal-gyrus"),
    (311, "Left-PHG---parahippocampal-gyrus"),
    (312, "Right-PIns--posterior-insula"),
    (313, "Left-PIns--posterior-insula"),
    (314, "Right-PO----parietal-operculum"),
    (315, "Left-PO----parietal-operculum"),
    (316, "Right-PoG---postcentral-gyrus"),
    (317, "Left-PoG---postcentral-gyrus"),
    (318, "Right-POrG--posterior-orbital-gyrus"),
    (319, "Left-POrG--posterior-orbital-gyrus"),
    (320, "Right-PP----planum-polare"),
    (321, "Left-PP----planum-polare"),
    (322, "Right-PrG---precentral-gyrus"),
    (323, "Left-PrG---precentral-gyrus"),
    (324, "Right-PT----planum-temporale"),
    (325, "Left-PT----planum-temporale"),
    (326, "Right-SCA---subcallosal-area"),
    (327, "Left-SCA---subcallosal-area"),
    (328, "Right-SFG---superior-frontal-gyrus"),
    (329, "Left-SFG---superior-frontal-gyrus"),
    (330, "Right-SMC---supplementary-motor-cortex"),
    (331, "Left-SMC---supplementary-motor-cortex"),
    (332, "Right-SMG---supramarginal-gyrus"),
    (333, "Left-SMG---supramarginal-gyrus"),
    (334, "Right-SOG---superior-occipital-gyrus"),
    (335, "Left-SOG---superior-occipital-gyrus"),
    (336, "Right-SPL---superior-parietal-lobule"),
    (337, "Left-SPL---superior-parietal-lobule"),
    (338, "Right-STG---superior-temporal-gyrus"),
    (339, "Left-STG---superior-temporal-gyrus"),
    (340, "Right-TMP---temporal-pole"),
    (341, "Left-TMP---temporal-pole"),
    (342, "Right-TrIFG-triangular-part-of-the-IFG"),
    (343, "Left-TrIFG-triangular-part-of-the-IFG"),
    (344, "Right-TTG---transverse-temporal-gyrus"),
    (345, "Left-TTG---transverse-temporal-gyrus"),
)
# NV-Segment-CTMR ``everything_labels``: global IDs (second column) of
# configs/label_mappings.json, in file order. ``mri_body`` is TotalSegmentatorMR;
# ``mri_brain`` is LUMIR, whose IDs are exactly 214-345 in increasing order.
MRI_BODY_LABEL_IDS: tuple[int, ...] = (
    9,
    8,
    6,
    104,
    105,
    22,
    91,
    92,
    62,
    13,
    11,
    93,
    94,
    10,
    98,
    99,
    100,
    101,
    102,
    103,
    115,
    95,
    96,
    87,
    88,
    58,
    59,
    60,
    61,
    106,
    107,
    7,
    134,
    14,
    5,
    1,
    135,
    136,
    4,
    17,
    118,
    97,
    89,
    90,
    19,
    121,
    3,
    12,
    15,
    146,
)
MRI_BRAIN_LABEL_IDS: tuple[int, ...] = tuple(range(214, 346))


@dataclass(frozen=True, slots=True)
class LabelRules:
    """One released checkpoint's vocabulary and its bundle's prompt rules.

    ``max_label`` and ``automatic`` are ``scripts/evaluator.py``
    ``check_prompts_format``: the label-prompt bound and the ``supported_list`` that
    label prompts without points must belong to. IDs from ``zero_shot_start`` up
    need points. ``everything`` maps each modality to its ``everything_labels``
    in bundle order.
    """

    checkpoint: str
    labels: tuple[tuple[int, str], ...]
    max_label: int
    automatic: frozenset[int]
    zero_shot_start: int
    everything: tuple[tuple[str, tuple[int, ...]], ...]
    default_modality: str

    def everything_labels(self, modality: str | None = None) -> tuple[int, ...]:
        """The automatic all-classes prompt for ``modality`` (default: the bundle's)."""
        selected = self.default_modality if modality is None else modality
        for name, label_ids in self.everything:
            if name == selected:
                return label_ids
        raise InvalidInputError(
            f"{self.checkpoint} modality must be one of "
            f"{', '.join(name for name, _ in self.everything)}: {modality!r}"
        )


CT_RULES = LabelRules(
    checkpoint="NV-Segment-CT",
    labels=LABELS,
    max_label=254,
    automatic=frozenset(range(1, 133)) - {16, 18, 129, 130, 131},
    zero_shot_start=133,
    everything=(("CT_BODY", AUTOMATIC_LABEL_IDS),),
    default_modality="CT_BODY",
)
CTMR_RULES = LabelRules(
    checkpoint="NV-Segment-CTMR",
    labels=CTMR_LABELS,
    max_label=511,
    automatic=frozenset(range(1, 346)) - {16, 129, 130, 131, 133, *range(137, 146), 162},
    zero_shot_start=346,
    everything=(
        ("CT_BODY", AUTOMATIC_LABEL_IDS),
        ("MRI_BODY", MRI_BODY_LABEL_IDS),
        ("MRI_BRAIN", MRI_BRAIN_LABEL_IDS),
    ),
    default_modality="MRI_BODY",
)
RULES: dict[str, LabelRules] = {rules.checkpoint: rules for rules in (CT_RULES, CTMR_RULES)}


def label_rules(checkpoint: str) -> LabelRules:
    """Select rules for ``"NV-Segment-CT"`` or ``"NV-Segment-CTMR"``."""
    rules = RULES.get(checkpoint)
    if rules is None:
        raise InvalidInputError(f"checkpoint must be one of {', '.join(RULES)}: {checkpoint!r}")
    return rules


def normalize_label_name(value: str) -> str:
    """Return the canonical form used to match NV-Segment-CT label names."""
    return " ".join(value.strip().lower().split())


@cache
def _vocabulary(rules: LabelRules) -> tuple[dict[str, int], dict[int, str]]:
    """Build normalized name-to-ID and ID-to-name lookups for the selected rules."""
    by_name = {normalize_label_name(name): value for value, name in rules.labels}
    names = dict(rules.labels)
    if len(by_name) != len(rules.labels) or len(names) != len(rules.labels):
        raise RuntimeError(f"{rules.checkpoint} label vocabulary contains duplicates")
    return by_name, names


def label_name(value: int, rules: LabelRules = CT_RULES) -> str | None:
    """The vocabulary name of one label value, or None outside the vocabulary."""
    return _vocabulary(rules)[1].get(value)


def resolve_label_name(name: str, rules: LabelRules = CT_RULES) -> int:
    """Resolve one canonical anatomy name to its label value."""
    normalized = normalize_label_name(name)
    value = _vocabulary(rules)[0].get(normalized)
    if value is None:
        raise InvalidInputError(
            f"Unknown {rules.checkpoint} label {name!r}. Use a canonical metadata label name "
            "or a numeric label prompt."
        )
    return value


def split_label_names(labels: Sequence[str]) -> tuple[str, ...]:
    """Split comma-separated names, dropping empty parts."""
    values: list[str] = []
    for label in labels:
        values.extend(part.strip() for part in str(label).split(",") if part.strip())
    return tuple(values)


@dataclass(frozen=True, slots=True)
class Prompt:
    """A label prompt, point prompt, or their valid combination.

    ``label_ids`` are global label IDs admitted by ``rules``. ``points`` are
    (x, y, z) voxel indices on the source image and ``point_labels`` their codes
    (see ``InteractivePrompt``). ``segment_all`` requests the checkpoint's all-classes
    set and excludes label and point prompts.
    """

    label_ids: tuple[int, ...] = ()
    label_names: tuple[str, ...] = ()
    points: tuple[tuple[float, float, float], ...] = ()
    point_labels: tuple[int, ...] = ()
    segment_all: bool = False
    rules: LabelRules = CT_RULES

    def __post_init__(self) -> None:
        """Validate automatic label and segment-all prompt choices."""
        if not self.label_ids and not self.points and not self.segment_all:
            raise InvalidInputError(
                "NV-Segment-CT requires segment_all, a label prompt, or a point prompt"
            )
        if self.segment_all and (self.label_ids or self.points):
            raise InvalidInputError(
                "NV-Segment-CT segment_all cannot be combined with label or point prompts"
            )
        if bool(self.points) != bool(self.point_labels):
            raise InvalidInputError(
                "NV-Segment-CT points and point labels must be provided together"
            )
        if len(self.points) != len(self.point_labels):
            raise InvalidInputError("NV-Segment-CT points must match point labels")
        if self.points and len(self.label_ids) > 1:
            raise InvalidInputError("NV-Segment-CT accepts at most one label id with point prompts")
        rules = self.rules
        invalid_ids = [value for value in self.label_ids if value < 1 or value > rules.max_label]
        if invalid_ids:
            raise InvalidInputError(
                f"{rules.checkpoint} label prompts must be in [1, {rules.max_label}]: {invalid_ids}"
            )
        zero_shot = [value for value in self.label_ids if value >= rules.zero_shot_start]
        if zero_shot and not self.points:
            raise InvalidInputError(
                f"{rules.checkpoint} zero-shot label ids {rules.zero_shot_start} through "
                f"{rules.max_label} require point prompts"
            )
        unsupported = [
            value for value in self.label_ids if not self.points and value not in rules.automatic
        ]
        if unsupported:
            raise InvalidInputError(
                f"{rules.checkpoint} does not define automatic label prompts: {unsupported}"
            )
        coarse = [value for value in self.label_ids if self.points and value in {2, 20, 21}]
        if coarse:
            raise InvalidInputError(
                "NV-Segment-CT point prompts cannot use coarse kidney, lung, or bone labels; "
                f"use a documented subclass label instead: {coarse}"
            )
        if any(not all(math.isfinite(axis) for axis in point) for point in self.points):
            raise InvalidInputError("NV-Segment-CT point coordinates must be finite")


@dataclass(frozen=True, slots=True)
class InteractivePrompt:
    """One object's clicks and optional global class, without a mask.

    ``points`` are (x, y, z) voxel indices on whichever grid the caller holds: the
    source image before ``transform_points`` and the preprocessed grid after it.
    ``point_labels`` are -1 (padding), 0 (exclude), 1 (include), or 2/3 (special
    exclude/include); for special classes ``arrays`` maps 0/1 to 2/3.
    ``class_label`` is a global label ID, or None for a class-free object that is
    written as label 1.
    """

    points: tuple[tuple[float, float, float], ...]
    point_labels: tuple[int, ...]
    class_label: int | None = None
    rules: LabelRules = CT_RULES

    def __post_init__(self) -> None:
        """Validate click coordinates, codes, and optional target class."""
        if not isinstance(cast("object", self.points), tuple) or not isinstance(
            cast("object", self.point_labels), tuple
        ):
            raise InvalidInputError("Interactive prompt points and labels must be immutable tuples")
        if not self.points:
            raise InvalidInputError("Interactive segmentation requires at least one point")
        if any(
            not isinstance(cast("object", point), tuple) or len(point) != 3 for point in self.points
        ):
            raise InvalidInputError("Interactive points must have three coordinates (x, y, z)")
        if any(not isinstance(axis, Real) for point in self.points for axis in point):
            raise InvalidInputError("Interactive point coordinates must be numeric")
        if any(type(code) is not int or code not in (-1, 0, 1, 2, 3) for code in self.point_labels):
            raise InvalidInputError("Interactive point_labels must be -1, 0, 1, 2, or 3")
        if self.class_label is not None and type(self.class_label) is not int:
            raise InvalidInputError("Interactive class label must be an integer")
        Prompt(
            label_ids=() if self.class_label is None else (self.class_label,),
            points=self.points,
            point_labels=self.point_labels,
            rules=self.rules,
        )

    @property
    def uses_automatic(self) -> bool:
        # Vista3dInferer drops the class vector above vista3d132's last_supported (132).
        """Return whether this click prompt combines automatic class scores."""
        return self.class_label is not None and (
            self.class_label <= 132 or all(code == -1 for code in self.point_labels)
        )

    @property
    def output_label(self) -> int:
        # Label value written for this object: the class ID, or 1 when class-free.
        """Return the target class ID or one for a class-free prompt."""
        return 1 if self.class_label is None else self.class_label

    def arrays(self) -> tuple[HostArray, HostArray]:
        """Bundle VistaPreTransformd codes and FP64 geometry, with the object axis."""
        codes = (
            self.point_labels
            if self.class_label is None
            else effective_point_labels(self.class_label, self.point_labels)
        )
        return np.asarray([self.points], np.float64), np.asarray([codes], np.int64)


def class_ids_for_prompt(
    label_prompt: Sequence[int] | None = None,
    rules: LabelRules = CT_RULES,
    modality: str | None = None,
) -> tuple[int, ...]:
    """Apply the pinned class-prompt subclass transform in request order.

    Without a prompt, return the bundle's ``everything_labels`` for ``modality``.
    Coarse kidney, lung, and bone prompts produce their source subclass IDs.
    Repeated identical channels select the same global label, so keep their
    first occurrence for the bounded reducer's unique-label workspace.
    """
    if label_prompt is None:
        return rules.everything_labels(modality)
    prompt = Prompt(label_ids=tuple(label_prompt), rules=rules)
    resolved: list[int] = []
    for value in prompt.label_ids:
        resolved.extend(SUBCLASS.get(value, (value,)))
    selected = tuple(dict.fromkeys(int(value) for value in resolved))
    Prompt(label_ids=selected, rules=rules)
    return selected


def effective_point_labels(target: int, point_labels: Sequence[int]) -> list[int]:
    """``VistaPreTransformd``: special targets encode clicks as 2 (exclude) and 3 (include)."""
    labels = [int(value) for value in point_labels]
    if target in SPECIAL_INDEX:
        return [{0: 2, 1: 3}.get(value, value) for value in labels]
    return labels
