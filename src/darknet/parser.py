import dataclasses
import re
from typing import List, Dict, Tuple


def parse_config(path: str) -> List:
    with open(path, "r") as cfg:
        raw_sections = []

        section = {}
        header_re = re.compile(r"^\[(?P<name>\w+)\]$")
        for line in cfg:
            if line.startswith("#") or line.isspace():
                continue

            match = header_re.match(line)
            if match:
                if section:
                    raw_sections.append(section)
                section = {"type": match.group("name")}
            else:
                key, val = [s.strip() for s in line.split("=")]
                if key in section:
                    raise ValueError(
                        f"duplicate section key: section={section['type']}, key={key}"
                    )
                section[key] = val
        raw_sections.append(section)

    sections = []
    section_types = {
        "convolutional": ConvolutionalSection,
        "shortcut": ShortcutSection,
        "upsample": UpsampleSection,
        "maxpool": MaxPoolSection,
        "route": RouteSection,
        "yolo": YOLOSection,
    }
    for raw in raw_sections:
        type_ = raw["type"]
        if type_ == "net":
            # skipping net section because it is used for training
            continue

        if type_ not in section_types:
            raise ValueError(f"unknown section type: type={type_}")

        cls = section_types[type_]
        sections.append(cls.from_dict(raw))

    return sections


@dataclasses.dataclass
class ConvolutionalSection:
    filters: int
    size: int
    stride: int
    pad: int
    activation: str
    batch_normalize: bool

    @classmethod
    def from_dict(cls, dict: Dict[str, str]):
        return cls(
            filters=int(dict.get("filters", "1")),
            size=int(dict.get("size", "1")),
            stride=int(dict.get("stride", "1")),
            pad=int(dict.get("pad", "0")),
            activation=dict.get("activation", "logistic"),
            batch_normalize=bool(dict.get("batch_normalize", False)),
        )


@dataclasses.dataclass
class ShortcutSection:
    from_: int
    activation: str

    @classmethod
    def from_dict(cls, dict: Dict[str, str]):
        return cls(
            from_=int(dict["from"]),
            activation=dict.get("activation", "linear"),
        )


@dataclasses.dataclass
class YOLOSection:
    classes: int
    mask: List[int]
    anchors: List[Tuple[int, int]]

    @classmethod
    def from_dict(cls, dict: Dict[str, str]):
        raw_mask = dict.get("mask", "0")
        mask = [int(m) for m in raw_mask.split(",")]

        raw_anchors = dict.get("anchors", "0")
        anchors = [int(a) for a in raw_anchors.split(",")]
        anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]

        return cls(classes=int(dict.get("classes", "20")), mask=mask, anchors=anchors)


@dataclasses.dataclass
class RouteSection:
    layers: List[int]

    @classmethod
    def from_dict(cls, dict: Dict[str, str]):
        layers = dict["layers"].split(",")
        return cls(layers=[int(l) for l in layers])


@dataclasses.dataclass
class UpsampleSection:
    stride: int

    @classmethod
    def from_dict(cls, dict: Dict[str, str]):
        return cls(stride=int(dict.get("stride", "2")))


@dataclasses.dataclass
class MaxPoolSection:
    stride: int
    size: int

    @classmethod
    def from_dict(cls, dict: Dict[str, str]):
        stride = int(dict.get("stride", "1"))
        return cls(stride=stride, size=int(dict.get("size", stride)))
