import dataclasses


@dataclasses.dataclass
class DetectionResult:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    clazz: int
