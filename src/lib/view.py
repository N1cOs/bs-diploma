import dataclasses
import random
from typing import List

import cv2
import numpy as np


@dataclasses.dataclass
class DetectionResult:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    clazz: int


class DetectionWriter:
    def __init__(self, classes: List[str]):
        # ToDo: use class names
        self.classes = classes
        self.colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in classes]

    def write_detection(self, img: np.ndarray, detection: DetectionResult):
        try:
            color = self.colors[detection.clazz]
        except IndexError:
            raise ValueError(f"unknown detected class: index={detection.clazz}")

        p1 = (int(detection.x1), int(detection.y1))
        p2 = (int(detection.x2), int(detection.y2))
        cv2.rectangle(img, p1, p2, color, thickness=2)
