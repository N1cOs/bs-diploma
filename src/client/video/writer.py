import asyncio
import dataclasses
import random
from typing import List

import cv2
import numpy as np

from . import reader


class AsyncFrameWriter:
    def __init__(self, path: str, fourcc: str, stat: reader.VideoStat):
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(path, fourcc, stat.fps, (stat.width, stat.height))
        self.loop = asyncio.get_running_loop()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.writer.release()

    async def write(self, frame: np.ndarray):
        await self.loop.run_in_executor(None, self.writer.write, frame)


class DetectionWriter:
    def __init__(self, classes: List[str]):
        # ToDo: use class names
        self.classes = classes
        self.colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in classes]

    def write(self, img: np.ndarray, detection: "DetectionResult"):
        try:
            color = self.colors[detection.clazz]
        except IndexError:
            raise ValueError(f"unknown detected class: index={detection.clazz}")

        p1 = (int(detection.x1), int(detection.y1))
        p2 = (int(detection.x2), int(detection.y2))
        cv2.rectangle(img, p1, p2, color, thickness=2)


class AsyncDetectionWriter:
    def __init__(self, classes: List[str]):
        self.writer = DetectionWriter(classes)
        self.loop = asyncio.get_running_loop()

    async def write(self, img: np.ndarray, detection: "DetectionResult"):
        await self.loop.run_in_executor(None, self.writer.write, img, detection)


@dataclasses.dataclass
class DetectionResult:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    clazz: int
