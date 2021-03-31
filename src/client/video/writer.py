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
    def __init__(self, classes: List[str], write_label: bool = True):
        self.classes = classes
        self.colors = [tuple(random.randint(0, 255) for _ in range(3)) for _ in classes]
        self.write_label = write_label

    def write(self, img: np.ndarray, detection: "DetectionResult"):
        try:
            color = self.colors[detection.clazz]
            label = self.classes[detection.clazz]
        except IndexError:
            raise ValueError(f"unknown detected class: index={detection.clazz}")

        p1 = (detection.x1, detection.y1)
        p2 = (detection.x2, detection.y2)
        cv2.rectangle(img, p1, p2, color, thickness=2)

        if self.write_label:
            label_size, base_line = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            top = max(detection.y1, label_size[1])

            p1 = (detection.x1, top - label_size[1])
            p2 = (detection.x1 + label_size[0], top + base_line)
            cv2.rectangle(img, p1, p2, color, cv2.FILLED)
            cv2.putText(
                img,
                label,
                (detection.x1, top),
                cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0, 0, 0),
            )


class AsyncDetectionWriter:
    def __init__(self, classes: List[str], write_label: bool = True):
        self.writer = DetectionWriter(classes, write_label)
        self.loop = asyncio.get_running_loop()

    async def write(self, img: np.ndarray, detection: "DetectionResult"):
        await self.loop.run_in_executor(None, self.writer.write, img, detection)


@dataclasses.dataclass
class DetectionResult:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    clazz: int
