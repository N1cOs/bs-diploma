import asyncio
import dataclasses
from typing import Tuple

import cv2
import numpy as np


@dataclasses.dataclass
class VideoStat:
    width: int
    height: int
    fps: float
    frames: int
    duration: float

    @classmethod
    def from_video_capture(cls, cap: cv2.VideoCapture):
        fps = cap.get(cv2.CAP_PROP_FPS)
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return cls(
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=fps,
            frames=frames,
            duration=frames / fps,
        )

    def __str__(self):
        return (
            f"VideoStat: resolution={self.width}x{self.height}, "
            f"duration={self.duration:.2f}s, fps={self.fps:.2f}"
        )


class AsyncFrameReader:
    def __init__(self, video: str):
        self.cap = cv2.VideoCapture(video)
        self.loop = asyncio.get_running_loop()

    @property
    def video_stat(self) -> VideoStat:
        return VideoStat.from_video_capture(self.cap)

    @property
    def closed(self):
        return not self.cap.isOpened()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        self.cap.release()

    async def read(self) -> Tuple[bool, np.ndarray]:
        has_frame, frame = await self.loop.run_in_executor(None, self.cap.read)
        return has_frame, frame
