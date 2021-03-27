import asyncio

import cv2
import numpy as np

from . import reader


class AsyncFrameWriter:
    def __init__(self, path: str, fourcc: str, stat: reader.VideoStat):
        fourcc = cv2.VideoWriter_fourcc(*fourcc)
        self.writer = cv2.VideoWriter(path, fourcc, stat.fps, (stat.width, stat.height))
        self.loop = asyncio.get_running_loop()

    def __aenter__(self):
        return self

    def __aexit__(self, exc_type, exc_val, exc_tb):
        self.writer.release()

    async def write(self, frame: np.ndarray):
        await self.loop.run_in_executor(None, self.writer.write, frame)
