import dataclasses

import cv2


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
