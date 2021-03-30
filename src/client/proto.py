import dataclasses
import io
import struct
from typing import Tuple

import numpy as np

import video

VERSION = 1


@dataclasses.dataclass
class DetectRequest:
    id: int
    img: np.ndarray


@dataclasses.dataclass
class DetectResponse:
    id: int
    detections: Tuple[video.DetectionResult]

    def __lt__(self, other):
        return self.id < other.id


def dump_detect_request(req: DetectRequest) -> bytes:
    buf = io.BytesIO()
    buf.write(struct.pack("!BI", VERSION, req.id))

    h, w, c = req.img.shape
    buf.write(struct.pack("!hhB", h, w, c))
    buf.write(req.img.tobytes())

    return buf.getvalue()


def parse_detect_response(data: bytes) -> DetectResponse:
    buf = io.BytesIO(data)
    id_, len_ = struct.unpack("!Ih", buf.read(6))
    return DetectResponse(id_, tuple(_parse_detection(buf) for _ in range(len_)))


def _parse_detection(buf: io.BytesIO) -> video.DetectionResult:
    x1, y1 = struct.unpack("!ff", buf.read(8))
    x2, y2 = struct.unpack("!ff", buf.read(8))
    score, clazz = struct.unpack("!fh", buf.read(6))
    return video.DetectionResult(x1=x1, y1=y1, x2=x2, y2=y2, score=score, clazz=clazz)
