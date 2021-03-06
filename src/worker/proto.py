import dataclasses
import io
import struct
from typing import List

import detector
import numpy as np


@dataclasses.dataclass
class DetectRequest:
    id: int
    img: np.ndarray


@dataclasses.dataclass
class DetectResponse:
    id: int
    elapsed_sec: float
    detections: List[detector.DetectionResult]


def parse_detect_request(req: bytes) -> DetectRequest:
    buf = io.BytesIO(req)

    (version,) = struct.unpack("!B", buf.read(1))
    if version != 1:
        raise ValueError(f"unknown request version: {version}")
    (id_,) = struct.unpack("!I", buf.read(4))

    height, width, chans = struct.unpack("!hhB", buf.read(5))
    raw_img = np.frombuffer(buf.read(), dtype=np.uint8)
    img = raw_img.reshape((height, width, chans))

    return DetectRequest(id_, img.copy())


def dump_detect_response(resp: DetectResponse) -> bytes:
    buf = io.BytesIO()
    buf.write(struct.pack("!If", resp.id, resp.elapsed_sec))

    detections = resp.detections
    buf.write(struct.pack("!h", len(detections)))
    for d in detections:
        _dump_detection(d, buf)
    return buf.getvalue()


def _dump_detection(d: detector.DetectionResult, buf: io.BytesIO):
    buf.write(struct.pack("!hhhh", d.x1, d.y1, d.x2, d.y2))
    buf.write(struct.pack("!f", d.score))
    buf.write(struct.pack("!h", d.clazz))
