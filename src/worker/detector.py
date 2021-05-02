import dataclasses
import re
from typing import List, Dict, Tuple

import cv2
import numpy as np

TARGET_MAPPING = {
    cv2.dnn.DNN_TARGET_CPU: "cpu",
    cv2.dnn.DNN_TARGET_OPENCL: "opencl",
    cv2.dnn.DNN_TARGET_OPENCL_FP16: "opencl_fp16",
}


def get_available_targets() -> Dict[str, int]:
    targets = {}
    for target in cv2.dnn.getAvailableTargets(cv2.dnn.DNN_BACKEND_DEFAULT):
        val = target[0]
        key = TARGET_MAPPING.get(val)
        if key is None:
            raise ValueError(f"unknown target: {val}")
        targets[key] = val
    return targets


class DarknetObjectDetector:
    _SCALE_FACTOR = 0.00392
    _RGB_MEAN = (0, 0, 0)

    def __init__(self, cfg: str, weights: str, target: int, warm_up: bool = True):
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        net.setPreferableTarget(target)

        self._net = net
        self._size = self._parse_net_size(cfg)
        self._out_names = net.getUnconnectedOutLayersNames()

        if warm_up:
            shape = self._size + (3,)
            dummy_img = np.random.randint(0, 255, shape, dtype=np.uint8)
            self.detect(dummy_img)

    def detect(
        self, img: np.ndarray, conf_threshold: float = 0.5, nms_threshold: float = 0.4
    ) -> List["DetectionResult"]:
        blob = cv2.dnn.blobFromImage(
            img,
            self._SCALE_FACTOR,
            self._size,
            self._RGB_MEAN,
            swapRB=True,
            crop=False,
        )
        self._net.setInput(blob)
        outs = self._net.forward(self._out_names)

        layers = self._net.getLayerNames()
        last_layer_id = self._net.getLayerId(layers[-1])
        last_layer = self._net.getLayer(last_layer_id)

        boxes = []
        classes = []
        confidences = []
        img_h, img_w = img.shape[0], img.shape[1]
        if last_layer.type == "Region":
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    clazz = np.argmax(scores)
                    confidence = scores[clazz]

                    if confidence > conf_threshold:
                        center_x, center_y = detection[0] * img_w, detection[1] * img_h
                        w, h = detection[2] * img_w, detection[3] * img_h
                        left, top = center_x - w / 2, center_y - h / 2

                        classes.append(clazz)
                        # important: confidence must be float
                        confidences.append(float(confidence))
                        # important: boxes must be ints
                        boxes.append(list(map(int, [left, top, w, h])))
        else:
            raise ValueError(f"unknown last layer type: {last_layer.type}")

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
        results = []
        for i in indices:
            i = i[0]
            box = boxes[i]
            left = box[0]
            top = box[1]
            w, h = box[2], box[3]

            res = DetectionResult(
                x1=left,
                y1=top,
                x2=left + w,
                y2=top + h,
                score=confidences[i],
                clazz=classes[i],
            )
            results.append(res)

        return results

    @staticmethod
    def _parse_net_size(cfg_path: str) -> Tuple[int, int]:
        with open(cfg_path, "r") as cfg:
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

        for raw in raw_sections:
            type_ = raw["type"]
            if type_ == "net":
                width = raw.get("width")
                if width is None:
                    raise ValueError(f"missing width in net section: {cfg_path}")

                height = raw.get("height")
                if height is None:
                    raise ValueError(f"missing height in net section: {cfg_path}")

                return int(width), int(height)
        raise ValueError(f"missing net section: {cfg_path}")


@dataclasses.dataclass
class DetectionResult:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    clazz: int
