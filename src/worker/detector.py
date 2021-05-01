import dataclasses
from typing import List, Dict

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

    def __init__(self, cfg: str, weights: str, target: int):
        net = cv2.dnn.readNetFromDarknet(cfg, weights)
        net.setPreferableTarget(target)
        self._net = net
        self._out_names = net.getUnconnectedOutLayersNames()

    def detect(
        self, img: np.ndarray, conf_threshold: float = 0.5, nms_threshold: float = 0.4
    ) -> List["DetectionResult"]:
        # ToDo: use size from config
        blob = cv2.dnn.blobFromImage(
            img, self._SCALE_FACTOR, (416, 416), self._RGB_MEAN, swapRB=True, crop=False
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


@dataclasses.dataclass
class DetectionResult:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    clazz: int
