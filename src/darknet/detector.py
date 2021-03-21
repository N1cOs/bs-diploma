import dataclasses
from typing import List

import torch
from torchvision import ops

from . import network


class ObjectDetector:
    def __init__(self, net: network.Network, classes: List[str]):
        net.eval()
        self.net = net
        self.classes = classes

    def detect(
        self, img: torch.Tensor, conf_threshold: float = 0.5, iou_threshold: float = 0.4
    ) -> List["DetectionResult"]:
        predictions = self.net(img)
        # from (center x, center y, width, height) to (x1, y1, x2, y2)
        predictions[..., :4] = self._xywh2xyxy(predictions[..., :4])

        results = []
        for prediction in predictions:
            # filter out confidence scores below threshold
            prediction = prediction[prediction[:, 4] >= conf_threshold]

            coordinates = prediction[:, :4]
            # multiply object confidence by class confidence
            score = prediction[:, 4] * prediction[:, 5:].max(1)[0]
            prediction = torch.index_select(
                prediction, 0, ops.nms(coordinates, score, iou_threshold)
            )

            coordinates = prediction[:, :4]
            class_confs, class_preds = prediction[:, 5:].max(1, keepdim=True)
            objects = torch.cat(
                (coordinates, class_confs.float(), class_preds.float()), 1
            )

            for obj in objects:
                clazz_i = int(obj[5].item())
                try:
                    clazz = self.classes[clazz_i]
                except IndexError:
                    raise ValueError(f"unknown class with index {clazz_i}")

                res = DetectionResult(obj[0], obj[1], obj[2], obj[3], obj[4], clazz)
                results.append(res)

        return results

    @staticmethod
    def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
        y = x.new(x.shape)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y


@dataclasses.dataclass
class DetectionResult:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    clazz: str
