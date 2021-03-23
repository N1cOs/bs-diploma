import dataclasses
from typing import List, Tuple

import numpy as np
import torch
from torchvision import ops
from torchvision import transforms as tr

from lib import view
from . import network


class ObjectDetector:
    def __init__(self, net: network.Network):
        net.eval()
        self.net = net
        # ToDo: get value from network definition
        self.img_shape = ImageShape(416, 416)

    def detect(
        self, img: np.ndarray, conf_threshold: float = 0.5, iou_threshold: float = 0.4
    ) -> List[view.DetectionResult]:
        transforms = tr.Compose(
            [
                tr.ToTensor(),
                tr.Resize(self.img_shape.hw_tuple()),
            ]
        )
        tr_img = transforms(img)

        predictions = self.net(tr_img.unsqueeze(0))
        # from (center x, center y, width, height) to (x1, y1, x2, y2)
        predictions[..., :4] = self._xywh2xyxy(predictions[..., :4])

        results = []
        orig_shape = ImageShape.from_numpy(img)
        for prediction in predictions:
            # filter out object confidence below threshold
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
                obj = [o.item() for o in obj]
                res = view.DetectionResult(
                    obj[0], obj[1], obj[2], obj[3], obj[4], int(obj[5])
                )
                results.append(self._rescaled_result(res, orig_shape))

        return results

    @staticmethod
    def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
        y = x.new(x.shape)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def _rescaled_result(
        self, res: view.DetectionResult, orig_shape: "ImageShape"
    ) -> view.DetectionResult:
        scale_x = orig_shape.width / self.img_shape.width
        scale_y = orig_shape.height / self.img_shape.height

        return dataclasses.replace(
            res,
            x1=res.x1 * scale_x,
            y1=res.y1 * scale_y,
            x2=res.x2 * scale_x,
            y2=res.y2 * scale_y,
        )


@dataclasses.dataclass
class ImageShape:
    width: int
    height: int

    @classmethod
    def from_numpy(cls, arr: np.ndarray):
        try:
            shape = arr.shape
            return cls(shape[1], shape[0])
        except IndexError:
            raise ValueError("invalid numpy array: shape must be greater or equal to 2")

    def hw_tuple(self) -> Tuple[int]:
        return tuple([self.height, self.width])
