from typing import List

import numpy as np
import torch
from torch import nn

from . import parser, factory, layers


class Network(nn.Module):
    def __init__(self, sections: List, module_list: nn.ModuleList):
        super().__init__()
        # ToDo: think about removing sections
        self.sections = sections
        self.module_list = module_list

    @classmethod
    def from_config(cls, path: str):
        modules = nn.ModuleList()
        output_filters = [3]

        sections = parser.parse_config(path)
        for i, section in enumerate(sections):
            if isinstance(section, parser.ConvolutionalSection):
                filters = section.filters
                module = factory.create_conv_layer(section, output_filters[-1], i)
            elif isinstance(section, parser.UpsampleSection):
                module = factory.create_upsample_layer(section, i)
            elif isinstance(section, parser.YOLOSection):
                module = factory.create_yolo_layer(section, i)
            elif isinstance(section, parser.ShortcutSection):
                filters = output_filters[1:][section.from_]
                module = nn.Sequential()
                module.add_module(f"shortcut_{i}", nn.Sequential())
            elif isinstance(section, parser.RouteSection):
                filters = sum([output_filters[1:][i] for i in section.layers])
                module = nn.Sequential()
                module.add_module(f"route_{i}", nn.Sequential())
            else:
                raise ValueError(f"unknown section: {section}")
            modules.append(module)
            # ToDo: fix using before assignment
            output_filters.append(filters)
        return cls(sections, modules)

    def load_weights(self, path: str):
        with open(path, "rb") as f:
            # skip header
            np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)

        for module in self.module_list:
            if isinstance(module, layers.Convolutional):
                read = module.load_weights(weights)
                weights = weights[read:]

    def forward(self, x, targets=None):
        img_dim = x.shape[2]
        loss = 0
        layer_outputs, yolo_outputs = [], []
        for section, module in zip(self.sections, self.module_list):
            if isinstance(
                section,
                (
                    parser.ConvolutionalSection,
                    parser.UpsampleSection,
                    parser.MaxPoolSection,
                ),
            ):
                x = module(x)
            elif isinstance(section, parser.RouteSection):
                x = torch.cat([layer_outputs[i] for i in section.layers], 1)
            elif isinstance(section, parser.ShortcutSection):
                x = layer_outputs[-1] + layer_outputs[section.from_]
            elif isinstance(section, parser.YOLOSection):
                x, layer_loss = module[0](x, targets, img_dim)
                loss += layer_loss
                yolo_outputs.append(x)
            layer_outputs.append(x)
        yolo_outputs = torch.cat(yolo_outputs, 1).detach().cpu()
        return yolo_outputs if targets is None else (loss, yolo_outputs)
