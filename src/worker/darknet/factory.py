from torch import nn

from . import parser, layers


def create_conv_layer(
    section: parser.ConvolutionalSection, in_channels: int, index: int
) -> nn.Module:
    return layers.Convolutional(section, in_channels, index)


def create_upsample_layer(section: parser.UpsampleSection, index: int) -> nn.Module:
    module = nn.Sequential()
    module.add_module(f"upsample_{index}", layers.Upsample(section.stride))
    return module


def create_yolo_layer(section: parser.YOLOSection, index: int) -> nn.Module:
    anchors = [section.anchors[i] for i in section.mask]
    yolo = layers.YOLO(anchors, section.classes)

    module = nn.Sequential()
    module.add_module(f"yolo_{index}", yolo)
    return module
