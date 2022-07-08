from typing import Union, Tuple

import torchvision.models
from efficientnet_pytorch import EfficientNet
from torch import nn, Tensor
from torch.nn import functional as F

from . import Backbone


class EfficientNet_B7(Backbone):

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        efficientnet_b7 = torchvision.models.efficientnet_b7(pretrained=self.pretrained)

        # 0 torch.Size([1, 64, 120, 180])
        # 1 torch.Size([1, 32, 120, 180])
        # 2 torch.Size([1, 48, 60, 90])
        # 3 torch.Size([1, 80, 30, 45])
        # 4 torch.Size([1, 160, 15, 23])
        # 5 torch.Size([1, 224, 15, 23])
        # 6 torch.Size([1, 384, 8, 12])
        # 7 torch.Size([1, 640, 8, 12])
        # 8 torch.Size([1, 2560, 8, 12])
        conv1 = efficientnet_b7.features[:2]
        conv2 = efficientnet_b7.features[2:3]
        conv3 = efficientnet_b7.features[3:6]
        conv4 = efficientnet_b7.features[6:8]
        conv5 = efficientnet_b7.features[8:]

        num_conv1_out = 32
        num_conv2_out = 48
        num_conv3_out = 224
        num_conv4_out = 640
        num_conv5_out = 2560

        return Backbone.Component(
            conv1, conv2, conv3, conv4, conv5,
            num_conv1_out, num_conv2_out, num_conv3_out, num_conv4_out, num_conv5_out
        )

    @staticmethod
    def normalization_means() -> Tuple[float, float, float]:
        return 0.485, 0.456, 0.406

    @staticmethod
    def normalization_stds() -> Tuple[float, float, float]:
        return 0.229, 0.224, 0.225
