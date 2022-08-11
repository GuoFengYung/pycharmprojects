from typing import Union, Tuple

import torchvision.models
from efficientnet_pytorch import EfficientNet
from torch import nn, Tensor
from torch.nn import functional as F

from . import Backbone


class EfficientNet_V2(Backbone):

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        efficientnet_v2_s = torchvision.models.efficientnet_v2_s(pretrained=self.pretrained)

        # 0
        # torch.Size([1, 24, 120, 120])
        # 1
        # torch.Size([1, 24, 120, 120])
        # 2
        # torch.Size([1, 48, 60, 60])
        # 3
        # torch.Size([1, 64, 30, 30])
        # 4
        # torch.Size([1, 128, 15, 15])
        # 5
        # torch.Size([1, 160, 15, 15])
        # 6
        # torch.Size([1, 256, 8, 8])
        # 7
        # torch.Size([1, 1280, 8, 8])

        conv1 = efficientnet_v2_s.features[:2]
        conv2 = efficientnet_v2_s.features[2:3]
        conv3 = efficientnet_v2_s.features[3:4]
        conv4 = efficientnet_v2_s.features[4:6]
        conv5 = efficientnet_v2_s.features[6:]

        num_conv1_out = 48
        num_conv2_out = 64
        num_conv3_out = 128
        num_conv4_out = 256
        num_conv5_out = 1280

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
