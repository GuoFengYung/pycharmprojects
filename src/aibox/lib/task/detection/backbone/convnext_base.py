from typing import Union, Tuple

import torchvision.models
from efficientnet_pytorch import EfficientNet
from torch import nn, Tensor
from torch.nn import functional as F

from . import Backbone


class ConvNeXt_Base(Backbone):

    def __init__(self, pretrained: bool, num_frozen_levels: int):
        super().__init__(pretrained, num_frozen_levels)

    def _build_component(self) -> Backbone.Component:
        convnext_base = torchvision.models.convnext_base(pretrained=self.pretrained)

        # 0
        # torch.Size([1, 128, 56, 56])
        # 1
        # torch.Size([1, 128, 56, 56])
        # 2
        # torch.Size([1, 256, 28, 28])
        # 3
        # torch.Size([1, 256, 28, 28])
        # 4
        # torch.Size([1, 512, 14, 14])
        # 5
        # torch.Size([1, 512, 14, 14])
        # 6
        # torch.Size([1, 1024, 7, 7])
        # 7
        # torch.Size([1, 1024, 7, 7])

        conv1 = convnext_base.features[:1]
        conv2 = convnext_base.features[1:2]
        conv3 = convnext_base.features[2:4]
        conv4 = convnext_base.features[4:6]
        conv5 = convnext_base.features[6:]

        num_conv1_out = 128
        num_conv2_out = 128
        num_conv3_out = 256
        num_conv4_out = 512
        num_conv5_out = 1024

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
