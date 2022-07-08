import torch
import torchvision
from efficientnet_pytorch import EfficientNet

efficientnet_b7 = torchvision.models.efficientnet_b7()
x = torch.randn(1, 3, 240, 360)
for i, feature in enumerate(efficientnet_b7.features):
    x = feature(x)
    print(i, x.shape)
# regnet_y_8gf = torchvision.models.regnet_y_8gf(pretrained=False)
# print(regnet_y_8gf)
