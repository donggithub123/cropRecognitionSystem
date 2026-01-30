# -*- coding: utf-8 -*-
# @Time : 2024-10-2024/10/8 15:35
# @Author : 林枫
# @File : AlexNet_model.py
import torch
from torch import nn
from torchsummary import summary


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 96, 11, 4),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 添加自适应池化层，将输出调整为 (256, 6, 6)
            nn.AdaptiveAvgPool2d((6, 6))
        )
        self.fc = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 5),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    print(torch.cuda.get_device_name(0))
    print(torch.rand(3, 3).cuda())

    # Check CUDA version
    cuda_version = torch.version.cuda
    print("cuDA Version:", cuda_version)

    # Check CuDNN version

    cudnn_version = torch.backends.cudnn.version()
    print("CuDNN Version:", cudnn_version)
    model = AlexNet().to(device)
    print(summary(model, (3, 227, 227)))
