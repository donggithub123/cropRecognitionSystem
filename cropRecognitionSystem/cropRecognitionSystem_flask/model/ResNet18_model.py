# -*- coding: utf-8 -*-
# @Time : 2024-09-2024/9/16 17:07
# @Author : 林枫
# @File : ResNet18_model.py

import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

'''
这是pytorch神经网络模型
'''


class Residual(nn.Module):  # 本类已保存在d2lzh_pytorch包中方便以后使用
    def __init__(self, in_channels, out_channels, use_1x1conv=False, stride=1):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        return F.relu(y + x)


class ResNet18(nn.Module):
    def __init__(self, Residual):
        super().__init__()
        self.b1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(
            Residual(64, 64, use_1x1conv=False, stride=1),
            Residual(64, 64, use_1x1conv=False, stride=1))

        self.b3 = nn.Sequential(
            Residual(64, 128, use_1x1conv=True, stride=2),
            Residual(128, 128, use_1x1conv=False, stride=1))

        self.b4 = nn.Sequential(
            Residual(128, 256, use_1x1conv=True, stride=2),
            Residual(256, 256, use_1x1conv=False, stride=1))

        self.b5 = nn.Sequential(
            Residual(256, 512, use_1x1conv=True, stride=2),
            Residual(512, 512, use_1x1conv=False, stride=1))

        self.b6 = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Flatten(),
                                nn.Linear(512, 5))

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.b6(x)
        return x


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
    model = ResNet18(Residual).to(device)
    print(summary(model, (3, 224, 224)))
