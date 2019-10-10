import torch
import torchvision as tv
import torch.nn as nn
import numpy as np


# 定义基本卷积层，由卷积层、归一化层和激活层组成
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, ksize, stride, dilation, padding):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, ksize, stride=stride,
                            dilation=dilation, padding=padding)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


# 上采样，利用插值函数实现
class UpsampleConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                dilation=1, padding=1, output_padding=1):
        super(UpsampleConv, self).__init__()

    def forward(self, x):
        x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
        return x


# 第一阶生成网络
class Generator1(nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()
        # 网络定义的第一种形式
        self.encoder = nn.Sequential(
            BasicConv2d(4, 32, 5, 1, 1, 2),
            BasicConv2d(32, 64, 3, 1, 1, 1),
            BasicConv2d(64, 64, 3, 2, 1, 1),
            BasicConv2d(64, 128, 3, 1, 1, 1),
            BasicConv2d(128, 128, 3, 2, 1, 1),
            BasicConv2d(128, 256, 3, 1, 1, 1),
            BasicConv2d(256, 256, 3, 2, 1, 1),
            BasicConv2d(256, 256, 3, 1, 1, 1)
        )

        # 并行网络层结构
        self.layer9_0 = BasicConv2d(256, 256, 3, 1, 1, 1)
        self.layer9_1 = BasicConv2d(256, 256, 3, 1, 2, 2)
        self.layer9_2 = BasicConv2d(256, 256, 3, 1, 4, 4)
        self.layer9_3 = BasicConv2d(256, 256, 3, 1, 8, 8)
        self.layer9_4 = BasicConv2d(256, 256, 3, 1, 16, 16)
        self.layer10 = BasicConv2d(1280, 256, 1, 1, 1, 0)
        self.layer11 = BasicConv2d(256, 256, 3, 1, 4, 4)
        self.layer12 = BasicConv2d(256, 256, 3, 1, 16, 16)

        # 上采样层
        self.decoder = nn.Sequential(
            # upsample1
            UpsampleConv(256, 256, 3, 2, 1, 1, 1),
            BasicConv2d(256, 128, 3, 1, 1, 1),
            BasicConv2d(128, 128, 3, 1, 1, 1),
            # upsample2
            UpsampleConv(128, 128, 3, 2, 1, 1, 1),
            BasicConv2d(128, 64, 3, 1, 1, 1),
            BasicConv2d(64, 64, 3, 1, 1, 1),
            # upsample3
            UpsampleConv(64, 64, 3, 2, 1, 1, 1),
            BasicConv2d(64, 3, 3, 1, 1, 1)
        )
        self.layer21 = nn.Conv2d(3, 3, 3, stride=1, dilation=1, padding=1)
        self.result = nn.Tanh()

    def forward(self, x):
        # 前向过程
        x = self.encoder(x)
        x_0 = self.layer9_0(x)
        x_1 = self.layer9_1(x)
        x_2 = self.layer9_2(x)
        x_3 = self.layer9_3(x)
        x_4 = self.layer9_4(x)
        x = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.decoder(x)
        x = self.layer21(x)
        x = self.result(x)
        return x


class Generator2(nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        self.encoder = nn.Sequential(
            BasicConv2d(3, 32, 5, 1, 1, 2),
            BasicConv2d(32, 64, 3, 1, 1, 1),
            BasicConv2d(64, 64, 3, 2, 1, 1),
            BasicConv2d(64, 128, 3, 1, 1, 1),
            BasicConv2d(128, 128, 3, 2, 1, 1),
            BasicConv2d(128, 256, 3, 1, 1, 1)
        )
        self.layer9_0 = BasicConv2d(256, 256, 3, 1, 1, 1)
        self.layer9_1 = BasicConv2d(256, 256, 3, 1, 2, 2)
        self.layer9_2 = BasicConv2d(256, 256, 3, 1, 4, 4)
        self.layer9_3 = BasicConv2d(256, 256, 3, 1, 8, 8)
        self.layer9_4 = BasicConv2d(256, 256, 3, 1, 16, 16)
        self.layer10 = BasicConv2d(1280, 256, 1, 1, 1, 0)
        self.layer11 = BasicConv2d(256, 256, 3, 1, 4, 4)
        self.layer12 = BasicConv2d(256, 256, 3, 1, 16, 16)

        self.decoder = nn.Sequential(
            # upsample1
            UpsampleConv(256, 256, 3, 2, 1, 1, 1),
            BasicConv2d(256, 128, 3, 1, 1, 1),
            BasicConv2d(128, 128, 3, 1, 1, 1),
            # upsample2
            UpsampleConv(128, 128, 3, 2, 1, 1, 1),
            BasicConv2d(128, 64, 3, 1, 1, 1),
            BasicConv2d(64, 3, 3, 1, 1, 1)
        )
        self.layer21 = nn.Conv2d(3, 3, 3, stride=1, dilation=1, padding=1)
        self.result = nn.Tanh()

    def forward(self, x):
        x = self.encoder(x)
        x_0 = self.layer9_0(x)
        x_1 = self.layer9_1(x)
        x_2 = self.layer9_2(x)
        x_3 = self.layer9_3(x)
        x_4 = self.layer9_4(x)
        x = torch.cat([x_0, x_1, x_2, x_3, x_4], 1)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.decoder(x)
        x = self.layer21(x)
        x = self.result(x)
        return x


# 判别网络
class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3, 64, 5, 2, 2, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 5, 2, 2, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 5, 2, 2, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 5, 2, 2, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 5, 2, 2, bias=False),
            nn.BatchNorm2d(1),
            # nn.LeakyReLU(0.2, inplace=True)
        )
        # self.fc = nn.Linear(512*16*16, 1)
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = self.main(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)
        x = x.view(-1, 1)
        x_s = self.sg(x)
        return x, x_s
