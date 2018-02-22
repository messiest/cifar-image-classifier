import torch
import torch.nn as nn
import torch.nn.functional as F  # access to common functions
from torch.autograd import Variable


class Inception(nn.Module):
    def __init__(self, slices, n1x1, n3x3r, n3x3, n5x5r, n5x5, pool_x):
        super(Inception, self).__init__()
        # 1x1 conv
        self.path1 = nn.Sequential(
            nn.Conv2d(slices, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(True),
        )

        # 1x1 conv -> 3x3 conv
        self.path2 = nn.Sequential(
            nn.Conv2d(slices, n3x3r, kernel_size=1),
            nn.BatchNorm2d(n3x3r),
            nn.ReLU(True),
            nn.Conv2d(n3x3r, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(True),
        )

        # 1x1 conv -> 5x5 conv
        self.path3 = nn.Sequential(
            nn.Conv2d(slices, n5x5r, kernel_size=1),
            nn.BatchNorm2d(n5x5r),
            nn.ReLU(True),
            nn.Conv2d(n5x5r, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(True),
        )

        # 3x3 pool -> 3x3 conv
        self.path4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(slices, pool_x, kernel_size=1),
            nn.BatchNorm2d(pool_x),
            nn.ReLU(True),
        )

        # collection of all paths, runs paths on init
        self.paths = [
            self.path1,
            self.path2,
            self.path3,
            self.path4,
        ]

    def forward(self, x):
        """overwritten for torch"""
        pack = [path(x) for path in self.paths]

        return torch.cat(pack, 1)


class GoogLeNet(nn.Module):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        self.pre_processing = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),  # layer a1
            nn.BatchNorm2d(192),  # layer a2
            nn.ReLU(True),
        )

        # first inception block
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)  # b for 2nd block

        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # second inception block
        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)  # c for 3rd etc.
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        # third inception block
        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.linear = nn.Linear(1024, 10)

    def forward(self, x):
        """forward must be overwritten for torch"""
        out = self.pre_processing(x)
        out = self.a3(out)
        out = self.b3(out)
        out = self.maxpool(out)
        out = self.a4(out)
        out = self.b4(out)
        out = self.c4(out)
        out = self.d4(out)
        out = self.e4(out)
        out = self.maxpool(out)
        out = self.a5(out)
        out = self.b5(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)  # reshape the output tensor
        out = self.linear(out)

        return out


def main():
    print("running at", __file__)
    net = GoogLeNet()

    print(net)


if __name__ == "__main__":
    main()
