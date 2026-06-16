import torch
import torch.nn as nn
import torch.nn.functional as F


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class Recurrent_block(nn.Module):
    def __init__(self, ch_out, t=2):
        super(Recurrent_block, self).__init__()
        self.t = t
        self.ch_out = ch_out
        self.conv = nn.Sequential(
            nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        for i in range(self.t):
            if i == 0:
                x1 = self.conv(x)
            x1 = self.conv(x + x1)
        return x1


class RRCNN_block(nn.Module):
    def __init__(self, ch_in, ch_out, t=2):
        super(RRCNN_block, self).__init__()
        self.RCNN = nn.Sequential(
            Recurrent_block(ch_out, t=t),
            Recurrent_block(ch_out, t=t)
        )
        self.Conv_1x1 = nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.Conv_1x1(x)
        x1 = self.RCNN(x)
        return x + x1


class R2U_Net(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, t=2):
        super(R2U_Net, self).__init__()
        self.t = t
        self.num_classes = num_classes

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Upsample = nn.Upsample(scale_factor=2)

        self.RRCNN1 = RRCNN_block(ch_in=in_channels, ch_out=64, t=self.t)
        self.RRCNN2 = RRCNN_block(ch_in=64, ch_out=128, t=self.t)
        self.RRCNN3 = RRCNN_block(ch_in=128, ch_out=256, t=self.t)
        self.RRCNN4 = RRCNN_block(ch_in=256, ch_out=512, t=self.t)
        self.RRCNN5 = RRCNN_block(ch_in=512, ch_out=1024, t=self.t)

        self.Up5 = up_conv(ch_in=1024, ch_out=512)
        self.Up_RRCNN5 = RRCNN_block(ch_in=1024, ch_out=512, t=self.t)

        self.Up4 = up_conv(ch_in=512, ch_out=256)
        self.Up_RRCNN4 = RRCNN_block(ch_in=512, ch_out=256, t=self.t)

        self.Up3 = up_conv(ch_in=256, ch_out=128)
        self.Up_RRCNN3 = RRCNN_block(ch_in=256, ch_out=128, t=self.t)

        self.Up2 = up_conv(ch_in=128, ch_out=64)
        self.Up_RRCNN2 = RRCNN_block(ch_in=128, ch_out=64, t=self.t)

        self.Conv_1x1 = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0)

        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        # 保存各编码阶段的特征图尺寸，用于后续调整
        x1 = self.RRCNN1(x)
        x1_size = x1.shape[2:]  # 保存x1的尺寸

        x2 = self.Maxpool(x1)
        x2 = self.RRCNN2(x2)
        x2_size = x2.shape[2:]  # 保存x2的尺寸

        x3 = self.Maxpool(x2)
        x3 = self.RRCNN3(x3)
        x3_size = x3.shape[2:]  # 保存x3的尺寸

        x4 = self.Maxpool(x3)
        x4 = self.RRCNN4(x4)
        x4_size = x4.shape[2:]  # 保存x4的尺寸

        x5 = self.Maxpool(x4)
        x5 = self.RRCNN5(x5)

        # 解码阶段：上采样后调整尺寸以匹配编码阶段的特征图
        d5 = self.Up5(x5)
        # 关键修改1：调整d5尺寸以匹配x4
        d5 = F.interpolate(d5, size=x4_size, mode='bilinear', align_corners=True)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_RRCNN5(d5)

        d4 = self.Up4(d5)
        # 关键修改2：调整d4尺寸以匹配x3
        d4 = F.interpolate(d4, size=x3_size, mode='bilinear', align_corners=True)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_RRCNN4(d4)

        d3 = self.Up3(d4)
        # 关键修改3：调整d3尺寸以匹配x2
        d3 = F.interpolate(d3, size=x2_size, mode='bilinear', align_corners=True)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_RRCNN3(d3)

        d2 = self.Up2(d3)
        # 关键修改4：调整d2尺寸以匹配x1
        d2 = F.interpolate(d2, size=x1_size, mode='bilinear', align_corners=True)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_RRCNN2(d2)

        d1 = self.Conv_1x1(d2)

        # 确保最终输出与输入尺寸一致
        if d1.shape[2:] != x.shape[2:]:
            d1 = F.interpolate(d1, size=x.shape[2:], mode='bilinear', align_corners=True)

        output = self.activation(d1)
        return {"out": output}
