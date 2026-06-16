import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)


class NestedUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, deepsupervision=False, nb_filter=[32, 64, 128, 256, 512]):
        super().__init__()
        self.num_classes = num_classes
        self.deepsupervision = deepsupervision
        self.nb_filter = nb_filter

        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv0_0 = DoubleConv(in_channels, self.nb_filter[0])
        self.conv1_0 = DoubleConv(self.nb_filter[0], self.nb_filter[1])
        self.conv2_0 = DoubleConv(self.nb_filter[1], self.nb_filter[2])
        self.conv3_0 = DoubleConv(self.nb_filter[2], self.nb_filter[3])
        self.conv4_0 = DoubleConv(self.nb_filter[3], self.nb_filter[4])

        self.conv0_1 = DoubleConv(self.nb_filter[0] + self.nb_filter[1], self.nb_filter[0])
        self.conv1_1 = DoubleConv(self.nb_filter[1] + self.nb_filter[2], self.nb_filter[1])
        self.conv2_1 = DoubleConv(self.nb_filter[2] + self.nb_filter[3], self.nb_filter[2])
        self.conv3_1 = DoubleConv(self.nb_filter[3] + self.nb_filter[4], self.nb_filter[3])

        self.conv0_2 = DoubleConv(self.nb_filter[0] * 2 + self.nb_filter[1], self.nb_filter[0])
        self.conv1_2 = DoubleConv(self.nb_filter[1] * 2 + self.nb_filter[2], self.nb_filter[1])
        self.conv2_2 = DoubleConv(self.nb_filter[2] * 2 + self.nb_filter[3], self.nb_filter[2])

        self.conv0_3 = DoubleConv(self.nb_filter[0] * 3 + self.nb_filter[1], self.nb_filter[0])
        self.conv1_3 = DoubleConv(self.nb_filter[1] * 3 + self.nb_filter[2], self.nb_filter[1])

        self.conv0_4 = DoubleConv(self.nb_filter[0] * 4 + self.nb_filter[1], self.nb_filter[0])

        if self.num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

        if self.deepsupervision:
            self.final1 = nn.Conv2d(self.nb_filter[0], self.num_classes, kernel_size=1)
            self.final2 = nn.Conv2d(self.nb_filter[0], self.num_classes, kernel_size=1)
            self.final3 = nn.Conv2d(self.nb_filter[0], self.num_classes, kernel_size=1)
            self.final4 = nn.Conv2d(self.nb_filter[0], self.num_classes, kernel_size=1)
        else:
            self.final = nn.Conv2d(self.nb_filter[0], self.num_classes, kernel_size=1)

    def forward(self, input):
        input_h, input_w = input.shape[2], input.shape[3]  # 保存输入原始尺寸

        # 编码阶段：保存每个特征图的尺寸，用于后续对齐
        x0_0 = self.conv0_0(input)
        x0_0_size = x0_0.shape[2:]  # (H, W)

        x1_0 = self.conv1_0(self.pool(x0_0))
        x1_0_size = x1_0.shape[2:]

        x2_0 = self.conv2_0(self.pool(x1_0))
        x2_0_size = x2_0.shape[2:]

        x3_0 = self.conv3_0(self.pool(x2_0))
        x3_0_size = x3_0.shape[2:]

        x4_0 = self.conv4_0(self.pool(x3_0))
        x4_0_size = x4_0.shape[2:]

        # 解码阶段：上采样后强制对齐尺寸
        # 第1层解码（与x0_0拼接）
        up_x1_0 = self.up(x1_0)  # 上采样x1_0
        # 关键修改1：调整上采样结果尺寸，与x0_0完全一致
        up_x1_0 = F.interpolate(up_x1_0, size=x0_0_size, mode='bilinear', align_corners=True)
        x0_1 = self.conv0_1(torch.cat([x0_0, up_x1_0], dim=1))  # 拼接

        # 第2层解码（与x1_0拼接）
        up_x2_0 = self.up(x2_0)
        # 关键修改2：对齐x1_0尺寸
        up_x2_0 = F.interpolate(up_x2_0, size=x1_0_size, mode='bilinear', align_corners=True)
        x1_1 = self.conv1_1(torch.cat([x1_0, up_x2_0], dim=1))

        # 第2层解码（与x0_0、x0_1拼接）
        up_x1_1 = self.up(x1_1)
        # 关键修改3：对齐x0_0尺寸
        up_x1_1 = F.interpolate(up_x1_1, size=x0_0_size, mode='bilinear', align_corners=True)
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, up_x1_1], dim=1))

        # 第3层解码（与x2_0拼接）
        up_x3_0 = self.up(x3_0)
        # 关键修改4：对齐x2_0尺寸
        up_x3_0 = F.interpolate(up_x3_0, size=x2_0_size, mode='bilinear', align_corners=True)
        x2_1 = self.conv2_1(torch.cat([x2_0, up_x3_0], dim=1))

        # 第3层解码（与x1_0、x1_1拼接）
        up_x2_1 = self.up(x2_1)
        # 关键修改5：对齐x1_0尺寸
        up_x2_1 = F.interpolate(up_x2_1, size=x1_0_size, mode='bilinear', align_corners=True)
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, up_x2_1], dim=1))

        # 第3层解码（与x0_0、x0_1、x0_2拼接）
        up_x1_2 = self.up(x1_2)
        # 关键修改6：对齐x0_0尺寸
        up_x1_2 = F.interpolate(up_x1_2, size=x0_0_size, mode='bilinear', align_corners=True)
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, up_x1_2], dim=1))

        # 第4层解码（与x3_0拼接）
        up_x4_0 = self.up(x4_0)
        # 关键修改7：对齐x3_0尺寸
        up_x4_0 = F.interpolate(up_x4_0, size=x3_0_size, mode='bilinear', align_corners=True)
        x3_1 = self.conv3_1(torch.cat([x3_0, up_x4_0], dim=1))

        # 第4层解码（与x2_0、x2_1拼接）
        up_x3_1 = self.up(x3_1)
        # 关键修改8：对齐x2_0尺寸
        up_x3_1 = F.interpolate(up_x3_1, size=x2_0_size, mode='bilinear', align_corners=True)
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, up_x3_1], dim=1))

        # 第4层解码（与x1_0、x1_1、x1_2拼接）
        up_x2_2 = self.up(x2_2)
        # 关键修改9：对齐x1_0尺寸
        up_x2_2 = F.interpolate(up_x2_2, size=x1_0_size, mode='bilinear', align_corners=True)
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, up_x2_2], dim=1))

        # 第4层解码（与x0_0、x0_1、x0_2、x0_3拼接）
        up_x1_3 = self.up(x1_3)
        # 关键修改10：对齐x0_0尺寸
        up_x1_3 = F.interpolate(up_x1_3, size=x0_0_size, mode='bilinear', align_corners=True)
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, up_x1_3], dim=1))

        # 输出处理
        if self.deepsupervision:
            output1 = self.activation(self.final1(x0_1))
            output2 = self.activation(self.final2(x0_2))
            output3 = self.activation(self.final3(x0_3))
            output4 = self.activation(self.final4(x0_4))

            # 最终对齐到输入尺寸
            output1 = F.interpolate(output1, size=(input_h, input_w), mode='bilinear', align_corners=True)
            output2 = F.interpolate(output2, size=(input_h, input_w), mode='bilinear', align_corners=True)
            output3 = F.interpolate(output3, size=(input_h, input_w), mode='bilinear', align_corners=True)
            output4 = F.interpolate(output4, size=(input_h, input_w), mode='bilinear', align_corners=True)

            return {"out": output4, "aux1": output1, "aux2": output2, "aux3": output3}
        else:
            output = self.activation(self.final(x0_4))
            output = F.interpolate(output, size=(input_h, input_w), mode='bilinear', align_corners=True)
            return {"out": output}
