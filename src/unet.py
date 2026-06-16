from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(r"E:\unet")
from module.ECA import ECA_layer
from module.EMA import EMA
from module.LSK import LSKNet
from module.ELA import ELA
from module.Biformer import BiLevelRoutingAttention as BRA

class DoubleConv(nn.Module): # 定义一个名为 DoubleConv 的类，继承自 nn.Sequential
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__() # 调用父类的构造函数
        if mid_channels is None: # 如果未指定中间通道数，则默认为输出通道数
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False), # 第一个卷积层：输入通道数到中间通道数的卷积操作
            nn.BatchNorm2d(mid_channels), # 对中间通道数进行批量归一化操作
            nn.ReLU(inplace=True), # 使用 ReLU 激活函数进行非线性变换
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # self.eca = ECA_layer(out_channels)
        # self.ema = EMA(channels=out_channels)
        # self.ela = ELA(out_channels,phi="T")
        # self.bra = BRA(out_channels)


    def forward(self, x):
        x = self.double_conv(x)
        # x = self.eca(x)
        # x = self.ema(x)
        # x = self.ela(x)
        # x = self.bra(x)
        return x


class Down(nn.Sequential): # 定义一个名为 Down 的类，继承自 nn.Sequential
    def __init__(self, in_channels, out_channels):
         # 调用父类的构造函数
        super(Down, self).__init__(
            # 最大池化层，用于下采样，将特征图尺寸缩小一半
            nn.MaxPool2d(2, stride=2),
            # 使用定义的 DoubleConv 类来构建一个特征提取块
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module): # 定义一个名为 Up 的类，继承自 nn.Module
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        # 根据输入的参数决定使用双线性插值还是转置卷积
        if bilinear:
            # 使用双线性插值进行上采样
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # 使用定义的 DoubleConv 类构建一个特征提取块，其中中间通道数为输入通道数的一半
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 使用转置卷积进行上采样
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            # 使用定义的 DoubleConv 类构建一个特征提取块
            self.conv = DoubleConv(in_channels, out_channels)
        # self.ema = EMA(out_channels)

    # 定义前向传播函数，实现特征图的上采样和连接
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # 计算两个特征图的尺寸差异
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # 使用 F.pad 对 x1 进行填充，使其与 x2 的尺寸相同
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        # 将两个特征图按通道连接
        x = torch.cat([x2, x1], dim=1)
        # x = self.ema(x)
        # 经过特征提取块进行特征提取和处理
        x = self.conv(x)
        return x


class OutConv(nn.Sequential): # 定义一个名为 OutConv 的类，继承自 nn.Sequential
    def __init__(self, in_channels, num_classes):
         # 调用父类的构造函数
        super(OutConv, self).__init__(
            # 1x1 卷积层，用于生成最终的输出特征图
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class UNet(nn.Module): # 定义一个名为 UNet 的类，继承自 nn.Module
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        # 调用父类的构造函数
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        # 定义 U-Net 的各个组件
        self.in_conv = DoubleConv(in_channels, base_c)              # 输入通道数: in_channels -> base_c (64)
        self.down1 = Down(base_c, base_c * 2)                       # base_c (64) -> base_c * 2 (128)
        self.down2 = Down(base_c * 2, base_c * 4)                   # base_c * 2 (128) -> base_c * 4 (256)
        self.down3 = Down(base_c * 4, base_c * 8)                   # base_c * 4 (256) -> base_c * 8 (512)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)        # base_c * 8 (512) -> base_c * 16 // factor (512 or 1024)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)  # base_c * 16 (512 or 1024) -> base_c * 8 // factor (512 or 1024)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)   # base_c * 8 (512 or 1024) -> base_c * 4 // factor (256 or 512)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)   # base_c * 4 (256 or 512) -> base_c * 2 // factor (128 or 256)
        self.up4 = Up(base_c * 2, base_c, bilinear)                 # base_c * 2 (128 or 256) -> base_c (64)
        self.out_conv = OutConv(base_c, num_classes)                # base_c (64) -> num_classes (2)
    # 定义前向传播函数
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # U-Net 的前向传播过程
        # 编码器路径
        x1 = self.in_conv(x)       # 输入尺寸: (N, in_channels, H, W)，输出尺寸: (N, base_c, H, W)
        x2 = self.down1(x1)        # 输入尺寸: (N, base_c, H/2, W/2)，输出尺寸: (N, base_c*2, H/2, W/2)
        x3 = self.down2(x2)        # 输入尺寸: (N, base_c*2, H/4, W/4)，输出尺寸: (N, base_c*4, H/4, W/4)
        x4 = self.down3(x3)        # 输入尺寸: (N, base_c*4, H/8, W/8)，输出尺寸: (N, base_c*8, H/8, W/8)
        x5 = self.down4(x4)        # 输入尺寸: (N, base_c*8, H/16, W/16)，输出尺寸: (N, base_c*16//factor, H/16, W/16)
         # 解码器路径
        x = self.up1(x5, x4)       # 输入尺寸: (N, base_c*16//factor, H/8, W/8)，输出尺寸: (N, base_c*8//factor, H/8, W/8)
        x = self.up2(x, x3)        # 输入尺寸: (N, base_c*8//factor, H/4, W/4)，输出尺寸: (N, base_c*4//factor, H/4, W/4)
        x = self.up3(x, x2)        # 输入尺寸: (N, base_c*4//factor, H/2, W/2)，输出尺寸: (N, base_c*2//factor, H/2, W/2)
        x = self.up4(x, x1)        # 输入尺寸: (N, base_c*2//factor, H, W)，输出尺寸: (N, base_c, H, W)
        # 输出通道数变换
        logits = self.out_conv(x)  # 输入尺寸: (N, base_c, H, W)，输出尺寸: (N, num_classes, H, W)
        # 返回输出的字典，包含了最终的预测结果
        return {"out": logits}

if __name__ == "__main__":
    model = UNet(in_channels=3, num_classes=2)
    input_tensor = torch.randn(1, 3, 256, 256)  # 输入大小
    output = model(input_tensor)
    print(output["out"].shape)