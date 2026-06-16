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
from module.MEEM import MEEM, DetailEnhancement
from module.Biformer import BiLevelRoutingAttention as BRA

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.meem = MEEM(in_dim=out_channels, hidden_dim=out_channels//2)

    def forward(self, x):
        x = self.double_conv(x)
        x = self.meem(x)  # 应用边缘增强
        return x


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
        self.msde = DetailEnhancement(img_dim=in_channels//2, feature_dim=in_channels//2)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        enhanced = self.msde(x1, x1, x2)  # 输入：上采样特征、主分支特征、跳跃连接特征
        x = torch.cat([x2, enhanced], dim=1)  # 用增强后的特征融合

        x = self.conv(x)
        return x


class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class MMUNet(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 bilinear: bool = True,
                 base_c: int = 64):
        super(MMUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear
        self.base_c = base_c
        factor = 2 if bilinear else 1

        # 原UNet结构
        self.in_conv = DoubleConv(in_channels, base_c)
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        self.down4 = Down(base_c * 8, base_c * 16 // factor)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.out_conv = OutConv(base_c, num_classes)

        # MEEM模块（保持不变）
        self.meem = MEEM(
            in_dim=base_c * 4,
            hidden_dim=base_c * 2
        )

        # MSDE模块（保持不变）
        self.msde = DetailEnhancement(
            img_dim=base_c * 4 // factor,
            feature_dim=base_c * 2
        )

        # 新增：辅助卷积层（修改处1：添加下采样确保尺寸匹配）
        self.feature_adjust = nn.Sequential(
            nn.Conv2d(
                in_channels=base_c * 16 // factor,
                out_channels=base_c * 2,
                kernel_size=1
            ),
            # x5尺寸为H/16，无需额外下采样
        )
        self.b_feature_adjust = nn.Sequential(
            nn.Conv2d(
                in_channels=base_c * 8,
                out_channels=base_c * 2,
                kernel_size=1
            ),
            # 关键修改：x4尺寸为H/8，下采样1次到H/16，与x5尺寸匹配
            nn.MaxPool2d(2, stride=2)  # 新增下采样层，确保尺寸一致
        )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 编码器路径
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x3 = self.meem(x3)  # 应用MEEM
        x4 = self.down3(x3)  # 尺寸：H/8, W/8
        x5 = self.down4(x4)  # 尺寸：H/16, W/16

        # 解码器路径
        x = self.up1(x5, x4)
        x = self.up2(x, x3)  # 此时x尺寸：H/4, W/4

        # 准备MSDE输入（修改处2：确保feature和b_feature尺寸一致）
        feature = self.feature_adjust(x5)  # 尺寸：H/16, W/16（来自x5）
        b_feature = self.b_feature_adjust(x4)  # 尺寸：H/16, W/16（x4经下采样后）

        # 应用MSDE（此时feature和b_feature尺寸匹配）
        x_enhanced = self.msde(img=x, feature=feature, b_feature=b_feature)
        x = x + x_enhanced

        # 后续解码过程
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)
        return {"out": logits}


if __name__ == "__main__":
    model = MMUNet(in_channels=3, num_classes=2)
    input_tensor = torch.randn(1, 3, 256, 256)
    output = model(input_tensor)
    print(output["out"].shape)  # 预期：torch.Size([1, 2, 256, 256])