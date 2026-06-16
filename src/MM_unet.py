import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import sys

sys.path.append(r"E:\unet")  # 根据实际项目路径调整

# 导入自定义模块
from module.MEEM import MEEM, DetailEnhancement
from module.FEM import FEM
from module.MEGA import MEGA


class DoubleConv(nn.Module):
    """(卷积 => BN => ReLU) * 2 + MEEM + FEM"""

    def __init__(self, in_channels, out_channels, mid_channels=None, use_fem=True):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        # 基础双卷积块（保持尺寸不变）
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # 边缘增强模块
        self.meem = MEEM(in_dim=out_channels, hidden_dim=out_channels // 2)
        # FEM特征增强模块（适配TGRS 2024版本）
        self.use_fem = use_fem
        if self.use_fem:
            self.fem = FEM(
                in_planes=out_channels,  # 输入通道数
                out_planes=out_channels,  # 输出通道数（与输入一致）
                stride=1,  # 不改变尺寸
                scale=0.1,  # 残差缩放因子
                map_reduce=8  # 通道缩减因子（需能整除输入通道）
            )

    def forward(self, x):
        x = self.double_conv(x)  # 基础特征提取
        x = self.meem(x)  # 边缘增强
        if self.use_fem:
            x = self.fem(x)  # FEM特征增强（细血管强化）
        return x


class Down(nn.Module):
    """下采样模块：MaxPool + DoubleConv"""

    def __init__(self, in_channels, out_channels, use_fem=True):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),  # 尺寸减半
            DoubleConv(in_channels, out_channels, use_fem=use_fem)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样模块：上采样 + 特征融合 + DoubleConv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # 双线性上采样或转置卷积上采样
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_fem=False)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_fem=False)

        # 多尺度细节增强模块
        self.msde = DetailEnhancement(
            img_dim=in_channels // 2 if bilinear else in_channels // 2,
            feature_dim=in_channels // 2
        )

    def forward(self, x1, x2):
        # 上采样并对齐尺寸
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        # 细节增强后融合
        enhanced = self.msde(x1, x1, x2)
        x = torch.cat([x2, enhanced], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出卷积：将特征映射到分割类别"""

    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class MMUNet(nn.Module):
    """完整的多模块UNet（适配眼底血管分割）"""

    def __init__(self,
                 in_channels: int = 1,  # DRIVE为单通道
                 num_classes: int = 2,  # 血管/背景二分类
                 bilinear: bool = True,
                 base_c: int = 64):  # 基础通道数
        super(MMUNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_c = base_c
        factor = 2 if bilinear else 1  # 双线性上采样时的通道调整因子

        # 编码器（前3层使用FEM增强细血管特征）
        self.inc = DoubleConv(in_channels, base_c, use_fem=True)
        self.down1 = Down(base_c, base_c * 2, use_fem=True)
        self.down2 = Down(base_c * 2, base_c * 4, use_fem=True)
        self.down3 = Down(base_c * 4, base_c * 8, use_fem=False)
        self.down4 = Down(base_c * 8, base_c * 16 // factor, use_fem=False)

        # 解码器
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)
        self.outc = OutConv(base_c, num_classes)

        # 额外特征增强模块
        self.meem_x3 = MEEM(in_dim=base_c * 4, hidden_dim=base_c * 2)

        self.msde_enhance = DetailEnhancement(
            img_dim=base_c * 2,
            feature_dim=base_c * 2
        )

        # 特征调整层（确保尺寸/通道兼容）
        self.feature_adjust = nn.Sequential(
            nn.Conv2d(base_c * 16 // factor, base_c * 2, kernel_size=1),
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU(inplace=True)
        )
        self.b_feature_adjust = nn.Sequential(
            nn.Conv2d(base_c * 8, base_c * 2, kernel_size=1),
            nn.BatchNorm2d(base_c * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )

        # 激活函数（二分类用Sigmoid）
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        # 记录输入尺寸（确保输出对齐）
        input_h, input_w = x.shape[2], x.shape[3]

        # 编码器特征提取
        x1 = self.inc(x)  # (B, 64, H, W)
        x2 = self.down1(x1)  # (B, 128, H/2, W/2)
        x3 = self.down2(x2)  # (B, 256, H/4, W/4)
        x3 = self.meem_x3(x3)  # 增强x3的边缘特征
        x4 = self.down3(x3)  # (B, 512, H/8, W/8)
        x5 = self.down4(x4)  # (B, 512/factor, H/16, W/16)

        # 解码器特征融合
        x = self.up1(x5, x4)  # (B, 256/factor, H/8, W/8)
        x = self.up2(x, x3)  # (B, 128/factor, H/4, W/4)

        # 深层特征增强（确保尺寸匹配）
        feature = self.feature_adjust(x5)
        feature = F.interpolate(feature, size=x.shape[2:], mode='bilinear', align_corners=True)
        b_feature = self.b_feature_adjust(x4)
        b_feature = F.interpolate(b_feature, size=x.shape[2:], mode='bilinear', align_corners=True)

        x_enhanced = self.msde_enhance(img=x, feature=feature, b_feature=b_feature)
        x = x + x_enhanced  # 残差融合

        # 完成解码并对齐输出尺寸
        x = self.up3(x, x2)  # (B, 64/factor, H/2, W/2)
        x = self.up4(x, x1)  # (B, 64, H, W)
        x = F.interpolate(x, size=(input_h, input_w), mode='bilinear', align_corners=True)

        # 输出分割结果
        logits = self.outc(x)
        out = self.activation(logits)
        return {"out": out}


# 测试代码
if __name__ == "__main__":
    # 检查设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = MMUNet(in_channels=1, num_classes=2).to(device)

    # 测试输入（DRIVE数据集典型尺寸：565x565单通道）
    test_input = torch.randn(1, 1, 565, 565).to(device)
    output = model(test_input)

    # 验证输出
    print(f"输入尺寸: {test_input.shape}")
    print(f"输出尺寸: {output['out'].shape}")

    # 检查尺寸匹配
    assert output['out'].shape[2:] == test_input.shape[2:], "输出尺寸与输入不匹配！"
    print("✅ 模型结构验证通过，尺寸匹配正常")

    # 检查参数设备一致性
    param_device = next(model.parameters()).device
    assert param_device == test_input.device, "模型参数与输入设备不一致！"
    print("✅ 设备一致性验证通过")
