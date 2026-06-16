import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.resnet import ResNet34_Weights
import torch.nn.functional as F

from functools import partial

# 统一ReLU激活函数（减少冗余）
nonlinearity = partial(F.relu, inplace=True)


# -------------------------- 1. 特征增强模块（DAC系列） --------------------------
class DACblock(nn.Module):
    """带空洞卷积的特征增强模块（CE-Net核心）"""
    def __init__(self, channel):
        super(DACblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=3, padding=3)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=5, padding=5)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        # 初始化卷积层偏置为0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # 残差融合
        return out


class DACblock_without_atrous(nn.Module):
    """无空洞卷积的DAC模块（降低计算量）"""
    def __init__(self, channel):
        super(DACblock_without_atrous, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.conv1x1(self.dilate2(x)))
        dilate3_out = nonlinearity(self.conv1x1(self.dilate2(self.dilate1(x))))
        dilate4_out = nonlinearity(self.conv1x1(self.dilate3(self.dilate2(self.dilate1(x)))))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DACblock_with_inception(nn.Module):
    """结合Inception结构的DAC模块（多尺度特征融合）"""
    def __init__(self, channel):
        super(DACblock_with_inception, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv1x1 = nn.Conv2d(2 * channel, channel, kernel_size=1, dilation=1, padding=0)
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate3(self.dilate1(x)))
        dilate_concat = nonlinearity(self.conv1x1(torch.cat([dilate1_out, dilate2_out], 1)))  # 通道拼接
        dilate3_out = nonlinearity(self.dilate1(dilate_concat))
        out = x + dilate3_out  # 残差融合
        return out


class DACblock_with_inception_blocks(nn.Module):
    """多尺度卷积+池化的Inception-DAC模块"""
    def __init__(self, channel):
        super(DACblock_with_inception_blocks, self).__init__()
        self.conv1x1 = nn.Conv2d(channel, channel, kernel_size=1, dilation=1, padding=0)
        self.conv3x3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.conv5x5 = nn.Conv2d(channel, channel, kernel_size=5, dilation=1, padding=2)
        self.pooling = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)  # 保持尺寸的池化
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)) and m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.conv1x1(x))
        dilate2_out = nonlinearity(self.conv3x3(self.conv1x1(x)))
        dilate3_out = nonlinearity(self.conv5x5(self.conv1x1(x)))
        dilate4_out = self.pooling(x)  # 池化保留全局信息
        out = dilate1_out + dilate2_out + dilate3_out + dilate4_out  # 多路径融合
        return out


# -------------------------- 2. 空间池化模块 --------------------------
class PSPModule(nn.Module):
    """金字塔场景池化（PSP）：融合多尺度空间信息"""
    def __init__(self, features, out_features=1024, sizes=(2, 3, 6, 14)):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(size, size)),
            nn.Conv2d(features, features, kernel_size=1, bias=False)
        )

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        # 多尺度池化后上采样到原尺寸，与原特征拼接
        priors = [F.interpolate(stage(feats), size=(h, w), mode='bilinear', align_corners=True)
                  for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))  # 通道压缩
        return self.relu(bottle)


class SPPblock(nn.Module):
    """空间金字塔池化（SPP）：CE-Net默认使用，增强全局上下文"""
    def __init__(self, in_channels):
        super(SPPblock, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=[2, 2], stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=[3, 3], stride=3)
        self.pool3 = nn.MaxPool2d(kernel_size=[5, 5], stride=5)
        self.pool4 = nn.MaxPool2d(kernel_size=[6, 6], stride=6)
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1, padding=0)  # 压缩为1通道

    def forward(self, x):
        in_channels, h, w = x.size(1), x.size(2), x.size(3)
        # 多尺度池化→卷积→上采样到原尺寸
        layer1 = F.interpolate(self.conv(self.pool1(x)), size=(h, w), mode='bilinear', align_corners=True)
        layer2 = F.interpolate(self.conv(self.pool2(x)), size=(h, w), mode='bilinear', align_corners=True)
        layer3 = F.interpolate(self.conv(self.pool3(x)), size=(h, w), mode='bilinear', align_corners=True)
        layer4 = F.interpolate(self.conv(self.pool4(x)), size=(h, w), mode='bilinear', align_corners=True)
        # 拼接：4个1通道池化特征 + 原512通道特征 → 516通道
        out = torch.cat([layer1, layer2, layer3, layer4, x], 1)
        return out


# -------------------------- 3. 解码器基础块 --------------------------
class DecoderBlock(nn.Module):
    """CE-Net解码器块：上采样+通道调整（适配编码器特征维度）"""
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)  # 通道压缩为1/4
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(
            in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1
        )  # 2倍上采样
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)  # 通道调整为目标维度
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


# -------------------------- 4. CE-Net系列模型（修改核心） --------------------------
class CE_Net_Base(nn.Module):
    """CE-Net基础类（提取公共逻辑，减少冗余）"""
    def __init__(self, num_classes=1, num_channels=3, pretrained=True, dac_module=None):
        super(CE_Net_Base, self).__init__()
        self.num_classes = num_classes
        self.filters = [64, 128, 256, 512]

        # 1. 初始化ResNet34 backbone（适配PyTorch新版本预训练权重）
        if pretrained:
            self.backbone = models.resnet34(weights=ResNet34_Weights.DEFAULT)
        else:
            self.backbone = models.resnet34(weights=None)
        # 提取ResNet34的编码器部分
        self.firstconv = self.backbone.conv1
        self.firstbn = self.backbone.bn1
        self.firstrelu = self.backbone.relu
        self.firstmaxpool = self.backbone.maxpool
        self.encoder1 = self.backbone.layer1  # 输出64通道
        self.encoder2 = self.backbone.layer2  # 输出128通道
        self.encoder3 = self.backbone.layer3  # 输出256通道
        self.encoder4 = self.backbone.layer4  # 输出512通道

        # 2. 特征增强模块（由子类传入具体DAC模块）
        self.dblock = dac_module(512) if dac_module else None

        # 3. 解码器（公共结构）
        self.decoder4 = DecoderBlock(512, self.filters[2])  # 512→256
        self.decoder3 = DecoderBlock(self.filters[2], self.filters[1])  # 256→128
        self.decoder2 = DecoderBlock(self.filters[1], self.filters[0])  # 128→64
        self.decoder1 = DecoderBlock(self.filters[0], self.filters[0])  # 64→64

        # 4. 最终输出层
        self.finaldeconv1 = nn.ConvTranspose2d(self.filters[0], 32, 4, 2, 1)  # 2倍上采样
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

        # 5. 自动选择激活函数（二分类/多分类兼容）
        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]  # 保存输入尺寸，用于最终对齐

        # -------------------------- 编码器：提取特征 --------------------------
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)  # 1/4尺寸, 64通道
        e2 = self.encoder2(e1)  # 1/8尺寸, 128通道
        e3 = self.encoder3(e2)  # 1/16尺寸, 256通道
        e4 = self.encoder4(e3)  # 1/32尺寸, 512通道

        # -------------------------- 特征增强：DAC模块 --------------------------
        if self.dblock is not None:
            e4 = self.dblock(e4)

        # -------------------------- 解码器：特征融合+上采样 --------------------------
        # 解码器4 → 与e3融合（强制尺寸对齐）
        d4 = self.decoder4(e4)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d4 += e3  # 残差融合

        # 解码器3 → 与e2融合
        d3 = self.decoder3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d3 += e2

        # 解码器2 → 与e1融合
        d2 = self.decoder2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 += e1

        # 解码器1 → 最终上采样
        d1 = self.decoder1(d2)

        # -------------------------- 最终输出：尺寸对齐+激活 --------------------------
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        # 强制输出尺寸与输入一致（解决上采样误差）
        out = F.interpolate(out, size=(input_h, input_w), mode='bilinear', align_corners=True)
        out = self.activation(out)

        # 适配训练框架：返回字典格式
        return {"out": out}


# -------------------------- CE-Net变体（基于基础类实现） --------------------------
class CE_Net_(CE_Net_Base):
    """原始CE-Net：DACblock + SPPblock"""
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__(num_classes, num_channels, pretrained, dac_module=DACblock)
        self.spp = SPPblock(512)  # 新增SPP模块
        # 重写解码器4的输入通道（SPP后通道为512+4=516）
        self.decoder4 = DecoderBlock(516, self.filters[2])

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]

        # 编码器
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        # 特征增强：DAC + SPP
        e4 = self.dblock(e4)
        e4 = self.spp(e4)  # SPP后通道变为516

        # 解码器（与基础类一致，但decoder4输入为516）
        d4 = self.decoder4(e4)
        d4 = F.interpolate(d4, size=e3.shape[2:], mode='bilinear', align_corners=True)
        d4 += e3

        d3 = self.decoder3(d4)
        d3 = F.interpolate(d3, size=e2.shape[2:], mode='bilinear', align_corners=True)
        d3 += e2

        d2 = self.decoder2(d3)
        d2 = F.interpolate(d2, size=e1.shape[2:], mode='bilinear', align_corners=True)
        d2 += e1

        d1 = self.decoder1(d2)

        # 最终输出
        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)
        out = F.interpolate(out, size=(input_h, input_w), mode='bilinear', align_corners=True)
        out = self.activation(out)

        return {"out": out}


class CE_Net_backbone_DAC_without_atrous(CE_Net_Base):
    """CE-Net变体：无空洞卷积的DAC模块（降低计算量）"""
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__(num_classes, num_channels, pretrained, dac_module=DACblock_without_atrous)


class CE_Net_backbone_DAC_with_inception(CE_Net_Base):
    """CE-Net变体：Inception结构的DAC模块"""
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__(num_classes, num_channels, pretrained, dac_module=DACblock_with_inception)


class CE_Net_backbone_inception_blocks(CE_Net_Base):
    """CE-Net变体：多尺度Inception-DAC模块"""
    def __init__(self, num_classes=1, num_channels=3, pretrained=True):
        super().__init__(num_classes, num_channels, pretrained, dac_module=DACblock_with_inception_blocks)


class CE_Net_OCT(CE_Net_Base):
    """CE-Net-OCT：适配OCT图像多分类（原代码无激活函数，已修复）"""
    def __init__(self, num_classes=12, num_channels=3, pretrained=True):
        super().__init__(num_classes, num_channels, pretrained, dac_module=DACblock)
        self.spp = SPPblock(512)
        self.decoder4 = DecoderBlock(516, self.filters[2])  # SPP后通道516


# -------------------------- 5. 经典UNet（修改核心） --------------------------
class double_conv(nn.Module):
    """双卷积块：Conv→BN→ReLU→Conv→BN→ReLU（UNet基础组件）"""
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class inconv(nn.Module):
    """UNet输入层：单双卷积块"""
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    """UNet下采样层：MaxPool→双卷积块"""
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.max_pool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        return self.max_pool_conv(x)


class up(nn.Module):
    """UNet上采样层：上采样→尺寸对齐→通道拼接→双卷积块（修复尺寸匹配逻辑）"""
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        # 双线性上采样（轻量）或转置卷积上采样（高精度）
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        # 上采样x1（解码器特征）
        x1 = self.up(x1)
        # 计算尺寸差异（高度和宽度）
        diff_h = x1.size(2) - x2.size(2)
        diff_w = x1.size(3) - x2.size(3)
        # 对称pad x2（编码器特征），确保尺寸匹配（修复奇数差异问题）
        x2 = F.pad(
            x2,
            (diff_w // 2, diff_w - diff_w // 2,  # 宽度方向pad
             diff_h // 2, diff_h - diff_h // 2)   # 高度方向pad
        )
        # 通道拼接（编码器特征在前，解码器特征在后）
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class outconv(nn.Module):
    """UNet输出层：1x1卷积调整通道数"""
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """经典UNet（适配训练框架，支持二分类/多分类）"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # 编码器
        self.inc = inconv(n_channels, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        # 最深层：若双线性上采样，通道数不变；否则减半（转置卷积）
        factor = 2 if bilinear else 1
        self.down4 = down(512, 1024 // factor)

        # 解码器
        self.up1 = up(1024, 512 // factor, bilinear)
        self.up2 = up(512, 256 // factor, bilinear)
        self.up3 = up(256, 128 // factor, bilinear)
        self.up4 = up(128, 64, bilinear)

        # 输出层
        self.outc = outconv(64, n_classes)

        # 自动选择激活函数
        if n_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]

        # 编码器下采样
        x1 = self.inc(x)       # 原尺寸, 64通道
        x2 = self.down1(x1)    # 1/2尺寸, 128通道
        x3 = self.down2(x2)    # 1/4尺寸, 256通道
        x4 = self.down3(x3)    # 1/8尺寸, 512通道
        x5 = self.down4(x4)    # 1/16尺寸, 512/1024通道

        # 解码器上采样+融合
        x = self.up1(x5, x4)   # 1/8尺寸, 256通道
        x = self.up2(x, x3)    # 1/4尺寸, 128通道
        x = self.up3(x, x2)    # 1/2尺寸, 64通道
        x = self.up4(x, x1)    # 原尺寸, 64通道

        # 输出调整
        out = self.outc(x)
        # 强制尺寸与输入一致（避免边缘误差）
        out = F.interpolate(out, size=(input_h, input_w), mode='bilinear', align_corners=True)
        out = self.activation(out)

        # 适配训练框架：字典格式输出
        return {"out": out}


# -------------------------- 6. 模型工厂函数（快速创建模型） --------------------------
def create_ce_net(model_type="base", num_classes=1, pretrained=True):
    """
    快速创建CE-Net系列模型
    Args:
        model_type: CE-Net变体，可选"base"(基础DAC)、"without_atrous"(无空洞DAC)、"with_inception"(Inception DAC)、"inception_blocks"(多尺度DAC)、"oct"(OCT专用)
        num_classes: 类别数（二分类=1，多分类>1）
        pretrained: 是否加载ResNet34预训练权重
    Returns:
        CE-Net模型实例
    """
    if model_type == "base":
        return CE_Net_(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "without_atrous":
        return CE_Net_backbone_DAC_without_atrous(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "with_inception":
        return CE_Net_backbone_DAC_with_inception(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "inception_blocks":
        return CE_Net_backbone_inception_blocks(num_classes=num_classes, pretrained=pretrained)
    elif model_type == "oct":
        return CE_Net_OCT(num_classes=num_classes, pretrained=pretrained)
    else:
        raise ValueError(f"不支持的CE-Net类型: {model_type}")


def create_unet(n_channels=3, num_classes=1, bilinear=True):
    """快速创建经典UNet模型"""
    return UNet(n_channels=n_channels, n_classes=num_classes, bilinear=bilinear)