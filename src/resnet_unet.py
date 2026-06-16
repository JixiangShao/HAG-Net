import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18,ResNet18_Weights
# from model.resnet import resnet18
from module.CBAM import CBAM

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),  # 卷积层：输入 (B, in_channels, H, W) -> 输出 (B, mid_channels, H, W)
            nn.BatchNorm2d(mid_channels),  # 批量归一化：输入 (B, mid_channels, H, W) -> 输出 (B, mid_channels, H, W)
            nn.ReLU(inplace=True),  # ReLU 激活函数：输入 (B, mid_channels, H, W) -> 输出 (B, mid_channels, H, W)
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),  # 卷积层：输入 (B, mid_channels, H, W) -> 输出 (B, out_channels, H, W)
            nn.BatchNorm2d(out_channels),  # 批量归一化：输入 (B, out_channels, H, W) -> 输出 (B, out_channels, H, W)
            nn.ReLU(inplace=True)  # ReLU 激活函数：输入 (B, out_channels, H, W) -> 输出 (B, out_channels, H, W)
        )
        self.cbam = CBAM(out_channels)

    def forward(self, x):
        return self.cbam(self.double_conv(x))  # 前向传播：输入 (B, in_channels, H, W) -> 输出 (B, out_channels, H, W)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)  # 双线性插值上采样：输入 (B, in_channels, H, W) -> 输出 (B, in_channels, 2*H, 2*W)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)  # 使用 DoubleConv 处理上采样后的特征图
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)  # 转置卷积上采样：输入 (B, in_channels, H, W) -> 输出 (B, in_channels // 2, 2*H, 2*W)
            self.conv = DoubleConv(in_channels, out_channels)  # 使用 DoubleConv 处理上采样后的特征图

    def forward(self, x1, x2):
        x1 = self.up(x1)  # 上采样：输入 (B, in_channels, H, W) -> 输出 (B, in_channels, 2*H, 2*W)
        diff_y = x2.size()[2] - x1.size()[2]  # 计算高度差
        diff_x = x2.size()[3] - x1.size()[3]  # 计算宽度差
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])  # 填充：调整 x1 尺寸以匹配 x2
        x = torch.cat([x2, x1], dim=1)  # 拼接：在通道维度上连接 x2 和 x1，输入 (B, x2_channels, H, W) + (B, x1_channels, H, W) -> 输出 (B, x2_channels + x1_channels, H, W)
        return self.conv(x)  # 经过 DoubleConv：输入 (B, x2_channels + x1_channels, H, W) -> 输出 (B, out_channels, H, W)

class OutConv(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)  # 1x1 卷积层：输入 (B, in_channels, H, W) -> 输出 (B, num_classes, H, W)

    def forward(self, x):
        return self.conv(x)  # 前向传播：输入 (B, in_channels, H, W) -> 输出 (B, num_classes, H, W)

class ResNetUNet(nn.Module):
    def __init__(self, num_classes, bilinear=True):
        super(ResNetUNet, self).__init__()
        self.bilinear = bilinear
        # resnet = resnet18(weights=None)
        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.base_layers = list(resnet.children()) # 将ResNet18的所有子模块转换成一个列表，并将其存储在self.base_layers中

        # 修改 ResNet 的第一层卷积以保持分辨率
        self.layer0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False),  # 卷积层：输入 (B, 3, H, W) -> 输出 (B, 64, H, W)
            nn.BatchNorm2d(64),  # 批量归一化：输入 (B, 64, H, W) -> 输出 (B, 64, H, W)
            nn.ReLU(inplace=True)  # ReLU 激活函数：输入 (B, 64, H, W) -> 输出 (B, 64, H, W)
        )
        self.layer1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 最大池化：输入 (B, 64, H, W) -> 输出 (B, 64, H, W)
            self.base_layers[4]  # 包含 64 个通道的 ResNet 层：输出 (B, 64, H, W)
        )
        self.layer2 = self.base_layers[5]  # 包含 128 个通道的 ResNet 层：输出 (B, 128, H/2, W/2)
        self.layer3 = self.base_layers[6]  # 包含 256 个通道的 ResNet 层：输出 (B, 256, H/4, W/4)
        self.layer4 = self.base_layers[7]  # 包含 512 个通道的 ResNet 层：输出 (B, 512, H/8, W/8)

        self.up1 = Up(512 + 256, 256, bilinear)  # 上采样块：输入 (B, 512 + 256, H/8, W/8) -> 输出 (B, 256, H/4, W/4)
        self.up2 = Up(256 + 128, 128, bilinear)  # 上采样块：输入 (B, 256 + 128, H/4, W/4) -> 输出 (B, 128, H/2, W/2)
        self.up3 = Up(128 + 64, 64, bilinear)  # 上采样块：输入 (B, 128 + 64, H/2, W/2) -> 输出 (B, 64, H, W)
        self.up4 = Up(64 + 64, 64, bilinear)  # 上采样块：输入 (B, 64 + 64, H, W) -> 输出 (B, 64, H, W)
        self.outc = OutConv(64, num_classes)  # 输出卷积层：输入 (B, 64, H, W) -> 输出 (B, num_classes, H, W)

    def forward(self, x):
        x1 = self.layer0(x)  # 输入： (B, 3, H, W) -> 输出： (B, 64, H, W)
        x2 = self.layer1(x1)  # 输入： (B, 64, H, W) -> 输出： (B, 64, H, W)
        x3 = self.layer2(x2)  # 输入： (B, 64, H, W) -> 输出： (B, 128, H/2, W/2)
        x4 = self.layer3(x3)  # 输入： (B, 128, H/2, W/2) -> 输出： (B, 256, H/4, W/4)
        x5 = self.layer4(x4)  # 输入： (B, 256, H/4, W/4) -> 输出： (B, 512, H/8, W/8)
       
        x = self.up1(x5, x4)  # 输入： (B, 512, H/8, W/8) 和 (B, 256, H/4, W/4) -> 输出： (B, 256, H/4, W/4)
        x = self.up2(x, x3)  # 输入： (B, 256, H/4, W/4) 和 (B, 128, H/2, W/2) -> 输出： (B, 128, H/2, W/2)
        x = self.up3(x, x2)  # 输入： (B, 128, H/2, W/2) 和 (B, 64, H, W) -> 输出： (B, 64, H, W)
        x = self.up4(x, x1)  # 输入： (B, 64, H, W) 和 (B, 64, H, W) -> 输出： (B, 64, H, W)
        logits = self.outc(x)  # 输入： (B, 64, H, W) -> 输出： (B, num_classes, H, W)
        return {"out": logits}

# # 示例用法
# if __name__ == "__main__":
#     model = ResNetUNet(num_classes=2)  # 创建模型实例
#     input_tensor = torch.randn(1, 3, 256, 256)  # 示例输入张量
#     output = model(input_tensor)  # 前向传播
#     print(output["out"].shape)  # 打印输出尺寸，应该是 torch.Size([1, 2, 256, 256])
