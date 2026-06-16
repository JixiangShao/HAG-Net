import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import make_layers
from torch.hub import load_state_dict_from_url


class VGGNet(nn.Module):
    """VGG特征提取器：输出5个maxpool后的特征图（x1~x5）"""

    def __init__(self, model_name='vgg16', pretrained=True, requires_grad=True, show_params=False):
        super().__init__()
        self.model_name = model_name  # 保存模型名称

        self.cfg = {
            'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
            'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512,
                      'M'],
        }
        self.ranges = {
            'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
            'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
            'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
            'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
        }

        self.features = make_layers(self.cfg[model_name])
        if pretrained:
            model_urls = {
                'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
                'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
                'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
                'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
            }
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
            self.load_state_dict(state_dict, strict=False)

        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

        if show_params:
            for name, param in self.named_parameters():
                print(f"参数名：{name}, 尺寸：{param.size()}")

    def forward(self, x):
        output = {}
        for idx, (start, end) in enumerate(self.ranges[self.model_name]):
            for layer in range(start, end):
                x = self.features[layer](x)
            output[f"x{idx + 1}"] = x
        return output


class FCN32s(nn.Module):
    """FCN32s：仅用x5做32倍上采样"""

    def __init__(self, num_classes=2, model_name='vgg16', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = VGGNet(model_name=model_name, pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        backbone_out = self.backbone(x)
        x5 = backbone_out['x5']  # 1/32尺寸

        # 上采样并对齐尺寸
        score = self.relu(self.deconv1(x5))  # 1/16尺寸
        score = self.bn1(score)
        score = self.relu(self.deconv2(score))  # 1/8尺寸
        score = self.bn2(score)
        score = self.relu(self.deconv3(score))  # 1/4尺寸
        score = self.bn3(score)
        score = self.relu(self.deconv4(score))  # 1/2尺寸
        score = self.bn4(score)
        score = self.relu(self.deconv5(score))  # 原尺寸
        score = self.bn5(score)

        # 最终分类和尺寸对齐
        score = self.classifier(score)
        score = F.interpolate(score, size=(input_h, input_w), mode='bilinear', align_corners=True)
        score = self.activation(score)
        return {"out": score}


class FCN8s(nn.Module):
    """FCN8s：融合x3/x4/x5特征（修复尺寸不匹配问题）"""

    def __init__(self, num_classes=2, model_name='vgg16', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone = VGGNet(model_name=model_name, pretrained=pretrained)
        self.relu = nn.ReLU(inplace=True)
        # 上采样模块
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

        if num_classes == 1:
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Softmax(dim=1)

    def forward(self, x):
        input_h, input_w = x.shape[2], x.shape[3]
        backbone_out = self.backbone(x)

        # 提取backbone特征并保存尺寸
        x3 = backbone_out['x3']  # 1/8尺寸
        x3_size = x3.shape[2:]  # (H, W)
        x4 = backbone_out['x4']  # 1/16尺寸
        x4_size = x4.shape[2:]
        x5 = backbone_out['x5']  # 1/32尺寸
        x5_size = x5.shape[2:]

        # 1. x5上采样到1/16尺寸，与x4融合
        score = self.relu(self.deconv1(x5))  # 上采样后理论上是1/16尺寸
        # 关键修复1：强制对齐x4的尺寸
        score = F.interpolate(score, size=x4_size, mode='bilinear', align_corners=True)
        score = self.bn1(score + x4)  # 现在尺寸匹配，可以正常相加

        # 2. 上采样到1/8尺寸，与x3融合
        score = self.relu(self.deconv2(score))  # 理论上是1/8尺寸
        # 关键修复2：强制对齐x3的尺寸
        score = F.interpolate(score, size=x3_size, mode='bilinear', align_corners=True)
        score = self.bn2(score + x3)  # 尺寸匹配

        # 3. 继续上采样到原尺寸
        score = self.relu(self.deconv3(score))  # 1/4尺寸
        score = self.bn3(score)
        score = self.relu(self.deconv4(score))  # 1/2尺寸
        score = self.bn4(score)
        score = self.relu(self.deconv5(score))  # 原尺寸
        score = self.bn5(score)

        # 4. 最终分类和尺寸对齐
        score = self.classifier(score)
        score = F.interpolate(score, size=(input_h, input_w), mode='bilinear', align_corners=True)
        score = self.activation(score)
        return {"out": score}


def get_fcn_model(model_type='fcn8s', num_classes=2, backbone='vgg16', pretrained=True):
    if model_type == 'fcn32s':
        return FCN32s(num_classes=num_classes, model_name=backbone, pretrained=pretrained)
    elif model_type == 'fcn8s':
        return FCN8s(num_classes=num_classes, model_name=backbone, pretrained=pretrained)
    else:
        raise ValueError(f"不支持的模型类型：{model_type}")
