import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights

# 实现简单的CNN编码器
class SmallCNN(nn.Module):
    def __init__(self, in_ch=6, feat_dim=256):
        super().__init__()
        self.feat_dim = feat_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),   nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),  nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, feat_dim)

    def forward(self, x):
        x = self.net(x).flatten(1)
        return self.fc(x)

from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch


class ModifiedResNet(nn.Module):
    def __init__(self, in_channels=6, feat_dim=256, pretrained=True):
        super().__init__()
        # 加载预训练的 resnet18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        self.feat_dim = feat_dim

        # 处理输入层
        if in_channels != 3:
            # 替换输入层以适配 in_channels
            old_conv1 = resnet.conv1
            resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

            # 初始化新输入层的权重
            if pretrained:
                if in_channels == 6:
                    # 将3通道复制两份（适用于两个RGB相机）
                    resnet.conv1.weight.data[:, :3] = old_conv1.weight.data
                    resnet.conv1.weight.data[:, 3:] = old_conv1.weight.data
                else:
                    # 对于非3或6通道，平均复制或截断已有权重
                    for i in range(in_channels):
                        resnet.conv1.weight.data[:, i] = old_conv1.weight.data[:, i % 3]
        else:
            # in_channels = 3 时直接使用原始 conv1
            pass

        # 去掉最后的分类头，只保留 avgpool 之前的部分
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, feat_dim)

    def forward(self, x):
        x = self.feature_extractor(x)  # (B, 512, H/32, W/32)
        x = self.pool(x).flatten(1)    # (B, 512)
        x = self.fc(x)                 # (B, output_dim)
        return x

class LateFusionResNet(nn.Module):
    def __init__(self, in_channels=3, feat_dim=256, pretrained=True, dropout_rate=0.2):
        super().__init__()
        # 加载两个独立的 ResNet18 编码器（共享或不共享参数都可以）
        weights = ResNet18_Weights.DEFAULT if pretrained else None

        def build_branch():
            resnet = resnet18(weights=weights)
            layers = list(resnet.children())[:-2]  # 去除分类头
            return nn.Sequential(*layers), nn.AdaptiveAvgPool2d((1, 1)), nn.Linear(512, feat_dim)

        self.encoder_left, self.pool_left, self.fc_left = build_branch()
        self.encoder_right, self.pool_right, self.fc_right = build_branch()

        self.dropout_left = nn.Dropout(p=dropout_rate)
        self.dropout_right = nn.Dropout(p=dropout_rate)

        self.feat_dim = feat_dim
        self.fusion = nn.Linear(feat_dim * 2, feat_dim)

    def forward(self, x):
        """
        支持：
        - 单相机输入 (B, 3, H, W)
        - 双相机拼接输入 (B, 6, H, W)
        """
        if x.shape[1] == 3:
            # 单相机输入：只用左分支
            feat = self.encoder_left(x)
            feat = self.pool_left(feat).flatten(1)
            feat = self.dropout_left(feat)
            return self.fc_left(feat)
        elif x.shape[1] == 6:
            # 双相机输入：拆成两个视角
            x_left = x[:, :3, :, :]
            x_right = x[:, 3:, :, :]

            feat_left = self.encoder_left(x_left)
            feat_right = self.encoder_right(x_right)

            feat_left = self.pool_left(feat_left).flatten(1)
            feat_right = self.pool_right(feat_right).flatten(1)

            feat_left = self.dropout_left(feat_left)
            feat_right = self.dropout_right(feat_right)

            left_proj = self.fc_left(feat_left)
            right_proj = self.fc_right(feat_right)

            # 融合两个特征
            fused = torch.cat([left_proj, right_proj], dim=1)  # (B, 512)
            return self.fusion(fused)  # (B, output_dim)
        else:
            raise ValueError(f"Expected 3 or 6 channels, got {x.shape[1]}.")


if __name__ == "__main__":
    encoder = ModifiedResNet()
    late_fusion_encoder = LateFusionResNet()
