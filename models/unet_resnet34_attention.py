import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        print(f"[DEBUG] AttentionBlock input g: {g.shape}, x: {x.shape}")
        print(f"[DEBUG] W_g Conv2d: kernel_size={self.W_g[0].kernel_size}, stride={self.W_g[0].stride}, in_channels={self.W_g[0].in_channels}, out_channels={self.W_g[0].out_channels}")
        print(f"[DEBUG] W_x Conv2d: kernel_size={self.W_x[0].kernel_size}, stride={self.W_x[0].stride}, in_channels={self.W_x[0].in_channels}, out_channels={self.W_x[0].out_channels}")
        print(f"[DEBUG] g min/max: {g.min().item()}/{g.max().item()}, x min/max: {x.min().item()}/{x.max().item()}")
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class UNetResNet34(nn.Module):
    def __init__(self, n_classes=1, deep_supervision=True, dropout_p=0.3, se_reduction=16):
        super().__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.input_layer = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.encoder1 = resnet.layer1  # 64
        self.encoder2 = resnet.layer2  # 128
        self.encoder3 = resnet.layer3  # 256
        self.encoder4 = resnet.layer4  # 512
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.att1 = AttentionBlock(F_g=256, F_l=256, F_int=128)  # d1, x4
        self.drop1 = nn.Dropout2d(p=dropout_p)
        self.se1 = SEBlock(256, reduction=se_reduction)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att2 = AttentionBlock(F_g=128, F_l=128, F_int=64)   # d2, x3
        self.drop2 = nn.Dropout2d(p=dropout_p)
        self.se2 = SEBlock(128, reduction=se_reduction)
        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att3 = AttentionBlock(F_g=64, F_l=64, F_int=32)     # d3, x2
        self.drop3 = nn.Dropout2d(p=dropout_p)
        self.se3 = SEBlock(64, reduction=se_reduction)
        self.up4 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.drop4 = nn.Dropout2d(p=dropout_p)
        self.se4 = SEBlock(32, reduction=se_reduction)
        self.conv_last = nn.Conv2d(32, n_classes, 1)
        # Deep supervision heads
        self.ds1 = nn.Conv2d(256, n_classes, 1)
        self.ds2 = nn.Conv2d(128, n_classes, 1)
        self.ds3 = nn.Conv2d(64, n_classes, 1)
        self.ds4 = nn.Conv2d(32, n_classes, 1)
        self.deep_supervision = deep_supervision

    def forward(self, x):
        x0 = self.input_layer(x)         # [B, 64, 128, 128]
        x1 = self.maxpool(x0)            # [B, 64, 64, 64]
        x2 = self.encoder1(x1)           # [B, 64, 64, 64]
        x3 = self.encoder2(x2)           # [B, 128, 32, 32]
        x4 = self.encoder3(x3)           # [B, 256, 16, 16]
        x5 = self.encoder4(x4)           # [B, 512, 8, 8]
        d1 = self.up1(x5)
        d1 = F.interpolate(d1, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        d1 = self.att1(d1, x4)
        d1 = self.drop1(d1)
        d1 = self.se1(d1)
        d2 = self.up2(d1)
        d2 = F.interpolate(d2, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        d2 = self.att2(d2, x3)
        d2 = self.drop2(d2)
        d2 = self.se2(d2)
        d3 = self.up3(d2)
        d3 = F.interpolate(d3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        d3 = self.att3(d3, x2)
        d3 = self.drop3(d3)
        d3 = self.se3(d3)
        d4 = self.up4(d3)
        d4 = F.interpolate(d4, size=x.shape[-2:], mode='bilinear', align_corners=False)
        d4 = self.drop4(d4)
        d4 = self.se4(d4)
        out = self.conv_last(d4)
        if self.deep_supervision and self.training:
            ds1_out = F.interpolate(self.ds1(d1), size=x.shape[-2:], mode='bilinear', align_corners=False)
            ds2_out = F.interpolate(self.ds2(d2), size=x.shape[-2:], mode='bilinear', align_corners=False)
            ds3_out = F.interpolate(self.ds3(d3), size=x.shape[-2:], mode='bilinear', align_corners=False)
            ds4_out = F.interpolate(self.ds4(d4), size=x.shape[-2:], mode='bilinear', align_corners=False)
            return [torch.sigmoid(out), torch.sigmoid(ds1_out), torch.sigmoid(ds2_out), torch.sigmoid(ds3_out), torch.sigmoid(ds4_out)]
        else:
            return torch.sigmoid(out)
