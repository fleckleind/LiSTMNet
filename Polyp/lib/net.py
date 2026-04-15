import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from torchvision import models
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class DSConv3d(nn.Module):
    def __init__(self, in_chan, out_chan, stride=1):
        super(DSConv3d, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chan, in_chan, kernel_size=3, padding=1,
                      stride=stride, dilation=1, groups=in_chan),
            nn.GELU(), nn.Conv3d(in_chan, out_chan, kernel_size=1))
        
    def forward(self, x):
        return self.conv(x)
    

class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, drop=0.):
        super(MLP, self).__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or in_dim

        self.drop = nn.Dropout(drop)
        self.conv1 = DSConv3d(in_dim, hidden_dim)
        self.conv2 = DSConv3d(hidden_dim, out_dim)

    def forward(self, x, t, h, w):
        b, n, c = x.shape
        x = x.transpose(1, 2).view(b, c, t, h, w)
        x = self.drop(self.conv2(self.drop(self.conv1(x))))
        x = x.flatten(2).transpose(1, 2)
        return x
    

class STMamba(nn.Module):
    def __init__(
            self, dim, d_state=16, d_conv=4,
            expand=2, mlp_ratio=4, drop=0., drop_path=0.):
        super(STMamba, self).__init__()

        self.norm1, self.norm2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mamba = Mamba(d_model=dim, d_state=d_state, 
                           d_conv=d_conv, expand=expand, bimamba_type="v3")
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(dim, mlp_hidden_dim, drop)

    def forward(self, x, t):
        b, c, h, w = x.shape
        x = x.view(-1, t, c, h, w).transpose(1, 2)
        n_tokens = x.shape[2:].numel()

        x_flat = x.reshape(b//t, c, n_tokens).transpose(1, 2)
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))  # stmamba
        x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), t, h, w))  # fstmamba
        
        out = x_mamba.transpose(1, 2).reshape(-1, c, t, h, w)
        out = out.permute(0, 2, 1, 3, 4).reshape(-1, c, h, w)
        if not out.is_contiguous():
            out = out.contiguous()
        return out
    

class STAttn(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, kernel_size=7):
        super(STAttn, self).__init__()
        assert kernel_size in (3, 7), 'kernel size should be either 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Sequential(nn.Conv3d(
            in_ch, out_ch, kernel_size=kernel_size, 
            padding=padding, bias=True), nn.Sigmoid(),)

    def forward(self, xh, xs):
        avg_out_h = torch.mean(xh, dim=1, keepdim=True)
        avg_out_s = torch.mean(xs, dim=1, keepdim=True)
        nax_out_h, _ = torch.max(xh, dim=1, keepdim=True)
        nax_out_s, _ = torch.max(xs, dim=1, keepdim=True)
        out = torch.cat([avg_out_h, avg_out_s, nax_out_h, nax_out_s], dim=1)
        return self.conv(out)
    

class ResNet18_FSTMamba_BSTAFusion(nn.Module):
    def __init__(self, in_chan, n_classes, backbone='resnet18'):
        super(ResNet18_FSTMamba_BSTAFusion, self).__init__()

        resnet = models.resnet18(pretrained=True)  # 18/34/50
        channels = [64, 128, 256, 512]  # 18/34
        # channels = [256, 512, 1024, 2048]  # 50

        if in_chan == 3:
            conv1 = resnet.conv1
        else:
            conv1 = nn.Conv2d(in_chan, 64, kernel_size=7, stride=2, padding=3, bias=False)
            nn.init.kaiming_normal_(conv1.weight, mode="fan_out", nonlinearity="relu")

        self.encoder_layer0 = nn.Sequential(
            conv1, resnet.bn1, resnet.relu, resnet.maxpool,)
        
        self.encoder_layer1 = resnet.layer1
        self.encoder_layer2 = resnet.layer2
        self.encoder_layer3 = resnet.layer3
        self.encoder_layer4 = resnet.layer4

        self.encoder_mamba1 = STMamba(channels[0])
        self.encoder_mamba2 = STMamba(channels[1])
        self.encoder_mamba3 = STMamba(channels[2])
        self.encoder_mamba4 = STMamba(channels[3])

        self.STAttn = STAttn()  # BSTAFusion
        self.decode = DSConv3d(channels[0], channels[0]//16)
        self.out4 = nn.Conv3d(channels[3], channels[0], kernel_size=1)
        self.output = nn.Conv3d(channels[0]//16, n_classes, kernel_size=1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x0 = self.encoder_layer0(x)
        x1 = self.encoder_mamba1(self.encoder_layer1(x0), t)
        x2 = self.encoder_mamba2(self.encoder_layer2(x1), t)
        x3 = self.encoder_mamba3(self.encoder_layer3(x2), t)
        x4 = self.encoder_mamba4(self.encoder_layer4(x3), t)

        x1 = x1.view(b, t, -1, h//4, w//4).transpose(1, 2)
        x4 = x4.view(b, t, -1, h//32, w//32).transpose(1, 2)
        x4 = F.interpolate(x4, scale_factor=(1, 8, 8), mode='trilinear')
        out = self.decode(torch.add(x1, self.out4(x4)) * self.STAttn(x1, x4))
        out = F.interpolate(out, scale_factor=(1, 4, 4), mode='trilinear')
        out = self.output(out).transpose(1, 2)
        return out


if __name__ == "__main__":
    x = torch.randn(4, 10, 3, 256, 256).cuda()
    model = ResNet18_FSTMamba_BSTAFusion(in_chan=3, n_classes=1, backbone='resnet18').cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model(x).shape, pytorch_total_params)