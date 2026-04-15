import timm
import math
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from einops import rearrange
from torchvision import models
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_
from transformers import SegformerForSemanticSegmentation
    

class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, nf, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, nf, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class Vivim_Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)  # dstmamba
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, nf, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, nf, H, W)  # dstmamba
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
    

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
        self.mlp = MLP(dim, mlp_hidden_dim, drop)  # fstmamba
        # self.mlp = Vivim_Mlp(dim, mlp_hidden_dim)  # d/mstmamba(vivim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, t):
        b, c, h, w = x.shape
        x = x.view(-1, t, c, h, w).transpose(1, 2)
        n_tokens = x.shape[2:].numel()

        x_flat = x.reshape(b//t, c, n_tokens).transpose(1, 2)
        x_mamba = x_flat + self.drop_path(self.mamba(self.norm1(x_flat)))  # stmamba
        x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), t, h, w))  # fstmamba
        # x_mamba = x_mamba + self.drop_path(self.mlp(self.norm2(x_mamba), t, h, w))  # dstmamba

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
    

class MultiSTAttn(nn.Module):
    def __init__(self, in_ch=6, out_ch=1, kernel_size=7):
        super(MultiSTAttn, self).__init__()
        assert kernel_size in (3, 7), 'kernel size should be either 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Sequential(nn.Conv3d(
            in_ch, out_ch, kernel_size=kernel_size, 
            padding=padding, bias=True), nn.Sigmoid(),)

    def forward(self, x1, x2, x3):
        avg_out_1 = torch.mean(x1, dim=1, keepdim=True)
        avg_out_2 = torch.mean(x2, dim=1, keepdim=True)
        avg_out_3 = torch.mean(x3, dim=1, keepdim=True)
        nax_out_1, _ = torch.max(x1, dim=1, keepdim=True)
        nax_out_2, _ = torch.max(x2, dim=1, keepdim=True)
        nax_out_3, _ = torch.max(x3, dim=1, keepdim=True)
        out = torch.cat([
            avg_out_1, avg_out_2, avg_out_3,
            nax_out_1, nax_out_2, nax_out_3,], dim=1)
        return self.conv(out)


class BiSeNet_FFM(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(BiSeNet_FFM, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv3d(
            in_ch, out_ch, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(out_ch, out_ch, kernel_size=1, bias=False), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, kernel_size=1, bias=False), nn.Sigmoid(),)

    def forward(self, xl, xh):
        x = self.conv1(torch.cat([xl, xh], dim=1))
        x = x + x * self.attn(x)
        return x


class ResNet18_FSTMamba_BSTAFusion(nn.Module):
    def __init__(self, in_chan, n_classes, backbone='resnet18'):
        super(ResNet18_FSTMamba_BSTAFusion, self).__init__()

        resnet = models.resnet18(pretrained=True)
        channels = [64, 128, 256, 512]

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

    #     segformer spatial feature extraction
    #     feat_size=[64, 128, 320, 512]
    #     backbone = SegformerForSemanticSegmentation.from_pretrained(
    #         "/root/MemSAM/models/segformer_b3", local_files_only=True)
    #     self.decoder = backbone.decode_head
    #     self.out = nn.Conv2d(768, n_classes, kernel_size=1)
    #     self.encoder_chan1 = nn.Conv2d(channels[0], feat_size[0], kernel_size=1)
    #     self.encoder_chan2 = nn.Conv2d(channels[1], feat_size[1], kernel_size=1)
    #     self.encoder_chan3 = nn.Conv2d(channels[2], feat_size[2], kernel_size=1)
    #     self.encoder_chan4 = nn.Conv2d(channels[3], feat_size[3], kernel_size=1)

        self.encoder_mamba1 = STMamba(channels[0])
        self.encoder_mamba2 = STMamba(channels[1])
        self.encoder_mamba3 = STMamba(channels[2])
        self.encoder_mamba4 = STMamba(channels[3])

        self.STAttn = STAttn()  # BSTAFusion
        # self.STAttn = MultiSTAttn()  # Multi-STAFusion
        self.decode = DSConv3d(channels[0], channels[0]//16)
        # self.decode = DSConv3d(sum(channels), channels[0]//16)  # Concat 1-4
        # self.decode = DSConv3d(channels[0]+channels[3], channels[0]//16)  # Concat 1,4
        # self.decode = BiSeNet_FFM(channels[0] + channels[3], channels[0]//16)
        # self.out2 = nn.Conv3d(channels[1], channels[0], kernel_size=1)  # Concat 1,2
        # self.out3 = nn.Conv3d(channels[2], channels[0], kernel_size=1)  # Concat 1,3
        self.out4 = nn.Conv3d(channels[3], channels[0], kernel_size=1)
        self.output = nn.Conv3d(channels[0]//16, n_classes, kernel_size=1)

    # def decode(self, encoder_hidden_states, bz, nf):
    #     # def forward(self, encoder_hidden_states: torch.FloatTensor) -> torch.Tensor:
    #     batch_size = encoder_hidden_states[-1].shape[0]

    #     all_hidden_states = ()
    #     for encoder_hidden_state, mlp in zip(encoder_hidden_states, self.decoder.linear_c):
    #         if self.decoder.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
    #             height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
    #             encoder_hidden_state = (
    #                 encoder_hidden_state.reshape(batch_size, height, width, -1).permute(0, 3, 1, 2).contiguous()
    #             )

    #         # unify channel dimension
    #         height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
    #         encoder_hidden_state = mlp(encoder_hidden_state)
    #         encoder_hidden_state = encoder_hidden_state.permute(0, 2, 1)
    #         encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
    #         # upsample
    #         encoder_hidden_state = nn.functional.interpolate(
    #             encoder_hidden_state, size=encoder_hidden_states[0].size()[2:], mode="bilinear", align_corners=False
    #         )
    #         all_hidden_states += (encoder_hidden_state,)
    #     concat_hidden_states = torch.cat(all_hidden_states[::-1], dim=1)
    #     hidden_states = self.decoder.linear_fuse(concat_hidden_states)
    #     hidden_states = self.decoder.batch_norm(hidden_states)
    #     hidden_states = self.decoder.activation(hidden_states)
    #     hidden_states = self.decoder.dropout(hidden_states)
        
    #     logits = self.out(hidden_states)
    #     return logits

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x0 = self.encoder_layer0(x)
        x1 = self.encoder_mamba1(self.encoder_layer1(x0), t)
        x2 = self.encoder_mamba2(self.encoder_layer2(x1), t)
        x3 = self.encoder_mamba3(self.encoder_layer3(x2), t)
        x4 = self.encoder_mamba4(self.encoder_layer4(x3), t)

        x1 = x1.view(b, t, -1, h//4, w//4).transpose(1, 2)
        # x2 = x2.view(b, t, -1, h//8, w//8).transpose(1, 2)
        # x2 = F.interpolate(x2, scale_factor=(1, 2, 2), mode='trilinear')        
        # x3 = x3.view(b, t, -1, h//16, w//16).transpose(1, 2)
        # x3 = F.interpolate(x3, scale_factor=(1, 4, 4), mode='trilinear')
        x4 = x4.view(b, t, -1, h//32, w//32).transpose(1, 2)
        x4 = F.interpolate(x4, scale_factor=(1, 8, 8), mode='trilinear')
        out = self.decode(torch.add(x1, self.out4(x4)) * self.STAttn(x1, x4))
        # out = self.decode(torch.add(x1, torch.add(self.out2(x2), self.out3(x3)))\
        #         * self.STAttn(x1, x2, x3))
        # out = self.decode(torch.cat([x1, x2, x3, x4], dim=1))
        # out = self.decode(torch.cat([x1, x4], dim=1))
        # out = self.decode(x1, x4)
        out = F.interpolate(out, scale_factor=(1, 4, 4), mode='trilinear')
        out = self.output(out).transpose(1, 2)

        # vivim-decoder
        # outs = [self.encoder_chan1(x1), self.encoder_chan2(x2),
        #         self.encoder_chan3(x3), self.encoder_chan4(x4)]
        # logits = self.decode(outs, b, t)
        # out = nn.functional.interpolate(
        #         logits, size=(h,w), mode="bilinear", align_corners=False)
        # out = rearrange(out, "(b t) c h w -> b t c h w", b=b)
        return out


if __name__ == "__main__":
    # x = torch.randn(4, 3, 256, 256)
    # mobilevit = timm.create_model(
    #     'mobilevitv2_125', pretrained=True, features_only=True, out_indices=(1, 2, 3, 4))
    # x1, x2, x3, x4 = mobilevit(x)
    # print(x1.shape, x2.shape, x3.shape, x4.shape)
    # model = models.mobilenet_v2(weights="DEFAULT")
    # print(model)

    x = torch.randn(4, 10, 3, 256, 256).cuda()
    model = ResNet18_FSTMamba_BSTAFusion(in_chan=3, n_classes=1, backbone='resnet18').cuda()
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(model(x).shape, pytorch_total_params)