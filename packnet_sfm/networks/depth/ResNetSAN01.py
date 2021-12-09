import torch
import torch.nn as nn

from packnet_sfm.networks.layers.minkowski_encoder import MinkowskiEncoder
from packnet_sfm.networks.layers.packnet.layers01 import Conv2D, ResidualBlock2, InvDepth, ResidualConv

class Encoder(nn.Module):
    def __init__(self, version, in_channels, ni, n1, n2, n3, n4, n5,
                 num_blocks, dropout):
        super().__init__()
        # Encoder
        self.version = version

        self.pre_calc = Conv2D(in_channels, ni, 5, 1)

        self.conv1 = Conv2D(ni, n1, 7, 2)
        self.conv2 = ResidualBlock2(n1, n2, num_blocks[0], 1, dropout=dropout)
        self.conv3 = ResidualBlock2(n2, n3, num_blocks[1], 1, dropout=dropout)
        self.conv4 = ResidualBlock2(n3, n4, num_blocks[2], 1, dropout=dropout)
        self.conv5 = ResidualBlock2(n4, n5, num_blocks[3], 1, dropout=dropout)

    def forward(self, rgb):

        x = self.pre_calc(rgb)

        # Encoder

        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)

        # Skips
        return x5, [x, x1, x2, x3, x4]


class Decoder(nn.Module):
    def __init__(self, version, out_channels, ni, n1, n2, n3, n4, n5,
                 iconv_kernel):
        super().__init__()
        # Decoder
        self.version = version

        # n1o, n1i = n1, n1 + ni + out_channels
        # n2o, n2i = n2, n2 + n1 + out_channels
        # n3o, n3i = n3, n3 + n2 + out_channels
        # n4o, n4i = n4, n4 + n3
        # n5o, n5i = n5, n5 + n4
        n1o, n1i = n1, n1 + ni + out_channels
        n2o, n2i = n1, n2 + n1 + out_channels
        n3o, n3i = n2, n3 + n2 + out_channels
        n4o, n4i = n3, n4 + n3
        n5o, n5i = n4, n5 + n4

        self.unpack5 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack1 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.iconv5 = Conv2D(n5i, n5o, iconv_kernel[0], 1)
        self.iconv4 = Conv2D(n4i, n4o, iconv_kernel[1], 1)
        self.iconv3 = Conv2D(n3i, n3o, iconv_kernel[2], 1)
        self.iconv2 = Conv2D(n2i, n2o, iconv_kernel[3], 1)
        self.iconv1 = Conv2D(n1i, n1o, iconv_kernel[4], 1)

        # Depth Layers

        self.unpack_disps = nn.PixelShuffle(2)
        self.unpack_disp4 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp3 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)
        self.unpack_disp2 = nn.Upsample(scale_factor=2, mode='nearest', align_corners=None)

        self.disp4_layer = InvDepth(n4o, out_channels=out_channels)
        self.disp3_layer = InvDepth(n3o, out_channels=out_channels)
        self.disp2_layer = InvDepth(n2o, out_channels=out_channels)
        self.disp1_layer = InvDepth(n1o, out_channels=out_channels)

    def forward(self, x5, skips):
        skip1, skip2, skip3, skip4, skip5 = skips

        unpack5 = self.unpack5(x5)
        if self.version == 'A':
            concat5 = torch.cat((unpack5, skip5), 1)
        else:
            concat5 = unpack5 + skip5
        iconv5 = self.iconv5(concat5)

        unpack4 = self.unpack4(iconv5)
        if self.version == 'A':
            concat4 = torch.cat((unpack4, skip4), 1)
        else:
            concat4 = unpack4 + skip4
        iconv4 = self.iconv4(concat4)
        inv_depth4 = self.disp4_layer(iconv4)
        up_inv_depth4 = self.unpack_disp4(inv_depth4)

        unpack3 = self.unpack3(iconv4)
        if self.version == 'A':
            concat3 = torch.cat((unpack3, skip3, up_inv_depth4), 1)
        else:
            concat3 = torch.cat((unpack3 + skip3, up_inv_depth4), 1)
        iconv3 = self.iconv3(concat3)
        inv_depth3 = self.disp3_layer(iconv3)
        up_inv_depth3 = self.unpack_disp3(inv_depth3)

        unpack2 = self.unpack2(iconv3)
        if self.version == 'A':
            concat2 = torch.cat((unpack2, skip2, up_inv_depth3), 1)
        else:
            concat2 = torch.cat((unpack2 + skip2, up_inv_depth3), 1)
        iconv2 = self.iconv2(concat2)
        inv_depth2 = self.disp2_layer(iconv2)
        up_inv_depth2 = self.unpack_disp2(inv_depth2)

        unpack1 = self.unpack1(iconv2)
        if self.version == 'A':
            concat1 = torch.cat((unpack1, skip1, up_inv_depth2), 1)
        else:
            concat1 = torch.cat((unpack1 + skip1, up_inv_depth2), 1)
        iconv1 = self.iconv1(concat1)
        inv_depth1 = self.disp1_layer(iconv1)

        if self.training:
            inv_depths = [inv_depth1, inv_depth2, inv_depth3, inv_depth4]
        else:
            inv_depths = [inv_depth1]

        return inv_depths


class ResNetSAN01(nn.Module):
    """
    PackNet-SAN network, from the paper (https://arxiv.org/abs/2103.16690)

    Parameters
    ----------
    dropout : float
        Dropout value to use
    version : str
        Has a XY format, where:
        X controls upsampling variations (not used at the moment).
        Y controls feature stacking (A for concatenation and B for addition)
    kwargs : dict
        Extra parameters
    """
    def __init__(self, dropout=None, version=None, **kwargs):
        super().__init__()
        self.version = version[1:]
        # Input/output channels
        in_channels = 3
        out_channels = 1
        # Hyper-parameters
        # ni, n1, n2, n3, n4, n5 = 32, 32, 64, 128, 256, 512
        ni, n1, n2, n3, n4, n5 = 32, 32, 64, 128, 128, 256
        num_blocks = [2, 2, 3, 3]
        iconv_kernel = [3, 3, 3, 3, 3]

        self.encoder = Encoder(self.version, in_channels, ni, n1, n2, n3, n4, n5,
                               num_blocks, dropout)
        self.decoder = Decoder(self.version, out_channels, ni, n1, n2, n3, n4, n5,
                               iconv_kernel)

        self.mconvs = MinkowskiEncoder([n1, n2, n3, n4, n5], with_uncertainty=False)

        self.weight = torch.nn.parameter.Parameter(torch.ones(5), requires_grad=True)
        self.bias = torch.nn.parameter.Parameter(torch.zeros(5), requires_grad=True)

        self.init_weights()

    def init_weights(self):
        """Initializes network weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv3d)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def run_network(self, rgb, input_depth=None):
        """
        Runs the network and returns inverse depth maps
        (4 scales if training and 1 if not).
        """
        x5p, skips = self.encoder(rgb)

        if input_depth is not None:
            self.mconvs.prep(input_depth)

            skips[1] = skips[1] * self.weight[0].view(1, 1, 1, 1) + self.mconvs(skips[1]) + self.bias[0].view(1, 1, 1, 1)
            skips[2] = skips[2] * self.weight[1].view(1, 1, 1, 1) + self.mconvs(skips[2]) + self.bias[1].view(1, 1, 1, 1)
            skips[3] = skips[3] * self.weight[2].view(1, 1, 1, 1) + self.mconvs(skips[3]) + self.bias[2].view(1, 1, 1, 1)
            skips[4] = skips[4] * self.weight[3].view(1, 1, 1, 1) + self.mconvs(skips[4]) + self.bias[3].view(1, 1, 1, 1)
            x5p      = x5p      * self.weight[4].view(1, 1, 1, 1) + self.mconvs(x5p)      + self.bias[4].view(1, 1, 1, 1)

        return self.decoder(x5p, skips), skips + [x5p]

    def forward(self, rgb, input_depth=None, **kwargs):

        if not self.training:
            inv_depths, _ = self.run_network(rgb, input_depth)
            return {
                'inv_depths': inv_depths,
            }

        output = {}

        inv_depths_rgb, skip_feat_rgb = self.run_network(rgb)
        output['inv_depths'] = inv_depths_rgb

        if input_depth is None:
            return {
                'inv_depths': inv_depths_rgb,
            }

        inv_depths_rgbd, skip_feat_rgbd = self.run_network(rgb, input_depth)
        output['inv_depths_rgbd'] = inv_depths_rgbd

        loss = sum([((srgbd.detach() - srgb) ** 2).mean()
                    for srgbd, srgb in zip(skip_feat_rgbd, skip_feat_rgb)]) / len(skip_feat_rgbd)
        output['depth_loss'] = loss

        return output