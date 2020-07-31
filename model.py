import torch.nn as nn
import torch
from torch import autograd
import torch.nn.functional as F
from torchvision import models
from functools import partial

nonlinearity = partial(F.relu, inplace=True)

def dwt_init(x):
    '''
    haar wavelet decomposition
    '''
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    ll = x1 + x2 + x3 + x4
    hl = -(-x1 - x2 + x3 + x4)
    lh = -(-x1 + x2 - x3 + x4)
    hh = -(x1 - x2 - x3 + x4)

    #Normalization
    amin, amax = ll.min(), ll.max()
    ll = (ll - amin) / (amax - amin)
    amin, amax = lh.min(), lh.max()
    lh = (lh - amin) / (amax - amin)
    amin, amax = hl.min(), hl.max()
    hl = (hl - amin) / (amax - amin)
    amin, amax = hh.min(), hh.max()
    hh = (hh - amin) / (amax - amin)

    return torch.cat((ll, lh, hl, hh), 1)

def dwt_init_N(x):
    '''
    haar wavelet decomposition
    '''
    x01 = x[:, :, 0::2, :] / 2
    x02 = x[:, :, 1::2, :] / 2
    x1 = x01[:, :, :, 0::2]
    x2 = x02[:, :, :, 0::2]
    x3 = x01[:, :, :, 1::2]
    x4 = x02[:, :, :, 1::2]

    ll = x1 + x2 + x3 + x4
    hl = -(-x1 - x2 + x3 + x4)
    lh = -(-x1 + x2 - x3 + x4)
    hh = -(x1 - x2 - x3 + x4)

    #Normalization
    amin, amax = ll.min(), ll.max()
    ll = (ll - amin) / (amax - amin)
    amin, amax = lh.min(), lh.max()
    lh = (lh - amin) / (amax - amin)
    amin, amax = hl.min(), hl.max()
    hl = (hl - amin) / (amax - amin)
    amin, amax = hh.min(), hh.max()
    hh = (hh - amin) / (amax - amin)

    return torch.cat((lh, hl, hh), 1)

class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def forward(self, x):
        return dwt_init(x)

class WBRefineBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(WBRefineBlock, self).__init__()
        self.dwt = DWT()
        self.conv1_1 = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        x = self.dwt(x)
        x = self.conv1_1(x)
        return x


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
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


class Our_WRB(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(Our_WRB, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        #self.firstconv = resnet.conv1
        self.firstconv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        self.dwt1 = WBRefineBlock(64 * 4, 64)
        self.dwt2 = WBRefineBlock(128 * 4, 128)
        self.dwt3 = WBRefineBlock(256 * 4, 256)

        self.decoder4 = DecoderBlock(512+256, filters[2])
        self.decoder3 = DecoderBlock(filters[2]+128, filters[1])
        self.decoder2 = DecoderBlock(filters[1]+64, filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 4, 2, 1)
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, num_classes, 3, padding=1)

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        dwt1 = self.dwt1(e1)
        e2 = self.encoder2(e1)
        dwt2 = self.dwt2(e2)
        e3 = self.encoder3(e2)
        dwt3 = self.dwt3(e3)
        e4 = self.encoder4(e3)

        e4 = torch.cat([e4, dwt3], dim=1)
        d4 = torch.cat([self.decoder4(e4), dwt2], dim=1) 
        d3 = torch.cat([self.decoder3(d4), dwt1], dim=1)
        d2 = self.decoder2(d3)
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return F.sigmoid(out)

