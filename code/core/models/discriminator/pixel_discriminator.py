import torch
import torch.nn as nn
import torch.nn.functional as F

from ..ops.snconv import SNConv2d
from ..ops.spp import PPM

from ..registry import DISCRIMINATOR

class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=512, num_classes=1):
        super(PixelDiscriminator, self).__init__()

        self.D = nn.Sequential(
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(ndf, ndf//2, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
		)
        self.cls1 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)
        self.cls2 = nn.Conv2d(ndf//2, num_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x, size=None):
        if isinstance(x, list) or isinstance(x, tuple):
            x = x[-1]

        out = self.D(x)
        src_out = self.cls1(out)
        tgt_out = self.cls2(out)
        out = torch.cat((src_out, tgt_out), dim=1)
        # print(out.shape)
        if size is not None:
            out = F.interpolate(out, size=size, mode='bilinear', align_corners=True)
        # print(out.shape)
        return out

@DISCRIMINATOR.register("Pixel-Predictor")
def pixel_discriminator(cfg, channels):
    return PixelDiscriminator(channels[-1], 256, num_classes=cfg.MODEL.PREDICTOR.NUM_CLASSES)