import torch 
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url

from ..ops.ibn import IBN
from ..registry import BACKBONE

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class SD_BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(SD_BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

    def forward(self, x):
        
        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(),torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)
                out = self.conv2(out)
                out = self.bn2(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # SD_ResNet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                
                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:         
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out


class SD_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, prob, multFlag, inplanes, planes, stride=1, downsample=None):
        super(SD_Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.prob = prob
        self.m = torch.distributions.bernoulli.Bernoulli(torch.Tensor([self.prob]))
        self.multFlag = multFlag

    def forward(self, x):

        identity = x.clone()

        if self.training:
            if torch.equal(self.m.sample(),torch.ones(1)):
                self.conv1.weight.requires_grad = True
                self.conv2.weight.requires_grad = True
                self.conv3.weight.requires_grad = True

                out = self.conv1(x)
                out = self.bn1(out)
                out = self.relu(out)

                out = self.conv2(out)
                out = self.bn2(out)
                out = self.relu(out)

                out = self.conv3(out)
                out = self.bn3(out)

                if self.downsample is not None:
                    identity = self.downsample(x)

                out += identity
            else:
                # SD_ResNet does not use bias terms
                self.conv1.weight.requires_grad = False
                self.conv2.weight.requires_grad = False
                self.conv3.weight.requires_grad = False

                if self.downsample is not None:
                    identity = self.downsample(x)

                out = identity
        else:
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.relu(out)

            out = self.conv3(out)
            out = self.bn3(out)

            if self.downsample is not None:
                identity = self.downsample(x)

            if self.multFlag:
                out = self.prob*out + identity
            else:
                out = out + identity

        out = self.relu(out)

        return out


class SD_ResNet(nn.Module):
    def __init__(self, block, prob_0_L, multFlag, layers, num_classes=1000, zero_init_residual=False, freeze_bn=False):
        super(SD_ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.multFlag = multFlag
        self.prob_now = prob_0_L[0]
        self.prob_delta = prob_0_L[0]-prob_0_L[1]
        self.prob_step = self.prob_delta/(sum(layers)-1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
                if not freeze_bn:
                    for i in m.parameters():
                        i.requires_grad = True

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, SD_Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, SD_BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes, stride, downsample))
        self.prob_now = self.prob_now - self.prob_step
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.prob_now, self.multFlag, self.inplanes, planes))
            self.prob_now = self.prob_now - self.prob_step

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        outs = []
        outs.append(x)          # 1/2
        x = self.maxpool(x)
        
        x = self.layer1(x)
        outs.append(x)          # 1/4

        x = self.layer2(x)
        outs.append(x)          # 1/8

        x = self.layer3(x)
        outs.append(x)          # 1/16

        x = self.layer4(x)
        outs.append(x)          # 1/32

        # x = self.avgpool(x)
        # x = x.reshape(x.size(0), -1)
        # x = self.fc(x)

        return tuple(outs)


def _sd_resnet(arch, block, layers, pretrained, progress, prob_0_L=[1,0.5], multFlag=True, freeze_bn=False, **kwargs):
    model = SD_ResNet(block, prob_0_L, multFlag, layers, 
                    freeze_bn=freeze_bn,
                    **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


@BACKBONE.register("R-SD-DL-101-C1-C5-FREEZEBN")
def build_resnet101_sd_aspp(pretrained=False, progress=True, **kwargs):
    return _sd_resnet('resnet101', SD_Bottleneck, [3, 4, 23, 3], pretrained, progress,
                    freeze_bn = True,
                    **kwargs)

@BACKBONE.register("R-SD-DL-101-C1-C5")
def build_resnet101_sd_aspp_bn(pretrained=False, progress=True, **kwargs):
    return _sd_resnet('resnet101', SD_Bottleneck, [3, 4, 23, 3], pretrained, progress,
                    **kwargs)



