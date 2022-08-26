import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
from torch.utils.model_zoo import load_url as load_state_dict_from_url
import torch
import torch.nn.functional as F
from ..registry import BACKBONE


model_urls = {
    'res2net50_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_4s-06e79181.pth',
    'res2net50_48w_2s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_48w_2s-afed724a.pth',
    'res2net50_14w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_14w_8s-6527dddc.pth',
    'res2net50_26w_6s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_6s-19041792.pth',
    'res2net50_26w_8s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_26w_8s-2c7c9f12.pth',
    'res2net101_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_26w_4s-02a759a1.pth',
    'res2net50_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth',
    'res2net101_v1b_26w_4s': 'https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net101_v1b_26w_4s-0812c246.pth',
}


class Bottle2neck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, baseWidth=26, scale = 4, stype='normal'):
        """ Constructor
        Args:
            inplanes: input channel dimensionality
            planes: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            downsample: None when stride = 1
            baseWidth: basic width of conv3x3
            scale: number of scale.
            type: 'normal': normal set. 'stage': first block of a new stage.
        """
        super(Bottle2neck, self).__init__()

        width = int(math.floor(planes * (baseWidth/64.0)))
        self.conv1 = nn.Conv2d(inplanes, width*scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width*scale)
        
        if scale == 1:
          self.nums = 1
        else:
          self.nums = scale -1
        if stype == 'stage':
            self.pool = nn.AvgPool2d(kernel_size=3, stride = stride, padding=1)
        convs = []
        bns = []
        for i in range(self.nums):
          convs.append(nn.Conv2d(width, width, kernel_size=3, stride = stride, padding=1, bias=False))
          bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(width*scale, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width  = width

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
          if i==0 or self.stype=='stage':
            sp = spx[i]
          else:
            sp = sp + spx[i]
          sp = self.convs[i](sp)
          sp = self.relu(self.bns[i](sp))
          if i==0:
            out = sp
          else:
            out = torch.cat((out, sp), 1)
        if self.scale != 1 and self.stype=='normal':
          out = torch.cat((out, spx[self.nums]),1)
        elif self.scale != 1 and self.stype=='stage':
          out = torch.cat((out, self.pool(spx[self.nums])),1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Res2Net(nn.Module):

    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2Net, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.out_channels = [64] + [x * block.expansion for x in [64, 128, 256, 512]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        outs = []
        outs.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outs.append(x)

        x = self.layer2(x)
        outs.append(x)

        x = self.layer3(x)
        outs.append(x)
        
        x = self.layer4(x)
        outs.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return tuple(outs)

class Res2NetV1b(nn.Module):
    
    def __init__(self, block, layers, baseWidth = 26, scale = 4, num_classes=1000):
        self.inplanes = 64
        super(Res2NetV1b, self).__init__()
        self.baseWidth = baseWidth
        self.scale = scale
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False)
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self.out_channels = [64] + [x * block.expansion for x in [64, 128, 256, 512]]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=stride, stride=stride, 
                    ceil_mode=True, count_include_pad=False),
                nn.Conv2d(self.inplanes, planes * block.expansion, 
                    kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample=downsample, 
                        stype='stage', baseWidth = self.baseWidth, scale=self.scale))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, baseWidth = self.baseWidth, scale=self.scale))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        outs = []
        outs.append(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        outs.append(x)

        x = self.layer2(x)
        outs.append(x)

        x = self.layer3(x)
        outs.append(x)

        x = self.layer4(x)
        outs.append(x)

        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return tuple(outs)





def _res2net(arch, block, layers, baseWidth, scale, pretrained, progress, **kwargs):
    if "v1b" not in arch:
        model = Res2Net(block, layers,
                        baseWidth, scale, 
                        **kwargs)
    else:
        model = Res2NetV1b(block, layers,
                        baseWidth, scale, 
                        **kwargs)

    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model

# -----------------------------------------------------------------------------
#  Res2Net 
# -----------------------------------------------------------------------------
@BACKBONE.register("R2-50")
def build_res2net50(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net50", Bottle2neck, [3, 4, 6, 3], 26, 4,
                    pretrained, progress, **kwargs)
    
@BACKBONE.register("R2-50-W26-S4")
def build_res2net50_26w_4s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net50_26w_4s", Bottle2neck, [3, 4, 6, 3], 26, 4, 
                    pretrained, progress, **kwargs)

@BACKBONE.register("R2-50-W26-S6")
def build_res2net50_26w_6s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net50_26w_6s", Bottle2neck, [3, 4, 6, 3], 26, 6, 
                    pretrained, progress, **kwargs)
    
@BACKBONE.register("R2-50-W26-S8")
def build_res2net50_26w_8s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net50_26w_8s", Bottle2neck, [3, 4, 6, 3], 26, 8, 
                    pretrained, progress, **kwargs)
    
@BACKBONE.register("R2-50-W48-S2")
def build_res2net50_48w_2s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_48w_2s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net50_48w_2s", Bottle2neck, [3, 4, 6, 3], 48, 2, 
                    pretrained, progress, **kwargs)
    
@BACKBONE.register("R2-50-W14-S8")
def build_res2net50_14w_8s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_14w_8s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net50_14w_8s", Bottle2neck, [3, 4, 6, 3], 14, 8, 
                    pretrained, progress, **kwargs)    

@BACKBONE.register("R2-101-W26-S4")
def build_res2net101_26w_4s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net101_26w_4s", Bottle2neck, [3, 4, 23, 3], 26, 4, 
                    pretrained, progress, **kwargs)


# -----------------------------------------------------------------------------
#  Res2NetV1b
# -----------------------------------------------------------------------------
@BACKBONE.register("R2-50-V1B")
def build_res2net50_v1b(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50 model.
    Res2Net-50 refers to the Res2Net-50_26w_4s.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net50_v1b_26w_4s", Bottle2neck, [3, 4, 6, 3], 26, 4,
                    pretrained, progress, **kwargs)
    
@BACKBONE.register("R2-50-V1B-W26-S4")
def build_res2net50_v1b_26w_4s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net50_v1b_26w_4s", Bottle2neck, [3, 4, 6, 3], 26, 4, 
                    pretrained, progress, **kwargs)

@BACKBONE.register("R2-101-V1B")
def build_res2net101_v1b(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net101_v1b_26w_4s", Bottle2neck, [3, 4, 23, 3], 26, 4, 
                    pretrained, progress, **kwargs)

@BACKBONE.register("R2-101-V1B-W26-S4")
def build_res2net101_v1b_26w_4s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net101_v1b_26w_4s", Bottle2neck, [3, 4, 23, 3], 26, 4, 
                    pretrained, progress, **kwargs)

@BACKBONE.register("R2-152-V1B-W26-S4")
def build_res2net152_v1b_26w_4s(pretrained=False, progress=True, **kwargs):
    """Constructs a Res2Net-50_26w_4s model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return _res2net("res2net152_v1b_26w_4s", Bottle2neck, [3, 8, 36, 3], 26, 4, 
                    pretrained, progress, **kwargs)
