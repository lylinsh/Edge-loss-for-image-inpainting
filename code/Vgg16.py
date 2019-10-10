'''
the code can be found in https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
'''
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np


cfg = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']


class VGG16(nn.Module):
    def __init__(self, features, num_classes=1000, init_weights=True):
        super(VGG16, self).__init__()
        self.features = features
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, 0, 0.01)
                    nn.init.constant_(m.bias, 0)

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def vgg16(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG16(make_layers(cfg), **kwargs)
    if pretrained:
        vgg_16_pretrained = models.vgg16(pretrained=True)
        pretrained_dict = vgg_16_pretrained.state_dict()
        # print(pretrained_dict["features.2.bias"])
        model_dict = model.state_dict()
        pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict.keys()}
        # print(pretrained_dict["features.2.bias"])
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG16(make_layers(cfg, batch_norm=True), **kwargs)
    if pretrained:
        vgg_16_pretrained = models.vgg16(pretrained=True)
        model.load("./../checkpoints/vgg16/vgg16-397923af.pth")
    return model

if __name__ == "__main__":
    vgg = vgg16(True)
