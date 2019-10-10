import torch
import torch.nn as nn
import cv2
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.models import vgg16
from collections import namedtuple
from utils import *


class Vgg16(torch.nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(vgg16(pretrained=True).features)[:24]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        name = ['pool1', 'pool2', 'pool3', 'pool4']
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {4, 9, 16, 23}:#{4, 9, 16, 23}:
                results.append(x)

        return results

        vgg_outputs = namedtuple("VggOutputs", name)
        return vgg_outputs(*results)


if __name__ == "__main__":
    # img = cv2.imread("./../example/places/data_irregular/20190316143319.png")
    # img = cv2.resize(img, (224, 224))
    # img0, img1, img2 = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    # img = np.array([img0, img1, img2])
    # img = np.expand_dims(img, 0)
    # img = img / 127.5 - 1
    # img = torch.from_numpy(img)
    losses = {}
    losses["l1"] = nn.L1Loss()
    losses["l2"] = nn.MSELoss()
    losses["gan"] = nn.BCELoss()

    img_dir = "./../data/train"
    data = dataSet(img_dir, batchsize=2)
    model = Vgg16().eval()
    vgg_16_pretrained = vgg16(pretrained=True)
    pretrained_dict = vgg_16_pretrained.state_dict()
    model_dict = model.state_dict()
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in model_dict.keys()}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    m = nn.Sigmoid()
    # vgg16.load_state_dict(torch.load("./../checkpoints/vgg16/vgg16-397923af.pth"))
    for i, (img, _) in enumerate(data):
        features_gen = model(img)
        features_real = model(img)
        loss = 0
        for f_real, f_gen in zip(features_real, features_gen):
            gram_real = gram_matrix(f_real)
            # gram_gen = utils.gram_matrix(f_gen)
            print(gram_real.shape)
            val = torch.ones_like(gram_real)
            loss += losses["gan"](m(gram_real), val)
            # loss += losses["l2"](gram_gen, gram_real)
        print(loss)
    print(result.data[0])
