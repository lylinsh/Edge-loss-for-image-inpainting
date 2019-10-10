import torchvision as tv
import torch
import os
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


def rate_compute(img):
    w, h, d = img.shape
    n = 0
    for i in range(w):
        for j in range(h):
            if (img[i:i+1, j:j+1, :] == np.zeros(3)).all():
                n += 1
    rate = n / (w * h)
    return rate


def gram_matrix(y):
    '''计算格拉姆矩阵'''
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w*h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch*h*w)
    return gram


def dataSet(dir_path, size=256, batchsize=4, shuffle=False):
    '''加载数据，归一化到[-1, 1]区间'''
    normalize = transforms.Normalize(mean=IMAGENET_MEAN,
                                     std=IMAGENET_STD)
    data = datasets.ImageFolder(
            dir_path,
            transforms.Compose([
            transforms.Resize(size, interpolation=2),
            transforms.ToTensor(),
            normalize,
        ]))

    data_loader = torch.utils.data.DataLoader(
        data, batch_size=batchsize, shuffle=shuffle,
        num_workers=4, pin_memory=True, drop_last=True
    )
    return data_loader


def normalize_batch(batch):
    mean = batch.data.new(IMAGENET_MEAN).view(1, -1, 1, 1)
    std = batch.data.new(IMAGENET_STD).view(1, -1, 1, 1)
    mean = torch.autograd.Variable(mean.expand_as(batch.data))
    std = torch.autograd.Variable(std.expand_as(batch.data))
    return (batch/255.0-mean)/std


def path_gen(data_dir):
    filenames = []
    if os.path.isdir(data_dir):
        img_pth = os.walk(data_dir)
        for i,item in enumerate(img_pth):
            for pth in item[2]:
                pth = pth.split(".")[0]
                filenames.append(pth)
        return filenames
    else:
        return filenames


# 网络模型结构
class Config(object):
    image_size = 256
    batchsize = 4
    gpu_num = '1'
    data_dir = "./../data/train_large/"
    num_workers = 8
    use_gpu = True
    result_pth = "./../Result/code8_3/"
    lr = 0.001
    epoches = 140
    model_pth = "./../model/code8_1/"
    model_pth_pre = "./../model/code8_1"
    test_pth = "./../data/test/"
    losses_weight = {}
    losses_weight["l1"] = 15
    losses_weight["l1_hole"] = 15
    losses_weight["l2"] = 5
    losses_weight["perpectual"] = 5
    losses_weight["style"] = 250
    losses_weight["edge"] = 2
    losses_weight["dcgan"] = 0.1
    losses_weight["tv"] = 10
    section1 = 60
    section2 = 90
