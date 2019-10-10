import torchvision as tv
import torch
import torch.nn as nn
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms

from utils import *
from Vgg_models import Vgg16


losses = {}
losses["l1"] = nn.L1Loss()
losses["l2"] = nn.MSELoss()
losses["gan"] = nn.BCELoss()
losses["l2_s"] = nn.MSELoss(reduction="sum")


def loss_l1(opt, data_real, data_gen):
    loss = losses["l1"](data_real, data_gen)
    if opt.use_gpu:
        loss.cuda()
    return loss


def loss_l2(opt, data_real, data_gen):
    loss = losses["l2"](data_real, data_gen)
    if opt.use_gpu:
        loss.cuda()
    return loss


def loss_perpectual(opt, data_real, data_gen, vgg):
    features_gen = vgg(data_gen)
    features_real = vgg(data_real)
    loss = 0
    for f_real, f_fake in zip(features_gen, features_real):
        loss += losses["l1"](f_real, f_fake)
    if opt.use_gpu:
        loss.cuda()
    return loss


def loss_style(opt, data_real, data_gen, vgg):
    features_gen = vgg(data_gen)
    features_real = vgg(data_real)
    loss = 0
    for f_real, f_gen in zip(features_real, features_gen):
        gram_real = gram_matrix(f_real)
        gram_gen = gram_matrix(f_gen)
        loss += losses["l2"](gram_gen, gram_real)
    if opt.use_gpu:
        loss.cuda()
    return loss


def loss_tv(opt, img_in):
    img_left = img_in[:, :, 1:, :]
    img_right = img_in[:, :, :-1, :]
    img_down = img_in[:, :, :, 1:]
    img_up = img_in[:, :, :, :-1]
    loss = losses["l2"](img_left, img_right) + losses["l2"](img_up, img_down)
    if opt.use_gpu:
        loss.cuda()
    return loss


def loss_edge(opt, data_real, data_gen, edgeGenerator):
    edge_real = edgeGenerator(data_real)
    edge_gen = edgeGenerator(data_gen)
    loss = losses["l1"](edge_real, edge_gen)
    if opt.use_gpu:
        loss.cuda()
    return loss


def loss_gan(opt, label, predict):
    if label == 1:
        labels = torch.ones_like(predict)
    else:
        labels = torch.zeros_like(predict)
    loss = losses["gan"](predict, labels)
    if opt.use_gpu:
        loss.cuda()
    return loss
