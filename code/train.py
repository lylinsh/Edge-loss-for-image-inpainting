import torch
import torch.nn as nn
import torchvision as tv
import numpy as np
import os

from torchvision.models import vgg16
from utils import *
from model import *
from losses import *
from edgeModel import *
from Vgg_models import Vgg16


def model_init(m):
    '''模型初始化'''
    if isinstance(m, nn.Conv2d):
        nn.init.normal_(m.weight.data)
        nn.init.xavier_normal_(m.weight.data)
        nn.init.kaiming_normal_(m.weight.data)
        # nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight.data)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
    opt = Config()

    # 指定GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_num

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # 初始化权重保存路径
    gen1_weights_path = os.path.join(opt.model_pth_pre, "generator1.pth")
    gen2_weights_path = os.path.join(opt.model_pth_pre, "generator2.pth")
    dis_weights_path = os.path.join(opt.model_pth_pre, "discriminator.pth")

    # 初始化模型
    netG1 = Generator1()
    netG2 = Generator2()
    netD = Discriminator(opt)

    # 判断是否存在预训练模型，若存在，则先加载模型，否则进行初始化
    # 常用于断点训练
    if os.path.exists(gen1_weights_path):
        gen1_pretrain = torch.load(gen1_weights_path)
        e1 = gen1_pretrain["iteration"]
        gen2_pretrain = torch.load(gen2_weights_path)
        e2 = gen2_pretrain["iteration"]
        dis_pretrain = torch.load(dis_weights_path)
        netG1.load_state_dict(gen1_pretrain["generator1"])
        netG2.load_state_dict(gen2_pretrain["generator2"])
        netD.load_state_dict(dis_pretrain["discriminator"])
        netG1.train()
        netG2.train()
        netD.train()
        e = max(e1, e2)
        print(e)
        # e = 0
    else:
        e = 0
        b = 0
        netG1.apply(model_init)
        netG2.apply(model_init)
        netD.apply(model_init)

    data_dir = opt.data_dir

    # 初始化优化函数
    optimizer_G1 = torch.optim.Adam(netG1.parameters(), opt.lr, betas=(0.5, 0.999))
    optimizer_G2 = torch.optim.Adam(netG2.parameters(), opt.lr, betas=(0.5, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), opt.lr, betas=(0.5, 0.999))

    true_labels = 1
    false_labels = 0

    losses_weight = opt.losses_weight
    losses = {}

    # 加载VGG16，用于计算损失函数
    vgg = Vgg16().eval()
    vgg_16_pretrained = vgg16(pretrained=True)
    pretrained_dict = vgg_16_pretrained.state_dict()
    vgg_dict = vgg.state_dict()
    pretrained_dict = {k:v for k, v in pretrained_dict.items() if k in vgg_dict.keys()}
    vgg_dict.update(pretrained_dict)
    vgg.load_state_dict(vgg_dict)

    edgeGenerator = EdgeGenerator().eval()
    if opt.use_gpu:
        edgeGenerator_pretrained = torch.load("./../checkpoints/place2/EdgeModel_gen.pth")
    else:
        edgeGenerator_pretrained = torch.load("./../checkpoints/place2/EdgeModel_gen.pth",
                                                map_location='cpu')
    edgeGenerator_dict = edgeGenerator.state_dict()
    value_set = {}
    for k in edgeGenerator_dict.keys():
        value_set[k] = edgeGenerator_pretrained["generator"][k]
    edgeGenerator_dict.update(value_set)
    edgeGenerator.load_state_dict(edgeGenerator_dict)

    if not os.path.isdir(opt.result_pth):
        os.mkdir(opt.result_pth)
    if not os.path.isdir(opt.model_pth):
        os.mkdir(opt.model_pth)

    if opt.use_gpu:
        netG1.cuda()
        netG2.cuda()
        netD.cuda()
        # true_labels, false_labels = true_labels.cuda(), false_labels.cuda()
        vgg = vgg.cuda()
        edgeGenerator = edgeGenerator.cuda()

    # 开始训练
    for i in range(e, opt.epoches):
        # 调用数据加载函数，加载训练集
        data = dataSet(data_dir, batchsize=opt.batchsize, shuffle=True)
        
        # 根据训练集，对网络模型的权重进行迭代更新
        for ii, (img, _) in enumerate(data):
            if opt.use_gpu:
                img = img.cuda()
            img_raw, msk_in, img_in = torch.chunk(img, 3, 3)
            msk_in = (msk_in + 1) // 2
            msk_in = msk_in[:, 0:1, :, :]
            if i < opt.section1:
                data_in = torch.cat((img_in, 1 - msk_in), 1)
                optimizer_G1.zero_grad()

                # 前向
                img_gen1 = netG1(data_in)

                # 计算损失函数
                losses["l1"] = loss_l1(opt, img_raw, img_gen1)
                losses["l2"] = loss_l2(opt, img_raw, img_gen1)
                losses["l1_hole"] = loss_l1(opt, img_raw*(1-msk_in), img_gen1*(1-msk_in))
                losses["perpectual"] = loss_perpectual(opt, img_raw, img_gen1, vgg)
                losses["style"] = loss_style(opt, img_raw, img_gen1, vgg)
                losses["edge"] = loss_edge(opt, img_raw, img_gen1, edgeGenerator)
                losses["tv"] = loss_tv(opt, img_gen1)
                loss_G1 = losses_weight["l1"] * losses["l1"] + \
                         losses_weight["l2"] * losses["l2"] + \
                         losses_weight["perpectual"] * losses["perpectual"] + \
                         losses_weight["style"] * losses["style"] + \
                         losses_weight["tv"] * losses["tv"] + \
                         losses_weight["edge"] * losses["edge"]
                print("epochs:{:d} batches:{:d} gloss:{:.3f}".format(i, ii, loss_G1.data))
                
                # 反向传播
                loss_G1.backward()
                optimizer_G1.step()

            elif i < opt.section2:
                # 第二阶段的训练
                data_in1 = torch.cat((img_in, 1 - msk_in), 1)
                optimizer_D.zero_grad()
                img_gen1 = netG1(data_in1).detach()
                data_in2 = img_in * msk_in + img_gen1 * (1 - msk_in)
                img_gen2 = netG2(data_in2).detach()

                dis_real, dis_real_s = netD(img_raw)
                dis_gen, dis_gen_s = netD(img_gen2)

                losses["dloss_gen"] = loss_gan(opt, false_labels, dis_gen_s)
                losses["dloss_img"] = loss_gan(opt, true_labels, dis_real_s)
                loss_D = (losses["dloss_gen"] + losses["dloss_img"])/2
                loss_D.backward()
                optimizer_D.step()

                optimizer_G2.zero_grad()
                data_in2 = img_in * msk_in + img_gen1 * (1 - msk_in)
                img_gen2 = netG2(data_in2)
                dis_gen, dis_gen_s = netD(img_gen2)
                losses["l1"] = loss_l1(opt, img_raw, img_gen2)
                losses["l2"] = loss_l2(opt, img_raw, img_gen2)
                losses["l1_hole"] = loss_l1(opt, img_raw*(1-msk_in), img_gen2*(1-msk_in))
                losses["perpectual"] = loss_perpectual(opt, img_raw, img_gen2, vgg)
                losses["style"] = loss_style(opt, img_raw, img_gen2, vgg)
                losses["edge"] = loss_edge(opt, img_raw, img_gen2, edgeGenerator)
                losses["tv"] = loss_tv(opt, img_gen2)
                losses["gloss"] = loss_gan(opt, true_labels, dis_gen_s)
                loss_G2 = losses_weight["l1"] * losses["l1"] + \
                         losses_weight["l2"] * losses["l2"] + \
                         losses_weight["perpectual"] * losses["perpectual"] + \
                         losses_weight["style"] * losses["style"] + \
                         losses_weight["tv"] * losses["tv"] + \
                         losses_weight["edge"] * losses["edge"] + \
                         losses_weight["dcgan"] * losses["gloss"]
                print("epochs:{:d} batches:{:d} gloss:{:.3f}".format(i, ii, loss_G2.data))
                loss_G2.backward()
                optimizer_G2.step()
            else:
                data_in1 = torch.cat((img_in, 1 - msk_in), 1)
                optimizer_D.zero_grad()
                img_gen1 = netG1(data_in1).detach()
                data_in2 = img_in * msk_in + img_gen1 * (1 - msk_in)
                img_gen2 = netG2(data_in2).detach()

                dis_real, dis_real_s = netD(img_raw)
                dis_gen, dis_gen_s = netD(img_gen2)
                losses["dloss_gen"] = loss_gan(opt, false_labels, dis_gen_s)
                losses["dloss_img"] = loss_gan(opt, true_labels, dis_real_s)

                loss_D = (losses["dloss_gen"] + losses["dloss_img"])/2
                loss_D.backward()
                optimizer_D.step()

                optimizer_G1.zero_grad()
                img_gen1 = netG1(data_in1)
                dis_gen, dis_gen_s = netD(img_gen1)
                losses["l1"] = loss_l1(opt, img_raw, img_gen1)
                losses["l2"] = loss_l2(opt, img_raw, img_gen1)
                losses["l1_hole"] = loss_l1(opt, img_raw*(1-msk_in), img_gen1*(1-msk_in))
                losses["perpectual"] = loss_perpectual(opt, img_raw, img_gen1, vgg)
                losses["style"] = loss_style(opt, img_raw, img_gen1, vgg)
                losses["edge"] = loss_edge(opt, img_raw, img_gen1, edgeGenerator)
                losses["tv"] = loss_tv(opt, img_gen1)
                loss_G1 = losses_weight["l1"] * losses["l1"] + \
                         losses_weight["l2"] * losses["l2"] + \
                         losses_weight["perpectual"] * losses["perpectual"] + \
                         losses_weight["style"] * losses["style"] + \
                         losses_weight["tv"] * losses["tv"] + \
                         losses_weight["edge"] * losses["edge"]
                print("epochs:{:d} batches:{:d} gloss:{:.3f}".format(i, ii, loss_G1.data))
                loss_G1.backward()
                optimizer_G1.step()

                optimizer_G2.zero_grad()
                img_gen1 = netG1(data_in1).detach()
                data_in2 = img_in * msk_in + img_gen1 * (1 - msk_in)
                img_gen2 = netG2(data_in2)
                dis_gen, dis_gen_s = netD(img_gen2)
                losses["l1"] = loss_l1(opt, img_raw, img_gen2)
                losses["l2"] = loss_l2(opt, img_raw, img_gen2)
                losses["l1_hole"] = loss_l1(opt, img_raw*(1-msk_in), img_gen2*(1-msk_in))
                losses["perpectual"] = loss_perpectual(opt, img_raw, img_gen2, vgg)
                losses["style"] = loss_style(opt, img_raw, img_gen2, vgg)
                losses["edge"] = loss_edge(opt, img_raw, img_gen2, edgeGenerator)
                losses["tv"] = loss_tv(opt, img_gen2)
                losses["gloss"] = loss_gan(opt, true_labels, dis_gen_s)
                loss_G2 = losses_weight["l1"] * losses["l1"] + \
                         losses_weight["l2"] * losses["l2"] + \
                         losses_weight["perpectual"] * losses["perpectual"] + \
                         losses_weight["style"] * losses["style"] + \
                         losses_weight["tv"] * losses["tv"] + \
                         losses_weight["edge"] * losses["edge"] + \
                         losses_weight["dcgan"] * losses["gloss"]
                print("epochs:{:d} batches:{:d} gloss:{:.3f}".format(i, ii, loss_G2.data))
                loss_G2.backward()
                optimizer_G2.step()
            
            if (i+1) % 2 == 0 and (ii+1) % 25 == 0:
                tv.utils.save_image(img_in[0], '%s/%s_img.png' %(opt.result_pth, ii+1), normalize=True)
                tv.utils.save_image(img_gen1[0], '%s/%s_rst1.png' %(opt.result_pth, ii+1), normalize=True)
                if i >= opt.section1:
                    tv.utils.save_image(img_gen2[0], '%s/%s_rst2.png' %(opt.result_pth, ii+1), normalize=True)

            if i < opt.section1:
                print("l1:{:.5f} l2:{:.5f} p:{:.5f} s:{:.5f} e:{:.5f} tv:{:.5f}".format(
                        losses["l1"].data, losses["l2"].data, losses["perpectual"].data,
                        losses["style"].data, losses["edge"].data, losses["tv"].data))
            else:
                print("l1:{:.5f} l2:{:.5f} p:{:.5f} s:{:.5f} e:{:.5f} tv:{:.5f} dc_g:{:.5f}".format(
                        losses["l1"].data, losses["l2"].data, losses["perpectual"].data,
                        losses["style"].data, losses["edge"].data, losses["tv"].data, losses["gloss"].data))

        # 保存模型
        if (i+1)%10 == 0:
            gen1_weights_path = os.path.join(opt.model_pth, "generator1_"+str(i+1)+".pth")
            gen2_weights_path = os.path.join(opt.model_pth, "generator2_"+str(i+1)+".pth")
            dis_weights_path = os.path.join(opt.model_pth, "discriminator_"+str(i+1)+".pth")
        else:
            gen1_weights_path = os.path.join(opt.model_pth, "generator1.pth")
            gen2_weights_path = os.path.join(opt.model_pth, "generator2.pth")
            dis_weights_path = os.path.join(opt.model_pth, "discriminator.pth")
        if not os.path.isdir(opt.model_pth):
            os.mkdir(opt.model_pth)
            
        torch.save({
            'iteration': i+1,
            'generator1': netG1.state_dict()
        }, gen1_weights_path)
        torch.save({
            'iteration': i+1,
            'generator2': netG2.state_dict()
        }, gen2_weights_path)
        torch.save({
            'discriminator': netD.state_dict()
        }, dis_weights_path)

