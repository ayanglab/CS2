from __future__ import print_function
from utils import LoadConfig
import os
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torch.autograd import Variable
import network
from utils import weights_init
from network import define_D,GANLoss
from visdom import Visdom
from collections import OrderedDict
from utils import tensor2im
from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='COVID')
cfg = parser.parse_args()
opt = LoadConfig(cfg.config)

# specify the gpu id
os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.train.gpu_id)

# make saving directories
try:
    opt.outf = os.path.join(opt.save_dir,opt.task_name)
    os.makedirs(opt.outf)
except OSError:
    pass
for folder in ['models','imgs']:
    if not os.path.exists(os.path.join(opt.save_dir,folder)):
        os.mkdir(os.path.join(opt.save_dir,folder))


if opt.train.manual_seed is None:
    opt.train.manual_seed = random.randint(1, 10000)
print("Random Seed: ", opt.train.manual_seed)
random.seed(opt.train.manual_seed)
torch.manual_seed(opt.train.manual_seed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.train.manual_seed)


# datase t
from dataset.biDataset import biDatasetCTUnpair
from torch.utils.data import DataLoader
train_data = biDatasetCTUnpair(opt.dataset.path,opt.dataset.resize)
train_loader = DataLoader(train_data, batch_size=opt.train.batch_size, shuffle=True,num_workers=8)
dataset_size = train_data.__len__()


# Define the generator and initialize the weights
from AdaIN import Netv2
netG = Netv2(opt.generator.input_nc,opt.generator.output_nc,ngf=opt.ngf,
             n_downsampling=opt.generator.n_downsampling,n_blocks=opt.generator.n_blocks)
netG = torch.nn.DataParallel(netG)
netG = netG.cuda()
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))


# Define the discriminator and initialize the weights
netD = define_D(opt.discriminator.input_nc,opt.discriminator.ndf,opt.discriminator.n_layers)
netD = torch.nn.DataParallel(netD).cuda()
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))


# loss functions
dis_criterion = GANLoss()
mseloss = nn.MSELoss()
vgg_loss = network.VGGLoss(opt.train.gpu_id)

netD.cuda()
netG.cuda()
dis_criterion.cuda()


# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.train.lr, betas=opt.train.betas)
optimizerG = optim.Adam(netG.parameters(), lr=opt.train.lr, betas=opt.train.betas)

avg_loss_D = 0.0
avg_loss_G = 0.0


def trainEpoch(epoch):
    for i, data in enumerate(train_loader, 0):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        for p in netD.parameters():  # reset requires_grad
            p.requires_grad = True  # they are set to False below in netG update

        netD.zero_grad()
        bgseg = Variable(data['instance']).cuda()
        realimage = Variable(data['image']).cuda()

        dis_output = netD.forward(realimage.detach())

        dis_errD_real = dis_criterion(dis_output,True)

        errD_real = dis_errD_real
        errD_real.backward()
        D_x = dis_output[0][0].data.mean()


        # train with fake
        fakeimage = netG(bgseg,realimage)


        #fake_concat = torch.cat((bgseg.detach(), fakeimage.detach()), dim=1)
        dis_output = netD.forward(fakeimage.detach())

        dis_errD_fake = dis_criterion(dis_output,False)

        errD_fake = dis_errD_fake
        errD_fake.backward()
        optimizerD.step()
        errD = errD_real + errD_fake
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        for p in netD.parameters():
            p.requires_grad = False  # to avoid computation
        netG.zero_grad()
        dis_output = netD.forward(fakeimage)

        dis_errG = dis_criterion(dis_output,True)
        errG_vgg = vgg_loss(fakeimage.repeat(1,3,1,1),realimage.repeat(1,3,1,1))
        err_MSE = mseloss(fakeimage,bgseg)

        errG = dis_errG + errG_vgg + 0.001 * err_MSE
        errG.backward()

        optimizerG.step()

        # compute the average loss
        curr_iter = epoch * len(train_loader) + i
        global avg_loss_G
        global avg_loss_D

        all_loss_G = avg_loss_G * curr_iter
        all_loss_D = avg_loss_D * curr_iter

        all_loss_G += errG.data
        all_loss_D += errD.data
        avg_loss_G = all_loss_G / (curr_iter + 1)
        avg_loss_D = all_loss_D / (curr_iter + 1)


        print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f '
              % (epoch, opt.niter, i, len(train_loader),
                 errD.data, avg_loss_D, errG.data, avg_loss_G, D_x))

    # do checkpointing
    if epoch%opt.train.save_epochs == 0:
        torch.save(netG.state_dict(), '%s/models/netG_epoch_%d.pth' % (opt.save_dir, epoch))
        torch.save(netD.state_dict(), '%s/models/netD_epoch_%d.pth' % (opt.save_dir, epoch))

        visuals = OrderedDict([('input_label', tensor2im(bgseg[0, 0, :, :].detach().cpu(),
                                                         )),
                               ('synthesized_image', tensor2im(fakeimage.data[0, 0, :, :])),
                               ('real_image', tensor2im(data['image'][0, 0, :, :])),
                               ('filepath', data['path'][0])])
        for tag in ['synthesized_image', 'real_image', 'input_label']:
            image_pil = Image.fromarray(visuals[tag])
            image_pil.save('%s/imgs/Epoch%i_%s.png' % (opt.save_dir,epoch, tag))

    return float(errD.data), float(errG.data)

def Train():
    if opt.train.vis_on:
        vis = Visdom() #use_incoming_socket=False
        assert vis.check_connection()
        win_lossD = vis.line(np.arange(10))  # create the window
        win_lossG = vis.line(np.arange(10))

        x_index = []
        lossesD = []
        lossesG = []

        for epoch in range(opt.train.epochs):
            lossD, lossG = trainEpoch(epoch)

            x_index.append(epoch)
            lossesD.append(lossD)
            vis.line(X=np.array(x_index),Y=np.array(lossesD),
                     win=win_lossG,
                     opts=dict(title='LOSSD',
                               xlabel='epoch',
                               xtick=1,
                               ylabel='loss',
                               markersymbol='dot',
                               markersize=5,
                               legend=['train loss D']))
            lossesG.append(lossG)
            vis.line(X=np.array(x_index), Y=np.array(lossesG),
                     win=win_lossD,
                     opts=dict(title='LOSSG',
                               xlabel='epoch',
                               xtick=1,
                               ylabel='loss',
                               markersymbol='dot',
                               markersize=5,
                               legend=['train loss G']))

    else:
        for epoch in range(opt.train.epochs):
            lossD, lossG = trainEpoch(epoch)


if __name__ == '__main__':
    Train()