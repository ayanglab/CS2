import os
import importlib
import numpy as np
import torch.nn as nn
from AdaIN import calc_mean_std
# custom weights initialization called on netG and netD
mseloss = nn.MSELoss()
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

# compute the current classification accuracy
def compute_acc(preds, labels):
    correct = 0
    preds_ = preds.data.max(1)[1]
    correct = preds_.eq(labels.data).cpu().sum()
    acc = float(correct) / float(len(labels.data)) * 100.0
    return acc

def LoadConfig(config_file):
    config_path = os.path.dirname(config_file)
    config_base = os.path.basename(config_file)
    config_name, _ = os.path.splitext(config_base)
    os.sys.path.insert(0, config_path)
    lib = importlib.import_module(config_name)
    os.sys.path.pop(0)

    return lib.cfg

def calc_content_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    return mseloss(input, target)

def calc_style_loss(input, target):
    assert (input.size() == target.size())
    assert (target.requires_grad is False)
    input_mean, input_std = calc_mean_std(input)
    target_mean, target_std = calc_mean_std(target)
    return mseloss(input_mean, target_mean) + \
           mseloss(input_std, target_std)

def tensor2im(image_tensor, imtype=np.uint8, normalize=True,label_img=False):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy
    image_numpy = image_tensor.cpu().float().numpy()
    if label_img:
        image_numpy = (image_numpy/(label_img-1)) * 255.0
    else:
        image_numpy = (image_numpy*0.5+0.5) * 255.0
    # if normalize:
    #     image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    # else:
    #     image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    # image_numpy = np.clip(image_numpy, 0, 255)
    # if image_numpy.shape[2] == 1 or image_numpy.shape[2] > 3:
    #     image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)