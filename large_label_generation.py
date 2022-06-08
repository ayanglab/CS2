import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import Counter
from few_label_generation import pixel_classifier,dataGenerate
import argparse
from utils import  LoadConfig
parser = argparse.ArgumentParser()
parser.add_argument('--config', default='COVID')
parser.add_argument('--G_model_epoch', default=100)
parser.add_argument('--num_class', default=3)
parser.add_argument('--num_ensemble', default=10)
cfg = parser.parse_args()
opt = LoadConfig(cfg.config)



# datase t
from dataset.biDataset import biDatasetCTUnpair
from torch.utils.data import DataLoader

test_data = biDatasetCTUnpair(res=opt.dataset.resize,filepath=opt.dataset.filepath)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False,num_workers=1)
model_fname = '%s/model/netG_epoch_%i.pth'%(opt.save_dir,cfg.G_model_epoch)
dataset_size = test_data.__len__()

class pixel_classifier_ensemble(nn.Module):
    def __init__(self, numpy_class, dim,path,ensemble_num=10):
        super(pixel_classifier_ensemble, self).__init__()
        self.classifier = pixel_classifier(numpy_class,dim)
        self.path = path
        self.enum = ensemble_num
        self.classifier = nn.DataParallel(self.classifier)
        self.classifier = self.classifier.cuda()
    def forward(self, x):
        results = torch.zeros((x.shape[0],self.enum))
        for cla_idx in range(self.enum):
            cla_path = '%s_%i.pth'%(self.path,cla_idx)
            self.classifier.load_state_dict(torch.load(cla_path))
            sf_max_prediction = self.classifier(x)
            _, prediction_vector = torch.max(sf_max_prediction, dim=1)
            results[:,cla_idx] = prediction_vector
        results = [Counter(results[i,:]).most_common(1)[0][0] for i in range(x.shape[0])]
        return torch.Tensor(results)

def tensor2image(tensor):
    tensor = (tensor - torch.min(tensor))/(torch.max(tensor) - torch.min(tensor))
    return np.array(tensor*255,dtype=np.uint8)

def loadfile(img_name):
    from PIL import Image
    img = Image.open(img_name)
    img = torch.Tensor([[np.array(img)]])
    img = (img - torch.min(img))/(torch.max(img) - torch.min(img))
    return img

def testMLP(dataGenerator):
    dataGenerator.classifier.eval()
    upsample = torch.nn.Upsample((1024,1024))
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            label = data['label']
            fakeimage = dataGenerator.netG.forward(Variable(data['instance']).cuda(),
                                                   Variable(data['image']).cuda())
            input = dataGenerator.extract_features_AdaIN(Variable(data['instance']).cuda(),Variable(data['image']).cuda())
            #input = input.reshape(dataGenerator.dim, dataGenerator.res_new * dataGenerator.res_new).transpose(1, 0)
            label = dataGenerator.resize(label)
            prediction = torch.zeros([1,1,512,512])
            acc = 0
            for x_index in range(label.shape[2]):
                prediction_vector = dataGenerator.classifier(input[0,:,x_index,:].transpose(1,0))
                prediction[0,0,x_index,:] = prediction_vector
                correct_pred = (prediction_vector.cpu() == label[0, 0, x_index, :]).float()
                acc += correct_pred.sum() / len(correct_pred)
                if correct_pred.sum() / len(correct_pred) <= 0.8:
                    print(correct_pred.sum() / len(correct_pred))

            # cv2.imwrite('/home/xiaodan/PycharmProjects/'
            #             'comparisonMethod/'
            #             'ComparisonCodes/datasets_public/Ours/%s'% data['path'][0],
                        #np.array(upsample(prediction)[0, 0, :, :].detach().cpu()))
            cv2.imwrite('./image/%s' % data['path'][0],
                        tensor2image(fakeimage[0, 0, :, :].detach().cpu()))
            cv2.imwrite('./label/%s' % data['path'][0],
                        np.array(upsample(prediction)[0, 0, :, :].detach().cpu()))






if __name__ == '__main__':
    print(model_fname)
    dg = dataGenerate(model_fname,cfg.num_class,cfg.num_ensemble,opt.save_dir)
    dg.trainMLP()
