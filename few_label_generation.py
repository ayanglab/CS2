import numpy as np
import torch
from AdaIN import Netv2, adaptive_instance_normalization
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import argparse
from utils import LoadConfig

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

train_data = biDatasetCTUnpair(res=opt.dataset.resize,filepath=opt.dataset.filepath)
train_loader = DataLoader(train_data, batch_size=1, shuffle=False,num_workers=1)
model_fname = '%s/model/netG_epoch_%i.pth'%(opt.save_dir,cfg.G_model_epoch)
dataset_size = train_data.__len__()


class trainData(Dataset):

    def __init__(self,X_data,y_data):

        self.X_data = X_data
        self.y_data = y_data

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

class pixel_classifier(nn.Module):
    def __init__(self, numpy_class, dim):
        super(pixel_classifier, self).__init__()
        if numpy_class < 32:
            self.layers = nn.Sequential(
                nn.Linear(dim, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, 32),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=32),
                nn.Linear(32, numpy_class),
                nn.Softmax()
            )
        else:
            self.layers = nn.Sequential(
                nn.Linear(dim, 256),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=256),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.BatchNorm1d(num_features=128),
                nn.Linear(128, numpy_class),
                nn.Softmax()
            )

    def init_weights(self, init_type='normal', gain=0.02):
        '''
        initialize network's weights
        init_type: normal | xavier | kaiming | orthogonal
        https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/9451e70673400885567d08a9e97ade2524c700d0/models/networks.py#L39
        '''

        def init_func(m):
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    nn.init.normal_(m.weight.data, 0.0, gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight.data, gain=gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight.data, gain=gain)

                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias.data, 0.0)

            elif classname.find('BatchNorm2d') != -1:
                nn.init.normal_(m.weight.data, 1.0, gain)
                nn.init.constant_(m.bias.data, 0.0)

        self.apply(init_func)

    def forward(self, x):
        return self.layers(x)

class dataGenerate:
    def __init__(self,netG,num_class,num_ensemble,prefix):
        self.num_class = num_class
        self.num_ensemble = num_ensemble
        self.prefix = prefix
        netG = Netv2(opt.generator.input_nc, opt.generator.output_nc, ngf=opt.ngf,
                     n_downsampling=opt.generator.n_downsampling, n_blocks=opt.generator.n_blocks)
        netG = torch.nn.DataParallel(netG)
        self.netG = self.netG.cuda()
        self.netG.load_state_dict(torch.load(netG))
        self.netG.train()
        # self.netG = torch.nn.DataParallel(self.netG)

        self.dim = 13186

        self.res_new = 512
        self.resize = torch.nn.Upsample((self.res_new,self.res_new))

        self.upsample = nn.Sequential(nn.Upsample((self.res_new, self.res_new)), nn.Tanh())


    def extract_features_AdaIN(self,edge,image):
        content_vec =nn.DataParallel(self.netG.module.modeldown)(edge)
        style_vec = nn.DataParallel(self.netG.module.modeldown)(image)
        style_vec_pool = nn.DataParallel(self.netG.module.max_pool)(style_vec)
        features = torch.cat([self.upsample(edge).detach().cpu(),self.upsample(style_vec).detach().cpu(),
                              self.upsample(content_vec).detach().cpu()],1)
        for i in range(self.netG.module.n_blocks):
            content_vec = nn.DataParallel(self.netG.module.modelup[i])(content_vec)
            features = torch.cat([features.detach().cpu(),
                                  self.upsample(content_vec).detach().cpu()],1)
        for i in range(self.netG.module.n_downsampling):
            t = adaptive_instance_normalization(content_vec, style_vec_pool)
            features = torch.cat([features.detach().cpu(),self.upsample(t).detach().cpu()],1)
            content_vec = nn.DataParallel(self.netG.module.modelup[9+i*3:12+i*3])(t)
        # t = adaptive_instance_normalization(content_feats, style_feats_pool)
        # t = alpha * t + (1 - alpha) * content_feats
        content_vec= nn.DataParallel(self.netG.module.modelup[-3:])(content_vec)
        features = torch.cat([features, self.upsample(content_vec).detach().cpu()],1)
        return features


    def trainMLP(self):
        mlpclassifier = MLPclassification(self.num_class,self.prefix,self.dim)

        for ensemble in range(self.num_ensemble):
            for epoch in range(50):
                baselineacc = 0.7
                for i, data in enumerate(train_loader):
                    label = data['label']
                    with torch.no_grad():
                        input = self.extract_features_AdaIN(Variable(data['instance']).cuda(),Variable(data['image']).cuda())
                    input = input.reshape(self.dim,self.res_new*self.res_new).transpose(1,0)
                    label = self.resize(label).reshape(1,self.res_new*self.res_new).transpose(1,0)

                    train_data_mlp = trainData(input,label)
                    train_loader_mlp = DataLoader(train_data_mlp, batch_size=65536, shuffle=True,
                                                  num_workers=2)
                    if i % 2 == 0:
                        train_loss,train_acc1,train_acc2 = mlpclassifier.trainMLPepoch(epoch,train_loader_mlp)
                    else:
                        test_loss, test_acc1, test_acc2 = mlpclassifier.valMLPepoch(epoch,
                                                                                    train_loader_mlp,
                                                                                    ensemble,baselineacc)
                        baselineacc = (test_acc1 + test_acc2)/2
                        print('Epoch [%i/50], Step [%i/10], train loss %0.2f, train acc class1 %0.2f,train acc class2 %0.2f,'
                              'test loss %0.2f, test acc class1 %0.2f,test acc class2 %0.2f'%(epoch,i,train_loss,train_acc1,train_acc2,test_loss,test_acc1,test_acc2))


class MLPclassification:
    def __init__(self,num_class,prefix,dim):
        self.prefix = prefix
        self.dim = dim
        self.classifier = pixel_classifier(num_class,self.dim)
        self.classifier.init_weights()
        self.loss = nn.CrossEntropyLoss(weight=torch.Tensor([0.1,0.5,1]).cuda())
        self.optimizer = optim.Adam(self.classifier.parameters(),
                                    lr=0.001, betas=(0.5, 0.999))
        self.classifier = nn.DataParallel(self.classifier)
        self.classifier = self.classifier.cuda()

    def trainMLPepoch(self,epoch,train_loader_mlp):
        avg_train_loss = 0
        avg_train_acc1 = 0
        avg_train_acc2 = 0
        self.classifier.train()
        for step_train, (x,y) in enumerate(train_loader_mlp):
            x = x.cuda()
            y = y.squeeze(1).long().cuda()
            prediction = self.classifier(x)
            loss = self.loss(prediction,y)
            self.classifier.zero_grad()
            loss.backward()
            self.optimizer.step()
            acc1,acc2 = self.multi_acc(prediction, y)
            avg_train_loss += float(loss.data)
            avg_train_acc1 += float(acc1.data)
            avg_train_acc2 += float(acc2.data)

        return avg_train_loss / (step_train + 1), avg_train_acc1 / (step_train + 1),avg_train_acc2 / (step_train + 1)

    def valMLPepoch(self, epoch, test_loader_mlp,ensemble,baselineacc):
        avg_test_loss = 0
        avg_test_acc1 = 0
        avg_test_acc2 = 0
        self.classifier.eval()
        with torch.no_grad():
            for step_test, (x, y) in enumerate(test_loader_mlp):
                x = x.cuda()
                y = y.squeeze(1).long().cuda()
                prediction = self.classifier(x)
                loss = self.loss(prediction, y)
                acc1,acc2 = self.multi_acc(prediction, y)

                avg_test_loss += float(loss.data)
                avg_test_acc1 += float(acc1.data)
                avg_test_acc2 += float(acc2.data)
        if (avg_test_acc1 + avg_test_acc1)/(2*(step_test+1)) >= baselineacc:
            torch.save(self.classifier.state_dict(),
                       '%s/features/mlp_%i.pth' % (self.prefix,ensemble))
            np.savetxt('%s/features/mlp_%i.txt' % (self.prefix,ensemble),[(avg_test_acc1 + avg_test_acc1)/(2*(step_test+1))])
        return avg_test_loss/(step_test+1),avg_test_acc1/(step_test+1),avg_test_acc2/(step_test+1)

    def multi_acc(self,y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

        # correct_pred = (y_pred_tags == y_test).float()
        # acc = correct_pred.sum() / len(correct_pred)
        # correct_pred = (y_pred_tags[y_test == 3] == 3).float()
        # acc_class3 = correct_pred.sum() / len(correct_pred)

        correct_pred  = (y_pred_tags[y_test == 2] == 2).float()
        acc_class2 = correct_pred.sum() / len(correct_pred)

        correct_pred  = (y_pred_tags[y_test == 1] == 1).float()
        acc_class1 = correct_pred.sum() / len(correct_pred)
        # acc = acc * 100

        return acc_class1 ,acc_class2





if __name__ == '__main__':
    print(model_fname)
    dg = dataGenerate(model_fname,cfg.num_class,cfg.num_ensemble,opt.save_dir)
    dg.trainMLP()
