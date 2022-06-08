import os
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image,ImageOps
import torch.nn as nn


class biDatasetCTUnpair(Dataset):

    def __init__(self,filepath='./dataset/CTMontage/',res=1024):  # crop_size,
        self.filepath = filepath
        self.imlist = os.listdir(self.filepath)
        self.train_transform_lung = transforms.Compose([

            transforms.Resize([res, res]),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        self.train_transform_seg = nn.Upsample((res, res))

    def __getitem__(self, idx):
        if idx < len(self.imlist):
            ## load reference image
            idx_new = np.random.randint(0,len(self.imlist))
            filename_ref = self.imlist[idx_new]
            img_name = '%s/%s'%(self.filepath,filename_ref)
            img = Image.open(img_name)
            img = ImageOps.grayscale(img)
            img = self.train_transform_lung(img)

            ## load segmentation label; this label will not be used during synthesis
            filename = self.imlist[idx]
            seg_name = os.path.join(self.filepath.replace('CTMontage','Seg'),filename)
            seg = Image.open(seg_name)
            seg = torch.Tensor([[np.array(seg)]])
            seg = self.train_transform_seg(seg).squeeze(1)

            ## load unsupervised masks
            useg_name = os.path.join(self.filepath.replace('CTMontage','USeg'),filename)
            useg = Image.open(useg_name)
            useg = torch.Tensor([[np.array(useg)]])
            useg[useg==torch.min(useg)]=0
            useg = useg[:,:,:,:,0]/255
            useg = self.train_transform_seg(useg).squeeze(1)

            return {'label':seg,'image':img,'path':filename,'path_ref':filename_ref,
                    'instance':useg}



    def __len__(self):
        return len(self.imlist)
