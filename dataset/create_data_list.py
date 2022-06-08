import copy
import os
import numpy as np
import pandas as pd
import nibabel as nib
import cv2
from tqdm import tqdm,trange
import h5py
import matplotlib.pyplot as plt
import random
import multiprocessing
import json
from unsupervisedseg import run


def fancy_index_relu(m):
    if m<0:
        return 0
    else:
        return m
def check_position(meta):
    axial = [1, 0, 0, 0, 1, 0]
    indicator = 0
    for i in range(6):
        if meta['ImageOrientation(Patient)_%i'%(i+1)] != axial[i]:
            indicator+=1
    if indicator == 0:
        return 1
    else:
        return 0

def create_montage(np_array):
    if len(np_array.shape) == 3 and np_array.shape[-1] == 4:
        imgArrMonUp = np.vstack((np_array[:, :, 0], np_array[:, :, 1]))
        imgArrMonDown = np.vstack((np_array[:, :, 2], np_array[:, :, 3]))
        imgArrMon = np.hstack((imgArrMonUp, imgArrMonDown))
    else:
        imgArrMonUp = np.vstack((np_array, np_array))
        imgArrMonDown = np.vstack((np_array, np_array))
        imgArrMon = np.hstack((imgArrMonUp, imgArrMonDown))
    return imgArrMon

def save_one_montage(subID):
    filepath = '/media/NAS01/BraTS18/MICCAI_BraTS_2018_Data_Training/HGG/%s'%subID
    multi_model_list = os.listdir(filepath)

    seg = nib.load('%s/%s'%(filepath,multi_model_list[1]))
    segArr = seg.get_fdata()
    segMask = copy.deepcopy(segArr)
    segMask[np.where(segArr != 0)] = 1
    z_tumor = np.where(np.sum(np.sum(segMask, axis=0), axis=0)
                       ==np.max(np.sum(np.sum(segMask, axis=0), axis=0)) )[0]
    segMon = create_montage(segMask[:,:,z_tumor[0]])
    imgMon = np.zeros((segMask.shape[0],segMask.shape[1],4))
    idx_mode = 0
    for file in multi_model_list:
        if 'seg' not in file:
            img = nib.load('%s/%s'%(filepath,file))
            imgArr = img.get_fdata()
            imgArr = imgArr[:,:,z_tumor[0]]
            imgArr = imgArr.reshape(imgArr.shape[0],imgArr.shape[1])
            imgArr = (imgArr - np.min(imgArr)) \
                                  / (np.max(imgArr) -
                                     np.min(imgArr)) * 255

            imgMon[:,:,idx_mode] = imgArr
            idx_mode += 1
    imgMon = create_montage(imgMon)

    unsupervised_imgMon = \
        np.array(np.expand_dims(imgMon, 2).repeat(3, axis=2), dtype=np.uint8)
    unsupervised_imgMon = run(unsupervised_imgMon)

    cv2.imwrite('/media/NAS02/xiaodan/bratsSeg/img/%s.png' % subID, imgMon)
    cv2.imwrite('/media/NAS02/xiaodan/bratsSeg/seg/%s.png' % subID, segMon)
    cv2.imwrite('/media/NAS02/xiaodan/bratsSeg/unseg/%s.png' % subID, unsupervised_imgMon)

    print(subID)




def cambridgeCT(imgs):
    # seg = os.listdir('/media/NAS02/DataCam/CT/lung_segmentation')
    # img = os.listdir('/media/NAS02/DataCam/CT/Images')
    # seg_ggo = os.listdir('/media/NAS02/DataCam/CT/lung_ggo_consolid')
    # imgs = set(seg) & set(img)
    for file in imgs:
        save_one_montage(file)








if __name__ == '__main__':

    imgs = np.loadtxt('imgs.txt',dtype='str')
    cambridgeCT(imgs)
    # pool = multiprocessing.Pool(12)  # Create a multiprocessing Pool
    # pool.map(save_one_montage, imgs)
    # pool.close()
    # pool.join()



a = 1