import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import pandas as pd
import numpy as np
import os
from albumentations import (
Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
IAASharpen, IAAEmboss, Flip, OneOf, Compose,Resize,ImageCompression,MultiplicativeNoise,ChannelDropout,IAASuperpixels,GaussianBlur
)
from dataset.augmentations import *


config = {
    "Dataset" : "C:\\Users\\ACER\\Desktop\\Image-classification-pipeline\\dataset\\example.csv",
    "Augmentations" :  {
        "height": "h_val" , 
        "width": "w_val", 
        "alpha": "a_val",
        "cutout": {"val":"false",
                    'n_holes':12,
                    'length':10},
        "augmix": {"val":"true",
                    'width': 3,
                    'depth': 1,
                    'alph': 1.
                  },
        "techniques": {
                    "RandomBrightnessContrast":"true", 
                    "Blur":"true", 
                    "OpticalDistortion":"true", 
                    "ImageCompression":"false",
                    "MultiplicativeNoise":"true",
                    "IAASharpen":"false",
                    "IAAEmboss":"false",
                    "MotionBlur":"false",
                    "MedianBlur":"true",
                    }

}}

augs = {
'RandomBrightnessContrast' : RandomBrightnessContrast(brightness_limit=0.05,contrast_limit=0.05,p=1) ,
'Blur' : Blur(blur_limit=2,p=1),
'OpticalDistortion' : OpticalDistortion(p=1),
'ImageCompression': ImageCompression(p=1),
'MultiplicativeNoise' : MultiplicativeNoise(p=1),
'IAASharpen': IAASharpen(alpha=(0, 0.2) , p = 1),
'IAAEmboss' : IAAEmboss(alpha=(0, 0.3) , p = 1),
'MotionBlur': MotionBlur(blur_limit = 3,p=1),
'MedianBlur' :MedianBlur(blur_limit=3,p=1)
}
 
class classDataset(Dataset):
    
    def __init__(self, cfg, augmentation_list):  
       
        """
        Format of csv file:
        It contains 3 columns
        1. Path to image
        2. Label of image (String i.e Human Readable Form)
        3. Encoded label 
        """     
           
        self.csv_file = pd.read_csv(os.path.join(cfg.dataset.csvpath,mode+'.csv')).iloc[:,:].values
        self.cfg = cfg
        self.mode = mode
        augmentation_list = []
        
        for key in cfg.dataset.augmentation.techniques.keys():
            if(cfg.dataset.augmentation.techniques[key]):
                augmentation_list.append(augmentation_techniques_pool[key])
        
        if(cfg.dataset.augmentation.augmix.val):
            self.augmentation_list = augmentation_list
        else:
            self.augmentation_list = Compose(augmentation_list)
            
            
    def __len__(self): 
        return len(self.csv_file)

    def __getitem__(self,idx):  
        image = cv2.resize(cv2.imread(self.csv_file[idx,0]),(self.cfg.dataset.width,self.cfg.dataset.height))
        if(self.mode == 'train'):
            
            if(self.cfg.dataset.augmentation.augmix.val):
                image = augment_and_mix(image,self.augmentation_list,self.cfg)
            else:
                image = normalize(apply_op(image , self.augmentation_list),self.cfg)

            if(self.cfg.dataset.augmentation.cutout.val):
                apply_cutout = np.random.randint(0,2)
                if(apply_cutout == 1):
                    image = Cutout(image,self.cfg)
        else:
            image = normalize(image,self.cfg)
            
            
        sample = {'image':image,'label':self.csv_file[idx,2],'idx':idx}
        return sample
