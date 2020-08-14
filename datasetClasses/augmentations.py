import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
import cv2
from albumentations import (
Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, RandomBrightnessContrast, IAAPiecewiseAffine,
IAASharpen, IAAEmboss, Flip, OneOf, Compose,Resize,ImageCompression,MultiplicativeNoise,ChannelDropout,IAASuperpixels,GaussianBlur,
HorizontalFlip,RandomGamma,VerticalFlip,ShiftScaleRotate,CLAHE
)

import numpy as np
import torch
from torchvision import transforms

augmentation_techniques_pool = {
                'RandomBrightnessContrast' : RandomBrightnessContrast(brightness_limit=0.05,contrast_limit=0.05,p=1) ,
                'Blur' : Blur(blur_limit=2,p=1),
                'OpticalDistortion' : OpticalDistortion(p=1),
                'ImageCompression': ImageCompression(p=1),
                'MultiplicativeNoise' : MultiplicativeNoise(p=1),
                'IAASharpen': IAASharpen(alpha=(0, 0.2) , p = 1),
                'IAAEmboss' : IAAEmboss(alpha=(0, 0.3) , p = 1),
                'MotionBlur': MotionBlur(blur_limit = 3,p=1),
                'MedianBlur' :MedianBlur(blur_limit=3,p=1),
                'HorizontalFlip': HorizontalFlip(p=1),
                'GaussNoise':GaussNoise(),
                'RandomGamma':RandomGamma(p=1),
                'VerticalFlip': VerticalFlip(p=1),
                'ShiftScaleRotate': ShiftScaleRotate(),
                'HueSaturationValue':HueSaturationValue(),
                'CLAHE':CLAHE(),
                
                }


def Cutout(img , cfg):
    """
    Mask random part of image with black pixels
    https://arxiv.org/pdf/1708.04552.pdf
    
    Arguments:
        img {[numpy array]} -- [input image]
        cfg {[configuration file]} -- [description]
            n_holes {[int]} -- [Number of patches to cut out of each image.]
            length {[int]} -- [The length (in pixels) of each square patch.]
            
    Returns:
        [numpy array] -- [transformed image]
    """    
    h = cfg.Dataset.img_height
    w = cfg.Dataset.img_width
    length = cfg["Augumentations"]["cutout"]["length"]
    mask = np.ones((h, w), np.float32)
    n_holes = cfg.dataset.augmentation.cutout.n_holes
    length = cfg.dataset.augmentation.cutout.length

    for n in range(n_holes):
      y = np.random.randint(h)
      x = np.random.randint(w)

      y1 = np.clip(y - length // 2, 0, h)
      y2 = np.clip(y + length // 2, 0, h)
      x1 = np.clip(x - length // 2, 0, w)
      x2 = np.clip(x + length // 2, 0, w)

      mask[y1: y2, x1: x2] = 0.

      mask = torch.from_numpy(mask)
      mask = mask.expand_as(img)
      img = img * mask
      return img

def normalize(image):
  """Normalize input image channel-wise to zero mean and unit variance."""
  image = image.transpose(2, 0, 1)  # Switch to channel-first
  mean, std = np.array(MEAN), np.array(STD)
  image = (image - mean[:, None, None]) / std[:, None, None]
  return image.transpose(1, 2, 0)
  
def apply_op(image, op):
    """
    Apply augmentation function specifically for augmix
    
    Arguments:
        image {[Numpy array]} -- [description]
        op {[list]} -- [list of all the augmentations to be applied sequentially]
    
    Returns:
        [type] -- [transformed image]
    """    
    transformed_image = op(image = image)['image']
    return transformed_image

def augment_and_mix(image,augs,cfg):
    
    """
    Augmix - https://arxiv.org/abs/1912.02781
    
    Arguments:
        image {[numpy array]} -- []
        augs {[list of function(augmentations)]} -- [List of all augmentations applied to dataset]
        cfg {[Config File]} -- []
            Augmix hyperparameters:
            width {[int]} -- [Number of parallel augmentation paths]
            depth {[int]} -- [Number of augmentations applied to each image in each path]
            alpha {[float (0-1)]} -- [Probability coefficient for Beta and Dirichlet distributions.]
        tranform {[torchvision.compose]} -- [Pytorch transform for normalization]                   
    Returns:
        [torch tensor] -- [Transformed and normalized image]
    """
        
    
    """TODO: See what to do with transform function used(We will probably pass it as a parameter in augmix and create it in init of dataloader)
    """

    width = cfg.dataset.augmentation.augmix.width
    depth = cfg.dataset.augmentation.augmix.depth
    alpha = cfg.dataset.augmentation.augmix.alpha
    ops = []
    ws = np.float32(np.random.dirichlet([alpha] * width))
    m = np.float32(np.random.beta(alpha, alpha))
    for i in range(width):
        op = []
        ag = augs.copy()
        for j in range(depth):
            a = np.random.choice(ag)
            ag.remove(a)
            op.append(a)
        ops.append(Compose(op))
    mix = torch.zeros((3,cfg.dataset.height,cfg.dataset.width))
    for i in range(width):
        image_aug = image.copy()
        op = ops[i]
        image_aug = normalize(apply_op(image_aug, op),cfg)
        mix += ws[i] * image_aug
    output_image = (1 - m) * normalize(image,cfg) + m * mix
    return output_image

    
