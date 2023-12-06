from __future__ import print_function, division
from math import ceil, floor
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image, ImageOps
from random import random, randint, uniform
from scipy.ndimage import binary_closing, binary_erosion
# Ignore warnings
import warnings
from scipy.ndimage.measurements import center_of_mass
import pdb

from utils import getTargetSegmentation, min_max_normalize, predToSegmentation

warnings.filterwarnings("ignore")


class RandomZoomCrop:
    def __init__(self, size, scale=(1.1, 1.5), offcenter=(-10,10) ,interpolation=transforms.InterpolationMode.BILINEAR):
        super(RandomZoomCrop, self).__init__()
        self.size=size
        self.scale=scale
        self.offcenter = offcenter
        self.interpolation = interpolation
    def __call__(self, img, mask, center=None):
        # Resize both image and mask
        resc_factor = uniform(self.scale[0], self.scale[1])
        randsize = [ceil(self.size[0]*resc_factor), ceil(self.size[1]*resc_factor)]
        img_r = transforms.functional.resize(img,randsize, self.interpolation)
        mask_r = transforms.functional.resize(mask, randsize, self.interpolation)
        bin_mask = (np.array(mask_r)>0)
        if center==None:
            center = center_of_mass(bin_mask)
        # Calculate the crop parameters
        offcentering_x= uniform(self.offcenter[0], self.offcenter[1])
        offcentering_y= uniform(self.offcenter[0], self.offcenter[1])

        x = center[0] - ( self.size[0])//2 + offcentering_x
        y = center[1] - ( self.size[1])//2 + offcentering_y

        x = floor(min(max(0,x), randsize[0]-self.size[0]))
        y= floor(min(max(0,y), randsize[1]-self.size[1]))
        # Apply the same random crop to both image and mask
        img_c = transforms.functional.crop(img_r, x, y, self.size[0], self.size[1])
        mask_c = transforms.functional.crop(mask_r,  x, y, self.size[0], self.size[1])
        
        return img_c, mask_c

def make_dataset(root, mode):
    assert mode in ["train", "val", "test", "unlabeled"]
    items = []

    if mode == "train":
        train_img_path = os.path.join(root, "train", "Img")
        train_mask_path = os.path.join(root, "train", "GT")

        images = os.listdir(train_img_path)
        labels = os.listdir(train_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            if it_im[0] == "." or it_gt[0] == ".":
                continue
            item = (
                os.path.join(train_img_path, it_im),
                os.path.join(train_mask_path, it_gt),
            )
            items.append(item)

    elif mode == "val":
        val_img_path = os.path.join(root, "val", "Img")
        val_mask_path = os.path.join(root, "val", "GT")

        images = os.listdir(val_img_path)
        labels = os.listdir(val_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            if it_im[0] == "." or it_gt[0] == ".":
                continue
            item = (
                os.path.join(val_img_path, it_im),
                os.path.join(val_mask_path, it_gt),
            )
            items.append(item)
    elif mode == "unlabeled":
        ul_img_path = os.path.join(root, "train", "Img-Unlabeled")
        train_img_path = os.path.join(root, "train", "Img")

        ul_images = os.listdir(ul_img_path)
        ul_images.sort()
        for it_im in ul_images:
            if it_im[0] == ".":
                continue
            item = os.path.join(ul_img_path, it_im)
            items.append(item)

        train_images = os.listdir(train_img_path)
        train_images.sort()
        for it_im in train_images:
            if it_im[0] == ".":
                continue
            item = os.path.join(train_img_path, it_im)
            items.append(item)
    else:
        test_img_path = os.path.join(root, "test", "Img")
        test_mask_path = os.path.join(root, "test", "GT")

        images = os.listdir(test_img_path)
        labels = os.listdir(test_mask_path)

        images.sort()
        labels.sort()

        for it_im, it_gt in zip(images, labels):
            item = (
                os.path.join(test_img_path, it_im),
                os.path.join(test_mask_path, it_gt),
            )
            items.append(item)

    return items


class MedicalImageDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(
        self,
        mode,
        root_dir,
        transform=None,
        mask_transform=None,
        augment=False,
        equalize=False,
        max_translate = 50
    ):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.imgs = make_dataset(root_dir, mode)
        self.augmentation = augment
        self.equalize = equalize
        self.mode = mode
        self.max_translate = max_translate
        self.normalize = transforms.Normalize(0,1)
        if mode=="unlabeled":
            self.rzc = RandomZoomCrop([256,256], scale=(1.01,1.8), offcenter=(-15,15))
        else:
            self.rzc = RandomZoomCrop([256,256], scale=(1.01,1.8), offcenter=(-50,50))


    def __len__(self):
        return len(self.imgs)

    def augment(self, img, mask):
        if random() > 0.5:
            img = ImageOps.flip(img)
            mask = ImageOps.flip(mask)
        if random() > 0.5:
            img = ImageOps.mirror(img)
            mask = ImageOps.mirror(mask)
        if random() >0.8:
            left_right = (random()-0.5)*2*self.max_translate
            up_down = (random()-0.5)*2*self.max_translate
            translate = (1,0,left_right,0,1,up_down)
            img = img.transform(img.size, Image.AFFINE, translate)
            mask = mask.transform(mask.size, Image.AFFINE, translate)
        if random() > 0.5:
            angle = random() * 60 - 15
            img = img.rotate(angle)
            mask = mask.rotate(angle)
        if random()>0.3:
            if self.mode=="unlabeled":
                img,mask = self.rzc(img,mask, center=(128,128))
            else:
                img,mask = self.rzc(img,mask)
        return img, mask

    def __getitem__(self, index):
        if self.mode == "unlabeled":
            img_path = self.imgs[index]
            img = Image.open(img_path)
            

            if self.augmentation:
                img, mask = self.augment(img, img)

            if self.transform:
                img = self.transform(img)

            if self.equalize:
                img = min_max_normalize(img)
                # img= self.normalize(img)

            return [img, img_path]
        else:
            img_path, mask_path = self.imgs[index]
            img = Image.open(img_path)
            mask = Image.open(mask_path).convert("L")

            if self.augmentation:
                img, mask = self.augment(img, mask)

            if self.transform:
                img = self.transform(img)
                mask = self.mask_transform(mask)

            if self.equalize:
                # img = min_max_normalize(img)
                img= self.normalize(img)

           
            return [img, mask, img_path]
            
