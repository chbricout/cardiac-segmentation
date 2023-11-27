import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import os
import skimage.transform as skiTransf
from progressBar import printProgressBar
import scipy.io as sio
import pdb
import time
from os.path import isfile, join
import nibabel as nib
import statistics

from PIL import Image
from medpy.metric.binary import dc,hd,asd,assd
import SimpleITK as sitk
import scipy.spatial
#from scipy.spatial.distance import directed_hausdorff


labels = { 0: 'Background',1:'Foreground'}


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    
    denom = 0.33333334 # for ACDC this value
    return (batch / denom).round().long().squeeze()

from scipy import ndimage

def inferenceTest(net, img_batch, modelName):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()

    DSC_All_class1 = []
    DSC_All_class2 = []
    DSC_All_class3 = []

    HD_All_class1 = []
    HD_All_class2 = []
    HD_All_class3 = []

    ASD_All_class1 = []
    ASD_All_class2 = []
    ASD_All_class3 = []
    for i, data in enumerate(img_batch):

        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        segmentation_classes = getTargetSegmentation(labels)

        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y,dim=1)

        masks=masks.view((256,256))

        DSC_image = []
        HD_image = []
        ASD_image = []
        for c_i in range(3):
            mask_pred = np.zeros((segmentation_classes.shape))
            mask_gt = np.zeros((segmentation_classes.shape))

            idx=np.where(masks.cpu()==c_i+1)
            mask_pred[idx]=1

            idx = np.where(segmentation_classes.cpu() == c_i + 1)
            mask_gt[idx] = 1

            DSC_image.append(dc(mask_pred,mask_gt))

            if mask_gt.sum() == 0 and mask_pred.sum() == 0:
                HD_image.append(0.0)
                ASD_image.append(0.0)
            elif mask_gt.sum() > 0 and mask_pred.sum() == 0:
                HD_image.append(50)
                ASD_image.append(20)
            elif mask_gt.sum() == 0 and mask_pred.sum() > 0:
                HD_image.append(50)
                ASD_image.append(20)
            else:
                HD_image.append(hd(mask_pred, mask_gt))
                ASD_image.append(asd(mask_pred, mask_gt))

        DSC_All_class1.append(DSC_image[0])
        DSC_All_class2.append(DSC_image[1])
        DSC_All_class3.append(DSC_image[2])

        HD_All_class1.append(HD_image[0])
        HD_All_class2.append(HD_image[1])
        HD_All_class3.append(HD_image[2])

        ASD_All_class1.append(ASD_image[0])
        ASD_All_class2.append(ASD_image[1])
        ASD_All_class3.append(ASD_image[2])

        path = os.path.join('./ResultsTest/Images/', modelName)

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(torch.cat([images.data, labels.data, masks.view(labels.shape[0],1,256,256).data/3.0]),os.path.join(path,str(i)+'.png'), padding=0)

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    return [statistics.mean(DSC_All_class1),
            statistics.stdev(DSC_All_class1),
            statistics.mean(DSC_All_class2),
            statistics.stdev(DSC_All_class2),
            statistics.mean(DSC_All_class3),
            statistics.stdev(DSC_All_class3),
            statistics.mean(HD_All_class1),
            statistics.stdev(HD_All_class1),
            statistics.mean(HD_All_class2),
            statistics.stdev(HD_All_class2),
            statistics.mean(HD_All_class3),
            statistics.stdev(HD_All_class3),
            statistics.mean(ASD_All_class1),
            statistics.stdev(ASD_All_class1),
            statistics.mean(ASD_All_class2),
            statistics.stdev(ASD_All_class2),
            statistics.mean(ASD_All_class3),
            statistics.stdev(ASD_All_class3)]




class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()
