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
import statistics
from PIL import Image
from medpy.metric.binary import dc, hd, asd, assd
import scipy.spatial

# from scipy.spatial.distance import directed_hausdorff


labels = {0: "Background", 1: "Foreground"}


def computeDSC(pred, gt):
    dscAll = []
    # pdb.set_trace()
    for i_b in range(pred.shape[0]):
        pred_id = pred[i_b, :, :]
        gt_id = gt[i_b, :, :]
        dscAll.append(dice_seg(pred_id, gt_id))

    DSC = torch.Tensor(dscAll)

    return (DSC * torch.Tensor([0.5,0.25,0.25])).mean()


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
        imageNames = [
            f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))
        ]

    imageNames.sort()

    return imageNames


def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    # pdb.set_trace()
    return (x == 1).float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.33333334, 0.6666667 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3

    denom = 0.33333334  # for ACDC this value
    return (batch / denom).round().long().squeeze()


from scipy import ndimage


def inference(net, img_batch, modelName, epoch):
    total = len(img_batch)
    net.eval()

    softMax = nn.Softmax().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()

    losses = []
    for i, data in enumerate(img_batch):
        printProgressBar(
            i, total, prefix="[Inference] Getting segmentations...", length=30
        )
        images, labels, img_names = data

        images = to_var(images)
        labels = to_var(labels)

        net_predictions = net(images)
        segmentation_classes = getTargetSegmentation(labels)

        CE_loss_value = CE_loss(net_predictions, segmentation_classes)
        losses.append(CE_loss_value.cpu().data.numpy())
        pred_y = softMax(net_predictions)
        masks = torch.argmax(pred_y, dim=1)

        path = os.path.join("./Results/Images/", modelName, str(epoch))

        if not os.path.exists(path):
            os.makedirs(path)

        torchvision.utils.save_image(
            torch.cat(
                [
                    images.data,
                    labels.data,
                    masks.view(labels.shape[0], 1, 256, 256).data / 3.0,
                ]
            ),
            os.path.join(path, str(i) + ".png"),
            padding=0,
        )

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    losses = np.asarray(losses)

    return losses.mean()


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()


def batch_one_hot_encode(batch_segmentation_maps, num_classes):
    """
    Perform one-hot encoding on a batch of segmentation maps.

    Args:
    - batch_segmentation_maps (torch.Tensor): Batch of segmentation map tensors with class indices.
      Shape: (batch_size, H, W)
    - num_classes (int): Number of classes in the segmentation task.

    Returns:
    - torch.Tensor: Batch of one-hot encoded tensors.
      Shape: (batch_size, num_classes, H, W)
    """
    # Ensure the batch_segmentation_maps is a PyTorch tensor
    batch_segmentation_maps = torch.tensor(batch_segmentation_maps)

    # Ensure the batch_segmentation_maps has the correct shape (batch_size, H, W)
    if len(batch_segmentation_maps.shape) != 3:
        raise ValueError(
            "Batch segmentation maps should be a 3D tensor (batch_size, H, W)."
        )

    # Create a zero-filled tensor with dimensions (batch_size, num_classes, H, W)
    batch_one_hot = torch.zeros(
        (
            batch_segmentation_maps.shape[0],
            num_classes,
            batch_segmentation_maps.shape[1],
            batch_segmentation_maps.shape[2],
        )
    )

    # Fill in the one-hot tensor based on class indices for each batch
    for batch_idx in range(batch_segmentation_maps.shape[0]):
        for class_idx in range(num_classes):
            batch_one_hot[batch_idx, class_idx, :, :] = (
                batch_segmentation_maps[batch_idx, :, :] == class_idx
            ).float()

    return to_var(batch_one_hot)


def out_to_seg(img, num_classes):
    out = torch.zeros((1, *img.shape[1:])).to(img.device)
    for i, layer in enumerate(img):
        out += layer * (i * 255 / (num_classes - 1))
    return out


def min_max_normalize(input_tensor):
    min_value = input_tensor.min().item()
    max_value = input_tensor.max().item()

    # Normalize between 0 and 1
    return (input_tensor - min_value) / (max_value - min_value)


def dice_seg(masks, segmentation_classes):
    DSC_image = []
    for c_i in range(3):
        DSC_image.append(
            dc(
                masks[c_i + 1].cpu().numpy(),
                segmentation_classes[c_i + 1].cpu().numpy(),
            )
        )
    return DSC_image
