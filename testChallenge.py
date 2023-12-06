from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar

#import JoseLoader
import medicalDataLoader
import argparse
from utils_test import *

from UNet_Base import *
import random
import torchvision.transforms.functional as TF
from random import random, randint
from PIL import Image, ImageOps



##### INSTRUCTIONS #######

def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        #b = -1.0 * b.sum() # If using this one, the value on the final loss for eloss is 0.0000001
        return b


def _createModel(segmentation_model_class, encoder, encoder_weights, num_class, activation):
    model = segmentation_model_class(
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        classes=num_class,
        activation=activation,
        in_channels=1,
    )
    return model

def runTesting(args):
    print('-' * 40)
    print('~~~~~~~~  Starting the testing... ~~~~~~')
    print('-' * 40)

    batch_size_val = 1
    root_dir = './Data/'

    # https://sparrow.dev/pytorch-normalize/
    transform = transforms.Compose([
        transforms.ToTensor()
        #transforms.Normalize((0.5), (0.20))
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_set = medicalDataLoader.MedicalImageDataset('train',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False)

    test_loader = DataLoader(test_set,
                            batch_size=batch_size_val,
                            num_workers=5,
                            shuffle=False)
                                                                    
    # Initialize
    num_classes = 4

    # Create and load model
    net = UNet(num_classes)

    # Load
    net.load_state_dict(torch.load('./models/'+args.modelName))
    net.eval()

    if torch.cuda.is_available():
        net.cuda()

    print("~~~~~~~~~~~ Starting the testing ~~~~~~~~~~")
    [DSC1, DSC1s,DSC2,DSC2s, DSC3,DSC3s, HD1, HD1s,HD2,HD2s, HD3,HD3s, ASD1,ASD1s, ASD2,ASD2s, ASD3,ASD3s] = inferenceTest(net, test_loader, args.modelName)

    print("###                                                       ###")
    print("###         TEST RESULTS                                  ###")
    print("###  Dice : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(DSC1,
                                                                                                                     DSC1s,
                                                                                                                     DSC2,
                                                                                                                     DSC2s,
                                                                                                                     DSC3,
                                                                                                                     DSC3s,
                                                                                                                     (DSC1+DSC2+DSC3)/3,
                                                                                                                     (DSC1s+DSC2s+DSC3s)/3))
    print("###  HD   : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(HD1,
                                                                                                                     HD1s,
                                                                                                                     HD2,
                                                                                                                     HD2s,
                                                                                                                     HD3,
                                                                                                                     HD3s,
                                                                                                                     (HD1 + HD2 + HD3) / 3,
                                                                                                                     (HD1s + HD2s + HD3s) / 3))
    print("###  ASD  : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(ASD1,
                                                                                                                     ASD1s,
                                                                                                                     ASD2,
                                                                                                                     ASD2s,
                                                                                                                     ASD3,
                                                                                                                     ASD3s,
                                                                                                                     (ASD1 + ASD2 + ASD3) / 3,
                                                                                                                     (ASD1s + ASD2s + ASD3s) / 3))
    print("###                                                       ###")



if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument("--modelName",default="Equipe1",type=str)
    args=parser.parse_args()
    runTesting(args)
