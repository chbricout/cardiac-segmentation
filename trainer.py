from typing import List
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from progressBar import printProgressBar

import medicalDataLoader
import argparse
from utils import *
import logging

logging.basicConfig(level=logging.INFO)
from UNet_Base import *
import random
import torch
import pdb
import warnings

from utils_test import inferenceTest

import matplotlib.pyplot as plt
from PIL import Image
from numpy import isnan
from scipy.ndimage import binary_closing, binary_dilation



def prop_by_layer(segmentations):
    element_by_layers = segmentations.sum(dim=(2, 3))
    props = element_by_layers / (element_by_layers.sum())
    return props + 1e-9



def get_parameters(id: int, *params: List[List[any]]):
    div = 1
    correct_id = id - 1
    res = []
    for par in params:
        res.append(par[(correct_id // div) % len(par)])
        div *= len(par)
    return res

def phr(const_loss:torch.Tensor, pho, lamb):
    if lamb + pho*const_loss >=0:
        return lamb*const_loss+ 0.5 *pho*const_loss.pow(2)
    else:
        return - lamb.pow(2)/(2*pho)

def runTraining(job_id=0):
    # torch.autograd.set_detect_anomaly(True)
    logging.info("-" * 40)
    logging.info("~~~~~~~~  Starting the training... ~~~~~~")
    logging.info("-" * 40)
    # BATCH_SIZE = [8,16,32]
    # LEARNING_RATES=[1e-2,1e-3,1e-4]
    # LEARNING_RATES_LOW=[1e-6,1e-7,1e-8]
    # LAMBDAS=np.linspace(1,10,20)
    # batch_size, batch_size_unsupervised, lr, lr_d, u_lr, lambda_c1_dice, lambda_c2_dice, lambda_supervised_prop, lambda_likely = get_parameters(job_id, [BATCH_SIZE, BATCH_SIZE, LEARNING_RATES,LEARNING_RATES, LEARNING_RATES, LAMBDAS, LAMBDAS, LAMBDAS, LAMBDAS])
    ## DEFINE HYPERPARAMETERS (batch_size > 1)
    batch_size = 32
    batch_size_unsupervised = 32
    batch_size_val = 1
    lr = 5e-4  # Learning Rate
    lr_d = 1e-4
    u_lr = 5e-8
    epoch = 1001  # Number of epochs
    pho = 0 # 0.1
    lambda_c1_dice = 0
    lambda_c2_dice = 0
    lambda_supervised_prop =  6
    lambda_prop_unsup = 6

    param_df = pd.DataFrame(
        [[
            lr,
            lr_d,
            u_lr,
            epoch,
            lambda_c1_dice,
            lambda_c2_dice,
            lambda_supervised_prop,
            lambda_prop_unsup,
            batch_size,
            batch_size_unsupervised,
        ]],
        columns=[
            "learning_rate",
            "learning_rate_discriminator",
            "learning_rate_unsupervised",
            "epochs",
            "lambda_c1_dice",
            "lambda_c2_dice",
            "lambda_supervised_prop",
            "lambda_prop_unsup",
            'batch_size',
            'batch_size_unsupervised',
        ],
    )
    test_metrics =[]
    root_dir = "./Data/"

    logging.info(" Dataset: {} ".format(root_dir))

    ## DEFINE THE TRANSFORMATIONS TO DO AND THE VARIABLES FOR TRAINING AND VALIDATION

    transform = transforms.Compose([transforms.ToTensor()])

    mask_transform = transforms.Compose([transforms.ToTensor()])

    train_set_unlabeled = medicalDataLoader.MedicalImageDataset(
        "unlabeled", root_dir, transform=transform, augment=True, equalize=True
    )
    train_loader_unlabeled = DataLoader(
        train_set_unlabeled,
        batch_size=batch_size_unsupervised,
        worker_init_fn=np.random.seed(0),
        num_workers=5,
        shuffle=True,
    )

    train_set_full = medicalDataLoader.MedicalImageDataset(
        "train",
        root_dir,
        transform=transform,
        mask_transform=mask_transform,
        augment=True,
        equalize=True,
    )

    train_loader_full = DataLoader(
        train_set_full,
        batch_size=batch_size,
        worker_init_fn=np.random.seed(0),
        num_workers=5,
        shuffle=True,
    )

    val_set = medicalDataLoader.MedicalImageDataset(
        "val",
        root_dir,
        transform=transform,
        mask_transform=mask_transform,
        equalize=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size_val,
        worker_init_fn=np.random.seed(0),
        num_workers=5,
        shuffle=False,
    )

    ## INITIALIZE YOUR MODEL
    num_classes = 4  # NUMBER OF CLASSES

    logging.info("~~~~~~~~~~~ Creating the UNet model ~~~~~~~~~~")
    modelName = f"Net_with_no_lagrange"
    logging.info(" Model Name: {}".format(modelName))

    # CREATION OF YOUR MODEL
    net = UNet(num_classes)
    discNet = SegNet()

    logging.info(
        "Total params: {0:,}".format(
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        )
    )
    stat_prior = to_var(torch.FloatTensor([0.9717, 0.0098, 0.0102, 0.0081]))

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    softMax = torch.nn.Softmax()
    CE_loss = torch.nn.CrossEntropyLoss(weight=torch.FloatTensor([0.03, 0.33, 0.32, 0.32 ]))

    BCE_loss = torch.nn.BCELoss()
    ## PUT EVERYTHING IN GPU RESOURCES
    if torch.cuda.is_available():
        net.cuda()
        softMax.cuda()
        CE_loss.cuda()
        BCE_loss.cuda()
        discNet.cuda()

    ## DEFINE YOUR OPTIMIZER
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    unsupervised_optimizer = torch.optim.Adam(
        net.parameters(), lr=u_lr, betas=(0.8, 0.999)
    )

    discOptimizer = torch.optim.Adam(discNet.parameters(), lr=lr_d, betas=(0.95, 0.999))

    ### To save statistics ####
    lossTotalTraining = []
    Best_loss_val = 1000
    BestEpoch = 0

    label_gt = to_var(torch.full((batch_size,), 0, dtype=torch.float))
    label_seg = to_var(torch.full((batch_size,), 1, dtype=torch.float))

    # net= pretrain_autoencode(net, train_loader_unlabeled, epoch, optimizer)
    directory = "Results/Statistics/" + modelName
    logging.info("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    if os.path.exists(directory) == False:
        os.makedirs(directory)
    param_df.to_csv(directory+"/params.csv")
    ## START THE TRAINING

    ## FOR EACH EPOCH
    for i in range(epoch):
        net.train()
        lossEpoch = []
        DSCEpoch = []
        DSCEpoch_w = []
        num_batches = len(train_loader_full)

        ## FOR EACH BATCH
        for j, data in enumerate(train_loader_full):
            ## GET IMAGES, LABELS and IMG NAMES
            images, labels, img_names = data
            ### From numpy to torch variables
            labels = to_var(labels)
            images = to_var(images)

            ### TRAIN DISCRIMINATOR ###
            ### REAL EXAMPLES ###
            discNet.zero_grad()
            discOptimizer.zero_grad()
            segmentation_classes = getTargetSegmentation(labels).float()
            pred_y = discNet(segmentation_classes[:, None])
            loss_bce_real = BCE_loss(pred_y, label_gt[: pred_y.shape[0]])
            loss_bce_real.backward()
            ### Fake examples ####
            pred_single = net(images)
            pred_single = torch.nn.functional.softmax(pred_single / 0.0001)
            for c_i in range(4):
                pred_single[:, c_i] *= c_i
            sum_pred = pred_single.sum(axis=1, keepdim=True)
            pred_y = discNet(sum_pred.detach())
            loss_bce_fake = BCE_loss(pred_y, label_seg[: pred_y.shape[0]])
            loss_bce_fake.backward()
            discOptimizer.step()
            ################### Train ###################
            # -- The CNN makes its predictions (forward pass)
            ### Set to zero all the gradients
            net.zero_grad()
            optimizer.zero_grad()
            raw_predictions = net(images)

            net_predictions = softMax(raw_predictions)
            # -- Compute the losses --#
            # THIS FUNCTION IS TO CONVERT LABELS TO A FORMAT TO BE USED IN THIS CODE
            segmentation_classes = getTargetSegmentation(labels)
            ceil_pred = torch.nn.functional.softmax(raw_predictions.detach() / 0.0001)

            # COMPUTE THE LOSS
            CE_loss_value = CE_loss(
                net_predictions, segmentation_classes
            )  # XXXXXX and YYYYYYY are your inputs for the CE

            # # DICE
            # c1_label = torch.zeros_like(segmentation_classes)
            # c1_label[segmentation_classes == 1] = 1
            # num = (net_predictions[:, 1] * c1_label).sum()
            # den = (net_predictions[:, 1] + c1_label).sum()
            # c1_dice_loss = 1.0 - (2 * num + 1) / (den + 1)

            # c2_label = torch.zeros_like(segmentation_classes)
            # c2_label[segmentation_classes == 3] = 1
            # num = (net_predictions[:, 2] * c2_label).sum()
            # den = (net_predictions[:, 2] + c2_label).sum()
            # c2_dice_loss = 1.0 - (2 * num + 1) / (den + 1)

            # kl = torch.nn.functional.kl_div(torch.log(prop_by_layer(ceil_pred)), stat_prior,  reduction='batchmean')
            prop_diff_value = torch.pow(
                (prop_by_layer(ceil_pred) - stat_prior), 2
            ).mean()

            lossTotal = (
                CE_loss_value
                + lambda_supervised_prop * prop_diff_value
                # + lambda_c1_dice * c1_dice_loss
                # + lambda_c2_dice * c2_dice_loss
            )

            # for c_i in range(4):
            #     ceil_pred[:, c_i] *= c_i
            # sum_pred = ceil_pred.sum(axis=1, keepdim=True)
            # likely_loss = discNet(sum_pred)

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            if (i-1)%8==0 and False:
                lambda_supervised_prop = phr(prop_diff_value.detach(), pho, lambda_supervised_prop)
                # lambda_c1_dice = phr(c1_dice_loss.detach(), pho, lambda_c1_dice)
                # lambda_c2_dice = phr(c2_dice_loss.detach(), pho, lambda_c2_dice)

            lossTotal.backward()
            optimizer.step()

            

            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(lossTotal.cpu().data.numpy())
            printProgressBar(
                j + 1,
                num_batches,
                prefix="[Training] Epoch: {} ".format(i),
                length=15,
                suffix=" Loss: {:.4f}, prop diff : {:.4f}, gan : {:.4f}, lambda supervised : {:.4f}".format(
                    lossTotal,
                    prop_diff_value,
                    loss_bce_fake + loss_bce_real,
                    lambda_supervised_prop
                ),
            )
            # print(f" Dice ={computeDSC(net_predictions, segmentation_classes)} ")
            labels.cpu()
            images.cpu()

        num_batches = len(train_loader_unlabeled)
        for j, data in enumerate(train_loader_unlabeled):
            images, img_path = data
            images = to_var(images)

            ### TRAIN DISCRIMINATOR ###
            discNet.zero_grad()
            #####
            # -- The CNN makes its predictions (forward pass)
            ### Set to zero all the gradients
            net.zero_grad()
            unsupervised_optimizer.zero_grad()
            raw_predictions = net(images)

            # net_predictions = softMax(raw_predictions)
            ceil_pred_for_disc = torch.nn.functional.softmax(
                raw_predictions.detach() / 0.0001
            )
            ceil_pred_for_prop = torch.nn.functional.softmax(
                raw_predictions / 0.0001
            )

            # -- Compute the losses --#
            # kl = torch.nn.functional.kl_div(torch.log(prop_by_layer(ceil_pred)), stat_prior,  reduction='batchmean')
            prop_diff_value = torch.pow(
                (prop_by_layer(ceil_pred_for_prop) - stat_prior), 2
            ).mean()

            lossTotal = lambda_prop_unsup * prop_diff_value

            if i >= 2:
                for c_i in range(4):
                    ceil_pred_for_disc[:, c_i] *= c_i
                sum_pred = ceil_pred_for_disc.sum(axis=1, keepdim=True)
                likely_loss = discNet(sum_pred)
                lossTotal = likely_loss.sum()
                # lossTotal +=  BCE_loss(likely_loss, label_gt[:likely_loss.shape[0]])
                if (i-1)%8==0  and False:
                    lambda_prop_unsup = phr(prop_diff_value.detach().mean(), pho, lambda_prop_unsup)

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)

            lossTotal.backward()
            unsupervised_optimizer.step()
            printProgressBar(
                j + 1,
                num_batches,
                prefix="[Training] Unlabeled Epoch: {}".format(i),
                length=15,
                suffix=" Loss: {:.4f}, lambda likely {:.4f} ".format(lossTotal, lambda_prop_unsup),
            )

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        lossTotalTraining.append(lossEpoch)

        printProgressBar(
            num_batches,
            num_batches,
            done="[Training] Epoch: {}, LossG: {:.4f}".format(i, lossEpoch),
        )

        ## THIS IS HOW YOU WILL SAVE THE TRAINED MODELS AFTER EACH EPOCH.
        ## WARNING!!!!! YOU DON'T WANT TO SAVE IT AT EACH EPOCH, BUT ONLY WHEN THE MODEL WORKS BEST ON THE VALIDATION SET!!
        if not os.path.exists("./models/" + modelName):
            os.makedirs("./models/" + modelName)
        if i % 10 == 0:
            torch.save(
                net.state_dict(), "./models/" + modelName + "/" + str(i) + "_Epoch"
            )
            logging.info("~~~~~~~~~~~ Starting the testing ~~~~~~~~~~")
            inf = inferenceTest(net, val_loader, modelName + "/" + str(i) + "_Epoch")
            [
                DSC1,
                DSC1s,
                DSC2,
                DSC2s,
                DSC3,
                DSC3s,
                HD1,
                HD1s,
                HD2,
                HD2s,
                HD3,
                HD3s,
                ASD1,
                ASD1s,
                ASD2,
                ASD2s,
                ASD3,
                ASD3s,
            ]=inf

            logging.info(
                "###                                                       ###"
            )
            logging.info(
                "###         TEST RESULTS                                  ###"
            )
            logging.info(
                "###  Dice : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(
                    DSC1,
                    DSC1s,
                    DSC2,
                    DSC2s,
                    DSC3,
                    DSC3s,
                    (DSC1 + DSC2 + DSC3) / 3,
                    (DSC1s + DSC2s + DSC3s) / 3,
                )
            )
            logging.info(
                "###  HD   : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(
                    HD1,
                    HD1s,
                    HD2,
                    HD2s,
                    HD3,
                    HD3s,
                    (HD1 + HD2 + HD3) / 3,
                    (HD1s + HD2s + HD3s) / 3,
                )
            )
            logging.info(
                "###  ASD  : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(
                    ASD1,
                    ASD1s,
                    ASD2,
                    ASD2s,
                    ASD3,
                    ASD3s,
                    (ASD1 + ASD2 + ASD3) / 3,
                    (ASD1s + ASD2s + ASD3s) / 3,
                )
            )
            logging.info(
                "###                                                       ###"
            )
            test_metrics.append(inf)

        if i == epoch - 1:
            torch.save(
                net.state_dict(), "./models/" + modelName + "/" + str(i) + "_Epoch"
            )
            df= pd.DataFrame(test_metrics, index=[i for i in range(0,epoch, 10)], columns=["DSC1",
                "DSC1s",
                "DSC2",
                "DSC2s",
                "DSC3",
                "DSC3s",
                "HD1",
                "HD1s",
                "HD2",
                "HD2s",
                "HD3",
                "HD3s",
                "ASD1",
                "ASD1s",
                "ASD2",
                "ASD2s",
                "ASD3",
                "ASD3s",])
            df.to_csv(os.path.join(directory, "metrics.csv"))

        np.save(os.path.join(directory, "Losses.npy"), lossTotalTraining)
    return net


if __name__ == "__main__":
    warnings.simplefilter("ignore")
    mod = runTraining()
    with torch.no_grad():
        transform = transforms.Compose([transforms.ToTensor()])
        torch.cuda.empty_cache()
        val_set = medicalDataLoader.MedicalImageDataset(
            "val",
            "./Data/",
            transform=transform,
            mask_transform=transform,
            equalize=True,
        )

        val_loader = DataLoader(
            val_set,
            batch_size=2,
            worker_init_fn=np.random.seed(0),
            num_workers=0,
            shuffle=False,
        )
        logging.info(inference(mod, val_loader, "Test_Model", 500))

        i = 0
        for img, mask, path in val_set:
            if i == 10:
                break
            out = mod(to_var(img)[None, :])
            out_cpu = out.detach().cpu()[0]
            out_cpu = torch.nn.functional.softmax(out_cpu)
            # segmentation_image = out_to_seg(predToSegmentation(out_cpu),3)
            segmentation_image = torch.argmax(out_cpu, dim=0)
            # Now, concatenated_output has shape (batch_size, 1, num_classes, height, width)
            # If you want to remove the singleton dimension, you can use squeeze
            fig, plots = plt.subplots(2, 2)
            plots[0, 0].imshow(segmentation_image, cmap="gray")
            plots[0, 0].set_title("Network output")

            plots[1, 0].imshow(mask.permute(1, 2, 0), cmap="gray")
            plots[1, 0].set_title("Mask GT")

            plots[1, 1].imshow(img[0], cmap="gray")
            plots[1, 1].imshow(mask.permute(1, 2, 0), cmap="tab10", alpha=0.6)
            plots[1, 1].set_title("GT + Image")

            plots[0, 1].imshow(img[0], cmap="gray")
            plots[0, 1].imshow(segmentation_image, cmap="tab10", alpha=0.6)
            plots[0, 1].set_title("Prediction + Image")

            i += 1
        plt.show()
