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


def pretrain_autoencode(net: UNet, dataloader, epoch, optimizer):
    logging.info("~~~~~~~~~~~ Starting the pre-training ~~~~~~~~~~")
    net.pretraining = True

    ## START THE PRE TRAINING
    MSELoss = torch.nn.MSELoss(reduction="sum").cuda()
    ## FOR EACH EPOCH
    for i in range(epoch):
        net.train()
        lossEpoch = []
        num_batches = len(dataloader)

        ## FOR EACH BATCH
        for j, data in enumerate(dataloader):
            ### Set to zero all the gradients
            net.zero_grad()
            optimizer.zero_grad()

            ## GET IMAGES, LABELS and IMG NAMES
            images, erased_image, img_names = data

            ### From numpy to torch variables
            images = to_var(images)
            erased_image = to_var(erased_image)
            ################### Train ###################
            # -- The CNN makes its predictions (forward pass)
            net_predictions = net(erased_image)
            # -- Compute the losses --#
            # THIS FUNCTION IS TO CONVERT LABELS TO A FORMAT TO BE USED IN THIS CODE
            # COMPUTE THE LOSS
            lossTotal = MSELoss(
                net_predictions, images
            )  # XXXXXX and YYYYYYY are your inputs for the CE

            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)
            lossTotal.backward()
            optimizer.step()
            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(lossTotal.cpu().data.numpy())
            printProgressBar(
                j + 1,
                num_batches,
                prefix="[Pre-training] Epoch: {} ".format(i),
                length=15,
                suffix=" Loss: {:.4f}, ".format(lossTotal),
            )

        lossEpoch = np.asarray(lossEpoch)
        lossEpoch = lossEpoch.mean()

        printProgressBar(
            num_batches,
            num_batches,
            done="[Pre-training] Epoch: {}, LossG: {:.4f}".format(i, lossEpoch),
        )
    for image, erased, net_pred in zip(
        images.detach().cpu(),
        erased_image.detach().cpu(),
        net_predictions.detach().cpu(),
    ):
        fig, plots = plt.subplots(1, 3)
        plots[0].imshow(erased[0], cmap="gray")
        plots[0].set_title("erased image")

        plots[1].imshow(net_pred[0], cmap="gray")
        plots[1].set_title("Reconstruction")

        plots[2].imshow(image[0], cmap="gray")
        plots[2].set_title("GT")

        plt.show()
    net.pretraining = False

    return net


def prop_diff(segmentations, props):
    element_by_layers = segmentations.sum(dim=(2, 3))
    prob_by_layer = element_by_layers / (element_by_layers.sum())
    piecewise_diff = prob_by_layer - props.to(prob_by_layer.device)
    diff_sq = torch.pow(piecewise_diff, 2)
    return diff_sq.sum()

def prop_by_layer(segmentations):
    element_by_layers = segmentations.sum(dim=(2, 3))
    props= element_by_layers / (element_by_layers.sum())
    return props + 1e-9

def get_grav_center_by_class(batch_segmentations):
    batch_coord = []
    for segmentations in batch_segmentations:
        coord = []
        if (
            (segmentations[3] != 0).sum() > 1
            and (segmentations[2] != 0).sum() > 1
            and (segmentations[1] != 0).sum() > 1
        ):
            for seg in segmentations:
                grid_x, grid_y = torch.meshgrid(
                    torch.arange(seg.shape[1]), torch.arange(seg.shape[0])
                )

                flat_grid_x = grid_x.flatten().float().to(segmentations.device)
                flat_grid_y = grid_y.flatten().float().to(segmentations.device)
                flat_probabilities = seg.flatten()

                center_x = torch.sum(flat_grid_x * flat_probabilities) / torch.sum(
                    flat_probabilities
                )
                center_y = torch.sum(flat_grid_y * flat_probabilities) / torch.sum(
                    flat_probabilities
                )
                coord.append([center_x, center_y])
            batch_coord.append(coord)
    return torch.Tensor(batch_coord).to(segmentations.device)


def grav_center_loss(batch_segmentations):
    center_by_classe = get_grav_center_by_class(batch_segmentations)
    bet_2_3 = 0
    if len(center_by_classe.shape) == 3:
        bet_2_3 = torch.pow(center_by_classe[:, 3] - center_by_classe[:, 2], 2).mean()
    return bet_2_3




def runTraining():
    # torch.autograd.set_detect_anomaly(True)
    logging.info("-" * 40)
    logging.info("~~~~~~~~  Starting the training... ~~~~~~")
    logging.info("-" * 40)

    ## DEFINE HYPERPARAMETERS (batch_size > 1)
    batch_size = 32
    batch_size_unsupervised = 32
    batch_size_val = 1
    lr = 1e-3 # Learning Rate
    lr_d =1e-3
    u_lr = 1e-7
    epoch = 201  # Number of epochs

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
    modelName = "Test_Model"
    logging.info(" Model Name: {}".format(modelName))

    # CREATION OF YOUR MODEL
    net = UNet(num_classes)
    discNet = SegNet()

    logging.info(
        "Total params: {0:,}".format(
            sum(p.numel() for p in net.parameters() if p.requires_grad)
        )
    )
    stat_prior = to_var(torch.FloatTensor([ 0.9717, 0.0098, 0.0102, 0.0081]))

    # DEFINE YOUR OUTPUT COMPONENTS (e.g., SOFTMAX, LOSS FUNCTION, ETC)
    softMax = torch.nn.Softmax()
    CE_loss = torch.nn.CrossEntropyLoss(weight=1-stat_prior)
    
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
    unsupervised_optimizer = torch.optim.Adam(net.parameters(), lr=u_lr, betas=(0.5,0.999)) 

    discOptimizer = torch.optim.Adam(discNet.parameters(), lr=lr_d)

    ### To save statistics ####
    lossTotalTraining = []
    Best_loss_val = 1000
    BestEpoch = 0

    label_gt = to_var(torch.full((batch_size,), 0,  dtype=torch.float))
    label_seg = to_var(torch.full((batch_size,), 1,  dtype=torch.float))

    # net= pretrain_autoencode(net, train_loader_unlabeled, epoch, optimizer)
    directory = "Results/Statistics/" + modelName
    logging.info("~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    if os.path.exists(directory) == False:
        os.makedirs(directory)

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
            images, labels, out_el, img_names = data
            ### From numpy to torch variables
            labels = to_var(labels)
            images = to_var(images)

            ### TRAIN DISCRIMINATOR ###
            ### REAL EXAMPLES ### 
            # discNet.zero_grad()
            # discOptimizer.zero_grad()
            # segmentation_classes = getTargetSegmentation(labels).float()
            # pred_y = discNet(segmentation_classes[:,None])
            # loss_bce_real = BCE_loss(pred_y, label_gt[:pred_y.shape[0]])
            # loss_bce_real.backward()
            # ### Fake examples ####
            # pred_single = net(images)
            # pred_single =  torch.nn.functional.softmax(pred_single/0.0001)
            # for c_i in range(4):
            #     pred_single[:,c_i]*=c_i
            # sum_pred = pred_single.sum(axis=1, keepdim=True)
            # pred_y=discNet(sum_pred.detach())
            # loss_bce_fake = BCE_loss(pred_y, label_seg[:pred_y.shape[0]])
            # loss_bce_fake.backward()
            # discOptimizer.step()
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
            ceil_pred =  torch.nn.functional.softmax(raw_predictions.detach()/0.0001)

            # COMPUTE THE LOSS
            CE_loss_value = CE_loss(
                net_predictions, segmentation_classes
            )  # XXXXXX and YYYYYYY are your inputs for the CE
           
            # kl = torch.nn.functional.kl_div(torch.log(prop_by_layer(ceil_pred)), stat_prior,  reduction='batchmean')
            prop_diff_value = torch.pow((prop_by_layer(ceil_pred) - stat_prior),2).mean()

            lossTotal = CE_loss_value  + 6 * prop_diff_value

            # for c_i in range(4):
            #     ceil_pred[:,c_i]*=c_i
            # sum_pred = ceil_pred.sum(axis=1, keepdim=True)
            # likely_loss = discNet(sum_pred)
            
            # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)

            lossTotal.backward()
            optimizer.step()

            
            # THIS IS JUST TO VISUALIZE THE TRAINING
            lossEpoch.append(lossTotal.cpu().data.numpy())
            printProgressBar(
                j + 1,
                num_batches,
                prefix="[Training] Epoch: {} ".format(i),
                length=15,
                suffix=" Loss: {:.4f}, prop diff : {:.4f}".format(
                    lossTotal, prop_diff_value
                ),
            )
            # print(f" Dice ={computeDSC(net_predictions, segmentation_classes)} ")
            labels.cpu()
            images.cpu()

        if i >= 0:
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

                net_predictions = softMax(raw_predictions)
                ceil_pred=  torch.nn.functional.softmax(raw_predictions/0.0001)

                # -- Compute the losses --#
                # kl = torch.nn.functional.kl_div(torch.log(prop_by_layer(ceil_pred)), stat_prior,  reduction='batchmean')
                # prop_diff_value = prop_diff(segmented, torch.Tensor([0.7, 0.31, 0.3, 0.29]))
                # for c_i in range(4):
                #     ceil_pred[:,c_i]*=c_i
                # sum_pred = ceil_pred.sum(axis=1, keepdim=True)
                # likely_loss = discNet(sum_pred)
                lossTotal =   torch.pow((prop_by_layer(ceil_pred) - stat_prior),2).mean()

                
                
                # DO THE STEPS FOR BACKPROP (two things to be done in pytorch)

                lossTotal.backward()
                unsupervised_optimizer.step()
                printProgressBar(
                    j + 1,
                    num_batches,
                    prefix="[Training] Unlabeled Epoch: {} ".format(i),
                    length=15,
                    suffix=" Loss: {:.4f}".format(
                        lossTotal
                    ),
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
        if i%10 == 0:
            torch.save(
                net.state_dict(), "./models/" + modelName + "/" + str(i) + "_Epoch"
            )
            logging.info("~~~~~~~~~~~ Starting the testing ~~~~~~~~~~")
            [DSC1, DSC1s,DSC2,DSC2s, DSC3,DSC3s, HD1, HD1s,HD2,HD2s, HD3,HD3s, ASD1,ASD1s, ASD2,ASD2s, ASD3,ASD3s] = inferenceTest(net, val_loader,  modelName + "/" + str(i) + "_Epoch")
            logging.info("###                                                       ###")
            logging.info("###         TEST RESULTS                                  ###")
            logging.info("###  Dice : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(DSC1,
                                                                                                                            DSC1s,
                                                                                                                            DSC2,
                                                                                                                            DSC2s,
                                                                                                                            DSC3,
                                                                                                                            DSC3s,
                                                                                                                            (DSC1+DSC2+DSC3)/3,
                                                                                                                            (DSC1s+DSC2s+DSC3s)/3))
            logging.info("###  HD   : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(HD1,
                                                                                                                            HD1s,
                                                                                                                            HD2,
                                                                                                                            HD2s,
                                                                                                                            HD3,
                                                                                                                            HD3s,
                                                                                                                            (HD1 + HD2 + HD3) / 3,
                                                                                                                            (HD1s + HD2s + HD3s) / 3))
            logging.info("###  ASD  : c1: {:.4f} ({:.4f}) c2: {:.4f} ({:.4f}) c3: {:.4f} ({:.4f}) Mean: {:.4f} ({:.4f}) ###".format(ASD1,
                                                                                                                            ASD1s,
                                                                                                                            ASD2,
                                                                                                                            ASD2s,
                                                                                                                            ASD3,
                                                                                                                            ASD3s,
                                                                                                                            (ASD1 + ASD2 + ASD3) / 3,
                                                                                                                            (ASD1s + ASD2s + ASD3s) / 3))
            logging.info("###                                                       ###")



        if i == epoch - 1:
            torch.save(
                net.state_dict(), "./models/" + modelName + "/" + str(i) + "_Epoch"
            )

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
        logging.info(inference(mod, val_loader, "Test_Model", 99))

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
