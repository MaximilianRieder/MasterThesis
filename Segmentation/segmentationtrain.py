import torch
#from g_laufwerk.Masterarbeit.Code.Segmentation.legacy_dataset_org import DatasetSegOrg
#from g_laufwerk.Masterarbeit.Code.Segmentation.legacy_dataset_gen_smoke import DatasetSegSmoke
import sys
from utils import save_mask_grayscale_as_img
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config_seg
from tqdm import tqdm
from torchvision.utils import save_image
import random
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import Accuracy, JaccardIndex, Specificity, Recall, ConfusionMatrix
import network.modeling as modeling
import metric_calculation
import class_calculations
from average_meter import AverageMeter
from torch.utils.data import ConcatDataset
import time
import torchvision.transforms as transforms
from segmentation_models_pytorch.losses import LovaszLoss
from dataset_swap import DatasetSegSmokeSwap
import os
from utils import save_checkpoint
import copy

def add_to_mat(add_m, glob_m):
    if(pd.isna(glob_m)):
        return copy.deepcopy(add_m)
    else:
        glob_m += add_m
        return glob_m
        
def write_metrics(writer, global_conf_mat, unavailable_class_indices, num_classes, loss_meter, epoch, domain_label):
    #compute and write metrics
    global_acc, class_ious, class_dices, mean_iou_meter, mean_dice_meter = metric_calculation.calculate_iou_dice(global_conf_mat)
    metrics_dict = metric_calculation.calculate_mean_metrics(global_conf_mat, global_acc, class_ious, class_dices, mean_iou_meter, mean_dice_meter, unavailable_class_indices, None, loss_meter, True)
    for i in range(num_classes + 1):
        if ('IoU/' + str(i) in metrics_dict.keys()) or ('Dice/' + str(i) in metrics_dict.keys()):
            writer.add_scalar(f'{domain_label}/IoU/' + str(i), metrics_dict['IoU/' + str(i)], epoch)
            writer.add_scalar(f'{domain_label}/Dice/' + str(i), metrics_dict['Dice/' + str(i)], epoch)
    writer.add_scalar(f'{domain_label}/IoU/mIoU' , metrics_dict['IoU/mIoU'], epoch)
    writer.add_scalar(f'{domain_label}/Dice/mDice' , metrics_dict['Dice/mDice'], epoch)
    writer.add_scalar(f'{domain_label}/Global_Acc' , metrics_dict['Global_Acc'], epoch)
    writer.add_scalar(f'/Loss' , metrics_dict['Loss'], epoch)

def train_fn(model, loader, opt, loss_function, scaler, is_train, epoch, writer, num_classes, unavailable_class_indices, with_generated_imgs, scheduler, DEVICE, fold_video, LOVASZ_LOSS, custom_label):

    global_conf_mat_NS = None
    global_conf_mat_MS = None
    global_conf_mat_HS = None
    global_conf_mat = None
    
    loss_meter = AverageMeter()

    for image, mask, video_name, frame_n, domain in loader:
        
        #if is_train:
        opt.zero_grad()

        with torch.cuda.amp.autocast():
            image = image.float().to(DEVICE)
            if LOVASZ_LOSS:
                mask = mask.float().to(DEVICE)
            else:
                mask = mask.long().to(DEVICE)
            conf_mat = ConfusionMatrix(task="multiclass", num_classes=num_classes + 1).to(DEVICE)
            # segmentation / loss
            outputs = model(image)
            mask = mask.squeeze()
            loss = loss_function(outputs, mask)

        if is_train:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()           
            scheduler.step()
        
        with torch.no_grad():
            
            loss_meter.update(loss, n=config_seg.BATCH_SIZE)
            
            for idx, sample in enumerate(outputs):
                # Apply softmax
                output_sm = torch.nn.functional.softmax(sample, dim=0)
                output_preds = torch.argmax(output_sm, dim=0)
                
                conf_m = conf_mat(torch.flatten(output_preds), torch.flatten(mask[idx]))
                
                if domain[idx] == "not_smoked":
                    global_conf_mat_NS = add_to_mat(conf_m, global_conf_mat_NS)
                elif domain[idx] == "slightly_smoked":
                    global_conf_mat_MS = add_to_mat(conf_m, global_conf_mat_MS)
                elif domain[idx] == "heavily_smoked":
                    global_conf_mat_HS = add_to_mat(conf_m, global_conf_mat_HS)
                else: 
                    print("domain_error")
                    
                global_conf_mat = add_to_mat(conf_m, global_conf_mat)
            
                #output masks
                if is_train and with_generated_imgs and config_seg.SAVE_MASK and ((epoch+1) % 5 == 0):
                    if (epoch + 1) % config_seg.NUM_EPOCHS == 0:
                        save_mask_grayscale_as_img(output_preds,f"{config_seg.SAVE_IMGS_MASK_PATH}{custom_label}/train/with_gen/{epoch}/{fold_video}/{video_name[idx]}", frame_n[idx])
                if is_train and not with_generated_imgs:
                    if (epoch + 1) % config_seg.NUM_EPOCHS == 0:
                        save_mask_grayscale_as_img(output_preds,f"{config_seg.SAVE_IMGS_MASK_PATH}{custom_label}/train/normal/{epoch}/{fold_video}/{video_name[idx]}", frame_n[idx])
                if not is_train and with_generated_imgs:
                    save_mask_grayscale_as_img(output_preds,f"{config_seg.SAVE_IMGS_MASK_PATH}{custom_label}/val/with_gen/{epoch}/{fold_video}/{video_name[idx]}", frame_n[idx])
                if not is_train and not with_generated_imgs:
                    save_mask_grayscale_as_img(output_preds,f"{config_seg.SAVE_IMGS_MASK_PATH}{custom_label}/val/normal/{epoch}/{fold_video}/{video_name[idx]}", frame_n[idx])
    
    with torch.no_grad():
        ua = unavailable_class_indices["train"] if is_train else unavailable_class_indices["val"]
        if global_conf_mat_NS is not None:
            write_metrics(writer, global_conf_mat_NS, ua, num_classes, loss_meter, epoch, "no_smoke_domain")
        if global_conf_mat_MS is not None:
            write_metrics(writer, global_conf_mat_MS, ua, num_classes, loss_meter, epoch, "slightly_smoked_domain")
        if global_conf_mat_HS is not None:
            write_metrics(writer, global_conf_mat_HS, ua, num_classes, loss_meter, epoch, "heavy_smoke_domain")
        write_metrics(writer, global_conf_mat, ua, num_classes, loss_meter, epoch, "all_domains")
        if is_train:
            writer.add_scalar("LR", scheduler.get_last_lr()[0], epoch)
        
        if config_seg.SAVE_MODEL and ((epoch+1) % 5 == 0):
            save_model_path = f"{config_seg.SAVE_MODEL_PATH}{custom_label}/{fold_video}/{epoch + 1}/with_gen_{with_generated_imgs}/lov_loss{LOVASZ_LOSS}/"
            os.makedirs(save_model_path, exist_ok=True)
            save_checkpoint(model, filename=f"{save_model_path}seg_model.pth.tar")


##################################################################
### function that sets up the training and validation of the model
##################################################################
def train_validate(dataset_train, dataset_val, dataloader_train, writer_train, writer_val, num_classes, unavailable_class_indices, class_weights_train, class_weights_val, with_generated_imgs, DEVICE, fold_videos, LOVASZ_LOSS, custom_label):
    ## Model taken from https://github.com/VainF/DeepLabV3Plus-Pytorch.
    model = modeling.deeplabv3plus_resnet101(num_classes=19, output_stride=1, pretrained_backbone=True)
    last_layer = nn.Conv2d(
        in_channels=256,
        out_channels=num_classes + 1,
        kernel_size=1,
        stride=1
    )
    model.load_state_dict( torch.load(config_seg.SEG_MODEL_PAR_PRETRAIN)['model_state'])
    model.classifier.classifier[3] = last_layer
    model = model.to(DEVICE)
    
    # scaler
    # float 16 -> scale gradients so no information is lost
    scaler = torch.cuda.amp.GradScaler()

    opt = optim.Adam(
        model.parameters(),
        lr=config_seg.LEARNING_RATE
        #betas=(0.5, 0.999),
    )

    scheduler_train = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=config_seg.LEARNING_RATE, epochs=config_seg.NUM_EPOCHS, steps_per_epoch=len(dataloader_train), pct_start=0.05)
    
    if LOVASZ_LOSS:
        loss_train = LovaszLoss('multiclass', per_image=False, from_logits=True)
        loss_val  = LovaszLoss('multiclass', per_image=False, from_logits=True)
    else:
        loss_train = nn.CrossEntropyLoss(weight=class_weights_train)
        loss_val = nn.CrossEntropyLoss(weight=class_weights_val)
        
    dataloader_val = DataLoader(
            dataset_val,
            batch_size=config_seg.BATCH_SIZE,
            shuffle=True,
            num_workers=config_seg.NUM_WORKERS,
            pin_memory=True
    )
    
    for epoch in tqdm(range(config_seg.NUM_EPOCHS)):
        model.train()
        dataset_train.set_swap_indices()
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=config_seg.BATCH_SIZE,
            shuffle=True,
            num_workers=config_seg.NUM_WORKERS,
            pin_memory=True
        )
        train_fn(model, dataloader_train, opt, loss_train, scaler, True, epoch, writer_train, num_classes, unavailable_class_indices, with_generated_imgs, scheduler_train, DEVICE, fold_videos, LOVASZ_LOSS, custom_label)
        if ((epoch + 1) % 10 == 0) or epoch == 0:
            model.eval()
            with torch.no_grad():
                train_fn(model, dataloader_val, opt, loss_val, scaler, False, epoch, writer_val, num_classes, unavailable_class_indices, with_generated_imgs, None, DEVICE, fold_videos, LOVASZ_LOSS, custom_label)
    
def solve_segmentation(videos_train, videos_val, swap, DEVICE, fold_video, LOVASZ_LOSS, custom_label):
    num_classes = config_seg.NUM_CLASSES
    swap_label = "without_gen"
    if swap:
        swap_label = "with_gen"
    writer_train = SummaryWriter(f"{config_seg.SAVE_TENSORBAORD_PATH}{custom_label}/train/{LOVASZ_LOSS}/{config_seg.NUM_EPOCHS}/fold_{fold_video}/{swap_label}")
    writer_val = SummaryWriter(f"{config_seg.SAVE_TENSORBAORD_PATH}{custom_label}/val/{LOVASZ_LOSS}/{config_seg.NUM_EPOCHS}/fold_{fold_video}/{swap_label}")

    dataset_train = DatasetSegSmokeSwap(
        root_images=f"{config_seg.ROOT_PATH_IMG}",
        root_masks=f"{config_seg.ROOT_PATH_MASKS}",
        root_generated=f"{config_seg.ROOT_PATH_GENERATED_IMGS}fold_{fold_video}/epoch{config_seg.EPOCH_GENIMG}/Videos_25fps",
        videos=videos_train, 
        transform_image=config_seg.transformimage,
        transform_mask=config_seg.transformmask,
        with_swap=swap,
        ratio=config_seg.SWAP_RATIO
    )    

    # /data/home/rim36739/disk/saved_old_results/CycleGan/Correct_Baseline_22_2_23/Generated_images/fold_0003_148/Smoked_frames_lr_disc0.0001_LR_gen_0.0001_epoch20
    dataset_val = DatasetSegSmokeSwap(
        root_images=f"{config_seg.ROOT_PATH_IMG}",
        root_masks=f"{config_seg.ROOT_PATH_MASKS}",
        root_generated=f"{config_seg.ROOT_PATH_GENERATED_IMGS}fold_{fold_video}/epoch{config_seg.EPOCH_GENIMG}/Videos_25fps",
        videos=videos_val, 
        transform_image=config_seg.transformimage,
        transform_mask=config_seg.transformmask,
        with_swap=False
    )

    loader_train = DataLoader(
        dataset_train,
        batch_size=config_seg.BATCH_SIZE,
        shuffle=True,
        num_workers=config_seg.NUM_WORKERS,
        pin_memory=True
    )
    
    loader_val = DataLoader(
        dataset_val,
        batch_size=config_seg.BATCH_SIZE,
        shuffle=True,
        num_workers=config_seg.NUM_WORKERS,
        pin_memory=True
    )

    if not LOVASZ_LOSS:
        class_weights_train = class_calculations.get_class_weights(loader_train, num_classes, DEVICE, True)
        class_weights_val = class_calculations.get_class_weights(loader_val, num_classes, DEVICE, True)
    else:
        class_weights_train = None
        class_weights_val = None

    unavailable_class_indices = class_calculations.get_unavailable_class_indices(loader_train, loader_val, num_classes, DEVICE, True)
    print(unavailable_class_indices)

    #train and validate with normal images
    train_validate(dataset_train , dataset_val, loader_train, writer_train, writer_val, num_classes, unavailable_class_indices, class_weights_train, class_weights_val, swap, DEVICE, fold_video, LOVASZ_LOSS, custom_label)

    writer_train.flush() 
    writer_val.flush()
    
def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    print("cuda" if torch.cuda.is_available() else "cpu")

    video_fold_idx = sys.argv[1]
    with_img= sys.argv[2] == "True"
    
    custom_label = config_seg.CUSTOM_LABEL 
    LOVASZ_LOSS = config_seg.LOVASZ_LOSS
    videos_all = config_seg.ALL_VIDEO_NAMES

    video = videos_all[int(video_fold_idx)]
    videos_fold = [x for x in videos_all if x != video]

    print(f"segmentation with {videos_fold} validation with {video}, generated_imgs = {with_img}:")
    solve_segmentation(videos_fold, [video], with_img, config_seg.DEVICE, video, LOVASZ_LOSS, custom_label)

if __name__ == "__main__":
    main()

"""
pyhton segmentationtrain.py video gpu with_image lovasz 


"""