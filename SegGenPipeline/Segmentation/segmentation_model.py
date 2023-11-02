import torch
import sys
sys.path.append('..')
from .utils import save_mask_grayscale_as_img
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from SegGenPipeline import config_seg_gan as config_seg
from tqdm import tqdm
from torchvision.utils import save_image
import random
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import Accuracy, JaccardIndex, Specificity, Recall, ConfusionMatrix
from .network import modeling as modeling
from . import metric_calculation
from . import class_calculations
from .average_meter import AverageMeter
from torch.utils.data import ConcatDataset
import time
import torchvision.transforms as transforms
from segmentation_models_pytorch.losses import LovaszLoss
import os
from .utils import save_checkpoint
import torch.nn.utils as utils_torch
import copy

class SegmentationModel():
    
    def __init__(self, num_classes, loader_train, loader_val, DEVICE, load_path=None, class_weights_train=None, class_weights_val=None, load_model=True, training_steps=config_seg.BATCH_SIZE, cusotom_label="standard"):
        #def train_validate(dataloader_train, dataloader_val, writer_train, writer_val, num_classes, unavailable_class_indices, class_weights, with_generated_imgs):
        ## Model taken from https://github.com/VainF/DeepLabV3Plus-Pytorch.
        self.cusotom_label = cusotom_label
        self.DEVICE = DEVICE
        self.model = modeling.deeplabv3plus_resnet101(num_classes=19, output_stride=1, pretrained_backbone=True)
        self.num_classes = num_classes
        self.last_layer = nn.Conv2d(
            in_channels=256,
            out_channels=num_classes + 1,
            kernel_size=1,
            stride=1
        )
        self.model.load_state_dict( torch.load(config_seg.LOAD_SEG_MODEL_BASE)['model_state'])
        self.model.classifier.classifier[3] = self.last_layer

        if load_model:
            checkpoint = torch.load(load_path, map_location=DEVICE)
            self.model.load_state_dict(checkpoint["model_sd"])

        self.model = self.model.to(self.DEVICE)
        self.scaler = torch.cuda.amp.GradScaler()

        self.opt = optim.Adam(
            self.model.parameters(),
            lr=config_seg.LEARNING_RATE_SEG
        )

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=self.opt, max_lr=config_seg.LEARNING_RATE_SEG, epochs=50, steps_per_epoch=len(loader_train), pct_start=0.05)
        if load_model:
            self.opt.load_state_dict(checkpoint["optimizer_sd"])
            self.scheduler.load_state_dict(checkpoint["scheduler_sd"])
            
        #loss
        if config_seg.LOVASZ_LOSS:
            self.loss_train = LovaszLoss('multiclass', per_image=False, from_logits=True)
            self.loss_val  = LovaszLoss('multiclass', per_image=False, from_logits=True)
        else:
            self.loss_train = nn.CrossEntropyLoss(weight=class_weights_train)
            self.loss_val = nn.CrossEntropyLoss(weight=class_weights_val)

        self.global_conf_mat_sup = None
        self.global_conf_mat_NS_sup = None
        self.global_conf_mat_MS_sup = None
        self.global_conf_mat_HS_sup = None
        self.loss_meter_sup = AverageMeter()
        self.global_conf_mat_unsup = None
        self.global_conf_mat_NS_unsup = None
        self.global_conf_mat_MS_unsup = None
        self.global_conf_mat_HS_unsup = None
        self.loss_meter_unsup = AverageMeter()

    def get_model_par(self):
        return self.model.parameters()
    
    def set_train(self):
        self.model.train()

    def set_eval(self):
        self.model.eval()
        
    ########################################################
    # functions for training one step at a time
    # call compute loss metric after each epoch for metrics
    ########################################################
    def train_step(self, batch, is_train, epoch, fold_video, supervised=True, retain=False, just_pred=False):

        self.opt.zero_grad()
        
        if is_train or just_pred:
            self.model.train()
        else:
            self.model.eval()        
        (image, mask, video_name, frame_n, domain) = batch
        with torch.cuda.amp.autocast(enabled=False):
            image = image.float().to(self.DEVICE)
            mask = mask.long().to(self.DEVICE)
            conf_mat = ConfusionMatrix(task="multiclass", num_classes=self.num_classes + 1).to(self.DEVICE)
            outputs = self.model(image)
            
            # return just the prediction without tarining the weights
            if just_pred:
                return outputs
            mask = mask.squeeze()
            loss = self.loss_train(outputs, mask)

        if is_train:
            self.scaler.scale(loss).backward(retain_graph=retain)
            self.scaler.step(self.opt)
            self.scaler.update()           
            self.scheduler.step()
        
        # Apply softmax
        output_sm = torch.nn.functional.softmax(outputs, dim=1)
        output_preds = torch.argmax(output_sm, dim=1)
            
        self.loss_meter_sup.update(loss, n=config_seg.BATCH_SIZE_SEG)
        
        for idx, pred in enumerate(output_preds):
            #print(f"mask {mask[idx].shape}")
            conf_m = conf_mat(torch.flatten(pred), torch.flatten(mask[idx]))
            
            self.add_metrics(conf_m, domain[idx], supervised)  
            
            if not is_train:
                save_mask_grayscale_as_img(pred, f"{config_seg.SAVE_SEG_MASK_PATH}{self.cusotom_label}/val/with_gen/{fold_video}/{epoch}/sup{supervised}/{video_name[idx]}", frame_n[idx])
        
        return outputs
        
    #######
    # call this after each epoch
    #######
    def calculate_metrics(self, writer_sup, writer_unsup, epoch, unavailable_class_indices, is_train):
        ua = unavailable_class_indices["train"] if is_train else unavailable_class_indices["val"]
        if self.global_conf_mat_NS_sup is not None:
            self.write_metrics(writer_sup, self.global_conf_mat_NS_sup, ua, self.num_classes, self.loss_meter_sup, epoch, "no_smoke_domain")
        if self.global_conf_mat_MS_sup is not None:
            self.write_metrics(writer_sup, self.global_conf_mat_MS_sup, ua, self.num_classes, self.loss_meter_sup, epoch, "slightly_smoked_domain")
        if self.global_conf_mat_HS_sup is not None:
            self.write_metrics(writer_sup, self.global_conf_mat_HS_sup, ua, self.num_classes, self.loss_meter_sup, epoch, "heavy_smoke_domain")
        if self.global_conf_mat_sup is not None:
            self.write_metrics(writer_sup, self.global_conf_mat_sup, ua, self.num_classes, self.loss_meter_sup, epoch, "all_domains")
        if self.global_conf_mat_NS_unsup is not None:
            self.write_metrics(writer_unsup, self.global_conf_mat_NS_unsup, ua, self.num_classes, self.loss_meter_unsup, epoch, "no_smoke_domain")
        if self.global_conf_mat_MS_unsup is not None:
            self.write_metrics(writer_unsup, self.global_conf_mat_MS_unsup, ua, self.num_classes, self.loss_meter_unsup, epoch, "slightly_smoked_domain")
        if self.global_conf_mat_HS_unsup is not None:
            self.write_metrics(writer_unsup, self.global_conf_mat_HS_unsup, ua, self.num_classes, self.loss_meter_unsup, epoch, "heavy_smoke_domain")
        if self.global_conf_mat_unsup is not None:
            self.write_metrics(writer_unsup, self.global_conf_mat_unsup, ua, self.num_classes, self.loss_meter_unsup, epoch, "all_domains")
        self.global_conf_mat_sup = None
        self.global_conf_mat_NS_sup = None
        self.global_conf_mat_MS_sup = None
        self.global_conf_mat_HS_sup = None
        self.loss_meter_sup = AverageMeter()
        self.global_conf_mat_unsup = None
        self.global_conf_mat_NS_unsup = None
        self.global_conf_mat_MS_unsup = None
        self.global_conf_mat_HS_unsup = None
        self.loss_meter_unsup = AverageMeter()
        
    def add_metrics(self, conf_m, domain, supervised):
        if supervised:
            if domain == "not_smoked":
                self.global_conf_mat_NS_sup = self.add_to_mat(conf_m, self.global_conf_mat_NS_sup)
            elif domain == "slightly_smoked":
                self.global_conf_mat_MS_sup = self.add_to_mat(conf_m, self.global_conf_mat_MS_sup)
            elif domain == "heavily_smoked":
                self.global_conf_mat_HS_sup = self.add_to_mat(conf_m, self.global_conf_mat_HS_sup)
            else: 
                print("domain_error")
            self.global_conf_mat_sup = self.add_to_mat(conf_m, self.global_conf_mat_sup)
        else:
            if domain == "not_smoked":
                self.global_conf_mat_NS_unsup = self.add_to_mat(conf_m, self.global_conf_mat_NS_unsup)
            elif domain == "slightly_smoked":
                self.global_conf_mat_MS_unsup = self.add_to_mat(conf_m, self.global_conf_mat_MS_unsup)
            elif domain == "heavily_smoked":
                self.global_conf_mat_HS_unsup = self.add_to_mat(conf_m, self.global_conf_mat_HS_unsup)
            else: 
                print("domain_error")
            self.global_conf_mat_unsup = self.add_to_mat(conf_m, self.global_conf_mat_unsup)

    def add_to_mat(self, add_m, glob_m):            
        if(pd.isna(glob_m)):
            return copy.deepcopy(add_m)
        else:
            glob_m += add_m
            return glob_m

    # call this after each epoch
    def write_metrics(self, writer, global_conf_mat, unavailable_class_indices, num_classes, loss_meter, epoch, domain_label):
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
    