import sys
from CycleGan.cycle_gan_model import CycleGANModel
from PIL import Image
from Segmentation.segmentation_model import SegmentationModel
from torchmetrics import Accuracy, JaccardIndex, Specificity, Recall, ConfusionMatrix
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config_seg_gan
from tqdm import tqdm
from torchvision.utils import save_image
import random
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics import Dice
import metric_calculation
import class_calculations
from average_meter import AverageMeter
from torch.utils.data import ConcatDataset
import time
import torchvision.transforms as transforms
from segmentation_models_pytorch.losses import LovaszLoss
from dataset_seg_gan import DatasetSegGan
from utils import normalize_manual, denormalize_manual, normalize_tensor, scale_max_pixel, save_image
import torch.nn.functional as F
from dataset_val import DatasetVal
from gen_buffer import GenBuffer
import os

class SegmentationGan():
    
    def __init__(self, videos_train, videos_val, DEVICE, fold, with_gan=True, seg_iter_per_step=4, custom_label="standard", segment_only=False):
        
        self.DEVICE = DEVICE
        self.with_gan = with_gan
        self.segment_only = segment_only

        self.writer_train_seggan = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/train/advers")
        self.writer_val_seggan = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/val/advers")

        self.writer_cycle_step_train = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/train/cycle_step") if self.with_gan else None
        self.writer_cycle_step_val = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/val/cycle_step") if self.with_gan else None
        self.writer_cycle_epoch_train = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/train/cycle_epoch") if self.with_gan else None
        self.writer_cycle_epoch_val = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/val/cycle_epoch") if self.with_gan else None
        self.loss_meter_penalty = AverageMeter() if self.with_gan else None
        self.loss_meter_g_loss = AverageMeter() if self.with_gan else None
    
        self.writer_segmentation_unsup_train = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/train/segmentation_unsup")
        self.writer_segmentation_unsup_val = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/val/segmentation_unsup")
        self.writer_segmentation_sup_train = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/train/segmentation_sup")
        self.writer_segmentation_sup_val = SummaryWriter(f"{config_seg_gan.TENSORBOARD_PATH}{custom_label}/with_cycle{with_gan}/{fold}/val/segmentation_sup")    
        
        dataset_train = DatasetVal(
            root_images=config_seg_gan.IMAGES_ROOT_PATH,
            root_masks=config_seg_gan.MASK_ROOT_PATH,
            videos=videos_train, 
            transform_image_z_norm=config_seg_gan.transformimage_znorm,
            tranform_image_bounded=config_seg_gan.transformimage_nbounded,
            transform_mask=config_seg_gan.transformmask,
        )

        dataset_val = DatasetVal(
            root_images=config_seg_gan.IMAGES_ROOT_PATH,
            root_masks=config_seg_gan.MASK_ROOT_PATH,
            videos=videos_val, 
            transform_image_z_norm=config_seg_gan.transformimage_znorm,
            tranform_image_bounded=config_seg_gan.transformimage_nbounded,
            transform_mask=config_seg_gan.transformmask,
        )
        
        self.loader_train = DataLoader(
            dataset_train,
            batch_size=config_seg_gan.BATCH_SIZE_SEG,
            shuffle=True,
            num_workers=config_seg_gan.NUM_WORKERS,
            pin_memory=True
        )
        
        self.loader_val = DataLoader(
            dataset_val,
            batch_size=config_seg_gan.BATCH_SIZE_SEG,
            shuffle=True,
            num_workers=config_seg_gan.NUM_WORKERS,
            pin_memory=True
        )

        self.gen_buffer = GenBuffer(config_seg_gan.BATCH_SIZE_SEG, self.with_gan, skip_number=3)
        
        self.loss = LovaszLoss('multiclass', per_image=False, from_logits=True)

        """LOAD_PATH_SEG = f"/data/home/rim36739/disk/saved_models/segmentation/BL_KLEIN_SEGGEN/{fold}/20/with_gen_False/lov_lossTrue/seg_model.pth.tar"
        LOAD_PATH_CYCLE = {"G_NS":f"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/{fold}/19/gen_ns.pth.tar", 
                   "G_HS":f"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/{fold}/19/gen_hs.pth.tar", 
                   "D_NS":f"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/{fold}/19/disc_ns.pth.tar", 
                   "D_HS":f"/data/home/rim36739/disk/saved_old_results/CycleGan/Cycle_small_images/{fold}/19/disc_hs.pth.tar"}"""
        LOAD_PATH_SEG = config_seg_gan.LOAD_PATH_SEG
        LOAD_PATH_CYCLE = config_seg_gan.LOAD_PATH_CYCLE

        self.unavailable_class_indices = class_calculations.get_unavailable_class_indices(self.loader_train, self.loader_val, config_seg_gan.NUM_CLASSES, DEVICE, True)
        self.seg_model = SegmentationModel(config_seg_gan.NUM_CLASSES, self.loader_train, self.loader_val, DEVICE, load_model=config_seg_gan.LOAD_MODEL, load_path=LOAD_PATH_SEG,training_steps=config_seg_gan.BATCH_SIZE_SEG if self.segment_only else config_seg_gan.BATCH_SIZE, cusotom_label=custom_label)
        
        if self.with_gan and not segment_only:
            self.cycle_model = CycleGANModel(self.loader_train, with_metric=True,load_model=True, DEVICE=DEVICE, load_paths=LOAD_PATH_CYCLE, custom_label=custom_label)
        

    def train(self, fold_name):
        for epoch in tqdm(range(config_seg_gan.NUM_EPOCHS)):
            if self.with_gan:
                self.loss_meter_penalty = AverageMeter()
                self.loss_meter_g_loss = AverageMeter()
            self.seg_model.set_train()
            self.train_epoch(self.loader_train, True, epoch, fold_name, self.writer_cycle_step_train)
            # calculate and reset the metrics for each epoch
            self.seg_model.calculate_metrics(self.writer_segmentation_sup_train, self.writer_segmentation_unsup_train, epoch, self.unavailable_class_indices, is_train=True)
            if self.with_gan and not self.segment_only:
                self.cycle_model.calculate_metrics_reset_buffer(self.writer_cycle_epoch_train, epoch)
                self.writer_train_seggan.add_scalar("loss_penalty", self.loss_meter_penalty.avg, epoch)
                self.writer_train_seggan.add_scalar("loss_g", self.loss_meter_g_loss.avg, epoch)
            
            if (epoch + 1) % config_seg_gan.EVAL_EPOCH == 0:
                self.loss_meter_penalty = AverageMeter() if self.with_gan else None
                self.loss_meter_g_loss = AverageMeter() if self.with_gan else None
                self.seg_model.set_eval()
                with torch.no_grad():
                    self.train_epoch(self.loader_val, is_train=False, epoch=epoch, fold=fold_name, writer_cycle_step=self.writer_cycle_step_val)
                    # calculate and reset the metrics for val
                    self.seg_model.calculate_metrics(self.writer_segmentation_sup_val, self.writer_segmentation_unsup_val, epoch, self.unavailable_class_indices, is_train=False)
        if self.with_gan:
            self.writer_train_seggan.flush()
            self.writer_val_seggan.flush()
            self.writer_cycle_step_train.flush()
            self.writer_cycle_step_val.flush()
            self.writer_cycle_epoch_train.flush()
            self.writer_cycle_epoch_val.flush()
        
        self.writer_segmentation_sup_train.flush()
        self.writer_segmentation_sup_val.flush()
        self.writer_segmentation_unsup_train.flush()
        self.writer_segmentation_unsup_val.flush()
        
    #"seg_info": {"mask_ns": data_ns_mask_list, "video": data_ns_video_list, "frame":data_ns_frame_list, "domain":data_ns_domain_list}
    def make_seg_data(self, buff_seg_info, fake_hs_img):
        fake_hs_denorm = scale_max_pixel(fake_hs_img, config_seg_gan.MAX_PIXEL_VAL)            
        img = normalize_tensor(fake_hs_denorm, config_seg_gan.mean_smokeset_hs_z, config_seg_gan.std_smokeset_hs_z, config_seg_gan.MAX_PIXEL_VAL)
        
        return (img, buff_seg_info["mask_ns"], buff_seg_info["video"], buff_seg_info["frame"], buff_seg_info["domain"])
            
    def train_epoch(self, dataloader, is_train, epoch, fold, writer_cycle_step):
        #torch.autograd.set_detect_anomaly(True)

        for idx, batch in enumerate(dataloader):
            
            data_bound_norm = batch["data_bound"]
            data_z_norm = batch["data_z"]

            if not is_train:
                self.seg_model.train_step(data_z_norm, False, epoch, fold, supervised=True)
                continue

            if self.segment_only:
                self.seg_model.train_step(data_z_norm, True, epoch, fold, True)
                continue
            
            step = len(dataloader) * (epoch) + idx

            buff_out = self.gen_buffer.push_pop(data_bound_norm if self.with_gan else data_z_norm)

            # skip normal seg training and instead use cycle data when buffer full
            if not buff_out["returned_cycle"]:
                self.seg_model.train_step(data_z_norm, is_train, epoch, fold, True)

            # Train with cyclegan on
            if buff_out["returned_cycle"] and is_train and self.with_gan:
                cycle_data = buff_out["cycle_data"]
                #"seg_info": {"mask_ns": data_ns_mask_list, "video": data_ns_video_list, "frame":data_ns_frame_list, "domain":data_ns_domain_list}
                buff_seg_info = buff_out["seg_info"]

                # generate fake data and loss for cyclegan 
                cycle_output = self.cycle_model.train_step_gen(cycle_data, writer_cycle_step, is_train, fold, epoch, step, False)
                fake_hs_img = cycle_output["fake_HS"]
                g_loss_cycle = cycle_output["g_loss"]
                discriminator_cycle_data = cycle_output["d_data"]
                
                seg_data_fake = self.make_seg_data(buff_seg_info, fake_hs_img)
                mask_fake_hs = buff_seg_info["mask_ns"]
                
                # TODO freeze weights instead of just_pred?
                # generate segmentation output without training
                output_seg_fake = self.seg_model.train_step(seg_data_fake, is_train, epoch, fold, True, retain=False, just_pred=True)
                
                # train the generator
                with torch.cuda.amp.autocast(enabled=False):
                    
                    mask_fake_hs = mask_fake_hs.to(device="cuda", dtype=torch.long)
                    loss_gen_penalty =  self.loss(output_seg_fake, mask_fake_hs)
                    
                    g_loss = g_loss_cycle - (loss_gen_penalty * config_seg_gan.LAMBDA_GEN_PENALTY)

                self.cycle_model.backprop_gen(g_loss)
                    
                self.loss_meter_g_loss.update(g_loss, config_seg_gan.BATCH_SIZE)
                self.loss_meter_penalty.update(loss_gen_penalty, config_seg_gan.BATCH_SIZE)
                # train the discriminator
                self.cycle_model.train_step_disc(discriminator_cycle_data, is_train, writer_cycle_step, step, epoch, fold)
                
                # Train segmentation network with fake image
                fake_hs_img_seg_train = self.make_seg_data(buff_seg_info, fake_hs_img.detach())
                self.seg_model.train_step(fake_hs_img_seg_train, is_train, epoch, fold)

            # Train with cyclegan off 
            if buff_out["returned_cycle"] and is_train and not self.with_gan and not self.segment_only:
                seg_batch_hs_buff = buff_out["batch_seg_hs"]
                self.seg_model.train_step(seg_batch_hs_buff, is_train, epoch, fold)

def main():
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    print("cuda" if torch.cuda.is_available() else "cpu")

    with_gan= True
    segment_only = False
    video_index = int(sys.argv[1])

    custom_label = config_seg_gan.CUSTOM_LABEL

    videos_all = config_seg_gan.VIDEOS_ALL
    video = videos_all[video_index]
    videos_fold = [x for x in videos_all if x != video]
    print(f"seg gan with {videos_fold} validation with {video} without generated images:")
    seg_gan = SegmentationGan(videos_fold, [video], "cuda", video, with_gan=with_gan, seg_iter_per_step=config_seg_gan.BATCH_SIZE, custom_label=custom_label, segment_only=segment_only)
    seg_gan.train(fold_name=video)

if __name__ == "__main__":
    main()
