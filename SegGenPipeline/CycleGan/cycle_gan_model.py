import sys
sys.path.append('..')
from SegGenPipeline import config_seg_gan as config_cyclegan
import torch
from utils import initialize_conv_weights_normal, denormalize_manual, normalize_manual
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torchvision.utils import save_image
from .discriminator_model import Discriminator
from .generator_model import Generator
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import ConcatDataset
import os
from .image_buffer import ImageBuffer
from PIL import Image
from .average_meter import AverageMeter
from torchmetrics import StructuralSimilarityIndexMeasure
from skimage.util import random_noise
import torch.nn.utils as utils_torch


class CycleGANModel():
    # load_paths: {"G_NS", "G_HS", "D_NS", "D_HS"}
    def __init__(self, dataloader, with_metric=True, load_model=False, DEVICE=None, load_paths=None, custom_label="standard"):
        torch.manual_seed(0)
        random.seed(0)
        np.random.seed(0)
        self.custom_label = custom_label
        
        self.DEVICE = DEVICE
        
        self.with_metric = with_metric
        
        self.L1 = nn.L1Loss()
        self.g_loss = nn.MSELoss()
        self.d_loss = nn.MSELoss()

        print("cuda" if torch.cuda.is_available() else "cpu")
        self.disc_HS = Discriminator(in_channels=3).to(DEVICE)
        self.disc_NS = Discriminator(in_channels=3).to(DEVICE)
        self.gen_NS = Generator(img_channels=3, num_residuals=9).to(DEVICE)
        self.gen_HS = Generator(img_channels=3, num_residuals=9).to(DEVICE)
        
        if load_model:
            checkpoint_G_NS = torch.load(load_paths["G_NS"], map_location=DEVICE)
            checkpoint_G_HS = torch.load(load_paths["G_HS"], map_location=DEVICE)
            checkpoint_D_NS = torch.load(load_paths["D_NS"], map_location=DEVICE)
            checkpoint_D_HS = torch.load(load_paths["D_HS"], map_location=DEVICE)
            
            self.gen_NS.load_state_dict(checkpoint_G_NS["state_dict"])
            self.gen_HS.load_state_dict(checkpoint_G_HS["state_dict"])
            self.disc_NS.load_state_dict(checkpoint_D_NS["state_dict"])
            self.disc_HS.load_state_dict(checkpoint_D_HS["state_dict"])
        else:
            # initialize weights like in paper with gaussian 0.02
            self.gen_NS.apply(initialize_conv_weights_normal)
            self.gen_HS.apply(initialize_conv_weights_normal)
            self.disc_HS.apply(initialize_conv_weights_normal)
            self.disc_NS.apply(initialize_conv_weights_normal)
        
        
        self.opt_disc = optim.Adam(
            list(self.disc_HS.parameters()) + list(self.disc_NS.parameters()),
            lr=config_cyclegan.LEARNING_RATE_DISC,
            betas=(0.5, 0.999),
        )

        self.opt_gen = optim.Adam(
            list(self.gen_NS.parameters()) + list(self.gen_HS.parameters()),
            lr=config_cyclegan.LEARNING_RATE_GEN,
            betas=(0.5, 0.999),
        )
            
        #for training float16
        self.g_scaler = torch.cuda.amp.GradScaler()
        self.d_scaler = torch.cuda.amp.GradScaler()

        self.G_loss_meter = AverageMeter()
        self.loss_G_NS_meter = AverageMeter()
        self.loss_G_HS_meter = AverageMeter()
        self.loss_D_NS_meter = AverageMeter()
        self.loss_D_HS_meter = AverageMeter()
        self.cycle_no_smoke_loss_meter = AverageMeter()
        self.cycle_heavy_smoke_loss_meter = AverageMeter()
        self.identity_heavy_smoke_loss_meter = AverageMeter()
        self.identity_no_smoke_loss_meter = AverageMeter()
        self.ssim_meter = AverageMeter()
        
        """self.img_buffer_no_smoke = ImageBuffer(config_cyclegan.BUFFER_SIZE)
        self.img_buffer_heavy_smoke = ImageBuffer(config_cyclegan.BUFFER_SIZE)"""
        
        if with_metric: 
            self.fid_hs = FrechetInceptionDistance()
            self.fid_hs = self.fid_hs.to(DEVICE)
            self.ssim = StructuralSimilarityIndexMeasure().to(DEVICE)
            
    def train_step_gen(self, batch, writer, is_train, fold, epoch, step, with_backprop_gen=True):
        
        (no_smoke_imgs, heavy_smoke_imgs) = batch
        no_smoke = no_smoke_imgs['img']
        heavy_smoke = heavy_smoke_imgs['img']
            
        # Train Generators
        self.opt_gen.zero_grad()
        # autocast for float16
        with torch.cuda.amp.autocast(enabled=False):
            no_smoke = no_smoke.to(self.DEVICE)
            heavy_smoke = heavy_smoke.to(self.DEVICE)           
                
            fake_heavy_smoke = self.gen_HS(no_smoke)
            fake_no_smoke = self.gen_NS(heavy_smoke)

            # adversarial loss for both generators
            D_HS_fake = self.disc_HS(fake_heavy_smoke)
            D_NS_fake = self.disc_NS(fake_no_smoke)
            loss_G_HS = self.g_loss(D_HS_fake, torch.ones_like(D_HS_fake))
            loss_G_NS = self.g_loss(D_NS_fake, torch.ones_like(D_NS_fake))

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_no_smoke = self.gen_NS(no_smoke)
            identity_heavy_smoke = self.gen_HS(heavy_smoke)
            
            identity_no_smoke_loss = self.L1(no_smoke, identity_no_smoke)
            identity_heavy_smoke_loss = self.L1(heavy_smoke, identity_heavy_smoke)

            cycle_no_smoke = self.gen_NS(fake_heavy_smoke)
            cycle_heavy_smoke = self.gen_HS(fake_no_smoke)
            cycle_no_smoke_loss = self.L1(no_smoke, cycle_no_smoke)
            cycle_heavy_smoke_loss = self.L1(heavy_smoke, cycle_heavy_smoke)
                
            # add all together
            G_loss = (
                loss_G_NS
                + loss_G_HS
                + cycle_no_smoke_loss * config_cyclegan.LAMBDA_CYCLE
                + cycle_heavy_smoke_loss * config_cyclegan.LAMBDA_CYCLE
                + identity_heavy_smoke_loss * config_cyclegan.LAMBDA_IDENTITY
                + identity_no_smoke_loss * config_cyclegan.LAMBDA_IDENTITY
            )
            
        if is_train and with_backprop_gen:
            
            self.g_scaler.scale(G_loss).backward(retain_graph=False)
            self.g_scaler.step(self.opt_gen)
            self.g_scaler.update()
            
        with torch.no_grad():
            # plotting
            writer.add_scalar("cycle_loss/no_smoke", cycle_no_smoke_loss, step)
            writer.add_scalar("cycle_loss/heavy_smoke", cycle_heavy_smoke_loss, step)
            writer.add_scalar("identity_loss/heavy_smoke", identity_heavy_smoke_loss, step)
            writer.add_scalar("identity_loss/no_smoke", identity_no_smoke_loss, step)
            writer.add_scalar("generator_loss/no_smoke", loss_G_NS, step)
            writer.add_scalar("generator_loss/heavy_smoke", loss_G_HS, step)
            writer.add_scalar("G_loss", G_loss, step)
            #writer.add_scalar("learning_rate", self.lr_scheduler_G.get_last_lr()[0], step)
            #for loss mean
            self.G_loss_meter.update(G_loss, config_cyclegan.BATCH_SIZE)
            self.loss_G_NS_meter.update(loss_G_NS, config_cyclegan.BATCH_SIZE)
            self.loss_G_HS_meter.update(loss_G_HS, config_cyclegan.BATCH_SIZE)
            self.cycle_no_smoke_loss_meter.update(cycle_no_smoke_loss, config_cyclegan.BATCH_SIZE)
            self.cycle_heavy_smoke_loss_meter.update(cycle_heavy_smoke_loss, config_cyclegan.BATCH_SIZE)
            self.identity_heavy_smoke_loss_meter.update(identity_heavy_smoke_loss, config_cyclegan.BATCH_SIZE)
            self.identity_no_smoke_loss_meter.update(identity_no_smoke_loss, config_cyclegan.BATCH_SIZE)
        
        discriminator_data = (heavy_smoke, fake_heavy_smoke, no_smoke, fake_no_smoke, no_smoke_imgs, cycle_no_smoke)
        return {"fake_HS": fake_heavy_smoke, "g_loss": G_loss, "d_data": discriminator_data}
        

    def train_step_disc(self, discriminator_data, is_train, writer, step, epoch, fold):
        # hand over triples for frame number memorization
        # original_hs_videos => [video, video, ...] (same with frames) for imgs in batch (changed due to swap in buffer)

        (heavy_smoke, fake_heavy_smoke, no_smoke, fake_no_smoke, no_smoke_data, cycle_no_smoke) = discriminator_data
        
        result = (fake_heavy_smoke > 1.0).any()
        result_2 = (fake_heavy_smoke < -1.0).any()

        if result or result_2:
            print("Fail")

        # Train Discriminators
        self.opt_disc.zero_grad()

        with torch.cuda.amp.autocast(enabled=False):

            D_HS_real = self.disc_HS(heavy_smoke)
            D_HS_fake = self.disc_HS(fake_heavy_smoke.detach())
            #D_HS_fake = self.disc_HS(fake_heavy_smoke_buffer.detach())
            D_HS_real_loss = self.d_loss(D_HS_real, torch.ones_like(D_HS_real))
            D_HS_fake_loss = self.d_loss(D_HS_fake, torch.zeros_like(D_HS_fake))
            D_HS_loss = (D_HS_real_loss + D_HS_fake_loss) / 2

            D_NS_real = self.disc_NS(no_smoke)
            D_NS_fake = self.disc_NS(fake_no_smoke.detach())
            D_NS_real_loss = self.d_loss(D_NS_real, torch.ones_like(D_NS_real))
            D_NS_fake_loss = self.d_loss(D_NS_fake, torch.zeros_like(D_NS_fake))
            D_NS_loss = (D_NS_real_loss + D_NS_fake_loss) / 2

            # put it together
            D_loss = (D_HS_loss + D_NS_loss)/2

        if is_train:
            self.d_scaler.scale(D_loss).backward()
            self.d_scaler.step(self.opt_disc)
            self.d_scaler.update()
            #self.lr_scheduler_D.step()
            
        with torch.no_grad():
            # plotting
            writer.add_scalar("discriminator_loss/heavy_smoke", D_HS_loss, step)
            writer.add_scalar("discriminator_loss/no_smoke", D_NS_loss, step)
            writer.add_scalar("discriminator_loss/heavy_smoke_real", D_HS_real_loss, step)
            writer.add_scalar("discriminator_loss/heavy_smoke_fake", D_HS_fake_loss, step)
            writer.add_scalar("discriminator_loss/no_smoke_real", D_NS_real_loss, step)
            writer.add_scalar("discriminator_loss/no_smoke_fake", D_NS_fake_loss, step)
            #for loss mean
            self.loss_D_NS_meter.update(D_NS_loss, config_cyclegan.BATCH_SIZE)
            self.loss_D_HS_meter.update(D_HS_loss, config_cyclegan.BATCH_SIZE)
            # save images to file and calculate metrics 
                
        with torch.no_grad():      
            # ITERATE OVER EACH IMG IN BATCH
            imgs_this_batch = fake_heavy_smoke.size(0)
            for idx in range(imgs_this_batch):
                ns_video = no_smoke_data['ns_video'][idx]
                ns_frame = no_smoke_data['ns_frame'][idx]
                if is_train and (int(ns_frame) % 25 == 0) and ((epoch + 1) % config_cyclegan.SAVE_METRICS_EPOCH == 0):
                    frame_directory_path = f"{config_cyclegan.OUTPUT_PATH_IMGS_CYCLE}/{self.custom_label}/fold_{fold}/Smoked_frames_lr_disc{config_cyclegan.LEARNING_RATE_DISC}_LR_gen_{config_cyclegan.LEARNING_RATE_GEN}_epoch{epoch+1}/Videos_25fps/{ns_video}/{self.framesegment_from_frame(ns_frame)}"
                    os.makedirs(frame_directory_path, exist_ok=True)
                    frame_path = os.path.join(frame_directory_path, f"f_{ns_frame}.png")
                    
                    mean = config_cyclegan.mean_bounded_norm
                    std = config_cyclegan.std_bounded_norm
                    fake_heavy_denorm = denormalize_manual(fake_heavy_smoke[idx], mean, std, 255)
                    img = fake_heavy_denorm.cpu().detach()
                    img = np.array(img)                    
                    img = np.transpose(img, (1, 2, 0))
                    img = (img).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(frame_path, compress_level=0)
                    
                    frame_path = os.path.join(frame_directory_path, f"f_{ns_frame}_cycle.png")
                    
                    fake_no_smoke_denorm = denormalize_manual(cycle_no_smoke[idx], mean, std, 255)
                    img = fake_no_smoke_denorm.cpu().detach()
                    img = np.array(img)                    
                    img = np.transpose(img, (1, 2, 0))
                    img = (img).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(frame_path, compress_level=0)
                
                    #fid and ssim
                    if self.with_metric:
                        mean = config_cyclegan.mean_bounded_norm
                        std = config_cyclegan.std_bounded_norm
                        heavy_denorm = denormalize_manual(heavy_smoke[idx], mean, std, 255)
                        self.fid_hs.update(heavy_denorm.type(torch.uint8).unsqueeze(0), real=True)
                        self.fid_hs.update(fake_heavy_denorm.type(torch.uint8).unsqueeze(0), real=False)
                        
                        ssim_hs = self.ssim(heavy_denorm.unsqueeze(0), fake_heavy_denorm.unsqueeze(0))
                        self.ssim_meter.update(ssim_hs)
        
    def backprop_gen(self, g_loss):
        self.g_scaler.scale(g_loss).backward()
        self.g_scaler.step(self.opt_gen)
        self.g_scaler.update()
                
    def framesegment_from_frame(self, frame):
        return str((int(frame) // 1000) * 1000)

    def calculate_metrics_reset_buffer(self, writer_epoch_end, epoch):
        if self.with_metric and ((epoch + 1) % config_cyclegan.SAVE_METRICS_EPOCH == 0):
            with torch.no_grad():
                fid_computed_hs_smoke = self.fid_hs.compute()
                writer_epoch_end.add_scalar("fid/heavy_smoke", fid_computed_hs_smoke, (epoch))
                writer_epoch_end.add_scalar("avg/ssim", self.ssim_meter.avg, (epoch + 1))
        writer_epoch_end.add_scalar("avg/discriminator_loss/no_smoke", self.loss_D_NS_meter.avg * 2, (epoch))
        writer_epoch_end.add_scalar("avg/discriminator_loss/heavy_smoke", self.loss_D_HS_meter.avg * 2, (epoch))
        writer_epoch_end.add_scalar("avg/cycle_loss/no_smoke", self.cycle_no_smoke_loss_meter.avg, (epoch))
        writer_epoch_end.add_scalar("avg/cycle_loss/heavy_smoke", self.cycle_heavy_smoke_loss_meter.avg, (epoch))
        writer_epoch_end.add_scalar("avg/identity_loss/heavy_smoke", self.identity_heavy_smoke_loss_meter.avg, (epoch))
        writer_epoch_end.add_scalar("avg/identity_loss/no_smoke", self.identity_no_smoke_loss_meter.avg, (epoch))
        writer_epoch_end.add_scalar("avg/generator_loss/no_smoke", self.loss_G_NS_meter.avg, (epoch))
        writer_epoch_end.add_scalar("avg/generator_loss/heavy_smoke", self.loss_G_HS_meter.avg, (epoch))
        writer_epoch_end.add_scalar("avg/G_loss", self.G_loss_meter.avg, (epoch))
        
        self.G_loss_meter = AverageMeter()
        self.loss_G_NS_meter = AverageMeter()
        self.loss_G_HS_meter = AverageMeter()
        self.loss_D_NS_meter = AverageMeter()
        self.loss_D_HS_meter = AverageMeter()
        self.cycle_no_smoke_loss_meter = AverageMeter()
        self.cycle_heavy_smoke_loss_meter = AverageMeter()
        self.identity_heavy_smoke_loss_meter = AverageMeter()
        self.identity_no_smoke_loss_meter = AverageMeter()
        self.ssim_meter = AverageMeter()
        
        """self.img_buffer_no_smoke = ImageBuffer(config_cyclegan.BUFFER_SIZE)
        self.img_buffer_heavy_smoke = ImageBuffer(config_cyclegan.BUFFER_SIZE)"""
        
        if self.with_metric and ((epoch + 1) % config_cyclegan.SAVE_METRICS_EPOCH == 0): 
            self.fid_hs = FrechetInceptionDistance()
            self.fid_hs = self.fid_hs.to(self.DEVICE)
            self.ssim = StructuralSimilarityIndexMeasure().to(self.DEVICE)
            