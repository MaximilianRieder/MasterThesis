import torch
from datasetcyclegan import SmokeDataset
import sys
from utils import save_checkpoint, initialize_conv_weights_normal, denormalize_manual
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config_cyclegan as config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator_model import Discriminator
from generator_model import Generator
import random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchmetrics.image.fid import FrechetInceptionDistance
from torch.utils.data import ConcatDataset
import os
from image_buffer import ImageBuffer
from PIL import Image
from average_meter import AverageMeter
from torchmetrics import StructuralSimilarityIndexMeasure
from skimage.util import random_noise

# thousand place of frame number
def framesegment_from_frame(frame):
    return str((int(frame) // 1000) * 1000)

# original cycle gan https://arxiv.org/pdf/1703.10593.pdf
def train_fn(disc_HS, disc_NS, gen_NS, gen_HS, loader, opt_disc, opt_gen, l1, g_loss, d_loss, d_scaler, g_scaler, is_train, epoch, writer, with_metric, lr_scheduler_G, lr_scheduler_D, fold):

    G_loss_meter = AverageMeter()
    loss_G_NS_meter = AverageMeter()
    loss_G_HS_meter = AverageMeter()
    loss_D_NS_meter = AverageMeter()
    loss_D_HS_meter = AverageMeter()
    cycle_no_smoke_loss_meter = AverageMeter()
    cycle_heavy_smoke_loss_meter = AverageMeter()
    identity_heavy_smoke_loss_meter = AverageMeter()
    identity_no_smoke_loss_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    img_buffer_no_smoke = ImageBuffer(config.BUFFER_SIZE)
    img_buffer_heavy_smoke = ImageBuffer(config.BUFFER_SIZE)

    # initialize objects for metric
    if with_metric and ((epoch + 1) % config.SAVE_METRICS_EPOCH == 0):
        fid_hs = FrechetInceptionDistance()
        fid_hs = fid_hs.to(config.DEVICE)
        ssim = StructuralSimilarityIndexMeasure().to(config.DEVICE)
        
    for idx, (no_smoke_imgs, heavy_smoke_imgs) in enumerate(loader):
        no_smoke = no_smoke_imgs['ns_img']
        heavy_smoke = heavy_smoke_imgs['hs_img']

        step = len(loader) * (epoch) + idx
        
        ###########################
        # Train Generators
        ###########################
        
        opt_gen.zero_grad()

        with torch.cuda.amp.autocast():
            no_smoke = no_smoke.to(config.DEVICE)
            heavy_smoke = heavy_smoke.to(config.DEVICE)           
                
            fake_heavy_smoke = gen_HS(no_smoke)
            fake_no_smoke = gen_NS(heavy_smoke)

            # adversarial loss for both generators
            D_HS_fake = disc_HS(fake_heavy_smoke)
            D_NS_fake = disc_NS(fake_no_smoke)
            loss_G_HS = g_loss(D_HS_fake, torch.ones_like(D_HS_fake))
            loss_G_NS = g_loss(D_NS_fake, torch.ones_like(D_NS_fake))

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_no_smoke = gen_NS(no_smoke)
            identity_heavy_smoke = gen_HS(heavy_smoke)
            identity_no_smoke_loss = l1(no_smoke, identity_no_smoke)
            identity_heavy_smoke_loss = l1(heavy_smoke, identity_heavy_smoke)

            # cycle loss
            cycle_no_smoke = gen_NS(fake_heavy_smoke)
            cycle_heavy_smoke = gen_HS(fake_no_smoke)
            cycle_no_smoke_loss = l1(no_smoke, cycle_no_smoke)
            cycle_heavy_smoke_loss = l1(heavy_smoke, cycle_heavy_smoke)
                
            # generator objective
            G_loss = (
                loss_G_NS
                + loss_G_HS
                + cycle_no_smoke_loss * config.LAMBDA_CYCLE
                + cycle_heavy_smoke_loss * config.LAMBDA_CYCLE
                + identity_heavy_smoke_loss * config.LAMBDA_IDENTITY
                + identity_no_smoke_loss * config.LAMBDA_IDENTITY
            )

        if is_train:
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            lr_scheduler_G.step()

        # original_hs_videos and frames returns the video and frame denotations for the images returned from the buffer
        fake_no_smoke, original_hs_videos, original_hs_frames = img_buffer_no_smoke.push_pop(fake_no_smoke, heavy_smoke_imgs['hs_video'], heavy_smoke_imgs['hs_frame'])
        fake_heavy_smoke, original_ns_videos, original_ns_frames = img_buffer_heavy_smoke.push_pop(fake_heavy_smoke, no_smoke_imgs['ns_video'], no_smoke_imgs['ns_frame'])

        ############################
        # Train Discriminators
        ############################

        opt_disc.zero_grad()

        with torch.cuda.amp.autocast():
            D_HS_real = disc_HS(heavy_smoke)
            D_HS_fake = disc_HS(fake_heavy_smoke.detach())
            D_HS_real_loss = d_loss(D_HS_real, torch.ones_like(D_HS_real))
            D_HS_fake_loss = d_loss(D_HS_fake, torch.zeros_like(D_HS_fake))
            D_HS_loss = (D_HS_real_loss + D_HS_fake_loss) / 2

            D_NS_real = disc_NS(no_smoke)
            D_NS_fake = disc_NS(fake_no_smoke.detach())
            D_NS_real_loss = d_loss(D_NS_real, torch.ones_like(D_NS_real))
            D_NS_fake_loss = d_loss(D_NS_fake, torch.zeros_like(D_NS_fake))
            D_NS_loss = (D_NS_real_loss + D_NS_fake_loss) / 2

            # divide by 2 like in original paper
            D_loss = (D_HS_loss + D_NS_loss)/2

        if is_train:
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            lr_scheduler_D.step()

        # plotting
        if with_metric:
            with torch.no_grad():
                writer.add_scalar(f"{is_train}/discriminator_loss/heavy_smoke", D_HS_loss, step)
                writer.add_scalar(f"{is_train}/discriminator_loss/no_smoke", D_NS_loss, step)
                writer.add_scalar(f"{is_train}/discriminator_loss/heavy_smoke_real", D_HS_real_loss, step)
                writer.add_scalar(f"{is_train}/discriminator_loss/heavy_smoke_fake", D_HS_fake_loss, step)
                writer.add_scalar(f"{is_train}/discriminator_loss/no_smoke_real", D_NS_real_loss, step)
                writer.add_scalar(f"{is_train}/discriminator_loss/no_smoke_fake", D_NS_fake_loss, step)
                writer.add_scalar(f"{is_train}/cycle_loss/no_smoke", cycle_no_smoke_loss, step)
                writer.add_scalar(f"{is_train}/cycle_loss/heavy_smoke", cycle_heavy_smoke_loss, step)
                writer.add_scalar(f"{is_train}/identity_loss/heavy_smoke", identity_heavy_smoke_loss, step)
                writer.add_scalar(f"{is_train}/identity_loss/no_smoke", identity_no_smoke_loss, step)
                writer.add_scalar(f"{is_train}/generator_loss/no_smoke", loss_G_NS, step)
                writer.add_scalar(f"{is_train}/generator_loss/heavy_smoke", loss_G_HS, step)
                writer.add_scalar(f"{is_train}/G_loss", G_loss, step)
                writer.add_scalar(f"{is_train}/learning_rate", lr_scheduler_G.get_last_lr()[0], step)
                #for loss mean
                loss_D_NS_meter.update(D_NS_loss, config.BATCH_SIZE)
                loss_D_HS_meter.update(D_HS_loss, config.BATCH_SIZE)
                G_loss_meter.update(G_loss, config.BATCH_SIZE)
                loss_G_NS_meter.update(loss_G_NS, config.BATCH_SIZE)
                loss_G_HS_meter.update(loss_G_HS, config.BATCH_SIZE)
                cycle_no_smoke_loss_meter.update(cycle_no_smoke_loss, config.BATCH_SIZE)
                cycle_heavy_smoke_loss_meter.update(cycle_heavy_smoke_loss, config.BATCH_SIZE)
                identity_heavy_smoke_loss_meter.update(identity_heavy_smoke_loss, config.BATCH_SIZE)
                identity_no_smoke_loss_meter.update(identity_no_smoke_loss, config.BATCH_SIZE)
                
        # save images to file and calculate metrics 
        with torch.no_grad():      
            # ITERATE OVER EACH IMG IN BATCH
            imgs_this_batch = fake_heavy_smoke.size(0)
            for idx in range(imgs_this_batch):
                ns_video = original_ns_videos[idx]
                ns_frame = original_ns_frames[idx]
                if not is_train or (is_train and (int(ns_frame) % 25 == 0) and ((epoch + 1) % config.SAVE_METRICS_EPOCH == 0)):
                    frame_directory_path = f"{config.OUTPUT_PATH_IMGS}/{is_train}/fold_{fold}/epoch{epoch+1}/Videos_25fps/{ns_video}/{framesegment_from_frame(ns_frame)}"
                    os.makedirs(frame_directory_path, exist_ok=True)
                    frame_path = os.path.join(frame_directory_path, f"f_{ns_frame}.png")
                    
                    # denormalize
                    mean = config.mean_smokeset_hs
                    std = config.std_smokeset_hs
                    fake_heavy_denorm = denormalize_manual(fake_heavy_smoke[idx], mean, std, 255)
                    img = fake_heavy_denorm.cpu().detach()
                    img = np.array(img)                    
                    img = np.transpose(img, (1, 2, 0))
                    img = (img).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(frame_path, compress_level=0)
                
                    #fid and ssim
                    if with_metric:
                        mean = config.mean_smokeset_hs
                        std = config.std_smokeset_hs
                        heavy_denorm = denormalize_manual(heavy_smoke[idx], mean, std, 255)
                        fid_hs.update(heavy_denorm.type(torch.uint8).unsqueeze(0), real=True)
                        fid_hs.update(fake_heavy_denorm.type(torch.uint8).unsqueeze(0), real=False)
                        
                        ssim_hs = ssim(heavy_denorm.unsqueeze(0), fake_heavy_denorm.unsqueeze(0))
                        ssim_meter.update(ssim_hs)

    # calculate metrics and plotting
    if with_metric:
        with torch.no_grad():
            if (epoch + 1) % config.SAVE_METRICS_EPOCH == 0:
                fid_computed_hs_smoke = fid_hs.compute()
                writer.add_scalar(f"{is_train}/fid/heavy_smoke", fid_computed_hs_smoke, (epoch + 1))
                writer.add_scalar(f"{is_train}/avg/ssim", ssim_meter.avg, (epoch + 1))
            
            writer.add_scalar(f"{is_train}/avg/discriminator_loss/no_smoke", loss_D_NS_meter.avg * 2, (epoch + 1))
            writer.add_scalar(f"{is_train}/avg/discriminator_loss/heavy_smoke", loss_D_HS_meter.avg * 2, (epoch + 1))
            writer.add_scalar(f"{is_train}/avg/cycle_loss/no_smoke", cycle_no_smoke_loss_meter.avg, (epoch + 1))
            writer.add_scalar(f"{is_train}/avg/cycle_loss/heavy_smoke", cycle_heavy_smoke_loss_meter.avg, (epoch + 1))
            writer.add_scalar(f"{is_train}/avg/identity_loss/heavy_smoke", identity_heavy_smoke_loss_meter.avg, (epoch + 1))
            writer.add_scalar(f"{is_train}/avg/identity_loss/no_smoke", identity_no_smoke_loss_meter.avg, (epoch + 1))
            writer.add_scalar(f"{is_train}/avg/generator_loss/no_smoke", loss_G_NS_meter.avg, (epoch + 1))
            writer.add_scalar(f"{is_train}/avg/generator_loss/heavy_smoke", loss_G_HS_meter.avg, (epoch + 1))
            writer.add_scalar(f"{is_train}/avg/G_loss", G_loss_meter.avg, (epoch + 1))

def cycle_gan(videos, fold):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    print("cuda" if torch.cuda.is_available() else "cpu")
    disc_HS = Discriminator(in_channels=3).to(config.DEVICE)
    disc_NS = Discriminator(in_channels=3).to(config.DEVICE)
    gen_NS = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_HS = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    
    # initialize weights like in paper with gaussian 0.02
    gen_NS.apply(initialize_conv_weights_normal)
    gen_HS.apply(initialize_conv_weights_normal)
    disc_HS.apply(initialize_conv_weights_normal)
    disc_NS.apply(initialize_conv_weights_normal)

    opt_disc = optim.Adam(
        list(disc_HS.parameters()) + list(disc_NS.parameters()),
        lr=config.LEARNING_RATE_DISC,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_NS.parameters()) + list(gen_HS.parameters()),
        lr=config.LEARNING_RATE_GEN,
        betas=(0.5, 0.999),
    )
    
    L1 = nn.L1Loss()
    g_loss = nn.MSELoss()
    d_loss = nn.MSELoss()

    datasets = []
    for video_name in videos:
        dataset_video = SmokeDataset(
            root_images=config.ROOT_VIDEO_PATH, video=video_name, 
            transform_hs=config.transform_hs, 
            transform_ns=config.transform_ns, 
        )
        datasets.append(dataset_video)
    
    dataset_with_gen = ConcatDataset(datasets)
    
    loader = DataLoader(
        dataset_with_gen,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True
    )

    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    lr_scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_gen, max_lr=config.LEARNING_RATE_GEN, epochs=config.NUM_EPOCHS, steps_per_epoch=len(loader), pct_start=0.05)
    lr_scheduler_D = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_disc, max_lr=config.LEARNING_RATE_DISC, epochs=config.NUM_EPOCHS, steps_per_epoch=len(loader), pct_start=0.05)

    #tensorboard
    save_folder_path = f"{config.SAVE_TENSOR_BOARD_PATH}fold_{fold}"
    os.makedirs(save_folder_path, exist_ok=True)
    writer = SummaryWriter(save_folder_path)

    for epoch in tqdm(range(config.NUM_EPOCHS)):
        disc_HS.train()
        disc_NS.train()
        gen_HS.train()
        gen_NS.train()
        train_fn(disc_HS, disc_NS, gen_NS, gen_HS, loader, opt_disc, opt_gen, L1, g_loss, d_loss, d_scaler, g_scaler, True, epoch, writer, config.WITH_METRIC, lr_scheduler_G, lr_scheduler_D, fold)

        if config.SAVE_MODEL and ((epoch+1) % config.SAVE_MODEL_EPOCH == 0):
            save_model_path = f"{config.SAVE_MODEL_PATH}/{fold}/{epoch}/"
            os.makedirs(save_model_path, exist_ok=True)
            save_checkpoint(gen_HS, filename=f"{save_model_path}gen_hs.pth.tar")
            save_checkpoint(gen_NS, filename=f"{save_model_path}gen_ns.pth.tar")
            save_checkpoint(disc_HS, filename=f"{save_model_path}disc_hs.pth.tar")
            save_checkpoint(disc_NS, filename=f"{save_model_path}disc_ns.pth.tar")
                      
    writer.flush()

def main():
    videos_all = config.ALL_VIDEO_NAMES
    fold_idx = int(sys.argv[1])
    video = videos_all[fold_idx] 
    videos_fold = [x for x in videos_all if x != video]
    print(f"generating images for videos {videos_fold}:")
    cycle_gan(videos_fold, video)
    print(f"finished generation for fold: without {video}.")

if __name__ == "__main__":
    main()