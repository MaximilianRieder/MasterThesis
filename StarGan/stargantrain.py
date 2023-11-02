import torch
from datasetstargan import SmokeDataset
import sys
from utils import save_checkpoint, load_checkpoint, initialize_conv_weights_normal, denormalize_manual, normalize_manual
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config_stargan as config
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
from random import choice

def framesegment_from_frame(frame):
    return str((int(frame) // 1000) * 1000)

def reset_grad(opt_gen, opt_disc):
    """Reset the gradient buffers."""
    opt_disc.zero_grad()
    opt_gen.zero_grad()
        
# source: https://necromuralist.github.io/Neurotic-Networking/posts/gans/wasserstein-gan-with-gradient-penalty/
def compute_gradient_penalty(discriminator, real_images, generated_images, device, lambda_gp=10.0):
    # Sample random interpolation points between real and generated images
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device)
    interpolated = (alpha * real_images + (1 - alpha) * generated_images).requires_grad_(True)

    # Compute discriminator output at interpolated points
    d_interpolated_output, _ = discriminator(interpolated)

    # Compute gradients of discriminator output with respect to interpolated points
    gradient = torch.autograd.grad(
        outputs=d_interpolated_output,
        inputs=interpolated,
        grad_outputs=torch.ones(d_interpolated_output.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    # Compute norm of gradients
    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    penalty = torch.mean(torch.square(gradient_norm - 1))
    return penalty * lambda_gp

def train_fn(D, G, loader, opt_disc, opt_gen, l1, classification_loss, d_scaler, g_scaler, is_train, epoch, writer, with_metric, lr_scheduler_G, lr_scheduler_D, fold):

    G_loss_meter = AverageMeter()
    loss_G_fake_meter = AverageMeter()
    loss_G_class_meter = AverageMeter()
    loss_D_real_meter = AverageMeter()
    loss_D_fake_meter = AverageMeter()
    gradient_penalty_meter = AverageMeter()
    cycle_loss_meter = AverageMeter()
    identity_loss_meter = AverageMeter()
    d_loss_class_meter = AverageMeter()
    ssim_meter = AverageMeter()
    
    D = D.to(config.DEVICE)
    G = G.to(config.DEVICE)
    
    #img_buffer = ImageBuffer(config.BUFFER_SIZE)

    if with_metric and ((epoch + 1) % config.SAVE_METRICS_EPOCH == 0):
        fid = FrechetInceptionDistance()
        fid = fid.to(config.DEVICE)
        ssim = StructuralSimilarityIndexMeasure().to(config.DEVICE)
        
    for idx, (image_data, video_data, frame_data) in enumerate(loader):
        
        img_original = image_data['image'].to(config.DEVICE)
        cls_original = image_data['class'].to(config.DEVICE) #[[1,0,0,], [...], ...]
        
        # Generate target domain labels randomly, that doesnt correspond to the original domain
        ###############################################
        # hier anpassen, dass auch 2 label manchmal hergenommen werden TODO
        ###############################################
        
        cls_target = torch.empty(0, config.NUM_CLASSES)
        for cls_single in cls_original:
            cls_original_index = (cls_single==1).nonzero().squeeze().item()
            rand_idx_not_org = choice([i for i in range(0,  config.NUM_CLASSES) if i not in [cls_original_index]])
            cls_target_single = torch.zeros(1, config.NUM_CLASSES)
            cls_target_single[0, rand_idx_not_org] = 1
            cls_target = torch.cat((cls_target, cls_target_single), 0)
        cls_target = cls_target.to(config.DEVICE)

        step = len(loader) * (epoch) + idx
        
        # Train Discriminators
        with torch.cuda.amp.autocast():
            # train on original
            D_org_pred, cls_org_pred = D(img_original)
            D_real_loss = - torch.mean(D_org_pred)
            D_loss_cls = classification_loss(cls_org_pred, cls_original)
            img_fake = G(img_original, cls_target)
            D_fake_pred, cls_fake_pred = D(img_fake.detach())
            D_fake_loss =  torch.mean(D_fake_pred)
            
            # Compute loss for gradient penalty. 
            d_loss_gp = compute_gradient_penalty(D, img_original, img_fake.detach(), config.DEVICE, config.LAMBDA_GP)
            
            D_loss = d_loss_gp + D_real_loss + D_fake_loss + D_loss_cls * config.LAMBDA_CLS

        if is_train:
            reset_grad(opt_gen, opt_disc)
            d_scaler.scale(D_loss).backward()
            d_scaler.step(opt_disc)
            d_scaler.update()
            lr_scheduler_D.step()
        
        # Train Generator
        # autocast for float16
        with torch.cuda.amp.autocast():
            img_fake = G(img_original, cls_target)

            D_fake_pred, cls_fake_pred = D(img_fake)
            g_loss_fake = - torch.mean(D_fake_pred)
            g_loss_cls = classification_loss(cls_fake_pred, cls_target)

            # identity loss 
            img_identity = G(img_original, cls_original)    
            identity_loss = l1(img_original, img_identity)

            img_reconstruct = G(img_fake, cls_original)
            reconstruct_loss = l1(img_original, img_reconstruct)
                
            # add all together
            G_loss = (
                g_loss_fake
                + g_loss_cls * config.LAMBDA_CLS
                + reconstruct_loss * config.LAMBDA_CYCLE
                + identity_loss * config.LAMBDA_IDENTITY
            )

        if is_train:
            reset_grad(opt_gen, opt_disc)
            g_scaler.scale(G_loss).backward()
            g_scaler.step(opt_gen)
            g_scaler.update()
            lr_scheduler_G.step()

        if with_metric:
            with torch.no_grad():
                # plotting
                writer.add_scalar("discriminator_loss/real", D_real_loss, step)
                writer.add_scalar("discriminator_loss/fake", D_fake_loss, step)
                writer.add_scalar("discriminator_loss/class", D_loss_cls, step)
                writer.add_scalar("discriminator_loss/gradient_penalty", d_loss_gp, step)

                writer.add_scalar("cycle_loss/reconstruct", reconstruct_loss, step)
                writer.add_scalar("identity_loss/", identity_loss, step)
                writer.add_scalar("generator_loss/fake", g_loss_fake, step)
                writer.add_scalar("generator_loss/class", g_loss_cls, step)
                writer.add_scalar("G_loss", G_loss, step)
                writer.add_scalar("learning_rate", lr_scheduler_G.get_last_lr()[0], step)
                #for loss mean
                loss_D_real_meter.update(D_real_loss, config.BATCH_SIZE)
                loss_D_fake_meter.update(D_fake_loss, config.BATCH_SIZE)
                d_loss_class_meter.update(D_loss_cls, config.BATCH_SIZE)
                G_loss_meter.update(G_loss, config.BATCH_SIZE)
                loss_G_fake_meter.update(g_loss_fake, config.BATCH_SIZE)
                loss_G_class_meter.update(g_loss_cls, config.BATCH_SIZE)
                gradient_penalty_meter.update(d_loss_gp, config.BATCH_SIZE)
                cycle_loss_meter.update(reconstruct_loss, config.BATCH_SIZE)
                identity_loss_meter.update(identity_loss, config.BATCH_SIZE)

                
        # save images to file and calculate metrics 
        with torch.no_grad():      
            # ITERATE OVER EACH IMG IN BATCH
            imgs_this_batch = img_fake.size(0)
            for idx in range(imgs_this_batch):
                video_name = video_data[idx]
                frame_name = frame_data[idx]
                if is_train and ((image_data['class_name'][idx] == "slightly") or (image_data['class_name'][idx] == "heavily")) and (int(frame_name) % 25 == 0) and ((epoch + 1) % config.SAVE_METRICS_EPOCH == 0):
                    mean = config.mean_smokeset_hs
                    std = config.std_smokeset_hs
                    org_img_denorm = denormalize_manual(img_original[idx], mean, std, 255)
                    fid.update(org_img_denorm.type(torch.uint8).unsqueeze(0), real=True)
                        
                if is_train and (image_data['class_name'][idx] == "not") and (int(frame_name) % 25 == 0) and ((epoch + 1) % config.SAVE_METRICS_EPOCH == 0):
                    frame_directory_path = f"{config.OUTPUT_PATH_IMGS}/fold_{fold}/Smoked_frames_lr_disc{config.LEARNING_RATE_DISC}_LR_gen_{config.LEARNING_RATE_GEN}_epoch{epoch+1}/Videos_25fps/{video_name}/{framesegment_from_frame(frame_name)}"
                    os.makedirs(frame_directory_path, exist_ok=True)
                    frame_path = os.path.join(frame_directory_path, f"f_{frame_name}_input_{cls_original[idx]}_target_{cls_target[idx]}.png")
                    
                    mean = config.mean_smokeset_hs
                    std = config.std_smokeset_hs
                    fake_img_denorm = denormalize_manual(img_fake[idx], mean, std, 255)
                    img = fake_img_denorm.cpu().detach()
                    img = np.array(img)                    
                    img = np.transpose(img, (1, 2, 0))
                    img = (img).astype(np.uint8)
                    img = Image.fromarray(img)
                    img.save(frame_path, compress_level=0)
                
                    #fid and ssim
                    if with_metric:
                        mean = config.mean_smokeset_hs
                        std = config.std_smokeset_hs
                        org_img_denorm = denormalize_manual(img_original[idx], mean, std, 255)
                        fid.update(fake_img_denorm.type(torch.uint8).unsqueeze(0), real=False)
                        
                        #fake_hs_32 = fake_heavy_smoke.type_as(no_smoke[idx])
                        ssim_hs = ssim(org_img_denorm.unsqueeze(0), fake_img_denorm.unsqueeze(0))
                        ssim_meter.update(ssim_hs)
          

    if with_metric:
        with torch.no_grad():
            if (epoch + 1) % config.SAVE_METRICS_EPOCH == 0:
                fid_computed_hs_smoke = fid.compute()
                writer.add_scalar("fid/generated", fid_computed_hs_smoke, (epoch + 1))
                writer.add_scalar("avg/ssim", ssim_meter.avg, (epoch + 1))
            
            writer.add_scalar("avg/loss_D_real/", loss_D_real_meter.avg * 2, (epoch + 1))
            writer.add_scalar("avg/loss_D_fake/", loss_D_fake_meter.avg * 2, (epoch + 1))
            writer.add_scalar("avg/gradient_penalty/", gradient_penalty_meter.avg, (epoch + 1))
            writer.add_scalar("avg/cycle_loss/", cycle_loss_meter.avg, (epoch + 1))
            writer.add_scalar("avg/identity/", identity_loss_meter.avg, (epoch + 1))
            writer.add_scalar("avg/d_loss_class/", d_loss_class_meter.avg, (epoch + 1))
            writer.add_scalar("avg/loss_G_fake/", loss_G_fake_meter.avg, (epoch + 1))
            writer.add_scalar("avg/loss_G_class/", loss_G_class_meter.avg, (epoch + 1))
            writer.add_scalar("avg/G_loss", G_loss_meter.avg, (epoch + 1))

def star_gan(videos, fold):
    """
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    """
    print("cuda" if torch.cuda.is_available() else "cpu")
    gen = Generator(img_channels=3, num_classes=3, num_residuals=6)
    disc = Discriminator(image_h=540, image_w=960, c_dim=3)
    
    # initialize weights like in paper with gaussian 0.02 TODO hier auch?
    gen.apply(initialize_conv_weights_normal)
    disc.apply(initialize_conv_weights_normal)

    # decrease betas for slower learning?
    opt_disc = optim.Adam(
        disc.parameters(),
        lr=config.LEARNING_RATE_DISC,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        gen.parameters(),
        lr=config.LEARNING_RATE_GEN,
        betas=(0.5, 0.999),
    )
    
    # Losses like in lsgan (stated in orinigal cyclegan paper)
    # LSGAN paper: https://arxiv.org/pdf/1611.04076.pdf 
    L1 = nn.L1Loss()
    classification_loss = nn.BCEWithLogitsLoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_SAVE, gen, opt_gen, config.LEARNING_RATE_GEN,
        )
        load_checkpoint(
            config.CHECKPOINT_DISC_SAVE, disc, opt_disc, config.LEARNING_RATE_DISC,
        )

    datasets = []
    for video_name in videos:
        dataset_video = SmokeDataset(
            root_images=config.ROOT_VIDEO_PATH, video=video_name, transform=config.transform, downsampling=config.DOWNSAMPLING
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

    #for training float16
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    # scheduler
    lr_scheduler_G = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_gen, max_lr=config.LEARNING_RATE_GEN, epochs=config.NUM_EPOCHS, steps_per_epoch=len(loader), pct_start=0.05)
    lr_scheduler_D = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt_disc, max_lr=config.LEARNING_RATE_DISC, epochs=config.NUM_EPOCHS, steps_per_epoch=len(loader), pct_start=0.05)

    #tensorboard
    save_folder_path = f"{config.TENSORBOARD_PATH}fold_{fold}/{video_name}"
    os.makedirs(save_folder_path, exist_ok=True)
    writer = SummaryWriter(save_folder_path)

    for epoch in tqdm(range(config.NUM_EPOCHS)):
        disc.train()
        gen.train()
        train_fn(disc, gen, loader, opt_disc, opt_gen, L1, classification_loss, d_scaler, g_scaler, True, epoch, writer, config.WITH_METRIC, lr_scheduler_G, lr_scheduler_D, fold)
        if config.SAVE_MODEL and (epoch % 5 == 0):
            save_checkpoint(gen, opt_gen, filename=config.CHECKPOINT_GEN_SAVE)
            save_checkpoint(disc, opt_disc, filename=config.CHECKPOINT_DISC_SAVE)

    writer.flush()

def main(): 
    videos_all = config.VIDEOS
    video_fold_idx = sys.argv[1]
    video = videos_all[int(video_fold_idx)]
    videos_fold = [x for x in videos_all if x != video]
    star_gan(videos_fold, video)

if __name__ == "__main__":
    main()