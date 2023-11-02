import random, torch, os, numpy as np
import torch.nn as nn
from torchvision.utils import save_image
import config_cyclegan as config
from PIL import Image

from torchvision import transforms

def save_image(image, idx, frame_path):
    mean = config.mean_smokeset_hs
    std = config.mean_smokeset_hs
    img = image[idx]
    img = img.cpu().detach()
    img = np.array(img)
    img[0] = img[0, : , : ] * std[0] + mean[0]
    img[1] = img[1, : , : ] * std[1] + mean[1]
    img[2] = img[2, : , : ] * std[2] + mean[2]
    
    img = np.transpose(img, (1, 2, 0))
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(frame_path, compress_level=0)

def denormalize_manual(img, mean, std, max_px):
    mean_m = [i * max_px for i in mean]
    std_m = [i * max_px for i in std]
    unorm = transforms.Compose([transforms.Normalize(mean = [0., 0., 0. ],std = [1/std_m[0], 1/std_m[1], 1/std_m[2]]),
                                transforms.Normalize(mean = [-mean_m[0], -mean_m[1], -mean_m[2]],std = [ 1., 1., 1. ]),])
    return unorm(img)
       
def normalize_manual(img, mean, std, max_px):
    mean_m = [i * max_px for i in mean]
    std_m = [i * max_px for i in std]
    norm = transforms.Normalize(mean=mean_m, std=std_m)
    return norm(img)    

def initialize_conv_weights_normal(m):
    classname = m.__class__.__name__
    if (type(m) == nn.Conv2d) or (type(m) == nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
        if hasattr(m, "bias") and m.bias is not None:
            torch.nn.init.constant_(m.bias.data, 0.0)
    
def save_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, filename)