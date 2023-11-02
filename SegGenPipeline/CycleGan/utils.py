import random, torch, os, numpy as np
import torch.nn as nn
import copy
from torchvision.utils import save_image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
from . import config_cyclegan
from PIL import Image

from torchvision import transforms

def save_image(image, idx, frame_path):
    mean = config_cyclegan.mean_smokeset_hs
    std = config_cyclegan.mean_smokeset_hs
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

#######################
# https://inside-machinelearning.com/en/why-and-how-to-normalize-data-object-detection-on-image-in-pytorch-part-1/
# https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
#######################
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
    '''
    elif classname.find("BatchNorm2d") != -1:
        print("batchnorm2d")
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    '''
    
def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, device):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

class LambdaLR:
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert (n_epochs - decay_start_epoch) > 0, "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch) / (self.n_epochs - self.decay_start_epoch)

"""
def seed_everything(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
"""