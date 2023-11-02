import random, torch, os, numpy as np
import torch.nn as nn
import copy
from torchvision.utils import save_image
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

def save_checkpoint(model, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict()
    }
    torch.save(checkpoint, filename)

def get_color_for_class(class_n, channel):
    colors_channel0 = [204, 255, 204, 153, 0, 0, 153, 255, 255, 30, 255, 255, 128, 255, 0, 60, 64]
    colors_channel1 = [102, 102, 153, 102, 255, 0, 0, 0, 130, 144, 255, 0, 128, 200, 80, 50, 64]
    colors_channel2 = [204, 255, 153, 0, 255, 255, 204, 153, 0, 1, 0, 0, 128, 170, 80, 50, 255]
    colors = [colors_channel0, colors_channel1, colors_channel2]
    if class_n == 0:
        return 0
    return colors[channel][class_n - 1]

#channel 0,1,2 for color image
def get_gray_to_color_lambda(channel):
    return lambda x: get_color_for_class(x, channel)

#save seg in last epoch
def save_mask_grayscale_as_img(mask_grayscale, folder, frame_number):
    #for idx, outpt in enumerate(mask_grayscale):
    os.makedirs(folder, exist_ok=True)
    output = mask_grayscale.to("cpu")
    lambda_channel0 = get_gray_to_color_lambda(0)
    lambda_channel1 = get_gray_to_color_lambda(1)
    lambda_channel2 = get_gray_to_color_lambda(2)
    output.unsqueeze_(0)
    mask_color = output.repeat(3, 1, 1)
    # for all three channels
    mask_color[0].apply_(lambda_channel0)
    mask_color[1].apply_(lambda_channel1)
    mask_color[2].apply_(lambda_channel2)  
    mask_color = mask_color.float() /255
    save_image(mask_color, f"{folder}/mask_pred_{frame_number}.png")
    """
    for idx, outpt in enumerate(mask_grayscale):
        # https://discuss.pytorch.org/t/solved-convert-color-of-tensors/58341
        color_map = #Tensor of shape(256,3)
        #gray_image = (gray_image * 255).long() # Tensor values between 0 and 255 and LongTensor and shape of (512,512)
        colored_mask = color_map[outpt]
        save_image(colored_mask, f"{folder}mask_{frame_numbers_batch[idx]}.png")"""

"""
def get_color_dict():
    cmap=ListedColormap(["light purple", "pink", "pink-orange", "dark orange", "Cyan", "" , "", "", "", "", "", "", ""])
    return cmap

def get_color_encoding_for_17_classes():
    '''
    color_dict = []
    color_dict[0] = {
        # Retrieval-Bags => light purple.
        1 : [204, 102, 204],
        # Palpation-Probe => pink.
        2 : [255, 102, 255],
        # Needle-Probe => pink-orange.
        3 : [204, 153, 153],
        # Trokar-Tip => dark orange.
        4 : [153, 102, 0],
        # Clips => Cyan.
        5 : [0, 255, 255],
        # Clip-Applicator Tip => Blue.
        6 : [0, 0, 255],
        # Suction-Rod Tip => Purple.
        7 : [153, 0, 204],
        # HFcoag-Probe Tip => Light Red.
        8 : [255, 0, 153],
        # Grasper Tip => Orange.
        9 : [255, 130, 0],
        # PE-Forceps Tip => Lawngreen.
        10 : [30, 144, 1],
        # Scissors Tip => Light Yellow. 
        11 : [255, 255, 0],
        # Drainage => Dark red. 
        12 : [255, 0, 0],
        # Blunt Grasper Tip => Gray.
        13 : [128, 128, 128],
        # Overholt Tip => Peach.
        14 : [255, 200, 170],
        # Hook Clamp Tip => Dark green.
        15 : [0, 80, 80],
        # Argonbeamer Tip => Brown.
        16 : [60, 50, 50],
        # Shaft => Light Blue.
        17 : [64, 64, 255],
    }'''

"""