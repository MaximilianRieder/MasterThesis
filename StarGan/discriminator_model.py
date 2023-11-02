import torch
import torch.nn as nn
import numpy as np

'''
paper used as foundation: https://arxiv.org/pdf/1711.09020.pdf
'''
class Discriminator(nn.Module):
    """Discriminator network with PatchGAN."""
    def __init__(self, image_h, image_w, c_dim=3, conv_dim=64, repeat_num=6):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(3, conv_dim, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.01))

        curr_dim = conv_dim
        for i in range(1, repeat_num):
            layers.append(nn.Conv2d(curr_dim, curr_dim*2, kernel_size=4, stride=2, padding=1))
            layers.append(nn.LeakyReLU(0.01))
            curr_dim = curr_dim * 2

        kernel_size = (image_h // 2**repeat_num, image_w // 2**repeat_num)
        
        self.main = nn.Sequential(*layers)
        self.conv_pred = nn.Conv2d(curr_dim, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv_label = nn.Conv2d(curr_dim, c_dim, kernel_size=kernel_size, bias=False)
        
    def forward(self, x):
        h = self.main(x)
        pred = self.conv_pred(h)
        pred_labels = self.conv_label(h)
        
        return pred, pred_labels.view(pred_labels.size(0), pred_labels.size(1))