# code for concat label with image
import torch.nn as nn
import torch
from random import choice

class ConvBlock(nn.Module):
    #kwargs -> keyword arguments (stride padding etc.)
    def __init__(self, in_channels, out_channels, down=True, use_act=True, **kwargs):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if down
            else nn.ConvTranspose2d(in_channels, out_channels, **kwargs),
            nn.InstanceNorm2d(out_channels),
            #inplace for performance? nn.identity -> nop
            nn.ReLU(inplace=True) if use_act else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, padding=1),
            ConvBlock(channels, channels, use_act=False, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

#######################################
# x = image (batchsize, channels, w, h)
# c = label (batchsize, labelvector([0,0,1]) -> one hot encoded)
#######################################
class Generator(nn.Module):
    #num_res = 9 (256) 6 (128)
    def __init__(self, img_channels, num_classes=3, num_features = 64, num_residuals=9):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(img_channels + num_classes, num_features, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(num_features),
            nn.ReLU(inplace=True),
        )
        #downsample
        self.down_blocks = nn.ModuleList(
            [
                ConvBlock(num_features, num_features*2, down=True, kernel_size=3, stride=2, padding=1),
                ConvBlock(num_features*2, num_features*4, down=True, kernel_size=3, stride=2, padding=1),
            ]
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_features*4) for _ in range(num_residuals)]
        )
        #upsample
        self.up_blocks = nn.ModuleList(
            [
                ConvBlock(num_features*4, num_features*2, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
                #upsampling bilinear 2d pytorch...
                #nn.UpsamplingBilinear2d(scale_factor=2),
                ConvBlock(num_features*2, num_features*1, down=False, kernel_size=3, stride=2, padding=1, output_padding=1),
            ]
        )

        self.last = nn.Conv2d(num_features*1, img_channels, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x, c):
        c = c.view(c.size(0), c.size(1), 1, 1)
        c = c.repeat(1, 1, x.size(2), x.size(3))
        x = torch.cat([x, c], dim=1)
        #print(x.shape)
        
        x = self.initial(x)
        for layer in self.down_blocks:
            x = layer(x)
        x = self.res_blocks(x)
        for layer in self.up_blocks:
            x = layer(x)
        return torch.tanh(self.last(x))
