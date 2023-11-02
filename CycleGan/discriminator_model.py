import torch
import torch.nn as nn

'''
paper used as foundation: https://arxiv.org/pdf/1703.10593.pdf
patch gan 70x70
'''
class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, use_instance_norm):
        super().__init__()
        if use_instance_norm:
            self.conv = nn.Sequential(
                #kernel size 4 -> in paper paddingmode=reflect -> reduce artifacts
                nn.Conv2d(in_channels, out_channels, 4, stride, 1, bias=True, padding_mode="reflect"),
                nn.InstanceNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
                nn.LeakyReLU(0.2, inplace=True),
            )
            

    def forward(self, x):
        return self.conv(x)


class Discriminator(nn.Module):
    # inChan -> rgb
    def __init__(self, in_channels=3, features=[64, 128, 256, 512]):
        super().__init__()
        # initial run without instance norm
        self.initial = Block(in_channels=in_channels, out_channels=features[0], stride=2, use_instance_norm=False)

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            # use stride 1 for last feature
            layers.append(Block(in_channels, feature, stride=1 if feature==features[-1] else 2, use_instance_norm=True))
            in_channels = feature
        layers.append(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect"))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = self.initial(x)
        return torch.sigmoid(self.model(x))

def test():
    x = torch.randn((5, 3, 560, 980))
    model = Discriminator(in_channels=3)
    preds = model(x)
    print(preds.size())


if __name__ == "__main__":
    test()
