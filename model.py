
import torch
import torch.nn as nn

class overparameterized_model(nn.Module):
    def __init__(self):
        super(overparameterized_model, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 10, kernel_size=1)
        self.conv1_2 = nn.Conv2d(10, 25, kernel_size=1)
        self.conv1_3 = nn.Conv2d(25, 10, kernel_size=1)
        self.conv1_4 = nn.Conv2d(3, 10, kernel_size=1)
        self.conv3 = nn.Conv2d(10, 3 * (2 ** 2), (3,3), (1,1), (1,1))
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        x = x.type(torch.float32)
        out = self.conv1_1(x)
        out = self.conv1_2(out)
        out_0 = self.conv1_3(out)
        out_1 = self.conv1_4(x)
        out = torch.add(out_0, out_1)
        out = self.conv3(out)
        out = self.pixel_shuffle(out)
        out = torch.round(torch.clip(out, 0, 255)).type(torch.uint8)

        return torch.permute(out, [0,2,3,1]).view(2160,3840,3)





