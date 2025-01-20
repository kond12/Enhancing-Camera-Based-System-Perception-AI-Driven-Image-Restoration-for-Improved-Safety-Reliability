
import torch
from torch import nn
import torch.nn.functional as F


class Srcnn(nn.Module):
    def __init__(self):
        super(Srcnn, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(64, 3, kernel_size=5, padding=2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x



class Autoencoders(nn.Module):
    def __init__(self):
        super().__init__()
        # encoding layers
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1, stride=2)  # 28x28 --> 14x14
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1, stride=2)  # 14x14 --> 7x7
        self.conv3 = nn.Conv2d(64, 128, 5)  # 7x7 --> 3x3

        # decoding layers
        self.tconv1 = nn.ConvTranspose2d(128, 64, 5)  # 3x3 --> 7x7
        self.tconv2 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)  # 7x7 --> 14x14
        self.tconv3 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)  # 14x14 --> 28x28


        self.relu = torch.nn.ReLU()
        self.activation = torch.nn.Sigmoid()
    # forward prop function
    def forward(self, x):
        x = self.relu(self.conv1(x))
        # print(x.size())
        x = self.relu(self.conv2(x))
        # print(x.size())
        x = self.relu(self.conv3(x))
        # print(x.size())

        x = self.relu(self.tconv1(x))
        # print(x.size())
        x = self.relu(self.tconv2(x))
        # print(x.size())
        x = self.activation(self.tconv3(x))  # final layer is applied sigmoid activation
        # print(x.size())

        return x
    
class MIRNet(nn.Module):
    def __init__(self):
        super(MIRNet, self).__init__()

        # Feature extraction network
        self.fe_net = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # Information exchange network
        self.ien_net = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Selective kernel feature fusion network
        self.skff_net = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # Output layer
        self.output_layer = nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.fe_net(x)
        x = self.ien_net(x)
        x = self.skff_net(x)
        x = self.output_layer(x)
        return x
