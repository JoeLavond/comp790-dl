# packages
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

def main():

    model = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, padding=3),
        nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
        nn.Sigmoid(),
        nn.MaxPool2d(kernel_size=2, stride=2),

        nn.Conv2d(in_channels=64, out_channels=100, kernel_size=1)
    ).cuda()

    summary(model, (3, 64, 64))


if __name__ == "__main__":
    main()

