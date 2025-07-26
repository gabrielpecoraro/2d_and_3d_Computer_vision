import torch

import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, int_c: int, out_c: int):
        super().__init__()

        # Conv Block 1
        self.conv1 = nn.Conv2d(int_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        # Conv Block 2
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        # Relu
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()

        self.block = ConvBlock(in_c, out_c)
        self.maxpool = nn.MaxPool2d((2, 2))

    def forward(self, x):
        x = self.block(x)
        p = self.maxpool(x)

        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, padding=0, stride=2)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        # Encode
        self.encode1 = EncoderBlock(3, 64)
        self.encode2 = EncoderBlock(64, 128)
        self.encode3 = EncoderBlock(128, 256)
        self.encode4 = EncoderBlock(256, 512)

        # Bottleneck
        self.bottle = ConvBlock(512, 1024)

        # Decode
        self.decode1 = DecoderBlock(1024, 512)
        self.decode2 = DecoderBlock(512, 256)
        self.decode3 = DecoderBlock(256, 128)
        self.decode4 = DecoderBlock(128, 64)

        # Classifier
        self.final = nn.Conv2d(64, 1, kernel_size=3, padding=1)

    def forward(self, x):
        # Encoder
        x1, s1 = self.encode1(x)
        x2, s2 = self.encode2(s1)
        x3, s3 = self.encode3(s2)
        x4, s4 = self.encode4(s3)

        # Bottleneck
        b = self.bottle(s4)

        # Decode
        d1 = self.decode1(b, x4)
        d2 = self.decode2(d1, x3)
        d3 = self.decode3(d2, x2)
        d4 = self.decode4(d3, x1)

        # Classifier
        classifier = self.final(d4)
        return classifier
