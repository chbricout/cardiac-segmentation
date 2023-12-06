import torch.nn.init as init
import torch.nn.functional as F
import math
import torch
import torch.nn.functional as F
from torch import nn


def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False, activation = nn.PReLU, batch_norm=True):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity(),
            activation(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels) if batch_norm else nn.Identity() ,
            activation(),
            
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, activation=nn.PReLU):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3),
            nn.BatchNorm2d(middle_channels) ,
            activation(),
            nn.Conv2d(middle_channels   , middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            activation(),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes, activation=nn.PReLU):
        super(UNet, self).__init__()
        self.pretraining = False
        self.enc1 = _EncoderBlock(1, 32, activation=activation)
        self.enc2 = _EncoderBlock(32, 64, activation=activation)
        self.enc3 = _EncoderBlock(64, 128, activation=activation)
        self.enc4 = _EncoderBlock(128, 256, activation=activation)
        # self.enc5 = _EncoderBlock(128, 256, activation=activation, dropout=True)


        self.center = _DecoderBlock(256, 512, 256, activation=activation)

        # self.dec5 = _DecoderBlock(512, 256, 128, activation=activation)
        self.dec4 = _DecoderBlock(512, 256, 128, activation=activation)
        self.dec3 = _DecoderBlock(256, 128, 64, activation=activation)
        self.dec2 = _DecoderBlock(128, 64, 32, activation=activation)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3),
            nn.BatchNorm2d(32),
            activation(),
            nn.Conv2d(32, 16, kernel_size=3),
            nn.BatchNorm2d(16),
            activation(),
        )
        self.final = nn.Conv2d(16, num_classes, kernel_size=1)

      

        initialize_weights(self)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        # enc5 = self.enc5(enc4)
        center = self.center(enc4)
        # dec5 = self.dec5(torch.cat([center, F.upsample(enc5, center.size()[2:], mode='bilinear')], 1))
        dec4 = self.dec4(torch.cat([center, F.upsample(enc4, center.size()[2:], mode='bilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.upsample(enc3, dec4.size()[2:], mode='bilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.upsample(enc2, dec3.size()[2:], mode='bilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.upsample(enc1, dec2.size()[2:], mode='bilinear')], 1))
        final = self.final(dec1)

        return F.upsample(final, x.size()[2:], mode='bilinear')


class SegNet(nn.Module):
    def __init__(self):
        super(SegNet, self).__init__()

        self.enc1 = _EncoderBlock(1, 32, batch_norm=False)
        self.enc2 = _EncoderBlock(32, 64, batch_norm=True)
        self.enc3 = _EncoderBlock(64, 128, batch_norm=True)
        self.enc4 = _EncoderBlock(128, 256, batch_norm=True)
        self.enc5 = _EncoderBlock(256, 512, dropout=True, batch_norm=True)
        self.enc6 = _EncoderBlock(512, 1024, dropout=True, batch_norm=False)

        self.fc = nn.Conv2d(1024,1,(2,2))
        self.logit = nn.Sigmoid()
    
    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        fc = self.fc(enc6)
        return self.logit(fc[:,0,0,0])