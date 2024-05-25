import torch
import torch.nn.functional as F
from .utils import N_CLASSES
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from torchvision import models
import torch.utils.model_zoo as model_zoo
import math
import sys, time, os, warnings 

 
class DoubleConv(torch.nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(mid_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(torch.nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = torch.nn.Sequential(
            torch.nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(torch.nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = torch.nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(torch.nn.Module):
    def __init__(self, n_channels=3, n_classes=N_CLASSES, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 16)
        self.down1 = Down(16, 32)
        self.down2 = Down(32, 64)
        self.down3 = Down(64, 128)
        factor = 2 if bilinear else 1
        self.down4 = Down(128, 256 // factor)
        self.up1 = Up(256, 128 // factor, bilinear)
        self.up2 = Up(128, 64 // factor, bilinear)
        self.up3 = Up(64, 32 // factor, bilinear)
        self.up4 = Up(32, 16, bilinear)
        self.outc = OutConv(16, n_classes)

    def forward(self, x):
        #print("0",x.shape)
        x1 = self.inc(x)
        #print("1",x1.shape)
        x2 = self.down1(x1)
        #print("2",x2.shape)
        x3 = self.down2(x2)
        #print("3",x3.shape)
        x4 = self.down3(x3)
        #print("4",x4.shape)
        x5 = self.down4(x4)
        #print("5",x5.shape)
        x = self.up1(x5, x4)
        #print("up1",x.shape)
        x = self.up2(x, x3)
        #print("up2",x.shape)
        x = self.up3(x, x2)
        #print("up3",x.shape)
        x = self.up4(x, x1)
        #print("up4",x.shape)
        logits = self.outc(x)
        #print("out",logits.shape)
        return logits
    
### FCN
def down_conv(small_channels, big_channels, pad):   ### contracting block
    return torch.nn.Sequential(
        torch.nn.Conv2d(small_channels, big_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(big_channels),
        torch.nn.Conv2d(big_channels, big_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(big_channels)
    )   ## consider stride = 2

def up_conv(big_channels, small_channels, pad):
    return torch.nn.Sequential(
        torch.nn.Conv2d(big_channels, small_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(small_channels),
        torch.nn.Conv2d(small_channels, small_channels, 3, padding=pad),
        torch.nn.ReLU(),
        torch.nn.BatchNorm2d(small_channels)
    )


class my_FCN(torch.nn.Module):
    def crop(self, a, b):
        ## a, b tensor shape = [batch, channel, H, W]
        Ha = a.size()[2]
        Wa = a.size()[3]
        Hb = b.size()[2]
        Wb = b.size()[3]

        adapt = torch.nn.AdaptiveMaxPool2d((Ha,Wa))
        crop_b = adapt(b) 
            
        return crop_b    
   
    
    def __init__(self):
        super().__init__()

        self.relu    = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=2, ceil_mode=True)         
        self.mean = torch.Tensor([0.5, 0.5, 0.5])
        self.std = torch.Tensor([0.25, 0.25, 0.25])
        
        a = 32
        b = a*2 #64
        c = b*2 #128
        d = c*2 #256
        
        n_class = N_CLASSES
        
        self.conv_down1 = down_conv(3, a, 1) # 3 --> 32
        self.conv_down2 = down_conv(a, b, 1)  # 32 --> 64
        self.conv_down3 = down_conv(b, c, 1)  # 64 --> 128
        self.conv_down4 = down_conv(c, d, 1)  # 128 --> 256
        
        self.bottleneck = torch.nn.ConvTranspose2d(d, c, kernel_size=3, stride=2, padding=1, output_padding=1)  
        self.conv_up3 = up_conv(c, b, 1)  # 128 --> 64
        self.upsample3 = torch.nn.ConvTranspose2d(b, a, kernel_size=3, stride=2, padding=1, output_padding=1)   
                 
        self.classifier = torch.nn.Conv2d(a, n_class, kernel_size=1) 
        
    
    def forward(self, x):
        H = x.shape[2]
        W = x.shape[3]
        z = (x - self.mean[None, :, None, None].to(x.device)) / self.std[None, :, None, None].to(x.device)
        #################### DOWN / ENCODER #############################
        conv1 =  self.conv_down1(z)   # 3 --> 32
        mx1 = self.maxpool(conv1)
        conv2 =  self.conv_down2(mx1)  # 32 --> 64
        mx2 = self.maxpool(conv2) 
        conv3 =  self.conv_down3(mx2) # 64 --> 128  
        mx3 = self.maxpool(conv3) 
        conv4 =  self.conv_down4(conv3) # 128 --> 256  ################### CHANGED THIS

        ########################### BOTTLENECK #############################
        score = self.bottleneck(conv4)  # 256 --> 128
       
        ######################### UP/DECODER #######################
        crop_conv3 = self.crop(score, conv3)    
        score = score + crop_conv3   ### add 128 
        
        ##########################
        score = self.conv_up3(score)  # 128 --> 64
        score = self.upsample3(score)  # 64 --> 32     
        crop_conv1 = self.crop(score, conv1)   
        score = score + crop_conv1   ### add 32           
        
        ############################
        score = self.classifier(score) 
        out = torch.nn.functional.interpolate(score, size=(H,W))
        out = out[:, :, :H, :W]
        return out  




###############################################################################################################################

model_factory = {
    'my_fcn': my_FCN, 
    'unet': UNet,
}


def save_model(model):
    from torch import save
    from os import path
    for n, m in model_factory.items():
        if isinstance(model, m):
            return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), '%s.th' % n))
    raise ValueError("model type '%s' not supported!" % str(type(model)))


def load_model(model):
    from torch import load
    from os import path
    r = model_factory[model]()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), '%s.th' % model), map_location='cpu'))
    return r