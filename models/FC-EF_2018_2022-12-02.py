import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

"""
paper site: https://arxiv.org/abs/1810.08462
paper name: Fully Convolutional Siamese Networks for Change Detection.
Publication year: 2018
contributors: Rodrigo Caye Daudt, Bertrand Le Saux, Alexandre Boulch.
github implementation: https://github.com/Bobholamovic/CDLab/blob/master/src/models/unet.py

paper index in our Notion site: 40 
"""


"""
 Identity:   A placeholder identity operator that is argument-insensitive.
 
 getattr:    It allows you to call methods based on the contents of 
             a string instead of typing the method name.
             
 Conv3x3:    Conv3x3(in_ch=16, out_ch=16, norm=True, act=True)
             sequential module consists of (zero padding layer,
                                            conv2d layer, BatchNorm layer,
                                            ReLU layer)
"""
# A placeholder identity operator that is argument-insensitive.
Identity = nn.Identity


def get_norm_layer():
    # TODO: select appropriate norm layer
    return nn.BatchNorm2d

def get_act_layer():
    # TODO: select appropriate activation layer
    return nn.ReLU


def make_norm(*args, **kwargs):
    norm_layer = get_norm_layer()
    return norm_layer(*args, **kwargs)

def make_act(*args, **kwargs):
    act_layer = get_act_layer()
    return act_layer(*args, **kwargs)


class BasicConv(nn.Module):
    def __init__(
            self, in_ch, out_ch,
            kernel_size, pad_mode='Zero',
            bias='auto', norm=False, act=False, **kwargs
        ):
        super().__init__()
        seq = []
        if kernel_size >= 2:
            seq.append( getattr(nn, pad_mode.capitalize()+'Pad2d')(kernel_size//2) )
        seq.append(
            nn.Conv2d(
                in_ch, out_ch, kernel_size, stride=1, padding=0,
                bias=(False if norm else True) if bias=='auto' else bias,
                **kwargs
            )    
        )
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq)
        
    def forward(self, x):
        return self.seq(x)
 
       
class Conv3x3(BasicConv):
    def __init__(self, in_ch, out_ch, pad_mode='Zero', bias='auto', norm=False, act=False, **kwargs):
        super().__init__(in_ch, out_ch, kernel_size=3, pad_mode=pad_mode, bias=bias, norm=norm, act=act, **kwargs)

class MaxPool2x2(nn.MaxPool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)

class MaxUnPool2x2(nn.MaxUnpool2d):
    def __init__(self, **kwargs):
        super().__init__(kernel_size=2, stride=(2,2), padding=(0,0), **kwargs)
        

class ConvTransposed3x3(nn.Module):
    def __init__(self, in_ch, out_ch, bias='auto', norm=False, act=False, **kwargs):
        super().__init__()
        seq = []
        seq.append(
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1,
                                   bias=(False if norm else True) if bias=='auto' else bias,
                                   **kwargs
                        )
            )        
        if norm:
            if norm is True:
                norm = make_norm(out_ch)
            seq.append(norm)
        if act:
            if act is True:
                act = make_act()
            seq.append(act)
        self.seq = nn.Sequential(*seq) 
        
    def forward(self, x):
        return self.seq(x)
            
"""
for testing the dimension of Conv3x3 block

conv_model = Conv3x3(in_ch=1, out_ch=16, norm=True, act=True)
print(summary(conv_model, input_size=(64, 1, 28, 28)))
"""
class UNet(nn.Module):
    def __init__(self, in_ch, out_ch, use_dropout=False):
        super().__init__()
        
        self.use_dropout = use_dropout
        
        self.conv11 = Conv3x3(in_ch, out_ch=16, norm=True, act=True)
        self.do11 = self.make_dropout()
        self.conv12 = Conv3x3(16, 16, norm=True, act=True)
        self.do12 = self.make_dropout()
        self.pool1 = MaxPool2x2()
        
        self.conv21 = Conv3x3(16, 32, norm=True, act=True)
        self.do21 = self.make_dropout()
        self.conv22 = Conv3x3(32, 32, norm=True, act=True)
        self.do22 = self.make_dropout()
        self.pool2 = MaxPool2x2()

        self.conv31 = Conv3x3(32, 64, norm=True, act=True)
        self.do31 = self.make_dropout()
        self.conv32 = Conv3x3(64, 64, norm=True, act=True)
        self.do32 = self.make_dropout()
        self.conv33 = Conv3x3(64, 64, norm=True, act=True)
        self.do33 = self.make_dropout()
        self.pool3 = MaxPool2x2()

        self.conv41 = Conv3x3(64, 128, norm=True, act=True)
        self.do41 = self.make_dropout()
        self.conv42 = Conv3x3(128, 128, norm=True, act=True)
        self.do42 = self.make_dropout()
        self.conv43 = Conv3x3(128, 128, norm=True, act=True)
        self.do43 = self.make_dropout()
        self.pool4 = MaxPool2x2()

        self.upconv4 = ConvTransposed3x3(128, 128, output_padding=1)

        self.conv43d = Conv3x3(256, 128, norm=True, act=True)
        self.do43d = self.make_dropout()
        self.conv42d = Conv3x3(128, 128, norm=True, act=True)
        self.do42d = self.make_dropout()
        self.conv41d = Conv3x3(128, 64, norm=True, act=True)
        self.do41d = self.make_dropout()

        self.upconv3 = ConvTransposed3x3(64, 64, output_padding=1)

        self.conv33d = Conv3x3(128, 64, norm=True, act=True)
        self.do33d = self.make_dropout()
        self.conv32d = Conv3x3(64, 64, norm=True, act=True)
        self.do32d = self.make_dropout()
        self.conv31d = Conv3x3(64, 32, norm=True, act=True)
        self.do31d = self.make_dropout()

        self.upconv2 = ConvTransposed3x3(in_ch=32, out_ch=32, output_padding=1)

        self.conv22d = Conv3x3(64, 32, norm=True, act=True)
        self.do22d = self.make_dropout()
        self.conv21d = Conv3x3(32, 16, norm=True, act=True)
        self.do21d = self.make_dropout()

        self.upconv1 = ConvTransposed3x3(in_ch=16, out_ch=16, output_padding=1)

        self.conv12d = Conv3x3(32, 16, norm=True, act=True)
        self.do12d = self.make_dropout()
        self.conv11d = Conv3x3(16, out_ch)

    def forward(self, x):
        
        #x = torch.cat([t1, t2], dim=1)
        
        # Stage 1
        x11 = self.do11(self.conv11(x))
        x12 = self.do12(self.conv12(x11))
        x1p = self.pool1(x12)

        # Stage 2
        x21 = self.do21(self.conv21(x1p))
        x22 = self.do22(self.conv22(x21))
        x2p = self.pool2(x22)

        # Stage 3
        x31 = self.do31(self.conv31(x2p))
        x32 = self.do32(self.conv32(x31))
        x33 = self.do33(self.conv33(x32))
        x3p = self.pool3(x33)

        # Stage 4
        x41 = self.do41(self.conv41(x3p))
        x42 = self.do42(self.conv42(x41))
        x43 = self.do43(self.conv43(x42))
        x4p = self.pool4(x43)

        # Stage 4d
        x4d = self.upconv4(x4p)
        pad4 = (0, x43.shape[3]-x4d.shape[3], 0, x43.shape[2]-x4d.shape[2])
        x4d = torch.cat([F.pad(x4d, pad=pad4, mode='replicate'), x43], 1)
        x43d = self.do43d(self.conv43d(x4d))
        x42d = self.do42d(self.conv42d(x43d))
        x41d = self.do41d(self.conv41d(x42d))

        # Stage 3d
        x3d = self.upconv3(x41d)
        pad3 = (0, x33.shape[3]-x3d.shape[3], 0, x33.shape[2]-x3d.shape[2])
        x3d = torch.cat([F.pad(x3d, pad=pad3, mode='replicate'), x33], 1)
        x33d = self.do33d(self.conv33d(x3d))
        x32d = self.do32d(self.conv32d(x33d))
        x31d = self.do31d(self.conv31d(x32d))

        # Stage 2d
        x2d = self.upconv2(x31d)
        pad2 = (0, x22.shape[3]-x2d.shape[3], 0, x22.shape[2]-x2d.shape[2])
        x2d = torch.cat([F.pad(x2d, pad=pad2, mode='replicate'), x22], 1)
        x22d = self.do22d(self.conv22d(x2d))
        x21d = self.do21d(self.conv21d(x22d))

        # Stage 1d
        x1d = self.upconv1(x21d)
        pad1 = (0, x12.shape[3]-x1d.shape[3], 0, x12.shape[2]-x1d.shape[2])
        x1d = torch.cat([F.pad(x1d, pad=pad1, mode='replicate'), x12], 1)
        x12d = self.do12d(self.conv12d(x1d))
        x11d = self.conv11d(x12d)

        return x11d
        
        
    def make_dropout(self):
        if self.use_dropout:
            return nn.Dropout2d(p=0.2)
        else:
            return Identity()

"""
a0 = torch.randn(64, 1, 28, 28)
a1 = torch.randn(64, 1, 28, 28)

x = torch.cat([a0, a1], dim=1)

#for testing the dimension of UNet Model

unet_model = UNet(in_ch=2, out_ch=1, use_dropout=False)
print(summary(unet_model, input_size=((64, 2, 256, 256))))
"""        












   