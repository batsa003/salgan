import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from torch.autograd import Variable
from ops import * 

def make_conv_layers(cfg):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [maxpool2d()]
        else:
            conv = conv2d(in_channels, v)
            layers += [conv, relu(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_deconv_layers(cfg):
    layers = []
    in_channels = 512
    for v in cfg:
        if v == 'U':
            layers += [nn.functional.interpolate(scale_factor=2)]
        else:
            deconv = deconv2d(in_channels, v)
            layers += [deconv]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'D': [512, 512, 512, 'U', 512, 512, 512, 'U', 256, 256, 256, 'U', 128, 128, 'U', 64, 64]
}

def encoder():
    return make_conv_layers(cfg['E'])

def decoder():
    return make_deconv_layers(cfg['D'])

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.encoder = encoder()
        self.decoder = decoder()
        self.mymodules = nn.ModuleList([
            deconv2d(64,1,kernel_size=1, padding = 0),
            nn.Sigmoid()
        ])

    def forward(self,x): #
        #print('Input x', x.size())
        x = self.encoder(x)
        #print('After encoder = ', x.size())
        x = self.decoder(x)
        #print('After decoder = ', x.size())
        x = self.mymodules[0](x)
        x = self.mymodules[1](x)
        #print('Final size = ', x.size())
        return x

#g = Generator()
#x = Variable(torch.rand([17, 3, 192, 256]))
#print('Input :', x.size())
#out = g(x)
#print('Output: ', out.size())
