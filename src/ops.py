import torch.nn as nn

def conv2d(in_channels, out_channels, kernel_size = 3, padding = 1):
    return nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)

def deconv2d(in_channels, out_channels, kernel_size = 3, padding = 1):
    return nn.ConvTranspose2d(in_channels, out_channels, kernel_size = kernel_size, padding = padding)

def relu(inplace = True): # Change to True?
    return nn.ReLU(inplace)

def maxpool2d():
    return nn.MaxPool2d(2)
