import torch.nn as nn
import torch
import math
from torch.autograd import Variable
from ops import *
#os.environ['CUDA_VISIBLE_DEVICES'] = '2'

class Discriminator(nn.Module):

    def __init__(self):
        super(Discriminator, self).__init__()
        self.convs = nn.Sequential( # [-1, 4, 256,192]
                conv2d(4,3,kernel_size=1),
                relu(),
                conv2d(3,32,kernel_size=3), #[-1, 32, 256, 192]
                relu(),
                maxpool2d(),
                conv2d(32,64,kernel_size=3), #[-1, 64, 128, 96]
                relu(),
                conv2d(64,64,kernel_size=3), #[-1, 64, 128, 96]
                relu(),
                maxpool2d(), #[-1,64,64,48]
                conv2d(64,64,kernel_size=3), #[-1,64,64,48]
                relu(),
                conv2d(64,64,kernel_size=3),
                relu(),
                maxpool2d(), #[-1,64,32,24]
        )
        self.mymodules = nn.ModuleList([
            nn.Sequential(nn.Linear(64*32*24, 100), nn.Tanh()),
            nn.Sequential(nn.Linear(100,2), nn.Tanh()),
            nn.Sequential(nn.Linear(2,1), nn.Sigmoid())
        ])
        #self._initialize_weights()

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self.num_flat_features(x))
        x = self.mymodules[0](x)
        x = self.mymodules[1](x)
        x = self.mymodules[2](x)
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                #print(m.weight.data.shape)
                #print('old conv2d layer!')
                #print(m.weight.data.min())
                #print(m.weight.data.max())
                m.weight.data.normal_(0, math.sqrt(2. / n))
                #print('new conv2d layer!')
                #print(m.weight.data.min())
                #print(m.weight.data.max())
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
#D = Discriminator()
#x = Variable(torch.rand([17, 4, 192, 256]))
#model = Discriminator()
#print('Discriminator input', x.size()) #[-1, 4, 192, 256] because 4 comes from 3 color channel + salience layer.
#out = model(x)
#print('Discriminator out ', out.size()) #[-1, 1]
