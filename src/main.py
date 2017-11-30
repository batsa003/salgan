from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
import time
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
from tqdm import tqdm
from torch.autograd import Variable
from logger import Logger

from data_loader import DataLoader
from discriminator import Discriminator
from generator import Generator
from utils import *

logger = Logger('./logs')
batch_size = 40
lr = 0.0003

discriminator = Discriminator()
generator = Generator()
if torch.cuda.is_available():
    discriminator.cuda()
    generator.cuda()
loss_function = nn.BCELoss()

d_optim = torch.optim.Adagrad(discriminator.parameters(), lr=lr)
g_optim = torch.optim.Adagrad(generator.parameters(), lr=lr)

num_epoch = 120
dataloader = DataLoader(batch_size)
num_batch = dataloader.num_batches# length of data / batch_size
print(num_batch)

def to_variable(x,requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x,requires_grad)

counter = 0
start_time = time.time()
DIR_TO_SAVE = "./generator_output/"
if not os.path.exists(DIR_TO_SAVE):
    os.makedirs(DIR_TO_SAVE)
validation_sample = cv2.imread("COCO_val2014_000000143859.png")

for current_epoch in tqdm(range(1,num_epoch+1)):
    n_updates = 1
    
    d_cost_avg = 0
    g_cost_avg = 0
    for idx in range(num_batch):
            
        (batch_img, batch_map) = dataloader.get_batch()
        batch_img = to_variable(batch_img,requires_grad=False)
        batch_map = to_variable(batch_map,requires_grad=False)
        real_labels = to_variable(torch.FloatTensor(np.ones(batch_size, dtype = float)),requires_grad=False)
        fake_labels = to_variable(torch.FloatTensor(np.zeros(batch_size, dtype = float)),requires_grad=False)
        
        if n_updates % 2 == 1:
            #print('Training Discriminator...')
            #discriminator.zero_grad()
            d_optim.zero_grad()
            inp_d = torch.cat((batch_img,batch_map),1)            
            #print(inp_d.size())
            outputs = discriminator(inp_d).squeeze()
            d_real_loss = loss_function(outputs,real_labels)
            #print('D_real_loss = ', d_real_loss.data[0])
            
            #print(outputs)
            real_score = outputs.data.mean()

#            fake_map = generator(batch_img)
#            inp_d = torch.cat((batch_img,fake_map),1)
#            outputs = discriminator(inp_d)
#            d_fake_loss = loss_function(outputs, fake_labels)
#            print('D_fake_loss = ', d_fake_loss.data[0])
            d_loss = torch.sum(torch.log(outputs))
            d_cost_avg += d_loss.data[0]
            
            d_loss.backward()
            d_loss.register_hook(print)
            d_optim.step()
            info = {
                 'd_loss' : d_loss.data[0],
                 'real_score_mean' : real_score,
            }
            for tag,value in info.items():
                logger.scalar_summary(tag, value, counter)
        else:
            #print('Training Generator...')
            #generator.zero_grad()
            g_optim.zero_grad()
            fake_map = generator(batch_img)
            inp_d = torch.cat((batch_img,fake_map),1)
            outputs = discriminator(inp_d)
            fake_score = outputs.data.mean()

            g_gen_loss = loss_function(fake_map,batch_map)
            g_dis_loss = -torch.log(outputs)
            alpha = 0.05
            g_loss = torch.sum(g_dis_loss + alpha * g_gen_loss)

            g_cost_avg += g_loss.data[0]
            
            g_loss.backward()
            g_optim.step()
            info = {
                  'g_loss' : g_loss.data[0],
                  'fake_score_mean' : fake_score,
            }
            for tag,value in info.items():
                logger.scalar_summary(tag, value, counter)

        n_updates += 1

        if (idx+1)%100 == 0:
            print("Epoch [%d/%d], Step[%d/%d], d_loss: %.4f, g_loss: %.4f, D(x): %2.f, D(G(x)): %.2f, time: %4.4f"
		        % (current_epoch, num_epoch, idx+1, num_batch, d_loss.data[0], g_loss.data[0],
		        real_score, fake_score, time.time()-start_time))
        counter += 1
    d_cost_avg /= num_batch
    g_cost_avg /= num_batch
        
    # Save weights every 3 epoch
    if (current_epoch + 1) % 3 == 0:
        print('Epoch:', current_epoch, ' train_loss->', (d_cost_avg, g_cost_avg))
        torch.save(generator.state_dict(), './generator.pkl')
        torch.save(discriminator.state_dict(), './discriminator.pkl')
    predict(generator, validation_sample, current_epoch, DIR_TO_SAVE)
torch.save(generator.state_dict(), './generator.pkl')
torch.save(discriminator.state_dict(), './discriminator.pkl')
print('Done')
