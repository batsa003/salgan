import glob
import os
import numpy as np
import cv2
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from constants import *

class DataLoader(object):

    def __init__(self, batch_size = 5):
        #reading data list
        self.list_img = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(pathToResizedImagesTrain, '*train*'))]
        self.batch_size = batch_size
        self.size = len(self.list_img)
        self.cursor = 0
        self.num_batches = self.size / batch_size

    def get_batch(self): # Returns 
        if self.cursor + self.batch_size > self.size:
            self.cursor = 0
            np.random.shuffle(self.list_img)
            
        img = torch.zeros(self.batch_size, 3, 192, 256)
        sal_map = torch.zeros(self.batch_size, 1, 192, 256)
        
        #to_tensor = transform.Compose(transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.
        to_tensor = transforms.ToTensor() # Transforms 0-255 numbers to 0 - 1.0.

        for idx in xrange(self.batch_size):
            curr_file = self.list_img[self.cursor]
            full_img_path = os.path.join(pathToResizedImagesTrain, curr_file + '.png')
            full_map_path = os.path.join(pathToResizedMapsTrain, curr_file + '.png')
            self.cursor += 1
            inputimage = cv2.imread(full_img_path) # (192,256,3)
            img[idx] = to_tensor(inputimage)
            
            saliencyimage = cv2.imread(full_map_path,0)
            saliencyimage = np.expand_dims(saliencyimage,axis=2)
            sal_map[idx] = to_tensor(saliencyimage)
            
        return (img,sal_map)
#d = DataLoader()
#(imgs,maps) = d.get_batch()
#I1 = imgs[4]
#S1 = maps[4]

#print('Data Loader implemented.')
#
#def show(img):
#    print(img.shape)
#    pilTrans = transforms.ToPILImage()
#    pilImg = pilTrans(img)
#    s = np.array(pilImg)
#    plt.figure()
#    plt.imshow(s)
#show(I1)
#show(S1)
