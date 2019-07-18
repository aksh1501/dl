import pandas as pd
import numpy as np
import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch import optim
from skimage import io, transform
import glob2
#import cv2
from skimage.util import random_noise

from PIL import Image

image_p=[]   
#for f in glob2.iglob(r"C:\Users\abheesht\Desktop\Code\Train001\*"):
#    image_p.append(f)

for f in glob2.iglob(r"alien_pred/train/fall/*"):
    image_p.append(f)


class ImageDataset(Dataset):
    

    def __init__(self, image_path, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.image_path = image_path
        #self.root_dir = root_dir
        self.transform = transform

        
    def __len__(self):
        return(len(self.image_path))
    
    def __getitem__(self, idx):
        img_name =(self.image_path)[idx]        
        image = Image.open(img_name).convert('RGB')
        


                                    
        if self.transform:            
            image = self.transform(image)

        label = np.array([1 for i in range(len(self.image_path))])
        
        sample = (image,label)      
        return sample


transform = transforms.Compose([transforms.Resize((64,64)),transforms.ToTensor(),transforms.Lambda(lambda x : x + 0.1*torch.randn_like(x))])
#transforms.Lambda(lambda x : x + torch.randn_like(x))
trainset = ImageDataset(image_path=image_p, transform=transform)

#trainloader = torch.utils.data.DataLoader(trainset, batch_size=8, shuffle=True, num_workers=0)
sample=trainset[0]
print(sample[0])
new_im = Image.fromarray(np.transpose(np.array(sample[0]), (1, 2, 0)))


#new_im.save(r"C:\Users\abheesht\Desktop\Code\numpy_altered_sample2.png")
new_im.save(r"alien_pred/numpy_altered_sample2.jpeg")
plt.show()
