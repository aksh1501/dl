from __future__ import print_function, division
import os
import numpy as np
import torch 
from data import VRVideo
import torchvision
from torchvision import datasets, models, transforms
from torch.utils import data as tdata
from torch.optim import SGD
import matplotlib.pyplot as plt
from torch.autograd import Variable
from argparse import ArgumentParser
from tqdm import trange, tqdm

import time

from spherical_unet import Final1
from sconv.module import SphericalConv, SphereMSE



def train_model(model, criterion, optimizer, num_epochs=25):
    #loader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
    #model = Final1()
    #optimizer = SGD(model.parameters(), lr, momentum=0.9, weight_decay=1e-5)
    #pmodel = nn.DataParallel(model).cuda()
    #criterion = SphereMSE(128, 256).float().cuda()
    #if resume:
     #   ckpt = th.load('ckpt-' + exp_name + '-latest.pth.tar')
      #  model.load_state_dict(ckpt['state_dict'])
       # start_epoch = ckpt['epoch']

   # log_file = open(exp_name +'.out', 'w+')
    #for epoch in trange(start_epoch, epochs, desc='epoch'):
    
    since = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


       # model.train()


        train_loss = 0
        #tic = time.time()
        #for i, (img_batch, last_batch, target_batch) in tqdm(enumerate(loader), desc='batch', total=len(loader)):
            #img_var = Variable(img_batch).cuda()
            #last_var = Variable(last_batch * 10).cuda()
            #t_var = Variable(target_batch * 10).cuda()
            #data_time = time.time() - tic
            #tic = time.time()

            #out = pmodel(img_var, last_var)
            #loss = criterion(out, t_var)
            #fwd_time = time.time() - tic
            #tic = time.time()

        for i,data in enumerate(dataloaders['train']):
            inputs,labels=data
           # inputs = Variable(inputs).cuda()
            #labels = Variable(labels).cuda()

            inputs = inputs.to(device)
            labels = labels.to(device)


            #out = pmodel(inputs)
            #loss = criterion(out, labels)
            
            #fwd_time = time.time() - tic
            #tic = time.time()


            optimizer.zero_grad()
          
            with torch.set_grad_enabled(True):
                outputs  = model(inputs)
                loss = criterion(outputs, labels)



            #optimizer.zero_grad()
            loss.backward()
            optimizer.step()
                
            train_loss+=loss.item() + inputs.size(0)

                 

            print('{} Loss: {:.4f}'.format('train', train_loss / dataset_sizes['train']))




            #bkw_time = time.time() - tic

            #msg = '[{:03d}|{:05d}/{:05d}] time: data={}, fwd={}, bkw={}, total={}\nloss: {:g}'.format(
            #epoch, i, len(loader), data_time, fwd_time, bkw_time, data_time+fwd_time+bkw_time, loss.data[0]
            #)
            #viz.images(target_batch.cpu().numpy() * 10, win='gt')
            #viz.images(out.data.cpu().numpy(), win='out')
            #viz.text(msg, win='log')
            #print(msg, file=log_file, flush=True)
            #print(msg, flush=True)

            #tic = time.time()

            #if (i + 1) % save_interval == 0:
                #state_dict = model.state_dict()
                #ckpt = dict(epoch=epoch, iter=i, state_dict=state_dict)
                #th.save(ckpt, 'ckpt-' + exp_name + '-latest.pth.tar')

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
       time_elapsed // 60, time_elapsed % 60))

    return model





def visualize_model(model, num_images=6):
   was_training = model.training
   #model.eval()
   images_so_far = 0
   fig = plt.figure()

   with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['validation']):
            #inputs = Variable(inputs).cuda()
            #labels = Variable(labels).cuda()

          inputs = inputs.to(device)
          labels = labels.to(device)

          outputs = model(inputs)
           
          print(outputs)
           
          _, preds = torch.max(outputs, 1)

          for j in range(inputs.size()[0]):
              images_so_far += 1
              ax = plt.subplot(num_images//2, 2, images_so_far)
              ax.axis('off')
              ax.set_title('predicted: {} truth: {}'.format(class_names[preds[j]], class_names[labels[j]]))
              img = inputs.cpu().data[j].numpy().transpose((1, 2, 0))
              img = std * img + mean
              ax.imshow(img)

              if images_so_far == num_images:
                  model.train(mode=was_training)
                  return
       #model.train(mode=was_training)















    
data_dir = "alien_pred"

transform = transforms.Compose([
           transforms.Resize((128, 256)),
           transforms.ToTensor()
           ])

#input_shape = 224
data_transforms = {
   'train': transforms.Compose([
       #transforms.CenterCrop(input_shape),
       transforms.Resize((128, 256)),
       transforms.ToTensor(),
      # transforms.Normalize(mean, std)
   ]),
   'validation': transforms.Compose([
       #transforms.CenterCrop(input_shape),
       transforms.Resize((128, 256)),
       transforms.ToTensor(),
       #transforms.Normalize(mean, std)
   ]),
}


image_datasets = {
   x: datasets.ImageFolder(
       os.path.join(data_dir, x),
       transform=data_transforms[x]
   )
   for x in ['train', 'validation']
}


   # dataset = VRVideo(data, 128, 256, 80, frame_interval=5, cache_gt=True, transform=transform, gaussian_sigma=np.pi/20, kernel_rad=np.pi/7)
    #if clear_cache:
     #   dataset.clear_cache()
    
dataloaders = {
   x: torch.utils.data.DataLoader(
       image_datasets[x], batch_size=32,
       shuffle=True, num_workers=4
   )
   for x in ['train', 'validation']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation']}


print(dataset_sizes)

print(dataloaders)

class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


images, labels = next(iter(dataloaders['train']))

#print(labels)

rows = 4
columns = 4

fig=plt.figure()

for i in range(16):
   fig.add_subplot(rows, columns, i+1)
   plt.title(class_names[labels[i]])
   img = images[i].numpy().transpose((1, 2, 0))
   #img = std * img + mean
   plt.imshow(img)
plt.show()







# loader = tdata.DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=16, pin_memory=True)
model = Final1()

## freeze the layers
#for param in vgg_based.parameters():
 #  param.requires_grad = False


#number_features = vgg_based.classifier[6].in_features
#features = list(vgg_based.classifier.children())[:-1] # Remove last layer
#features.extend([torch.nn.Linear(number_features, len(class_names))])


model = model.to(device)

optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-5)
#pmodel = nn.DataParallel(model).cuda()
#criterion = SphereMSE(128, 256).float().cuda()
criterion = SphereMSE(128, 256).float().to(device)   
    #if resume:
        #ckpt = th.load('ckpt-' + exp_name + '-latest.pth.tar')
        #model.load_state_dict(ckpt['state_dict'])
        #start_epoch = ckpt['epoch']

    #log_file = open(exp_name +'.out', 'w+')

model = train_model(model, criterion, optimizer ,num_epochs=25)

visualize_model(model)

plt.show()
