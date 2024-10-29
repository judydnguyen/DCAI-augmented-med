from __future__ import print_function
import argparse
import datetime
import os
import os.path
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.utils.data import SubsetRandomSampler
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from IPython.display import clear_output
from tqdm import tqdm

from src.utils import get_transform
from src.constants import *
from src.models import *
from src.train_utils import *

data_dir = '/home/judy/code/sys-med/AUGMENTATION_GAN/custom_covid_dataset/'
PATH='./gan_models/epoch_200/p_virus_200_2020-08-22_15:49:13.dat' #P_vir_200_opt
PATH='./gan_models/epoch_200/p_bacteria_200_2020-08-22_16:21:47.dat' #P_bac_200_opt
PATH='./gan_models/epoch_200/normal_200_2020-08-22_16:38:52.dat' #Normal_200_opt
PATH='./gan_models/epoch_200/covid_200_2020-08-22_16_58_21.dat' #Covid_200_opt

if __name__ == '__main__':
    # create new folders if not exist
    os.makedirs(os.path.join('.', 'augGAN'), exist_ok=True)
    os.makedirs(os.path.join('.', 'augGAN/model'), exist_ok=True)
    os.makedirs(os.path.join('.', 'augGAN/plots'), exist_ok=True)
    os.makedirs(os.path.join('.', 'augGAN/output_images'), exist_ok=True)
    METRIC_FIELDS = [
    'train.D_x',
    'train.D_G_z1',
    'train.D_G_z2',
    'train.G_losses',
    'train.D_losses',
    'train.D_matching_loss',
    'train.D_proto_loss',
    'train.D_var_loss'
    ]
    metrics = {field: list() for field in METRIC_FIELDS}

    transform = get_transform(nc)

    trainset0 = datasets.ImageFolder(os.path.join(
            data_dir, "train/"), transform=transform)
    trainset500 = datasets.ImageFolder(os.path.join(
            data_dir, "train_classic/500/"), transform=transform)
    trainset1000 = datasets.ImageFolder(os.path.join(
            data_dir, "train_classic/1000/"), transform=transform)
    trainset2000 = datasets.ImageFolder(os.path.join(
            data_dir, "train_classic/2000/"), transform=transform)
    listtrainset_no_aug = [trainset0]
    listtrainset_classic = [trainset500,trainset1000]#,trainset2000]
    listtrainset = listtrainset_no_aug + listtrainset_classic
    train_set = torch.utils.data.ConcatDataset(listtrainset)

    # train_set = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform=transform)

    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    gen_optimizer = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))


    if LOAD_MODEL:
        if torch.cuda.is_available():
            checkpoint = torch.load(PATH)
        else:
            checkpoint = torch.load(PATH, map_location=lambda storage, loc: storage)
                
        generator.load_state_dict(checkpoint['state_dict_generator'])
        discriminator.load_state_dict(checkpoint['state_dict_discriminator'])
        gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
        dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
        metrics=checkpoint['metrics']
        num_epochs=checkpoint['train_epoch']
        date=checkpoint['date']
        generator.train(mode=False)
        discriminator.train(mode=False)
        print('GAN loaded for epochs: ', num_epochs)
        print(generator)
        print(discriminator)
        print(gen_optimizer)
        print(dis_optimizer)
        print(date)
        test1(generator, discriminator, num_epochs, metrics)
    else:
        if TRAIN_ALL:
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                    shuffle=True)
            train_gan(generator, discriminator, gen_optimizer, dis_optimizer, train_loader,
                    num_epochs, metrics, device=device, criterion=criterion)
            test2(generator, discriminator, num_epochs, metrics, train_loader)
        else:
            # idx = []
            # idx = get_indices(train_set, 4, idx) #second argument is 0 for covid; 1 for normal; 2 for pneumonia_bacteria; 3 for pneumonia_virus for x-ray dataset
    
            train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                    shuffle=True)
            mask =[x[1]==0 for x in train_loader.dataset] #here is 0 for covid; 1 for normal; 2 for pneumonia_bacteria; 3 for pneumonia_virus for x-ray dataset
            idx= np.arange(len(train_loader.dataset))[mask]

            print("Total samples now are ",len(idx))
            selected_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                                        sampler = SubsetRandomSampler(idx))
            train_gan(generator, discriminator, gen_optimizer, dis_optimizer, selected_loader,
                    num_epochs, metrics, device, criterion)
            test2(generator, discriminator, num_epochs, metrics, selected_loader)