from __future__ import print_function
import argparse
import datetime
import json
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.display import clear_output
from tqdm import tqdm
import datetime

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torchvision.utils import save_image
from torch.autograd import Variable
from torch.utils.data import SubsetRandomSampler

from src.data_utils import load_train_test_df
from src.models import _netD, _netG, _netD_CIFAR10, _netG_CIFAR10
from src.constants import *
from src.utils import compute_acc, get_transform, plot, set_seed
import matplotlib.pyplot as plt

def plot_losses(loss_D, loss_G, loss_A, num_epochs):
    num_epochs = len(loss_D)
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), loss_D, label='Discriminator Loss')
    plt.plot(range(1, num_epochs + 1), loss_G, label='Generator Loss')
    plt.plot(range(1, num_epochs + 1), loss_A, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig('output/acgan_skin/losses.png')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def smooth_labels(labels, smoothing=0.1):
    """
    Apply label smoothing to a tensor of labels.

    Args:
        labels (torch.Tensor): The original label tensor (e.g., all 1s for real or 0s for fake).
        smoothing (float): The amount of smoothing to apply (0.0 means no smoothing).

    Returns:
        torch.Tensor: The smoothed labels.
    """
    with torch.no_grad():
        smoothed_labels = labels * (1.0 - smoothing) + 0.5 * smoothing
    return smoothed_labels

def train_acgan(discriminator, generator, optimizerD, optimizerG , train_loader, num_epochs, metrics, args):
    for param_group in optimizerG.param_groups:
        param_group['lr'] *= 1.1  # Increase by 10%
    for param_group in optimizerD.param_groups:
        param_group['lr'] *= 0.9  # Decrease by 10%

    # loss functions
    dis_criterion = nn.BCELoss()
    aux_criterion = nn.NLLLoss()
    
    avg_loss_D = 0
    avg_loss_G = 0
    avg_loss_A = 0
    
    # define shapes
    input = torch.FloatTensor(args.batch_size, args.nc, args.image_size, args.image_size)
    noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1)
    eval_noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1).normal_(0, 1)
    dis_label = torch.FloatTensor(args.batch_size)
    aux_label = torch.LongTensor(args.batch_size)
    real_label = 1
    fake_label = 0
    eval_label = np.random.randint(0, args.num_classes, args.batch_size)
    # Store losses for plotting
    epoch_loss_D = []
    epoch_loss_G = []
    epoch_accuracy = []
    
    for epoch in tqdm(range(num_epochs), desc='Epochs'):
        total_loss_D = 0
        total_loss_G = 0
        total_accuracy = 0
        for i, data in tqdm(enumerate(train_loader, 0), desc='Batches', total=len(train_loader)):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train with real
            discriminator.zero_grad()
            real_images, labels = data
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            labels = labels.to(device)
            
            noise = torch.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
            input = torch.FloatTensor(batch_size, args.nc, args.image_size, args.image_size)
            eval_noise = torch.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
            dis_label = torch.FloatTensor(batch_size)
            aux_label = torch.LongTensor(batch_size)
            
            input.data.resize_(real_images.size()).copy_(real_images)
            dis_label.data.resize_(batch_size).fill_(real_label)
            dis_label = smooth_labels(dis_label, smoothing=0.1)  # Apply 0.1 label smoothing
            aux_label.data.resize_(batch_size).copy_(labels)
            input = input.to(device)
            dis_label = dis_label.to(device)
            aux_label = aux_label.to(device)
            # import IPython; IPython.embed()
            dis_output, aux_output, _ = discriminator(input)
            # import IPython; IPython.embed()
            
            dis_errD_real = dis_criterion(dis_output, dis_label)
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward()
            D_x = dis_output.data.mean()
            
            # import IPython; IPython.embed()
            # compute current classification accuracy
            accuracy = compute_acc(aux_output, labels)
            
            # train with fake
            noise.data.resize_(batch_size, args.nz, 1, 1).normal_(0, 1)
            label = np.random.randint(0, args.num_classes, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, args.nz))
            class_onehot = np.zeros((batch_size, args.num_classes))
            class_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :args.num_classes] = class_onehot[np.arange(batch_size)]
            noise_ = (torch.from_numpy(noise_))
            

            noise.data.copy_(noise_.view(batch_size, args.nz, 1, 1))
            noise = noise.to(device)
            aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
            
            fake = generator(noise)
            dis_label.data.fill_(fake_label)
            dis_output, aux_output, _ = discriminator(fake.detach())
            dis_errD_fake = dis_criterion(dis_output, dis_label)
            aux_errD_fake = aux_criterion(aux_output, aux_label)
            errD_fake = dis_errD_fake + aux_errD_fake
            errD_fake.backward()
            D_G_z1 = dis_output.data.mean()
            errD = errD_real + errD_fake
            optimizerD.step()
            

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            dis_label.data.fill_(real_label)
            dis_output, aux_output, _ = discriminator(fake)
            dis_errG = dis_criterion(dis_output, dis_label)
            aux_errG = aux_criterion(aux_output, aux_label)
            errG = dis_errG + aux_errG
            errG.backward()
            D_G_z2 = dis_output.data.mean()
            optimizerG.step()
            
            # compute the average loss
            curr_iter = epoch * len(train_loader) + i
            all_loss_G = avg_loss_G * curr_iter
            all_loss_D = avg_loss_D * curr_iter
            all_loss_A = avg_loss_A * curr_iter
            all_loss_G += errG.item()
            all_loss_D += errD.item()
            all_loss_A += accuracy
            avg_loss_G = all_loss_G / (curr_iter + 1)
            avg_loss_D = all_loss_D / (curr_iter + 1)
            avg_loss_A = all_loss_A / (curr_iter + 1)
            # Accumulate losses
            total_loss_D += errD.item()
            total_loss_G += errG.item()
            total_accuracy += accuracy
            
            
            print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
                % (epoch, args.num_epochs, i, len(train_loader),
                    errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
            
            if i % 100 == 0:
                vutils.save_image(
                    real_images.cpu(), '%s/real_samples.png' % args.output_dir)
                print('Label for eval = {}'.format(eval_label))
                fake = generator(eval_noise.to(device))
                vutils.save_image(
                    fake.cpu(),
                    '%s/fake_samples_epoch_%03d.png' % (args.output_dir, epoch)
                )

        if epoch % 50 == 0:
            # save_checkpoint(generator, discriminator, args, epoch)
            # Get the current time for file naming
            current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            # Save the generator's state
            torch.save(generator.state_dict(), '%s/netG_epoch_%d_%s.pth' % (args.checkpoint_dir, epoch, current_time))

            # Save the discriminator's state
            torch.save(discriminator.state_dict(), '%s/netD_epoch_%d_%s.pth' % (args.checkpoint_dir, epoch, current_time))
        avg_loss_D = total_loss_D / len(train_loader)
        avg_loss_G = total_loss_G / len(train_loader)
        avg_accuracy = total_accuracy / len(train_loader)
        epoch_loss_D.append(avg_loss_D)
        epoch_loss_G.append(avg_loss_G)
        epoch_accuracy.append(avg_accuracy)
        plot_losses(epoch_loss_D, epoch_loss_G, epoch_accuracy, num_epochs)

def main(args):
    manualSeed = args.seed
    set_seed(manualSeed)
    torch.backends.cudnn.benchmark = True

    
    # load pre-processed skin lession dataset
    train_loader, val_dl, subset_train = load_train_test_df(input_size=args.image_size, batch_size=args.batch_size)
    # train_loader = subset_train
    # load generator and discriminator
    generator = _netG_CIFAR10(1, args.nz).to(device)
    discriminator = _netD_CIFAR10(1, args.num_classes).to(device)
    
    print(generator)
    print(discriminator)
    # netG.apply(weights_init)
    # netD.apply(weights_init)
    
    if args.load_checkpoint:
        generator.load_state_dict(torch.load("output/acgan_skin/saved_models/0.002/netG_epoch_99.pth"))
        discriminator.load_state_dict(torch.load("output/acgan_skin/saved_models/0.002/netD_epoch_99.pth"))
        # generator.load_state_dict(torch.load('%s/netG_epoch_%d.pth' % (args.checkpoint_dir, 99)))
        # discriminator.load_state_dict(torch.load('%s/netD_epoch_%d.pth' % (args.checkpoint_dir, 99)))
    generator.train()
    discriminator.train()
    # get corresponding optimizers
    optimizerD = optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(args.beta1, 0.999))
    optimizerG = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    
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
    train_acgan(discriminator, generator, optimizerD, optimizerG, train_loader, args.num_epochs, metrics, args)

if __name__=='__main__':
    # Set random seed for reproducibility
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='path to dataset')
    parser.add_argument('--num_epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--nz', type=int, default=110, help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64, help='size of feature maps in generator')
    parser.add_argument('--ndf', type=int, default=64, help='size of feature maps in discriminator')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
    parser.add_argument('--d_lr', type=float, default=0.0002, help='learning rate for discriminator')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam optimizer')
    parser.add_argument('--workers', type=int, default=54, help='number of data loading workers')
    parser.add_argument('--output_dir', type=str, default='output', help='output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='checkpoint directory')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='checkpoint interval')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--seed', type=int, default=999, help='random seed')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    parser.add_argument('--nc', type=int, default=3, help='number of channels')
    parser.add_argument('--image_size', type=int, default=64, help='image size')
    parser.add_argument('--config', type=str, default='config.json', help='config file')
    parser.add_argument('--load_checkpoint', type=bool, default=False, help='load checkpoint')
    args = parser.parse_args()
    
    # parse configs from json file
    config_file = args.config
    with open(config_file) as f:
        data = f.read()
    config = json.loads(data)
    for key, value in config.items():
        setattr(args, key, value)

    print(config)
    print(args)
    args.output_dir = os.path.join(args.output_dir, str(args.lr))
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, str(args.lr))
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    main(args)



    # test2(generator, discriminator, num_epochs, metrics, subset_loader)

    # trainset0 = datasets.ImageFolder(os.path.join(
    #         data_dir, "train/"), transform=transform)
    # trainset500 = datasets.ImageFolder(os.path.join(
    #         data_dir, "train_classic/500/"), transform=transform)
    # trainset1000 = datasets.ImageFolder(os.path.join(
    #         data_dir, "train_classic/1000/"), transform=transform)
    # trainset2000 = datasets.ImageFolder(os.path.join(
    #         data_dir, "train_classic/2000/"), transform=transform)
    # listtrainset_no_aug = [trainset0]
    # listtrainset_classic = [trainset500,trainset1000]#,trainset2000]
    # listtrainset = listtrainset_no_aug + listtrainset_classic
    # train_set = torch.utils.data.ConcatDataset(listtrainset)
    # train_set = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform=transform)

    # train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
    #                                       shuffle=True)
    
    