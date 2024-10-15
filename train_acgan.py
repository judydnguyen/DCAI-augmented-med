from __future__ import print_function
import argparse
import datetime
import os
import random
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import HTML
from IPython.display import clear_output
from tqdm import tqdm

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

from src.models import netD, netG
from src.constants import *
from src.utils import get_transform, plot

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_dir = '/home/judy/code/sys-med/AUGMENTATION_GAN/custom_covid_dataset/'
batch_size = 32

def save_model(generator, discriminator, gen_optimizer, dis_optimizer, metrics, num_epochs):
    now = datetime.datetime.now()
    g_losses = metrics['train.G_losses'][-1]
    d_losses = metrics['train.D_losses'][-1]
    name = "%+.3f_%+.3f_%d_%s.dat" % (g_losses, d_losses, num_epochs, now.strftime("%Y-%m-%d_%H:%M:%S"))
    # fname = os.path.join('.', 'augGAN/model', name)
    # states = {
    #         'state_dict_generator': generator.state_dict(),
    #         'state_dict_discriminator': discriminator.state_dict(),
    #         'gen_optimizer': gen_optimizer.state_dict(),
    #         'dis_optimizer': dis_optimizer.state_dict(),
    #         'metrics': metrics,
    #         'train_epoch': num_epochs,
    #         'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
    # }
    # torch.save(states, fname)
    path='augGAN/plots/ACGAN/train_%+.3f_%+.3f_%s'% (g_losses, d_losses, now.strftime("%Y-%m-%d_%H:%M:%S"))
    try:
      os.makedirs(os.path.join('.', path), exist_ok=True)
    except Exception as error:
      print(error)

    plot('G_losses', num_epochs, metrics['train.G_losses'], path, True)
    plot('D_losses', num_epochs, metrics['train.D_losses'], path, True)
    # plot('D_x', num_epochs, metrics['train.D_x'], path, True)
    # plot('D_G_z1', num_epochs, metrics['train.D_G_z1'], path, True)
    # plot('D_G_z2', num_epochs, metrics['train.D_G_z2'], path, True)
    
def test(predict, labels):
    correct = 0
    pred = predict.data.max(1)[1]
    correct = pred.eq(labels.data).cpu().sum()
    return correct, len(labels.data)

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import os

# Example variables (replace with your values)
nz = 100  # latent vector size
nb_label = 4  # number of classes (e.g., 4)
imageSize = 64
lr_d = 0.0002
lr = 0.0002
beta1, beta2 = 0.5, 0.999
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
real_label, fake_label = 1, 0

# Create a dictionary to store the running average prototypes for each class
class_prototypes = {i: torch.zeros(nz, device=device) for i in range(nb_label)}

def update_prototype(current_feat, current_label, total_feat_mean):
    """
    Updates the class-specific prototypes by averaging the features over the epochs.
    """
    # For each class, calculate its new prototype
    for cls in range(nb_label):
        # Get the mean feature for the current class
        mask = current_label == cls
        if mask.sum() > 0:  # If the class exists in the batch
            class_feat = current_feat[mask].mean(0)
            if total_feat_mean[cls].sum() == 0:
                total_feat_mean[cls] = class_feat
            else:
                total_feat_mean[cls] = (total_feat_mean[cls] + class_feat) / 2  # Moving average
    return total_feat_mean

def train_acgan(discriminator, generator, train_loader, num_epochs, metrics, batch_size=32):
    
    optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
    optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
    # Input to generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device) #batch of 64
    # Define Loss function
    s_criterion = nn.BCELoss().to(device) #For synthesizing
    c_criterion = nn.NLLLoss().to(device) #For classification
    
    input = torch.FloatTensor(batch_size, 3, imageSize, imageSize).to(device)
    s_label = torch.FloatTensor(batch_size).to(device)
    c_label = torch.LongTensor(batch_size).to(device)
    noise = torch.FloatTensor(batch_size, nz, 1, 1).to(device)
    
    fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
    
    total_feat_mean_real = {i: torch.zeros(nz, device=device) for i in range(nb_label)}  # Store class prototypes

    for epoch in range(num_epochs):
        for i, data in enumerate(tqdm(train_loader, 0)):
            ###########################
            # (1) Update D network
            ###########################
            # train with real
            discriminator.zero_grad()
            img, label = data
            batch_size = img.size(0)
            with torch.no_grad():
                input.resize_(img.size()).copy_(img)
                s_label.resize_(batch_size).fill_(real_label)
                c_label.resize_(batch_size).copy_(label)
            
            s_output, c_output, f_R = discriminator(input)
            s_output = s_output.view(-1)
            s_errD_real = s_criterion(s_output, s_label)
            c_errD_real = c_criterion(c_output, c_label)
            errD_real = s_errD_real + c_errD_real
            errD_real.backward()
            D_x = s_output.data.mean()
            
            correct, length = test(c_output, c_label)

            # Update prototypes for real images
            total_feat_mean_real = update_prototype(f_R.detach(), c_label, total_feat_mean_real)

            # train with fake
            with torch.no_grad():
                noise.resize_(batch_size, nz, 1, 1)
                noise.normal_(0, 1)

            label = np.random.randint(0, nb_label, batch_size)
            noise_ = np.random.normal(0, 1, (batch_size, nz))
            label_onehot = np.zeros((batch_size, nb_label))
            label_onehot[np.arange(batch_size), label] = 1
            noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
            
            noise_ = (torch.from_numpy(noise_))
            noise_ = noise_.resize_(batch_size, nz, 1, 1)
            noise.data.copy_(noise_)

            c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

            fake = generator(noise)
            s_label.data.fill_(fake_label)
            s_output, c_output, _ = discriminator(fake.detach())
            
            s_output = s_output.view(-1)
            s_errD_fake = s_criterion(s_output, s_label)
            c_errD_fake = c_criterion(c_output, c_label)
            errD_fake = s_errD_fake + c_errD_fake

            errD_fake.backward()
            D_G_z1 = s_output.data.mean()
            errD = s_errD_real + s_errD_fake
            optimizerD.step()

            ###########################
            # (2) Update G network
            ###########################
            generator.zero_grad()
            s_label.data.fill_(real_label)  # fake labels are real for generator cost
            
            s_output, c_output, f_F = discriminator(fake)
            s_output = s_output.view(-1)
            s_errG = s_criterion(s_output, s_label)
            c_errG = c_criterion(c_output, c_label)
            
            # Prototype loss: Compare generated features to the prototype of the class they belong to
            proto_loss = 0
            for cls in range(nb_label):
                mask = c_label == cls
                if mask.sum() > 0:
                    proto_loss += torch.mean((f_F[mask] - total_feat_mean_real[cls].detach()) ** 2)

            # Final Generator loss
            errG = s_errG + c_errG + 0.2*proto_loss
            errG.backward(retain_graph=True)
            D_G_z2 = s_output.data.mean()
            
            optimizerG.step()

            # Logging metrics
            metrics['train.G_losses'].append(errG.item())
            metrics['train.D_losses'].append(errD.item())
            metrics['train.D_proto_loss'].append(proto_loss.item())

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, Accuracy: %.4f / %.4f = %.4f'
                % (epoch, num_epochs, i, len(train_loader),
                    errD.data, errG.data, D_x, D_G_z1, D_G_z2,
                    correct, length, 100. * correct / length))
            
            # Save real and fake images periodically
            os.makedirs(os.path.join('.', 'augGAN/output_images_prototype/ACGAN'), exist_ok=True)
            if i % 100 == 0:
                vutils.save_image(img,
                        '%s/real_samples.png' % './augGAN/output_images_prototype/ACGAN', normalize=True)
                fake = generator(fixed_noise)
                vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % ('./augGAN/output_images_prototype/ACGAN', epoch), normalize=True)

    # Save model checkpoint
    save_model(generator, discriminator, optimizerG, optimizerD, metrics, num_epochs)


# def train_acgan(discriminator, generator, train_loader, num_epochs, metrics, batch_size=32):
    
#     optimizerD = optim.Adam(discriminator.parameters(), lr=lr_d, betas=(beta1, beta2))
#     optimizerG = optim.Adam(generator.parameters(), lr=lr, betas=(beta1, beta2))
#     # Input to generator
#     fixed_noise = torch.randn(64, nz, 1, 1, device=device) #batch of 64
#     # Define Loss function
#     s_criterion = nn.BCELoss().to(device) #For synthesizing
#     c_criterion = nn.NLLLoss().to(device) #For classification
    
#     input = torch.FloatTensor(batch_size, 3, imageSize, imageSize).to(device)
#     s_label = torch.FloatTensor(batch_size).to(device)
#     c_label = torch.LongTensor(batch_size).to(device)
#     noise = torch.FloatTensor(batch_size, nz, 1, 1).to(device)
    
#     fixed_noise = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1).to(device)
    
#     for epoch in range(num_epochs):
#         for i, data in enumerate(tqdm(train_loader, 0)):
#             ###########################
#             # (1) Update D network
#             ###########################
#             # train with real
#             discriminator.zero_grad()
#             img, label = data
#             batch_size = img.size(0)
#             with torch.no_grad():
#                 input.resize_(img.size()).copy_(img)
#                 s_label.resize_(batch_size).fill_(real_label)
#                 c_label.resize_(batch_size).copy_(label)
#             s_output, c_output, f_R = discriminator(input)
#             s_output = s_output.view(-1)
#             s_errD_real = s_criterion(s_output, s_label)
#             c_errD_real = c_criterion(c_output, c_label)
#             errD_real = s_errD_real + c_errD_real
#             errD_real.backward()
#             D_x = s_output.data.mean()
            
#             correct, length = test(c_output, c_label)

#             # train with fake
#             with torch.no_grad():
#                 noise.resize_(batch_size, nz, 1, 1)
#                 noise.normal_(0, 1)

#             label = np.random.randint(0, nb_label, batch_size)
#             noise_ = np.random.normal(0, 1, (batch_size, nz))
#             label_onehot = np.zeros((batch_size, nb_label))
#             label_onehot[np.arange(batch_size), label] = 1
#             noise_[np.arange(batch_size), :nb_label] = label_onehot[np.arange(batch_size)]
            
#             noise_ = (torch.from_numpy(noise_))
#             noise_ = noise_.resize_(batch_size, nz, 1, 1)
#             noise.data.copy_(noise_)

#             c_label.data.resize_(batch_size).copy_(torch.from_numpy(label))

#             fake = generator(noise)
#             s_label.data.fill_(fake_label)
#             s_output, c_output, _ = discriminator(fake.detach())
            
#             s_output = s_output.view(-1)
#             s_errD_fake = s_criterion(s_output, s_label)
#             c_errD_fake = c_criterion(c_output, c_label)
#             errD_fake = s_errD_fake + c_errD_fake

#             errD_fake.backward()
#             D_G_z1 = s_output.data.mean()
#             errD = s_errD_real + s_errD_fake
#             optimizerD.step()

#             ###########################
#             # (2) Update G network
#             ###########################
#             generator.zero_grad()
#             s_label.data.fill_(real_label)  # fake labels are real for generator cost
            
#             s_output, c_output, f_F = discriminator(fake)
#             s_output = s_output.view(-1)
#             s_errG = s_criterion(s_output, s_label)
#             c_errG = c_criterion(c_output, c_label)
#             # aggregate the features along training time 
#             current_feat_mean_real = f_R.mean(0)
#             if epoch == 0:
#                 total_feat_mean_real = current_feat_mean_real
#             else:
#                 total_feat_mean_real = (epoch * total_feat_mean_real +  current_feat_mean_real) / (epoch + 1)
            
#             # import IPython; IPython.embed()
#             total_feat_mean_real = total_feat_mean_real.detach()

#             f_R = f_R.detach()
#             mean_feat_R = torch.mean(f_R, 0)
#             mean_feat_F = torch.mean(f_F, 0)
            
#             var_feat_R = torch.var(f_R, 0)  
#             matching_loss =  f_F - f_R
#             proto_loss = mean_feat_F - total_feat_mean_real
#             var_loss = var_feat_R
            
#             errG = s_errG + c_errG
#             errG = errG + (epoch / num_epochs) * matching_loss.mean() + (epoch / num_epochs) * proto_loss.mean() - 2 * var_loss.mean()
            
#             errG.backward(retain_graph=True)
#             D_G_z2 = s_output.data.mean()
            
#             optimizerG.step()
#             metrics['train.G_losses'].append(errG.item())
#             metrics['train.D_losses'].append(errD.item())
#             metrics['train.D_matching_loss'].append(matching_loss.mean().item())
#             metrics['train.D_proto_loss'].append(proto_loss.mean().item())
#             metrics['train.D_var_loss'].append(var_loss.mean().item())

#             print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f, Accuracy: %.4f / %.4f = %.4f'
#                 % (epoch, num_epochs, i, len(train_loader),
#                     errD.data, errG.data, D_x, D_G_z1, D_G_z2,
#                     correct, length, 100.* correct / length))
#             os.makedirs(os.path.join('.', 'augGAN/output_images_prototype/ACGAN'), exist_ok=True)
#             if i % 100 == 0:
#                 vutils.save_image(img,
#                         '%s/real_samples.png' % './augGAN/output_images_prototype/ACGAN', normalize=True)
#                 #fake = netG(fixed_cat)
#                 fake = generator(fixed_noise)
#                 vutils.save_image(fake.data,
#                         '%s/fake_samples_epoch_%03d.png' % ('./augGAN/output_images_prototype/ACGAN', epoch), normalize=True)
#     save_model(generator, discriminator, optimizerG, optimizerD, metrics, num_epochs)
#         # do checkpointing
#         # torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (os.path.join('.', 'augGAN/model/ACGAN'), epoch))
#         # torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (os.path.join('.', 'augGAN/model/ACGAN'), epoch))

def test2(generator, discriminator, num_epochs, metrics, loader):
    print('Testing Block.........')
    now = datetime.datetime.now()
    g_losses = metrics['train.G_losses'][-1]
    d_losses = metrics['train.D_losses'][-1]
    path='augGAN/output_images_prototype/ACGAN'
    try:
      os.makedirs(os.path.join('.', path), exist_ok=True)
    except Exception as error:
      print(error)

    real_batch = next(iter(loader))
    
    test_img_list = []
    test_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    test_fake = generator(test_noise).detach().cpu()
    test_img_list.append(vutils.make_grid(test_fake, padding=2, normalize=True))

    fig = plt.figure(figsize=(15,15))
    ax1 = plt.subplot(1,2,1)
    ax1 = plt.axis("off")
    ax1 = plt.title("Real Images")
    ax1 = plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=5, normalize=True).cpu(),(1,2,0)))

    ax2 = plt.subplot(1,2,2)
    ax2 = plt.axis("off")
    ax2 = plt.title("Fake Images")
    ax2 = plt.imshow(np.transpose(test_img_list[-1],(1,2,0)))
    #ax2 = plt.show()
    fig.savefig('%s/image_%.3f_%.3f_%d_%s.png' %
                   (path, g_losses, d_losses, num_epochs, now.strftime("%Y-%m-%d_%H:%M:%S")))

def save_model(generator, discriminator, gen_optimizer, dis_optimizer, metrics, num_epochs):
    now = datetime.datetime.now()
    g_losses = metrics['train.G_losses'][-1]
    d_losses = metrics['train.D_losses'][-1]
    name = "%+.3f_%+.3f_%d_%s.dat" % (g_losses, d_losses, num_epochs, now.strftime("%Y-%m-%d_%H:%M:%S"))
    fname = os.path.join('.', 'augGAN/model', name)
    states = {
            'state_dict_generator': generator.state_dict(),
            'state_dict_discriminator': discriminator.state_dict(),
            'gen_optimizer': gen_optimizer.state_dict(),
            'dis_optimizer': dis_optimizer.state_dict(),
            'metrics': metrics,
            'train_epoch': num_epochs,
            'date': now.strftime("%Y-%m-%d_%H:%M:%S"),
    }
    torch.save(states, fname)
    path='augGAN/plots/ACGAN/train_%+.3f_%+.3f_%s'% (g_losses, d_losses, now.strftime("%Y-%m-%d_%H:%M:%S"))
    try:
      os.makedirs(os.path.join('.', path), exist_ok=True)
    except Exception as error:
      print(error)

    plot('G_losses', num_epochs, metrics['train.G_losses'], path, True)
    plot('D_losses', num_epochs, metrics['train.D_losses'], path, True)
    # plot('D_x', num_epochs, metrics['train.D_x'], path, True)
    # plot('D_G_z1', num_epochs, metrics['train.D_G_z1'], path, True)
    # plot('D_G_z2', num_epochs, metrics['train.D_G_z2'], path, True)
    
if __name__=='__main__':
    # Set random seed for reproducibility
    manualSeed = 999
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    cudnn.benchmark = True
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
    train_set = datasets.ImageFolder(os.path.join(data_dir, "train/"), transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                          shuffle=True)
    
    generator = netG(nz, ngf, nc).to(device)
    discriminator = netD(ndf, nc, nb_label).to(device)

    
    generator.train()
    discriminator.train()
    
    train_acgan(discriminator, generator, train_loader, num_epochs, metrics)
    test2(generator, discriminator, num_epochs, metrics, train_loader)

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
    
    