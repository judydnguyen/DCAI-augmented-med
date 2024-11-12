from __future__ import print_function
import argparse
import copy
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
from src.models import PerceptualLoss, _netD, _netG, _netD_CIFAR10, _netG_CIFAR10
from src.constants import *
from src.utils import compute_acc, get_transform, plot, set_seed
import matplotlib.pyplot as plt

HIDDEN_DIM = 8192
START_EPOCH = 150

def plot_losses(loss_D, loss_G, loss_A, loss_P, loss_C, num_epochs, args):
    num_epochs = len(loss_D)
    print(f"Number of epochs: {num_epochs}, end: {num_epochs+START_EPOCH+1}")
    plt.figure(figsize=(10, 5))
    plt.plot(range(START_EPOCH, num_epochs + START_EPOCH), loss_D, label='Discriminator Loss')
    plt.plot(range(START_EPOCH, num_epochs + START_EPOCH), loss_G, label='Generator Loss')
    plt.plot(range(START_EPOCH, num_epochs + START_EPOCH), loss_P, label="Prototype Loss")
    plt.plot(range(START_EPOCH, num_epochs + START_EPOCH), loss_C, label="Perceptual Loss")
    # plt.plot(range(1, num_epochs + 1), loss_A, label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.grid(True)
    # plt.show()
    os.makedirs(f'output/acgan_skin/plots', exist_ok=True)
    plt.savefig(f'output/acgan_skin/plots/{args.lr}_d_lr_{args.d_lr}_losses.png')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using DEVICE: {device}")

# Example usage
perceptual_loss_fn = PerceptualLoss().to(device)

def update_prototype(current_feat, current_label, total_feat_mean, nb_label=7, iteration=0):
    """
    Updates the class-specific prototypes by averaging the features over the epochs.
    """
    alpha = 0.25  # Weight for the new feature; adjust this as needed

    # For each class, calculate its new prototype
    # total_feat_mean = torch.mean(total_feat_mean, dim=0)
    for cls in range(nb_label):
        # Get the mean feature for the current class
        mask = current_label == cls
        # import IPython; IPython.embed()
            
        if mask.sum() > 0:  # If the class exists in the batch
            class_feat = current_feat[mask].mean(0)
            if iteration == 0:
                total_feat_mean[cls] = class_feat
            else:
                # total_feat_mean[cls] = (total_feat_mean[cls] + class_feat) / 2  # Moving average
                total_feat_mean[cls] = (iteration * total_feat_mean[cls] + class_feat)/(iteration+1)
    return total_feat_mean

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
    
    prototype_R = {i: torch.zeros(HIDDEN_DIM, device=device) for i in range(args.num_classes)}  # Store class prototypes
    
    real_label = 1
    fake_label = 0
    eval_label = np.random.randint(0, args.num_classes, args.batch_size)
    # Store losses for plotting
    epoch_loss_D = []
    epoch_loss_G = []
    epoch_loss_P = []
    epoch_loss_C = []
    epoch_accuracy = []
    
    for epoch in tqdm(range(num_epochs)):
        total_loss_D = 0
        total_loss_G = 0
        total_accuracy = 0
        total_loss_P = 0.
        total_perceptual_loss = 0.
        cur_prototype_R = copy.deepcopy(prototype_R)
        # for key, item in cur_prototype_R.items():
        #     item = item.to(device)
        for i, data in tqdm(enumerate(train_loader, 0)):
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
            dis_output, aux_output, feat_R = discriminator(input)
            # import IPython; IPython.embed()
            
            dis_errD_real = dis_criterion(dis_output, dis_label)
            aux_errD_real = aux_criterion(aux_output, aux_label)
            errD_real = dis_errD_real + aux_errD_real
            errD_real.backward()
            D_x = dis_output.data.mean()
            
            # import IPython; IPython.embed()
            # compute current classification accuracy
            accuracy = compute_acc(aux_output, labels)
            
            prototype_R = update_prototype(feat_R.detach(), aux_label, prototype_R, iteration=i)
            # import IPython; IPython.embed()
            
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
            dis_output, aux_output, feat_F = discriminator(fake)
            dis_errG = dis_criterion(dis_output, dis_label)
            aux_errG = aux_criterion(aux_output, aux_label)
            errG = dis_errG + aux_errG

            # Prototype loss: Compare generated features to the prototype of the class they belong to
            proto_loss = torch.tensor(0.0).to(device)
            
            if epoch >= 1:
                for cls in range(nb_label):
                    if cls not in [3, 5, 6]:
                        # print(f"cls = {cls}, skipping")
                        continue
                    mask = aux_label == cls
                    # import IPython; IPython.embed()
                    if mask.sum() > 0:
                        # import IPython; IPython.embed()
                        proto_loss += torch.mean((feat_F[mask] - prototype_R[cls].detach()) ** 2)
            # import IPython; IPython.embed()
            print(f"Proto loss: {proto_loss.item()}")
            proto_loss /= nb_label
            # prototype_coeff = i/(len(train_loader)+1)
            prototype_coeff = 0.05
            # increase the prototype loss as the training progresses
            # prototype_coeff = prototype_coeff * (1 + epoch/num_epochs)
            
            errG += prototype_coeff*proto_loss
            errG.backward()
            D_G_z2 = dis_output.data.mean()
            optimizerG.step()

            # Compute perceptual loss between real and generated images
            perceptual_loss = perceptual_loss_fn(fake, real_images)
            perceptual_coeff = 0.1  # Adjust as needed
            # errG += perceptual_coeff * perceptual_loss
            total_perceptual_loss += perceptual_loss.item()
            
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
            total_loss_P += proto_loss.item()
            total_accuracy += accuracy
            avg_prototype_loss = total_loss_P / len(train_loader)
            
            
            
            # print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)\n'
            #     % (epoch, args.num_epochs, i, len(train_loader),
            #         errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A, avg_prototype_loss))
            print('[%d/%d][%d/%d] Loss: %.4f (%.4f) Acc: %.4f (%.4f) Prototype_Loss: %.4f\n'
                % (epoch, args.num_epochs, i, len(train_loader),
                    errD.item(), avg_loss_D, accuracy, avg_loss_A, avg_prototype_loss))

            if i % 100 == 0 or i == len(train_loader) - 1:
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
        avg_prototype_loss = total_loss_P / len(train_loader)
        avg_loss_C = total_perceptual_loss / len(train_loader)
        epoch_loss_D.append(avg_loss_D)
        epoch_loss_G.append(avg_loss_G)
        epoch_loss_P.append(avg_prototype_loss)
        epoch_accuracy.append(avg_accuracy)
        epoch_loss_C.append(avg_loss_C)
        # import IPython; IPython.embed()
        plot_losses(epoch_loss_D, epoch_loss_G, epoch_accuracy, epoch_loss_P, epoch_loss_C, num_epochs, args)

# def train_acgan(discriminator, generator, optimizerD, optimizerG , train_loader, num_epochs, metrics, args):
#     for param_group in optimizerG.param_groups:
#         param_group['lr'] *= 1.1  # Increase by 10%
#     for param_group in optimizerD.param_groups:
#         param_group['lr'] *= 0.9  # Decrease by 10%

#     # loss functions
#     dis_criterion = nn.BCELoss()
#     aux_criterion = nn.NLLLoss()
    
#     avg_loss_D = 0
#     avg_loss_G = 0
#     avg_loss_A = 0
#     total_loss_P = 0.
    
#     # define shapes
#     input = torch.FloatTensor(args.batch_size, args.nc, args.image_size, args.image_size)
#     noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1)
#     eval_noise = torch.FloatTensor(args.batch_size, args.nz, 1, 1).normal_(0, 1)
#     dis_label = torch.FloatTensor(args.batch_size)
#     aux_label = torch.LongTensor(args.batch_size)
#     real_label = 1
#     fake_label = 0
#     eval_label = np.random.randint(0, args.num_classes, args.batch_size)
#     # Store losses for plotting
#     epoch_loss_D = []
#     epoch_loss_G = []
#     epoch_loss_P = []
#     epoch_loss_C = []
#     epoch_accuracy = []

#     # Initialize prototype mean for real images
#     total_feat_mean_real = None

#     for epoch in tqdm(range(START_EPOCH, START_EPOCH+num_epochs), desc="Epochs"):
#         total_loss_D = 0
#         total_loss_G = 0
#         total_accuracy = 0
#         total_perceptual_loss = 0
#         for i, data in tqdm(enumerate(train_loader, 0), desc="Batches"):
#             ############################
#             # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
#             ###########################
#             # train with real
#             discriminator.zero_grad()
#             real_images, labels = data
#             batch_size = real_images.size(0)
#             real_images = real_images.to(device)
#             labels = labels.to(device)
            
#             noise = torch.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
#             input = torch.FloatTensor(batch_size, args.nc, args.image_size, args.image_size)
#             eval_noise = torch.FloatTensor(batch_size, args.nz, 1, 1).normal_(0, 1)
#             dis_label = torch.FloatTensor(batch_size)
#             aux_label = torch.LongTensor(batch_size)
            
#             input.data.resize_(real_images.size()).copy_(real_images)
#             dis_label.data.resize_(batch_size).fill_(real_label)
#             dis_label = smooth_labels(dis_label, smoothing=0.1)  # Apply 0.1 label smoothing
#             aux_label.data.resize_(batch_size).copy_(labels)
#             input = input.to(device)
#             dis_label = dis_label.to(device)
#             aux_label = aux_label.to(device)

#             # Forward pass for real images
#             dis_output, aux_output, feat_R = discriminator(input)
            
#             dis_errD_real = dis_criterion(dis_output, dis_label)
#             aux_errD_real = aux_criterion(aux_output, aux_label)
#             errD_real = dis_errD_real + aux_errD_real
#             errD_real.backward()
#             D_x = dis_output.data.mean()
            
#             # compute current classification accuracy
#             accuracy = compute_acc(aux_output, labels)
            
#             # Update the total feature mean for real images
#             current_feat_mean_real = feat_R.mean(dim=0)
#             if total_feat_mean_real is None:
#                 total_feat_mean_real = current_feat_mean_real.detach()
#             else:
#                 total_feat_mean_real = (i * total_feat_mean_real + current_feat_mean_real) / (i + 1)
#                 total_feat_mean_real = total_feat_mean_real.detach()
            
#             # train with fake
#             noise.data.resize_(batch_size, args.nz, 1, 1).normal_(0, 1)
#             label = np.random.randint(0, args.num_classes, batch_size)
#             noise_ = np.random.normal(0, 1, (batch_size, args.nz))
#             class_onehot = np.zeros((batch_size, args.num_classes))
#             class_onehot[np.arange(batch_size), label] = 1
#             noise_[np.arange(batch_size), :args.num_classes] = class_onehot[np.arange(batch_size)]
#             noise_ = (torch.from_numpy(noise_))
            
#             noise.data.copy_(noise_.view(batch_size, args.nz, 1, 1))
#             noise = noise.to(device)
#             aux_label.data.resize_(batch_size).copy_(torch.from_numpy(label))
            
#             fake = generator(noise)
#             dis_label.data.fill_(fake_label)
#             dis_output, aux_output, _ = discriminator(fake.detach())
#             dis_errD_fake = dis_criterion(dis_output, dis_label)
#             aux_errD_fake = aux_criterion(aux_output, aux_label)
#             errD_fake = dis_errD_fake + aux_errD_fake
#             errD_fake.backward()
#             D_G_z1 = dis_output.data.mean()
#             errD = errD_real + errD_fake
#             optimizerD.step()
            

#             ############################
#             # (2) Update G network: maximize log(D(G(z)))
#             ###########################
#             generator.zero_grad()
#             dis_label.data.fill_(real_label)
#             dis_output, aux_output, feat_F = discriminator(fake)
#             dis_errG = dis_criterion(dis_output, dis_label)
#             aux_errG = aux_criterion(aux_output, aux_label)
            
#             # Prototype loss calculation
#             proto_loss = (feat_F.mean(dim=0) - total_feat_mean_real.detach()).mean()
#             # using l2 norm instead of mean to ensure its positive
#             proto_loss = torch.norm(feat_F.mean(dim=0) - total_feat_mean_real.detach())/feat_F.mean(dim=0).shape[0]
#             # proto_loss = torch.norm(feat_F.mean(dim=0) - total_feat_mean_real.detach())
            
#             # Matching loss calculation (feature matching between real and generated features)
#             # matching_loss = (feat_F - feat_R.detach()).mean()
#             matching_loss = torch.norm(feat_F - feat_R.detach())/feat_F.shape[0]
            
#             # Variance loss calculation (encourages diversity in generated features)
#             feat_var_F = feat_F.var(dim=0)
#             # make sure the variance is positive
#             var_loss = -feat_var_F.mean()  # Negative to encourage variance
            
#             # Total generator loss with added prototype, matching, and variance losses
#             # errG = 0.5*dis_errG + aux_errG + proto_loss + 0.2*matching_loss + var_loss
#             errG = dis_errG + aux_errG + proto_loss
#             print(f"Proto loss: {proto_loss.item()}\t Matching loss: {matching_loss.item()}\t Variance loss: {var_loss.item()}")
#             # Compute perceptual loss between real and generated images
#             perceptual_loss = perceptual_loss_fn(fake, real_images)
#             perceptual_coeff = 0.25  # Adjust as needed
#             # errG += perceptual_coeff * perceptual_loss
#             total_perceptual_loss += perceptual_loss.item()
            
#             errG.backward(retain_graph=True)
#             D_G_z2 = dis_output.data.mean()
#             # import IPython; IPython.embed()
#             optimizerG.step()
            
#             # compute the average loss
#             curr_iter = epoch * len(train_loader) + i
#             all_loss_G = avg_loss_G * curr_iter
#             all_loss_D = avg_loss_D * curr_iter
#             all_loss_A = avg_loss_A * curr_iter
#             all_loss_G += errG.item()
#             all_loss_D += errD.item()
#             all_loss_A += accuracy
#             total_loss_P += proto_loss.item()
#             avg_loss_G = all_loss_G / (curr_iter + 1)
#             avg_loss_D = all_loss_D / (curr_iter + 1)
#             avg_loss_A = all_loss_A / (curr_iter + 1)
#             avg_loss_P = total_loss_P / (curr_iter + 1)
#             # Accumulate losses
#             total_loss_D += errD.item()
#             total_loss_G += errG.item()
#             total_accuracy += accuracy
            
#             print('[%d/%d][%d/%d] Loss_D: %.4f (%.4f) Loss_G: %.4f (%.4f) D(x): %.4f D(G(z)): %.4f / %.4f Acc: %.4f (%.4f)'
#                 % (epoch, args.num_epochs, i, len(train_loader),
#                     errD.item(), avg_loss_D, errG.item(), avg_loss_G, D_x, D_G_z1, D_G_z2, accuracy, avg_loss_A))
#             print(f"D_loss: {round(avg_loss_D, 2)}\t G_loss: {round(avg_loss_G,2)}\t P_loss: {round(avg_loss_P,2)}\t Accuracy: {avg_loss_A}")
#             if i % 100 == 0:
#                 vutils.save_image(
#                     real_images.cpu(), '%s/real_samples.png' % args.output_dir)
#                 print('Label for eval = {}'.format(eval_label))
#                 fake = generator(eval_noise.to(device))
#                 vutils.save_image(
#                     fake.cpu(),
#                     '%s/fake_samples_epoch_%03d.png' % (args.output_dir, epoch)
#                 )

#         if epoch % 50 == 0:
#             # save_checkpoint(generator, discriminator, args, epoch)
#             # Get the current time for file naming
#             current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

#             # Save the generator's state
#             torch.save(generator.state_dict(), '%s/netG_epoch_%d_%s.pth' % (args.checkpoint_dir, epoch, current_time))

#             # Save the discriminator's state
#             torch.save(discriminator.state_dict(), '%s/netD_epoch_%d_%s.pth' % (args.checkpoint_dir, epoch, current_time))
#         avg_loss_D = total_loss_D / len(train_loader)
#         avg_loss_G = total_loss_G / len(train_loader)
#         avg_accuracy = total_accuracy / len(train_loader)
#         avg_loss_P = total_loss_P / len(train_loader)
#         avg_loss_C = total_perceptual_loss / len(train_loader)
#         epoch_loss_D.append(avg_loss_D)
#         epoch_loss_G.append(avg_loss_G)
#         epoch_loss_P.append(avg_loss_P)
#         epoch_accuracy.append(avg_accuracy)
#         epoch_loss_C.append(avg_loss_C)
#         # import IPython; IPython.embed()
#         plot_losses(epoch_loss_D, epoch_loss_G, epoch_accuracy, epoch_loss_P, epoch_loss_C, num_epochs, args)

def main(args):
    manualSeed = args.seed
    set_seed(manualSeed)
    torch.backends.cudnn.benchmark = True

    
    # load pre-processed skin lession dataset
    train_loader, val_dl, subset_train = load_train_test_df(input_size=args.image_size, batch_size=args.batch_size)
    train_loader = subset_train
    # load generator and discriminator
    generator = _netG_CIFAR10(1, args.nz).to(device)
    discriminator = _netD_CIFAR10(1, args.num_classes).to(device)
    
    print(generator)
    print(discriminator)
    # netG.apply(weights_init)
    # netD.apply(weights_init)
    
    if args.load_checkpoint:
        generator.load_state_dict(torch.load("output/acgan_skin/saved_models/ckpts/netG_epoch_150_baseline.pth"), strict=False)
        discriminator.load_state_dict(torch.load("output/acgan_skin/saved_models/ckpts/netG_epoch_150_baseline.pth"), strict=False)
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
    args.output_dir = os.path.join(args.output_dir, f"{str(args.lr)}_d_lr_{str(args.d_lr)}")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, f"{str(args.lr)}_d_lr_{str(args.d_lr)}")
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
    
    