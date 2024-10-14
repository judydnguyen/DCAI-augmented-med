import datetime
import os

import torch
import torchvision.utils as vutils
from torchvision.utils import save_image

from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append("./")

from src.utils import plot, plot_confusion_matrix
from sklearn.metrics import classification_report, confusion_matrix
from src.constants import *

def test1(generator, discriminator, num_epochs, metrics):
    print('Testing Block.........')
    now = datetime.datetime.now()
    g_losses = metrics['train.G_losses'][-1]
    d_losses = metrics['train.D_losses'][-1]
    path='augGAN/output_images'
    # try:
    #   os.mkdir(os.path.join('.', path))
    # except Exception as error:
    #   print(error)

    test_img_list = []
    test_noise = torch.randn(batch_size, nz, 1, 1, device=device)
    test_fake = generator(test_noise).detach().cpu()
    test_img_list.append(vutils.make_grid(test_fake, padding=2, normalize=True))
    fig = plt.figure(figsize=(15,15))
    fig = plt.axis("off")
    fig = plt.title("Fake Images")
    fig = plt.imshow(np.transpose(test_img_list[-1],(1,2,0)))
    get_fig = plt.gcf()
    fig = plt.show()
    get_fig.savefig('%s/image_%.3f_%.3f_%d_%s.png' %
                    (path, g_losses, d_losses, num_epochs, now.strftime("%Y-%m-%d_%H:%M:%S")))

def test2(generator, discriminator, num_epochs, metrics, loader):
    print('Testing Block.........')
    now = datetime.datetime.now()
    g_losses = metrics['train.G_losses'][-1]
    d_losses = metrics['train.D_losses'][-1]
    path='augGAN/output_images'
    try:
      os.mkdir(os.path.join('.', path))
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


def test_fake(generator, discriminator, metrics, n_images, folname):
    now = datetime.datetime.now()
    g_losses = metrics['train.G_losses'][-1]
    d_losses = metrics['train.D_losses'][-1]
    #path='augGAN/output_images/%+.3f_%+.3f_%d_%s'% (g_losses, d_losses, n_images, now.strftime("%Y-%m-%d_%H:%M:%S"))
    path='main_folder/'+str(n_images)+'/'+folname
    try:
      os.mkdir(os.path.join('.', path))
    except Exception as error:
      print(error)

    im_batch_size = 50
    #n_images=100
    for i_batch in range(0, n_images, im_batch_size):
        gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)
        gen_images = generator(gen_z)
        dis_result = discriminator(gen_images).view(-1)
        images = gen_images.to("cpu").clone().detach()
        images = images.numpy().transpose(0, 2, 3, 1)
        for i_image in range(gen_images.size(0)):
            save_image(gen_images[i_image, :, :, :], os.path.join(path, 
                        f'image_{i_batch+i_image:04d}.png'), normalize= True)

    print('Testing Block.........')
    print('Discriminator_mean: ', dis_result.mean().item())
    #import shutil
    #shutil.make_archive('images', 'zip', './augGAN/output_images')

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
    path='augGAN/plots/train_%+.3f_%+.3f_%s'% (g_losses, d_losses, now.strftime("%Y-%m-%d_%H:%M:%S"))
    try:
      os.mkdir(os.path.join('.', path))
    except Exception as error:
      print(error)

    plot('G_losses', num_epochs, metrics['train.G_losses'], path, True)
    plot('D_losses', num_epochs, metrics['train.D_losses'], path, True)
    plot('D_x', num_epochs, metrics['train.D_x'], path, True)
    plot('D_G_z1', num_epochs, metrics['train.D_G_z1'], path, True)
    plot('D_G_z2', num_epochs, metrics['train.D_G_z2'], path, True)
    
def train_gan(generator, discriminator, gen_optimizer, dis_optimizer, 
              train_loader, num_epochs, metrics, device, criterion):
    iters = 0
    print("GAN training started :D...")
    img_list = []
    G_losses = []
    D_losses = []
    for epoch in range(num_epochs):
        print("Epoch %d" %(epoch+1))
        # For each batch in the dataloader
        for i, data in enumerate(tqdm(train_loader, 0)):
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ## Train with all-real batch
            discriminator.zero_grad()
            # Format batch
            b_real = data[0].to(device)
            b_size = b_real.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Forward pass real batch through D
            output = discriminator(b_real).view(-1)
            # Calculate loss on all-real batch
            errD_real = criterion(output, label)
            # Calculate gradients for D in backward pass
            errD_real.backward()
            D_x = output.mean().item()
            metrics['train.D_x'].append(D_x)

            ## Train with all-fake batch
            # Generate batch of latent vectors
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            # Generate fake image batch with G
            fake = generator(noise)
            label.fill_(fake_label)
            # Classify all fake batch with D
            output = discriminator(fake.detach()).view(-1)
            # Calculate D's loss on the all-fake batch
            errD_fake = criterion(output, label)
            # Calculate the gradients for this batch
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            metrics['train.D_G_z1'].append(D_G_z1)
            # Add the gradients from the all-real and all-fake batches
            errD = errD_real + errD_fake
            # Update D
            dis_optimizer.step()
            # if i>0:
            #     if errD.item()>G_losses[i-1]:
            #         dis_optimizer.step()
            # else:
            #     dis_optimizer.step()

            # (2) Update G network: maximize log(D(G(z)))
            generator.zero_grad()
            label.fill_(real_label)  # fake labels are real for generator cost
            # Since we just updated D, perform another forward pass of all-fake batch through D
            output = discriminator(fake).view(-1)
            # Calculate G's loss based on this output
            errG = criterion(output, label)
            # Calculate gradients for G
            errG.backward()
            D_G_z2 = output.mean().item()
            metrics['train.D_G_z2'].append(D_G_z2)
            # Update G
            gen_optimizer.step()

            # Save Losses for plotting later
            G_losses.append(errG.item())
            D_losses.append(errD.item())
            metrics['train.G_losses'].append(errG.item())
            metrics['train.D_losses'].append(errD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(train_loader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1
        print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                % (epoch+1, num_epochs, i, len(train_loader),
                errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
    save_model(generator, discriminator, gen_optimizer, dis_optimizer, metrics, num_epochs)
    
def compute_metrics(cm):
    TP = np.diag(cm)
    FN = np.sum(cm, axis=1) - TP
    FP = np.sum(cm, axis=0) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    
    accuracy = (TP + TN) / np.sum(cm)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * (specificity * sensitivity) / (specificity + sensitivity)
    
    return accuracy, sensitivity, specificity, f1_score

def compute_class_metrics(cm, class_idx=0):
    TP = cm[class_idx, class_idx]
    FN = np.sum(cm[class_idx, :]) - TP
    FP = np.sum(cm[:, class_idx]) - TP
    TN = np.sum(cm) - (TP + FP + FN)
    
    accuracy = (TP + TN) / np.sum(cm)
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1_score = 2 * (specificity * sensitivity) / (specificity + sensitivity)
    
    return accuracy, sensitivity, specificity, f1_score

def test_classifier(model, test_loader, criterion, device, 
                    n_classes=4, log_folder="./", epoch=0):
    with torch.no_grad():
        # Set the model to evaluation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = list(0. for i in range(n_classes))
        class_total = list(0. for i in range(n_classes))
        all_labels = []
        all_preds = []
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            all_labels.append(labels)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted == labels).squeeze()
            all_preds.append(predicted)
            for i in range(4):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
        # calculate metrics
        # tot_acc = 100 * correct / total
        # calculate confusion matrix
        # confusion_matrix = []
        stacked_labels = torch.stack(all_labels, dim=0).view(-1).detach().cpu().numpy()
        stacked_preds = torch.stack(all_preds, dim=0).view(-1).detach().cpu().numpy()
        classes = ("covid", "normal", "pneumonia_bac", "pneumonia_vir")
        plot_confusion_matrix(stacked_labels, stacked_preds, classes=np.asarray(classes), normalize=True)
        
        cm = confusion_matrix(stacked_labels, stacked_preds)
        tot_acc=np.trace(cm)/np.sum(cm)
        # tot_acc_tot+=tot_acc
        accuracy_ls, sensitivity_ls, specificity_ls, f1_score_ls = [], [], [], []
        
        for cls_idx in range(n_classes):
            accuracy, sensitivity, specificity, f1_score = compute_class_metrics(cm, cls_idx)
            accuracy_ls.append(accuracy)
            sensitivity_ls.append(sensitivity)
            specificity_ls.append(specificity)
            f1_score_ls.append(f1_score)
            
        plt.gcf().savefig(log_folder + '/plots/confusion_matrix_%d_ep_%.3f_acc.png' % ((epoch + 1), tot_acc))
        
    model.train()
    return tot_acc, accuracy_ls, sensitivity_ls, specificity_ls, f1_score_ls, cm
        