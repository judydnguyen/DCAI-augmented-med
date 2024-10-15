import datetime
from more_itertools import tabulate
import numpy as np
import pandas as pd
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim

import os

from tqdm import tqdm
import wandb 

from src.constants import *
from src.train_utils import test_classifier
from src.models import CNN
from src.utils import set_seed
from sklearn.utils.multiclass import unique_labels

data_dir = './custom_covid_dataset/'
original_train_dir = 'train/'
train_classic_augmented_dir = 'train_classic/'
train_synthetic_augmented_dir = 'train_synthetic_proto/'
test_dir = 'test/'

def create_dataloader(epoch):
    if epoch < 200:
        # Use original, classic augmented, and GAN samples
        listtrainset = listtrainset_classic + listtrainset_gan
    else:
        # Use only original and classic augmented samples
        listtrainset = listtrainset_no_aug

    trainset_concat = torch.utils.data.ConcatDataset(listtrainset)
    sampler = torch.utils.data.RandomSampler(trainset_concat, replacement=True, num_samples=num_samples)

    return torch.utils.data.DataLoader(trainset_concat, sampler=sampler, batch_size=batch_size_train, drop_last=True)

transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5), (0.5))])
        

if __name__ == "__main__":
    set_seed(42)
    lr=0.01
    
    name_new_folder= 'Proto_CNN_'+str(channels_size)+'CH_O'+str(1)+'+C3'+'+G4'+'_batch'+str(batch_size_train)+'_lr'+str(lr)+'w_aug'+'_{}'.format(datetime.datetime.now())
    
    wandb.init(project='MedSyn-DCAI', 
               entity='duwgnt',
               name=name_new_folder)
    log_f = wandb.log
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Device: ', device)
    
    # TODO: stupid code, need to refactor
    # ORIGINAL TRAIN IMAGES
    trainset0 = torchvision.datasets.ImageFolder(os.path.join(
            data_dir, original_train_dir), transform=transform)

    # CLASSIC AUGMENTED TRAIN IMAGES
    trainset500 = torchvision.datasets.ImageFolder(os.path.join(
            data_dir+train_classic_augmented_dir, "500/"), transform=transform)
    trainset1000 = torchvision.datasets.ImageFolder(os.path.join(
            data_dir+train_classic_augmented_dir, "1000/"), transform=transform)
    trainset2000 = torchvision.datasets.ImageFolder(os.path.join(
            data_dir+train_classic_augmented_dir, "2000/"), transform=transform)
    trainset3000 = torchvision.datasets.ImageFolder(os.path.join(
            data_dir+train_classic_augmented_dir, "3000/"), transform=transform)

    # GAN AUGMENTED TRAIN IMAGES
    trainset100_gan = torchvision.datasets.ImageFolder(os.path.join(
            data_dir+train_synthetic_augmented_dir, "100/"), transform=transform)
    trainset500_gan = torchvision.datasets.ImageFolder(os.path.join(
            data_dir+train_synthetic_augmented_dir, "500/"), transform=transform)
    trainset1000_gan = torchvision.datasets.ImageFolder(os.path.join(
            data_dir+train_synthetic_augmented_dir, "1000/"), transform=transform)
    trainset2000_gan = torchvision.datasets.ImageFolder(os.path.join(
            data_dir+train_synthetic_augmented_dir, "2000/"), transform=transform)

    # CONCATENATION LISTS TRAIN IMAGES
    listtrainset_no_aug = [trainset0]
    # listtrainset_classic = [trainset2000]
    # listtrainset_gan = [trainset1000_gan]
    listtrainset_classic = [trainset500,trainset1000,trainset2000]
    listtrainset_gan = [trainset100_gan,trainset500_gan,trainset1000_gan,trainset2000_gan]
    
    # try O1 + C3 + G3
    
    listtrainset = listtrainset_no_aug + listtrainset_classic + listtrainset_gan
    # listtrainset = listtrainset_no_aug
    trainset_concat = torch.utils.data.ConcatDataset(listtrainset)

    sampler = torch.utils.data.RandomSampler(trainset_concat, replacement=True, 
                                             num_samples=num_samples)

    trainloader = torch.utils.data.DataLoader(trainset_concat, sampler=sampler,
                                            batch_size=batch_size_train, drop_last=True)

    # TEST IMAGES
    testset0 = torchvision.datasets.ImageFolder(os.path.join(data_dir, test_dir), transform=transform)

    testloader = torch.utils.data.DataLoader(testset0, batch_size=batch_size_test, shuffle=False)

    classes = ("covid", "normal", "pneumonia_bac", "pneumonia_vir")
    
    # Training hyper-parameters
    hidden_size=32
    channels_size=1
    class_number=4
    METRIC_FIELDS = [ 'loss', 'tot_acc', 'acc', 'sens', 'spec', 'f1_score']
    metrics = {field: list() for field in METRIC_FIELDS}
    epochs=750
    lr=0.01
    m=0.9
    
    net=CNN(hidden_size, channels_size, class_number)
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=m)
    # name_new_folder= 'CNN_'+str(channels_size)+'CH_O'+str(len(listtrainset_no_aug))+'+C3'+'+G3'+'_batch'+str(batch_size_train)+'_lr'+str(lr)+'w_aug'+'_{}'.format(datetime.datetime.now())
    
    # name_new_folder= 'CNN_'+str(channels_size)+'CH_O'+str(len(listtrainset_no_aug))+'+C'+str(len(listtrainset_classic))+'+G'+str(len(listtrainset_gan))+'_batch'+str(batch_size_train)+'_lr'+str(lr)+'w_aug'+'_{}'.format(datetime.datetime.now())
    print('NEW FOLDER: ', name_new_folder)
    results_folder = 'judy_results/'
    # make a new folder for the results
    os.makedirs(os.path.join('.', results_folder), exist_ok=True)
    os.makedirs(os.path.join('.', results_folder, name_new_folder), exist_ok=True)
    os.makedirs(os.path.join('.', results_folder, name_new_folder, 'models'), exist_ok=True)
    os.makedirs(os.path.join('.', results_folder, name_new_folder, 'plots'), exist_ok=True)
    
    eval_acc, eval_sens, eval_spec, eval_f1 = 0.0, 0.0, 0.0, 0.0
    tot_acc = 0.0
    for epoch in tqdm(range(epochs), desc='Epochs'):
        net.train()
        # print('Epoch: ', epoch)
        train_acc = 0.0
        running_loss = 0.0
        total = 0
        correct = 0
        
        
        for batch_idx, (inputs, labels) in tqdm(enumerate(trainloader), desc='Batches', total=len(trainloader)):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            preds = torch.argmax(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        print("Epoch: [%d]"%(epoch + 1),' => Loss: [%.3f]' %(running_loss / len(trainloader)))
        
        wandb.log({'loss': running_loss / len(trainloader),
                   'epoch': epoch,
                   'train_acc': correct / total})
        classes = ("covid", "normal", "pneumonia_bac", "pneumonia_vir")
        
        if (epoch+1) % 10 == 0:
            tot_acc, accuracy_ls, sensitivity_ls, specificity_ls, f1_score_ls, cm = test_classifier(net, testloader, criterion, device, 4, str(os.path.join('.', results_folder, name_new_folder)), epoch)
            
            eval_acc, eval_sens, eval_spec, eval_f1 = np.mean(accuracy_ls), np.mean(sensitivity_ls), np.mean(specificity_ls), np.mean(f1_score_ls)
            print("Epoch: [%d]"%(epoch + 1)," => Evaluation Confusion Matrix Test: [%d]" %(int((epoch+1)/10)))
            # classes = classes[unique_labels(y_true, y_pred)]
            
            table_df = pd.DataFrame({
                'Class': classes,
                'Accuracy': accuracy_ls,
                'Sensitivity': sensitivity_ls,
                'Specificity': specificity_ls,
                'F1 Score': f1_score_ls
            })

            # Print the DataFrame as a formatted table
            print(table_df.to_string(index=False))

            # Save the table to an image if needed
            if (epoch+1) % 150 == 0:
                fig, ax = plt.subplots(figsize=(8, 2.5))  # Set figure size
                ax.axis('off')  # Turn off the axis
                
                # Plot the table
                ax.table(cellText=table_df.values, colLabels=table_df.columns, cellLoc='center', loc='center')
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(results_folder + name_new_folder + '/plots/k_fold_%d_ep_%.3f_acc.png' % ((epoch + 1), tot_acc))
                # plt.show() # Uncomment to display the plot
              
        wandb.log({'tot_acc': tot_acc,
                   'acc': eval_acc,
                   'sens': eval_sens,
                   'spec': eval_spec,
                   'f1_score': eval_f1,
                   'epoch': epoch})
        
        
        
            
        
    
    
    