import argparse
from build_dataset_skin import build_data_loader_from_synthetic
import os, cv2,itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

# pytorch libraries
import torch
from torch import optim,nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

# sklearn libraries
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

from torchvision.models import ResNet50_Weights

from src.data_utils import load_train_test_df
from src.train_utils import AverageMeter, get_cls_model
from src.utils import set_seed
import wandb

SEED = 42
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def validate(val_loader, model, criterion, optimizer, epoch):
    model.eval()
    val_loss = AverageMeter()
    val_acc = AverageMeter()
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]

            val_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)

            val_loss.update(criterion(outputs, labels).item())

    print('------------------------------------------------------------')
    print('[epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, val_loss.avg, val_acc.avg))
    print('------------------------------------------------------------')
    return val_loss.avg, val_acc.avg

# def train(train_loader, model, criterion, optimizer, epoch):
#     total_loss_train, total_acc_train = [],[]
#     model.train()
#     train_loss = AverageMeter()
#     train_acc = AverageMeter()
#     curr_iter = (epoch - 1) * len(train_loader)
#     # make it appear the progress bar
#     for i, data in tqdm(enumerate(train_loader), desc=f"Training Epoch {epoch}"):
#         images, labels = data
#         N = images.size(0)
#         # print('image shape:',images.size(0), 'label shape',labels.size(0))
#         images = Variable(images).to(device)
#         labels = Variable(labels).to(device)

#         optimizer.zero_grad()
#         outputs = model(images)

#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#         prediction = outputs.max(1, keepdim=True)[1]
#         train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item()/N)
#         train_loss.update(loss.item())
#         curr_iter += 1
#         if (i + 1) % 100 == 0:
#             print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
#                 epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
#             total_loss_train.append(train_loss.avg)
#             total_acc_train.append(train_acc.avg)
#     return train_loss.avg, train_acc.avg

from tqdm import tqdm

def train(train_loader, model, criterion, optimizer, epoch, wandb_instance):
    total_loss_train, total_acc_train = [], []
    model.train()
    train_loss = AverageMeter()
    train_acc = AverageMeter()
    curr_iter = (epoch - 1) * len(train_loader)
    
    # make it appear the progress bar with total iterations
    with tqdm(total=len(train_loader), desc=f"Training Epoch {epoch}") as pbar:
        for i, data in enumerate(train_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            labels = Variable(labels).to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            prediction = outputs.max(1, keepdim=True)[1]
            train_acc.update(prediction.eq(labels.view_as(prediction)).sum().item() / N)
            train_loss.update(loss.item())
            curr_iter += 1

            # Update the progress bar and display current status every 100 iterations
            pbar.update(1)
            if (i + 1) % 10 == 0:
                print('[epoch %d], [iter %d / %d], [train loss %.5f], [train acc %.5f]' % (
                    epoch, i + 1, len(train_loader), train_loss.avg, train_acc.avg))
                total_loss_train.append(train_loss.avg)
                total_acc_train.append(train_acc.avg)
                
            wandb_instance.log({'train_loss': train_loss.avg, 'train_acc': train_acc.avg, 'epoch': epoch, 'iter': i})
                
    return train_loss.avg, train_acc.avg

def main(args):
    wandb_name = f"{args.model}_synthetic_{args.image_size}_lr_{args.lr}" if args.use_synthetic else f"{args.model}_no_synthetic_synthetic_{args.image_size}_lr_{args.lr}"
    wandb_instance = wandb.init(project="skin_lesion_classification", 
               entity="duwgnt",
               name=wandb_name)
    wandb_instance.config.update(args)
    
    set_seed(SEED)
    # load pre-processed skin lession dataset
    train_loader, val_dl, subset_train = load_train_test_df(input_size=args.image_size, batch_size=args.batch_size,
                                                            train=True)
    if args.use_synthetic:
        # train_loader = subset_train
        synthetic_loader = build_data_loader_from_synthetic(batch_size=args.batch_size)
        synthetic_dataset = synthetic_loader.dataset
        
        # merge the two datasets
        # import IPython; IPython.embed()
        
        train_loader = torch.utils.data.ConcatDataset([train_loader.dataset, synthetic_dataset])
        
        train_loader = DataLoader(train_loader, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        
    model = get_cls_model(args.model, args.num_classes, args.pretrained)
    model.to(device)
    print(f"Loaded model: {args.model},\n {model}")
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    
    epoch_num = 20
    best_val_acc = 0
    total_loss_val, total_acc_val = [],[]
    # model.to("cuda")
    
    for epoch in range(1, epoch_num+1):
        loss_train, acc_train = train(train_loader, model, criterion, optimizer, epoch, wandb_instance)
        loss_val, acc_val = validate(val_dl, model, criterion, optimizer, epoch)
        total_loss_val.append(loss_val)
        total_acc_val.append(acc_val)
        wandb_instance.log({'val_loss': loss_val, 'val_acc': acc_val, 'epoch': epoch, 
                            'loss_train': loss_train, 'acc_train': acc_train})
        if acc_val > best_val_acc:
            best_val_acc = acc_val
            print('*****************************************************')
            print('best record: [epoch %d], [val loss %.5f], [val acc %.5f]' % (epoch, loss_val, acc_val))
            print('*****************************************************')
            model_save_name = f'proto_{args.use_proto}_synthetic_{args.use_synthetic}_baseline_{args.model}_best_model'
            torch.save(model.state_dict(), os.path.join(args.output_dir, f'{model_save_name}.pth'))
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Skin Lesion Training')
    parser.add_argument('--image_size', type=int, default=32, help='image size')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--num_workers', type=int, default=54, help='number of workers')
    parser.add_argument('--seed', type=int, default=42, help='seed')
    parser.add_argument('--model', type=str, default='resnet', help='model')
    parser.add_argument('--pretrained', type=bool, default=True, help='pretrained')
    parser.add_argument('--resume', type=str, default='', help='resume')
    parser.add_argument('--epochs', type=int, default=10, help='epochs')
    parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
    parser.add_argument('--output_dir', type=str, default='log_classifier', help='output directory')
    parser.add_argument('--num_classes', type=int, default=7, help='number of classes')
    parser.add_argument('--use_synthetic', type=bool, default=False, help='use synthetic data')
    parser.add_argument('--use_proto', type=bool, default=False, help='use prototype synthetic data')
    args = parser.parse_args()
    main(args)