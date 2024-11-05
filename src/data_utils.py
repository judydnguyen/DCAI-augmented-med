import torch
from torch.utils.data import DataLoader,Dataset
from torchvision import models,transforms

import numpy as np
import pandas as pd
from tqdm import tqdm
from glob import glob
from PIL import Image

PARENT_PATH = "skin_lesson_dataset/2"

# Define a pytorch dataloader for this dataset
class HAM10000(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        # Load data and get label
        X = Image.open(self.df['path'][index])
        y = torch.tensor(int(self.df['cell_type_idx'][index]))

        if self.transform:
            X = self.transform(X)

        return X, y
    
# (224, 224, 3, 20030)
normMean = [0.763033, 0.5456458, 0.5700401]
normStd = [0.14092815, 0.15261315, 0.16997056]


def load_train_test_df(norm_mean=normMean, norm_std=normStd, input_size=64):
    
    df_train = pd.read_pickle(PARENT_PATH+'/train_data.pkl')
    df_val = pd.read_pickle(PARENT_PATH+'/val_data.pkl')


    train_transform = transforms.Compose([transforms.Resize((input_size,input_size)),transforms.RandomHorizontalFlip(),
                                        transforms.RandomVerticalFlip(),transforms.RandomRotation(20),
                                        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
                                            transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)])
    # define the transformation of the val images.
    val_transform = transforms.Compose([transforms.Resize((input_size,input_size)), transforms.ToTensor(),
                                        transforms.Normalize(norm_mean, norm_std)])
    # Define the training set using the table train_df and using our defined transitions (train_transform)
    training_set = HAM10000(df_train, transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=54)
    # Same for the validation set:
    validation_set = HAM10000(df_val, transform=train_transform)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=54)
    
    # Define the training set using the table train_df and using our defined transitions (train_transform)
    training_set = HAM10000(df_train, transform=train_transform)
    train_loader = DataLoader(training_set, batch_size=32, shuffle=True, num_workers=54)
    # Same for the validation set:
    validation_set = HAM10000(df_val, transform=train_transform)
    val_loader = DataLoader(validation_set, batch_size=32, shuffle=False, num_workers=54)
    return train_loader, val_loader
    
        
    