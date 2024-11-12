from glob import glob
import pandas as pd
import torch
import os
import torchvision
from torchvision.utils import save_image
import datetime

from src.models import _netG_CIFAR10, _netD_CIFAR10
from src.utils import get_transform, set_seed
from src.constants import *
from src.models import *
from src.train_utils import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.model_selection import train_test_split
from PIL import Image

# Paths to the saved models for baseline ACGAN
# GENERATOR_PATH = "output/acgan_skin/0.001/netG_epoch_350_last.pth"
# DISCRIMINATOR_PATH = "output/acgan_skin/0.001/netD_epoch_350_last.pth"

GENERATOR_PATH = "output/acgan_skin/saved_models/ckpts/netG_epoch_200_2024.pth"
DISCRIMINATOR_PATH = "output/acgan_skin/saved_models/ckpts/netD_epoch_200_2024.pth"
# # Paths to the saved models for baseline ACGAN + Proto
# GENERATOR_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/output/acgan_skin/saved_models/ckpts/netG_epoch_350_sz_32.pth"
# DISCRIMINATOR_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/output/acgan_skin/saved_models/ckpts/netD_epoch_350_sz_32.pth"

# Assuming 'generator', 'discriminator', and 'metrics' are predefined models and metrics
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# List of N values (number of images to generate per class)
# N_list = [100, 500, 1000, 2000]  # You can adjust this list with any number of desired N values
N_per_class = 500
nz = 110
num_classes = 7
# classes <- ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
OUTPUT_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/skin_lesson_dataset/2/train_synthetic_proto"

normMean = [0.763033, 0.5456458, 0.5700401]
normStd = [0.14092815, 0.15261315, 0.16997056]

import os
import datetime
import torch
import torch.nn as nn
from torchvision.utils import save_image

def generate_samples():
    generator = _netG_CIFAR10(1, nz).to(device)
    discriminator = _netD_CIFAR10(1, num_classes).to(device)
    generator.load_state_dict(torch.load(GENERATOR_PATH))
    discriminator.load_state_dict(torch.load(DISCRIMINATOR_PATH))
    generator.eval()
    discriminator.eval()
    now = datetime.datetime.now()
    im_batch_size = 50
    with torch.no_grad():
        for class_idx in range(num_classes):
            class_path = os.path.join(OUTPUT_PATH, str(class_idx))
            os.makedirs(class_path, exist_ok=True)
            valid_images_count = 0
            batch_index = 0

            while valid_images_count < N_per_class:
                gen_z = torch.FloatTensor(batch_size, nz, 1, 1).normal_(0, 1)
                gen_z = gen_z.to(device)
                # gen_z = torch.randn(im_batch_size, nz, 1, 1, device=device)
                class_tensor = torch.full((N_per_class,), class_idx, dtype=torch.long, device=device)
                gen_images = generator(gen_z)
                s, c, _ = discriminator(gen_images)
                pred_class = c.argmax(dim=1).view(-1)
                matching_images_indices = (pred_class == class_idx).nonzero(as_tuple=True)[0]
                matching_images = gen_images[matching_images_indices]
                num_matching_images = matching_images.size(0)

                if num_matching_images > 0:
                    for i in range(num_matching_images):
                        if valid_images_count >= N_per_class:
                            break
                        save_image(matching_images[i], os.path.join(class_path, f"{valid_images_count}.png"))
                        valid_images_count += 1

                    print(f"Generated {valid_images_count} images for class {class_idx}")

            print(f"Finished generating images for class {class_idx}")

# Check and improvements:
# 1. Removed duplicate counting of `valid_images_count`.
# 2. Ensured that saving images does not exceed `N_per_class`.
# 3. Fixed redundant code in saving images and incrementing `valid_images_count`.
# 4. Removed redundant double-saving logic.
# 5. Ensured logical flow in the `while` loop by stopping early if `valid_images_count` meets `N_per_class`. 

# Run the function
# generate_samples()

class SyntheticDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        for class_idx in range(num_classes):
            class_path = os.path.join(root_dir, str(class_idx))
            self.image_paths.extend([os.path.join(class_path, img) for img in os.listdir(class_path)])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = int(os.path.basename(os.path.dirname(img_path)))
        
        # change to tensor
        label = torch.tensor(label)
        
        if self.transform:
            image = self.transform(image)
        return image, label

def build_data_loader_from_synthetic(batch_size=32, input_size=32, use_proto=False):
    if use_proto:
        OUTPUT_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/skin_lesson_dataset/2/train_synthetic_proto"
    else:
        OUTPUT_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/skin_lesson_dataset/2/train_synthetic"
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])
    
    dataset = SyntheticDataset(OUTPUT_PATH, transform=train_transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return data_loader

def split_data_by_labels():
    set_seed(10)
    data_dir = '/home/judy/code/sys-med/AUGMENTATION_GAN/skin_lesson_dataset/2'
    all_image_path = glob(os.path.join(data_dir, '*', '*.jpg'))
    imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x for x in all_image_path}
    lesion_type_dict = {
        'nv': 'Melanocytic nevi',
        'mel': 'dermatofibroma',
        'bkl': 'Benign keratosis-like lesions ',
        'bcc': 'Basal cell carcinoma',
        'akiec': 'Actinic keratoses',
        'vasc': 'Vascular lesions',
        'df': 'Dermatofibroma'  
    }
    
    df_original = pd.read_csv(os.path.join(data_dir, 'HAM10000_metadata.csv'))
    df_original['path'] = df_original['image_id'].map(imageid_path_dict.get)
    df_original['cell_type'] = df_original['dx'].map(lesion_type_dict.get)
    df_original['cell_type_idx'] = pd.Categorical(df_original['cell_type']).codes

    # this will tell us how many images are associated with each lesion_id
    df_undup = df_original.groupby('lesion_id').count()
    # now we filter out lesion_id's that have only one image associated with it
    df_undup = df_undup[df_undup['image_id'] == 1]
    df_undup.reset_index(inplace=True)
    df_undup.head()
    
    # here we identify lesion_id's that have duplicate images and those that have only one image.
    def get_duplicates(x):
        unique_list = list(df_undup['lesion_id'])
        if x in unique_list:
            return 'unduplicated'
        else:
            return 'duplicated'

    # create a new colum that is a copy of the lesion_id column
    df_original['duplicates'] = df_original['lesion_id']
    # apply the function to this new column
    df_original['duplicates'] = df_original['duplicates'].apply(get_duplicates)
    df_original.head()
        

    df_undup = df_original[df_original['duplicates'] == 'unduplicated']
    
    y = df_undup['cell_type_idx']
    _, df_val = train_test_split(df_undup, test_size=0.2, random_state=101, stratify=y)

    def get_val_rows(x):
        # create a list of all the lesion_id's in the val set
        val_list = list(df_val['image_id'])
        if str(x) in val_list:
            return 'val'
        else:
            return 'train'

    # identify train and val rows
    # create a new colum that is a copy of the image_id column
    df_original['train_or_val'] = df_original['image_id']
    # apply the function to this new column
    df_original['train_or_val'] = df_original['train_or_val'].apply(get_val_rows)
    # filter out train rows
    df_train = df_original[df_original['train_or_val'] == 'train']
    
    # save all images for each class to a separate folder
    for key, value in lesion_type_dict.items():
        os.makedirs(os.path.join(data_dir, "split", key), exist_ok=True)
        # get images based on cell type
        df_temp = df_train[df_train['cell_type'] == value]
        for _, item in df_temp.iterrows():
            img = Image.open(item['path'])
            img.save(os.path.join(data_dir, "split", key, item['image_id'] + '.jpg'))
            
    # for idx in range(7):
    #     os.makedirs(os.path.join(data_dir, "split", str(idx)), exist_ok=True)
        
def get_synthetic_dataset_by_class(y=6, num_samples=1000):
    # data_path = "/home/judy/code/DCAI-augmented-med/skin_lesson_dataset/2/split_synthetic/eval_30000/img"
    data_path = "/home/judy/code/DCAI-augmented-med/skin_lesson_dataset/2/split_synthetic/vasc/eval_6500/img"
    if y == 5:
        data_path = "/home/judy/code/DCAI-augmented-med/skin_lesson_dataset/2/split_synthetic/vasc/eval_6500/img"
    elif y == 6:
        data_path = "/home/judy/code/DCAI-augmented-med/skin_lesson_dataset/2/split_synthetic/eval_30000/img"
    # Gather all samples
    all_samples = os.listdir(data_path)
    all_samples = [os.path.join(data_path, x) for x in all_samples]
    samples = all_samples[:num_samples]
    
    # Build a list of (image path, label)
    dataset = [(sample, y) for sample in samples]
    return dataset

# Dataset class
class SyntheticDataset2(Dataset):
    def __init__(self, transform=None, num_samples=1000, y=5):
        self.num_samples = num_samples
        self.data = get_synthetic_dataset_by_class(y, num_samples)
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, torch.tensor(label)

def get_df_class_dataset(input_size=32):
    train_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomVerticalFlip(),
        # transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(normMean, normStd)
    ])
    dataset1 = SyntheticDataset2(transform=train_transform, num_samples=1000, y=6)
    dataset2 = SyntheticDataset2(transform=train_transform, num_samples=200, y=5)
    
    # Plot some images using grid in torchvision
    # import matplotlib.pyplot as plt
    # import torchvision
    
    # concatenate the two datasets
    dataset = torch.utils.data.ConcatDataset([dataset1, dataset2])
    return dataset
    # return SyntheticDataset2(transform=train_transform)

# Function to visualize and save a batch of images
def save_batch_as_image(dataset, batch_size=16, filename="output_image_grid.png"):
    # Load a batch of images
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    images, labels = next(iter(dataloader))
    
    # Undo normalization for visualization
    images = images * torch.tensor(normStd).view(3, 1, 1) + torch.tensor(normMean).view(3, 1, 1)
    images = torch.clamp(images, 0, 1)  # Clamp to [0,1] range for saving

    # Create a grid of images
    grid_img = torchvision.utils.make_grid(images, nrow=4)

    # Convert to numpy array for saving
    np_img = grid_img.permute(1, 2, 0).numpy()

    # Save the image using matplotlib
    plt.imsave(filename, np_img)


if __name__ == "__main__":
    # Load dataset and save the image
    dataset = get_df_class_dataset(input_size=32)
    # save_batch_as_image(dataset, batch_size=16, filename="resized_images_grid.png")
    # parser = argparse.ArgumentParser()
    generate_samples()
    # data_loader = build_data_loader_from_synthetic()
    # for images, labels in data_loader:
    #     print(images.shape, labels)
    #     break
    # split_data_by_labels()
