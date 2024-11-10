import torch
import os
from torchvision.utils import save_image
import datetime

from src.models import _netG_CIFAR10, _netD_CIFAR10
from src.utils import get_transform
from src.constants import *
from src.models import *
from src.train_utils import *
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

# Paths to the saved models for baseline ACGAN
GENERATOR_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/output/acgan_skin/saved_models/0.001/netG_epoch_350_last.pth"
DISCRIMINATOR_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/output/acgan_skin/saved_models/0.001/netD_epoch_350_last.pth"

# Paths to the saved models for baseline ACGAN + Proto
GENERATOR_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/output/acgan_skin/saved_models/ckpts/netG_epoch_300_sz_32.pth"
DISCRIMINATOR_PATH = "/home/judy/code/sys-med/AUGMENTATION_GAN/output/acgan_skin/saved_models/ckpts/netD_epoch_300_sz_32.pth"

# Assuming 'generator', 'discriminator', and 'metrics' are predefined models and metrics
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# List of N values (number of images to generate per class)
# N_list = [100, 500, 1000, 2000]  # You can adjust this list with any number of desired N values
N_per_class = 1000
nz = 110
num_classes = 7
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

def build_data_loader_from_synthetic(batch_size=32, input_size=32):
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

if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    generate_samples()
    data_loader = build_data_loader_from_synthetic()
    for images, labels in data_loader:
        print(images.shape, labels)
        break
