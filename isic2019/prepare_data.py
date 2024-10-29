import torch
import os
import sys

from tqdm import tqdm
import numpy as np
from PIL import Image
from pathlib import Path
import torchvision
from torchvision import transforms

sys.path.append(os.path.abspath(".."))
from src.utils import set_seed

classes = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

img_dim = 64
N_rot = 2
N_flip = 2
N_tran = 4
N_scal = 4
N_color = 2

transform_list = []
transform_list.append(transforms.Resize((img_dim * 2, img_dim * 2), interpolation=2))

for i in range(N_color):
    transform_list.append(
        transforms.RandomApply(
            [
                transforms.ColorJitter(
                    brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5
                )
            ],
            p=0.5,
        )
    )

for i in range(N_rot):
    transform_list.append(
        transforms.RandomApply([transforms.RandomRotation(degrees=(-5, 5))], p=0.5)
    )

for i in range(N_flip):
    transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

for i in range(N_scal):
    transform_list.append(
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, scale=(0.9, 1.1))], p=0.5
        )
    )

for i in range(N_tran):
    transform_list.append(
        transforms.RandomApply(
            [transforms.RandomAffine(degrees=0, translate=(0.1, 0.1))], p=0.5
        )
    )

transform_list.append(transforms.Resize((img_dim, img_dim), interpolation=2))
transform_list.append(transforms.ToTensor())
transform = transforms.Compose(transform_list)

test_transform = transforms.Compose(
    [
        transforms.Resize((img_dim, img_dim), interpolation=2),
        transforms.ToTensor(),
    ]
)

class MyImageFolder(torchvision.datasets.ImageFolder):
    def __getitem__(self, index):
        return (
            super(MyImageFolder, self).__getitem__(index)[0],
            super(MyImageFolder, self).__getitem__(index)[1],
            self.imgs[index][0],
        )  # return image path

batch_size = 1
data_dir = "data"  # noqa

trainset = MyImageFolder(os.path.join(data_dir, "train/"), transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testset = MyImageFolder(os.path.join(data_dir, "test/"), transform=test_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)
normalset = MyImageFolder(os.path.join(data_dir, "train/"), transform=test_transform)
normalloader = torch.utils.data.DataLoader(
    normalset, batch_size=batch_size, shuffle=False
)

# sets = [4_000, 8_000, 12_000, 16_000]

fold_class_size = int(sys.argv[1])
class_count = np.zeros((len(classes),), dtype=int) # sequential

# reset seed because I had to restart the kernel
set_seed(42)

# tqdm empty progress bar
pbar = tqdm(total=fold_class_size * len(classes))

print("folder start: ", fold_class_size)
dataiter = iter(trainloader)

while np.any(class_count < fold_class_size):
    
    # loop back with random
    try:
        images, labels, path = next(dataiter)
    except StopIteration:
        dataiter = iter(trainloader)
        images, labels, path = next(dataiter)

    npimg = np.transpose(images.squeeze(0).numpy(), (1, 2, 0))
    img = Image.fromarray((npimg * 255).astype(np.uint8))

    if class_count[labels[0]] < fold_class_size:
        img_path = path[0].replace("/train/", f"/train_classic/{str(fold_class_size)}/")
        img_dir, img_name = img_path.rsplit("/", 1)
        img_path = img_dir + f"/{fold_class_size}.{class_count[labels[0]]}." + img_name
        # and ensure the folder exists
        Path(img_dir).mkdir(parents=True, exist_ok=True)
        class_count[labels[0]] += 1
        img.save(img_path)

        # update progress bar
        pbar.update(1)
