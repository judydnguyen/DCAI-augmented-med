import torch
import os
from torchvision.utils import save_image
import datetime

from src.utils import get_transform
from src.constants import *
from src.models import *
from src.train_utils import *

PATH = "augGAN/model/-21.495_+1.273_100_2024-10-14_21:29:25.dat"


# Assuming 'generator', 'discriminator', and 'metrics' are predefined models and metrics
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define your sets (number of images per class) and the desired number of samples per class
N = 100  # Number of images to generate per class
paths = [
    'custom_covid_dataset/train_synthetic_proto/covid',
    'custom_covid_dataset/train_synthetic_proto/normal',
    'custom_covid_dataset/train_synthetic_proto/pneumonia_bac',
    'custom_covid_dataset/train_synthetic_proto/pneumonia_vir'
]

# Define the class names
classes = ("covid", "normal", "pneumonia_bac", "pneumonia_vir")

# Modify the test_fake function to generate N samples per class with filtering
def test_fake(generator, discriminator, metrics, n_images_per_class, paths, classes):
    now = datetime.datetime.now()
    g_losses = metrics['train.G_losses'][-1]
    d_losses = metrics['train.D_losses'][-1]
    im_batch_size = 50  # Number of images per batch
    generator.eval()
    discriminator.eval()
    with torch.no_grad():

        for class_idx, class_name in enumerate(classes):  # Loop through each class
            class_path = paths[class_idx]
            try:
                os.makedirs(class_path, exist_ok=True)
                os.makedirs(os.path.join(class_path, str(N)), exist_ok=True)
            except Exception as error:
                print(error)

            valid_images_count = 0  # Track how many valid images have been saved
            batch_index = 0

            # Continue generating until we have enough valid images
            while valid_images_count < n_images_per_class:
                # Generate latent noise vectors
                gen_z = torch.randn(im_batch_size, 100, 1, 1, device=device)
                
                # Assuming the generator takes noise and class labels as input
                class_tensor = torch.full((im_batch_size,), class_idx, dtype=torch.long, device=device)
                gen_images = generator(gen_z)  # Modify this based on your GAN architecture
                
                # Discriminator output
                s, c,  _ = discriminator(gen_images)
                
                # import IPython; IPython.embed()
                # Get the predicted class
                pred_class = c.argmax(dim=1).view(-1)
                
                # pred_class = c.view(-1)
                print(f"pred_class: {pred_class}")
                
                # Filter only the images that the discriminator classifies as matching the class
                # import IPython; IPython.embed()
                matching_images_indices = (pred_class == class_idx).nonzero(as_tuple=True)[0]
                matching_images = gen_images[matching_images_indices]
                num_matching_images = matching_images.size(0)

                # Check if we have enough valid images to save
                if valid_images_count + num_matching_images > n_images_per_class:
                    num_to_save = n_images_per_class - valid_images_count
                    matching_images = matching_images[:num_to_save]
                    num_matching_images = num_to_save

                # Convert images to CPU for saving
                images = matching_images.to("cpu").clone().detach()
                images = images.numpy().transpose(0, 2, 3, 1)  # Convert to HWC format for saving

                # Save the generated images
                for i_image in range(num_matching_images):
                    image_path = os.path.join(class_path, str(N), f'image_{batch_index:04d}_{i_image}.png')
                    save_image(matching_images[i_image, :, :, :], image_path, normalize=True)
                    batch_index += 1
                
                # Update the count of valid images
                valid_images_count += num_matching_images

                print(f'Class {class_name}: {valid_images_count}/{n_images_per_class} images generated')

            print(f'Generated {n_images_per_class} images for class {class_name}')

# Assuming you have a generator, discriminator, and metrics objects available

# generator = Generator().to(device)
# discriminator = Discriminator().to(device)

generator = netG(nz, ngf, nc).to(device)
discriminator = netD(ndf, nc, nb_label).to(device)

checkpoint = torch.load(PATH)
generator.load_state_dict(checkpoint['state_dict_generator'])
discriminator.load_state_dict(checkpoint['state_dict_discriminator'])
metrics=checkpoint['metrics']
# Call the function to generate N samples per class
test_fake(generator, discriminator, metrics, N, paths, classes)
