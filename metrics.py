import os
import numpy as np
from PIL import Image

def calculate_psnr(original, generated):
    """Calculate the PSNR between two images."""
    mse = np.mean((np.array(original) - np.array(generated)) ** 2)
    if mse == 0:
        return float('inf')  # No noise, images are identical
    return 20 * np.log10(255.0 / np.sqrt(mse))

def process_folders(real_folder, fake_folder):
    """Process images in both folders and calculate PSNR."""
    psnr_values = []
    real_images = os.listdir(real_folder)
    
    for image_name in real_images:
        real_image_path = os.path.join(real_folder, image_name)
        fake_image_path = os.path.join(fake_folder, image_name)
        
        if os.path.exists(fake_image_path):
            real_image = Image.open(real_image_path)
            fake_image = Image.open(fake_image_path)
            psnr = calculate_psnr(real_image, fake_image)
            psnr_values.append((image_name, psnr))
            print(f'PSNR for {image_name}: {psnr:.2f} dB')
        else:
            print(f'Fake image not found for {image_name}')
    
    return psnr_values

REAL_FOLDER_PATH="custom_covid_dataset/train/covid"
FAKE_FOLDER_PATH="custom_covid_dataset/train_classic/2000/covid"
if __name__ == "__main__":
    # REAL_FOLDER_PATH = 'path_to_your_real_images'  # Replace with your real images folder path
    # FAKE_FOLDER_PATH = 'path_to_your_fake_images'  # Replace with your fake images folder path

    psnr_results = process_folders(REAL_FOLDER_PATH, FAKE_FOLDER_PATH)

    # Optionally, you can save the results to a file
    with open('psnr_results.txt', 'w') as f:
        for image_name, psnr in psnr_results:
            f.write(f'{image_name}: {psnr:.2f} dB\n')
