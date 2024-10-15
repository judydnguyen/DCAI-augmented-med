import torch
import torch.nn as nn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 64
# Number of training epochs
num_epochs = 200
LOAD_MODEL = False

#PATH='/AUGMENTATION_GAN/gan_models/epoch_200/p_virus_200_2020-08-22_15:49:13.dat' #P_vir_200_opt
#PATH='/AUGMENTATION_GAN/gan_models/epoch_200/p_bacteria_200_2020-08-22_16:21:47.dat' #P_bac_200_opt
#PATH='/AUGMENTATION_GAN/gan_models/epoch_200/normal_200_2020-08-22_16:38:52.dat' #Normal_200_opt
#PATH='/AUGMENTATION_GAN/gan_models/epoch_200/covid_200_2020-08-22_16:58:21.dat' #Covid_200_opt

TRAIN_ALL = True
#All images will be resized to this size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Learning rate for optimizers
lr = 0.002
lr_d = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Beta2 hyperparam for Adam optimizers
beta2 = 0.999

real_label = 1.
fake_label = 0.
# Input to generator
fixed_noise = torch.randn(64, nz, 1, 1, device=device) #batch of 64
# Define Loss function
criterion = nn.BCELoss()

batch_size_train=256
batch_size_test=36
num_samples=batch_size_train
img_dim=64

hidden_size=32
channels_size=1
class_number=4

batch_size = 32
# Number of training epochs
num_epochs = 100

#All images will be resized to this size using a transformer.
#image_size = 64
imageSize = 64

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# No of labels
nb_label = 4

# Learning rate for optimizers
lr = 0.002
lr_d = 0.0002
# Beta1 hyperparam for Adam optimizers
beta1 = 0.5
# Beta2 hyperparam for Adam optimizers
beta2 = 0.999

real_label = 1.
fake_label = 0.