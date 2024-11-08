# import sys
# import torch
# import torch.nn as nn

# sys.path.append("./")

# from src.constants import *

# # Number of channels in the training images. For color images this is 3
# # nc = 1
# nc = 3
# # Size of z latent vector (i.e. size of generator input)
# nz = 100
# # Size of feature maps in discriminator
# ndf = 64
# # Size of feature maps in generator
# ngf = 64


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()
#         self.main = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )

#     def forward(self, input):
#         return self.main(input)
    
# # class Discriminator(nn.Module):
# #     def __init__(self):
# #         super(Discriminator, self).__init__()
# #         self.main = nn.Sequential(
# #             # input is (nc) x 64 x 64
# #             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Dropout(p=0.5),
# #             # state size. (ndf) x 32 x 32
# #             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(ndf * 2),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # state size. (ndf*2) x 16 x 16
# #             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(ndf * 4),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             nn.Dropout(p=0.5),
# #             # state size. (ndf*4) x 8 x 8
# #             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
# #             nn.BatchNorm2d(ndf * 8),
# #             nn.Dropout(p=0.25),
# #             nn.LeakyReLU(0.2, inplace=True),
# #             # state size. (ndf*8) x 4 x 4
# #             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
# #             nn.Sigmoid()
# #         )

# #     def forward(self, input):
# #         return self.main(input)

# # import torch
# # import torch.nn as nn

# class Discriminator(nn.Module):
#     def __init__(self, nc=nc, ndf=64):
#         super(Discriminator, self).__init__()
#         # Define downsampling layers
#         self.down_blocks = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(p=0.5),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Dropout(p=0.5),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.Dropout(p=0.25),
#             nn.LeakyReLU(0.2, inplace=True),
#         )
        
#         # Final classification layer
#         self.final_conv = nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
#         self.sigmoid = nn.Sigmoid()

#     def forward(self, input):
#         # Pass through downsampling layers
#         features = self.down_blocks(input)
#         # print(features.shape)
#         feat_out = features.view(features.size(0), -1)
#         feat_out_mean = torch.mean(feat_out,dim=0)
#         feat_out_var = torch.var(feat_out,dim=0)
#         # Return the intermediate feature map as prototype
#         return self.sigmoid(self.final_conv(features)), (feat_out, feat_out_mean, feat_out_var)

# # # Example: Extracting prototypes
# # discriminator = Discriminator()
# # input_image = torch.randn(1, 3, 64, 64)  # Example input image
# # prototypes, output = discriminator(input_image)

# # print(f"Prototypes shape: {prototypes.shape}")

    
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
        
# class CNN(nn.Module):

#     def _name(self):
#         return "CNN"

#     def _conv2d(self, in_channels, out_channels):
#         return nn.Conv2d(
#             in_channels=in_channels,
#             out_channels=out_channels,
#             kernel_size=3,
#             padding=1
#         )

#     def _build_models(self):
#         self.conv1 = nn.Sequential(
#             self._conv2d(self.channels_size, self.hidden_size),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(self.hidden_size),
#             nn.MaxPool2d(2, 2)
#         )
#         self.conv2 = nn.Sequential(
#             self._conv2d(self.hidden_size , self.hidden_size * 2),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(self.hidden_size * 2),
#             nn.MaxPool2d(2, 2)
#         )
#         self.conv3 = nn.Sequential(
#             self._conv2d(self.hidden_size*2, self.hidden_size * 4),
#             nn.ReLU(inplace=True),
#             nn.BatchNorm2d(self.hidden_size * 4),
#             nn.MaxPool2d(2, 2)
#         )
#         self.dense = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(int(self.hidden_size * 4 * img_dim/8 * img_dim/8), img_dim*4),
#             nn.ReLU(),
#             nn.Dropout(p=0.5),
#             nn.Linear(img_dim*4, class_number),
#             nn.Softmax(dim=1)
#         )
#         return self.conv1, self.conv2, self.conv3, self.dense

#     def __init__(self, hidden_size, channels_size, class_number):
#         super().__init__()
#         self.hidden_size = hidden_size
#         self.channels_size = channels_size
#         self.class_number = class_number
#         self._models = self._build_models()
#         self.name = self._name()

#     def forward(self, image):
#         x = self._models[0](image)
#         x_1 = self._models[1](x)
#         x_2 = self._models[2](x_1)
#         x_3 = self._models[3](x_2)
#         return x_3
    
# class netG(nn.Module):

#     def __init__(self, nz, ngf, nc):

#         super(netG, self).__init__()
#         self.ReLU = nn.ReLU(True)
#         self.Tanh = nn.Tanh()
#         #self.DropOut = nn.Dropout(p=0.75)
#         #self.conv0 = nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 1, bias=False)
#         #self.BatchNorm0 = nn.BatchNorm2d(ngf * 16)
#         self.conv1 = nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False)
#         self.BatchNorm1 = nn.BatchNorm2d(ngf * 8)

#         self.conv2 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False)
#         self.BatchNorm2 = nn.BatchNorm2d(ngf * 4)

#         self.conv3 = nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False)
#         self.BatchNorm3 = nn.BatchNorm2d(ngf * 2)

#         self.conv4 = nn.ConvTranspose2d(ngf * 2, ngf * 1, 4, 2, 1, bias=False)
#         self.BatchNorm4 = nn.BatchNorm2d(ngf * 1)

#         self.conv5 = nn.ConvTranspose2d(ngf * 1, nc, 4, 2, 1, bias=False)

#         self.apply(weights_init)


#     def forward(self, input):
#         #x = self.conv0(input)
#         #x = self.BatchNorm0(x)
#         #x = self.ReLU(x)
#         x = self.conv1(input)
#         x = self.BatchNorm1(x)
#         x = self.ReLU(x)
#         #x = self.DropOut(x)

#         x = self.conv2(x)
#         x = self.BatchNorm2(x)
#         x = self.ReLU(x)
#         #x = self.DropOut(x)

#         x = self.conv3(x)
#         x = self.BatchNorm3(x)
#         x = self.ReLU(x)
#         #x = self.DropOut(x)

#         x = self.conv4(x)
#         x = self.BatchNorm4(x)
#         x = self.ReLU(x)
#         #x = self.DropOut(x)

#         x = self.conv5(x)
#         output = self.Tanh(x)
#         return output
    
# # class netD(nn.Module):
# #     def __init__(self, ndf=64, nc=3, nb_label=10):  # nc=3 for RGB
# #         super(netD, self).__init__()
# #         self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
# #         self.DropOut1 = nn.Dropout(p=0.5)
# #         self.DropOut2 = nn.Dropout(p=0.25)

# #         # Convolutional layers
# #         self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)  # 224 -> 112
# #         self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)  # 112 -> 56
# #         self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
# #         self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)  # 56 -> 28
# #         self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
# #         self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)  # 28 -> 14
# #         self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
# #         self.conv5 = nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False)  # 14 -> 7

# #         # Linear layers for discriminator and auxiliary classifier
# #         self.disc_linear = nn.Linear(ndf * 16 * 7 * 7, 1)
# #         self.aux_linear = nn.Linear(ndf * 16 * 7 * 7, nb_label)

# #         # Activation functions
# #         self.softmax = nn.Softmax(dim=1)
# #         self.sigmoid = nn.Sigmoid()

# #         self.ndf = ndf

# #         # Initialize weights
# #         self.apply(weights_init)

# #     def feature(self, input):
# #         x = self.conv1(input)  # 224 -> 112
# #         x = self.LeakyReLU(x)
# #         x = self.DropOut1(x)

# #         x = self.conv2(x)  # 112 -> 56
# #         x = self.BatchNorm2(x)
# #         x = self.LeakyReLU(x)

# #         x = self.conv3(x)  # 56 -> 28
# #         x = self.BatchNorm3(x)
# #         x = self.LeakyReLU(x)
# #         x = self.DropOut1(x)

# #         x = self.conv4(x)  # 28 -> 14
# #         x = self.BatchNorm4(x)
# #         x = self.LeakyReLU(x)
# #         x = self.DropOut2(x)

# #         x = self.conv5(x)  # 14 -> 7
# #         print(f"After conv5: {x.shape}")  # Debug

# #         # Flatten the tensor for linear layers
# #         batch_size = x.shape[0]
# #         x = x.view(batch_size, -1)
# #         print(f"After view: {x.shape}")  # Debug
# #         return x

# #     def forward(self, input):
# #         feat = self.feature(input)
# #         c = self.aux_linear(feat)
# #         c = self.softmax(c)
# #         s = self.disc_linear(feat)
# #         s = self.sigmoid(s)
# #         return s, c, feat

# # Weight initialization function
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
        
# # Weight initialization function (optional, customize as needed)
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
    
# class netD(nn.Module):

#     def __init__(self, ndf, nc, nb_label):

#         super(netD, self).__init__()
#         self.LeakyReLU = nn.LeakyReLU(0.2, inplace=True)
#         self.DropOut1 = nn.Dropout(p=0.5)
#         self.DropOut2 = nn.Dropout(p=0.25)
#         self.conv1 = nn.Conv2d(nc, ndf, 4, 2, 1, bias=False)
#         self.conv2 = nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False)
#         self.BatchNorm2 = nn.BatchNorm2d(ndf * 2)
#         self.conv3 = nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False)
#         self.BatchNorm3 = nn.BatchNorm2d(ndf * 4)
#         self.conv4 = nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False)
#         self.BatchNorm4 = nn.BatchNorm2d(ndf * 8)
#         self.conv5 = nn.Conv2d(ndf * 8, ndf * 1, 4, 1, 0, bias=False)
#         self.disc_linear = nn.Linear(ndf * 1, 1)
#         self.aux_linear = nn.Linear(ndf * 1, nb_label)
#         self.softmax = nn.Softmax(dim=1)
#         self.sigmoid = nn.Sigmoid()
#         self.ndf = ndf
#         self.apply(weights_init)
    
#     def feature(self, input):
#         x = self.conv1(input)
#         x = self.LeakyReLU(x)
#         x = self.DropOut1(x)

#         x = self.conv2(x)
#         x = self.BatchNorm2(x)
#         x = self.LeakyReLU(x)
#         #x = self.DropOut(x)

#         x = self.conv3(x)
#         x = self.BatchNorm3(x)
#         x = self.LeakyReLU(x)
#         x = self.DropOut1(x)

#         x = self.conv4(x)
#         x = self.BatchNorm4(x)
#         x = self.LeakyReLU(x)
#         x = self.DropOut2(x)

#         x = self.conv5(x)
#         x = x.view(-1, self.ndf * 1)
#         return x
    
#     def forward(self, input):
#         feat = self.feature(input)
#         c = self.aux_linear(feat)
#         c = self.softmax(c)
#         s = self.disc_linear(feat)
#         s = self.sigmoid(s)
#         return s, c, feat
    
#         x = self.conv1(input)
#         x = self.LeakyReLU(x)
#         x = self.DropOut1(x)

#         x = self.conv2(x)
#         x = self.BatchNorm2(x)
#         x = self.LeakyReLU(x)
#         #x = self.DropOut(x)

#         x = self.conv3(x)
#         x = self.BatchNorm3(x)
#         x = self.LeakyReLU(x)
#         x = self.DropOut1(x)

#         x = self.conv4(x)
#         x = self.BatchNorm4(x)
#         x = self.LeakyReLU(x)
#         x = self.DropOut2(x)

#         x = self.conv5(x)
#         x = x.view(-1, self.ndf * 1)
#         # import IPython; IPython.embed()
#         c = self.aux_linear(x)
#         c = self.softmax(c)
#         s = self.disc_linear(x)
#         s = self.sigmoid(s)
#         return s, c

import torch
import torch.nn as nn


class _netG(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 768)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(768, 384, 5, 2, 0, bias=False),
            nn.BatchNorm2d(384),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(384, 256, 5, 2, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(256, 192, 5, 2, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(192, 64, 5, 2, 0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
        # Transposed Convolution 5
        self.tconv6 = nn.Sequential(
            nn.ConvTranspose2d(64, 3, 8, 2, 0, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv6, tconv5, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 768, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            tconv5 = self.tconv6(tconv5)
            output = tconv5
        return output


class _netD(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(_netD, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 0, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(13*13*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(13*13*512, num_classes)
        # softmax and sigmoid
        self.softmax = nn.Softmax()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            flat6 = conv6.view(-1, 13*13*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes


class _netG_CIFAR10(nn.Module):
    def __init__(self, ngpu, nz):
        super(_netG_CIFAR10, self).__init__()
        self.ngpu = ngpu
        self.nz = nz

        # first linear layer
        self.fc1 = nn.Linear(110, 384)
        # Transposed Convolution 2
        self.tconv2 = nn.Sequential(
            nn.ConvTranspose2d(384, 192, 4, 1, 0, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(True),
        )
        # Transposed Convolution 3
        self.tconv3 = nn.Sequential(
            nn.ConvTranspose2d(192, 96, 4, 2, 1, bias=False),
            nn.BatchNorm2d(96),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv4 = nn.Sequential(
            nn.ConvTranspose2d(96, 48, 4, 2, 1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(True),
        )
        # Transposed Convolution 4
        self.tconv5 = nn.Sequential(
            nn.ConvTranspose2d(48, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            input = input.view(-1, self.nz)
            fc1 = nn.parallel.data_parallel(self.fc1, input, range(self.ngpu))
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = nn.parallel.data_parallel(self.tconv2, fc1, range(self.ngpu))
            tconv3 = nn.parallel.data_parallel(self.tconv3, tconv2, range(self.ngpu))
            tconv4 = nn.parallel.data_parallel(self.tconv4, tconv3, range(self.ngpu))
            tconv5 = nn.parallel.data_parallel(self.tconv5, tconv4, range(self.ngpu))
            output = tconv5
        else:
            input = input.view(-1, self.nz)
            fc1 = self.fc1(input)
            fc1 = fc1.view(-1, 384, 1, 1)
            tconv2 = self.tconv2(fc1)
            tconv3 = self.tconv3(tconv2)
            tconv4 = self.tconv4(tconv3)
            tconv5 = self.tconv5(tconv4)
            output = tconv5
        return output


class _netD_CIFAR10(nn.Module):
    def __init__(self, ngpu, num_classes=10):
        super(_netD_CIFAR10, self).__init__()
        self.ngpu = ngpu

        # Convolution 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 2
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 3
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 4
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 5
        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # Convolution 6
        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False),
        )
        # discriminator fc
        self.fc_dis = nn.Linear(4*4*512, 1)
        # aux-classifier fc
        self.fc_aux = nn.Linear(4*4*512, num_classes)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))

        # softmax and sigmoid
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            conv1 = nn.parallel.data_parallel(self.conv1, input, range(self.ngpu))
            conv2 = nn.parallel.data_parallel(self.conv2, conv1, range(self.ngpu))
            conv3 = nn.parallel.data_parallel(self.conv3, conv2, range(self.ngpu))
            conv4 = nn.parallel.data_parallel(self.conv4, conv3, range(self.ngpu))
            conv5 = nn.parallel.data_parallel(self.conv5, conv4, range(self.ngpu))
            conv6 = nn.parallel.data_parallel(self.conv6, conv5, range(self.ngpu))
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = nn.parallel.data_parallel(self.fc_dis, flat6, range(self.ngpu))
            fc_aux = nn.parallel.data_parallel(self.fc_aux, flat6, range(self.ngpu))
        else:
            conv1 = self.conv1(input)
            conv2 = self.conv2(conv1)
            conv3 = self.conv3(conv2)
            conv4 = self.conv4(conv3)
            conv5 = self.conv5(conv4)
            conv6 = self.conv6(conv5)
            conv6 = self.pool(conv6)
            flat6 = conv6.view(-1, 4*4*512)
            fc_dis = self.fc_dis(flat6)
            fc_aux = self.fc_aux(flat6)
            # import IPython; IPython.embed()
        classes = self.softmax(fc_aux)
        realfake = self.sigmoid(fc_dis).view(-1, 1).squeeze(1)
        return realfake, classes