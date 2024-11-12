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
# from train_classifier_skin import tsne_plot
import wandb

SEED = 42
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          path="."):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    os.makedirs(path, exist_ok=True)
    path_to_save = os.path.join(path, 'confusion_matrix.png')
    plt.savefig(path_to_save)
    
def eval(model, val_loader, path_to_save):
    model.eval()

    
    y_label = []
    y_predict = []
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            images, labels = data
            N = images.size(0)
            images = Variable(images).to(device)
            outputs = model(images)
            prediction = outputs.max(1, keepdim=True)[1]
            y_label.extend(labels.cpu().numpy())
            y_predict.extend(np.squeeze(prediction.cpu().numpy().T))

    # compute the confusion matrix
    confusion_mtx = confusion_matrix(y_label, y_predict)
    plot_labels = ['akiec', 'bcc', 'bkl', 'df', 'nv', 'vasc','mel']
    plot_confusion_matrix(confusion_mtx, plot_labels, path=path_to_save)
    # print a table of the classification results
    print(classification_report(y_label, y_predict, target_names=plot_labels))
    
    # tsne_plot(model, val_loader, save_path=path_to_save)
    
    
def main(args):
    # set seed
    set_seed(SEED)
    # load data
    df_train, df_val, _ = load_train_test_df()
    # define the model
    model = get_cls_model(args.model, args.num_classes, args.pretrained)
    model = model.to(device)
    
    # define path to load model
    model_to_load = f'proto_{args.use_proto}_synthetic_{args.use_synthetic}_baseline_{args.model}_best_model'
    path_to_save = os.path.join(args.output_dir, model_to_load)
    print(f"Loading model from {model_to_load}")
    model_to_load = os.path.join(args.output_dir, f"{model_to_load}.pth")
    model.load_state_dict(torch.load(model_to_load))
    model.to(device)
    model.eval()
    
    # load data
    eval(model, df_val, path_to_save)
    
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