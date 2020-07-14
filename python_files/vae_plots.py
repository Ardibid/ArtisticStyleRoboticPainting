import os
from os.path import join, dirname, exists
import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import torch.nn.functional as F
from torchvision.utils import make_grid


def imshow(img):
    img = img.numpy()
    plt.imshow(img.transpose((1,2,0)).squeeze(2), cmap='gray_r')
    plt.show()  

def visualize_one_batch(loader, logs=False):
    # Visualizes one full batch of images
    for i, (x, y) in enumerate(loader):
        if i > 0: break
        if logs:
            print('shape of batch of images: ', x.shape)
            print('Labels: ', y)

        for img, label in zip(x, y):
            imshow(img)
            if logs:
                img = img.numpy()
                print("Label: {}".format(label))
                print(img.transpose((1,2,0)).squeeze(2).shape)
                

def savefig(folder_name, show_figure=True):
    """
    Creates a folder to save the figures if it doesn't exist. Shows figures if opt=true
    """
    if not exists(dirname(folder_name)):
        os.makedirs(dirname(folder_name))
    plt.tight_layout()
    plt.savefig(folder_name)
    if show_figure:
        plt.show()
        
        
def show_samples_(samples, fname=None, nrow=10, title='Samples'):
    """
    Make a grid to show the samples. Rows with number of images = nrow
    """
    samples = torch.FloatTensor(samples)
    grid_img = make_grid(samples, nrow=nrow)
    print('grid img shape: ', grid_img.shape)
    plt.figure()
    plt.title(title)
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.axis('off')
    if fname is not None:
        savefig(fname, True)
    else:
        plt.show()
        
def plot_vae_training_plot(train_losses, test_losses, title, fname):
    elbo_train, recon_train, kl_train = train_losses[:, 0], train_losses[:, 1], train_losses[:, 2]
    elbo_test, recon_test, kl_test = test_losses[:, 0], test_losses[:, 1], test_losses[:, 2]
    plt.figure()
    n_epochs = len(test_losses) - 1
    x_train = np.linspace(0, n_epochs, len(train_losses))
    x_test = np.arange(n_epochs + 1)

    plt.plot(x_train, elbo_train, label='-elbo_train')
    plt.plot(x_train, recon_train, label='recon_loss_train')
    plt.plot(x_train, kl_train, label='kl_loss_train')
    plt.plot(x_test, elbo_test, label='-elbo_test')
    plt.plot(x_test, recon_test, label='recon_loss_test')
    plt.plot(x_test, kl_test, label='kl_loss_test')

    plt.legend()
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    savefig(fname)
