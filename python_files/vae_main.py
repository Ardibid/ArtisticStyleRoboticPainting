import os
import time
import torch
import torch.utils.data as data
import sys
import json
import torchvision
from torchvision import datasets, transforms

import numpy as np

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")
cuda

# Import models
from vae_models import ConvVAE, MLP_VAE

# Import plots
from vae_plots import show_samples_, plot_vae_training_plot

# Import train
from vae_train import train_epochs

# Import reconstructions and interpolations
from vae_recons_interps import reconstruct, interpolate

# import dataset
from vae_dataset import *


def train_procedure(epochs, batch_size, train_data, test_data, model, save_model_path, cnn, image_size=32):

    # 1) Get data
    train_loader = data.DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_data, batch_size=batch_size, drop_last=True)

    # 2) Train
    train_losses, test_losses = train_epochs(model, train_loader, test_loader, dict(epochs=epochs, lr=1e-3), save_model_path,
                                            cnn)

    train_losses = np.stack((train_losses['loss'], train_losses['recon'], train_losses['kl']),
                            axis=1)  # (total_iterations, 3)
    test_losses = np.stack((test_losses['loss'], test_losses['recon'], test_losses['kl']), axis=1)  # (total_epochs , 3)

    # 3) Sample from posterior distribution
    samples = model.sample(4)
    samples = samples.reshape(4, 1, image_size, image_size)

    # 4) Reconstructions
    reconstructions = reconstruct(model, test_loader, 1, cnn, image_size)

    # 5) Interpolations
    interps = interpolate(model, test_loader, 4, cnn, image_size)

    return train_losses, test_losses, samples, reconstructions, interps


def save_results(train_data, test_data, model, save_path, fn, epochs, batch_size, save_figures_path, cnn):
    train_losses, test_losses, samples, reconstructions, interpolations = fn(epochs, batch_size, train_data, test_data, model, save_path, cnn)
    samples, reconstructions, interpolations = samples.astype('float32'), reconstructions.astype(
        'float32'), interpolations.astype('float32')

    print('\n----- FINAL RESULTS ------')
    print(f'Final -ELBO: {test_losses[-1, 0]:.4f}, Recon Loss: {test_losses[-1, 1]:.4f}, '
          f'KL Loss: {test_losses[-1, 2]:.4f}')

    if cnn:
        name_dataset = 'BrushStrokes_CNN'
    else:
        name_dataset = 'BrushStrokes_MLP'

    plot_vae_training_plot(train_losses, test_losses, f'Dataset {name_dataset} Train Plot',
                           f'{save_figures_path}/plots/dset{name_dataset}_train_plot.png')

    show_samples_(samples, title=f'Dataset {name_dataset} Samples',
                  fname=f'{save_figures_path}/plots/dset{name_dataset}_samples.png')

    show_samples_(reconstructions, title=f'Dataset {name_dataset} Reconstructions',
                  fname=f'{save_figures_path}/plots/dset{name_dataset}_reconstructions.png')

    show_samples_(interpolations, title=f'Dataset {name_dataset} Interpolations',
                  fname=f'{save_figures_path}/plots/dset{name_dataset}_interpolations.png')

    # Write file
    results_folder =  save_figures_path + '/results/'
    if not exists(dirname(results_folder)):
        os.makedirs(dirname(results_folder))
    result_train_path = results_folder + 'results_train.txt'
    result_test_path = results_folder + 'results_test.txt'
    final_result_path = results_folder + 'final_results.txt'


    with open(final_result_path, 'w') as file:
        json.dump(final_result, file)
    with open(result_train_path, 'w') as file:
        json.dump(train_losses.tolist(), file)
    with open(result_test_path, 'w') as file2:
        json.dump(test_losses.tolist(), file2)    
    
    
    
if __name__ == '__main__':

    if len(sys.argv) != 5:
        print("Usage: python3 vae_main.py EPOCHS, BATCH_SIZE, Z_DIMS, CNN = 1 or 0 (if 1 model=CNN else MLP)")
        print("Example: python3 vae_main.py 5 32 32 1")
        print("Use python or python3 based on your environment parameters.")
        sys.exit(1)

    EPOCHS = int(sys.argv[1])
    BATCH_S = int(sys.argv[2])
    Z_DIMS = int(sys.argv[3])
    CNN = True if sys.argv[4] == '1' else False

    # Dataset
    brush_strokes = np.load('../data_brushstrokes/brush_strokes_16x32_vae.npy')

    training_data, test_data = partition_dataset(brush_strokes, 0.8, True)
    train_dataset = MyDataset(training_data)
    test_dataset = MyDataset(test_data)

    # Create runid
    des_model = 'cnn' if CNN else 'mlp'
    run_id = str(int(time.time()))
    if not os.path.exists('./experiments'):
        os.mkdir('./experiments')
    os.mkdir('./experiments/{}_{}'.format(run_id, des_model))
    SAVE_PATH = './experiments/{}_{}'.format(run_id, des_model)
    print("Saving models, and predictions to ./experiments/{}_{}".format(run_id, des_model))

    if CNN:
        model = ConvVAE(Z_DIMS)
    else:
        model = MLP_VAE(Z_DIMS)

    save_results(train_dataset, test_dataset, model, SAVE_PATH, train_procedure, EPOCHS, BATCH_S, SAVE_PATH, CNN)


