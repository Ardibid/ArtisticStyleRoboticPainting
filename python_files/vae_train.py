import torch
import numpy as np

from torch import nn, optim
from collections import OrderedDict
from vae_loss import loss_function
from vae_plots import show_samples_

cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def train(model, train_loader, optimizer, epoch, cnn):
    """
    Returns dictionary containing losses 
    """
    model.train()
    losses = OrderedDict()
    
    for batch_idx, (data, _) in enumerate(train_loader):        
        data = data.to(device)
        optimizer.zero_grad()

        # Pass image to the model
        if cnn != True:
            x_recon, mu, logvar = model(data)
        
        else:
            mu, logvar = model.encoder(data) # shape of each (batch_size, 16)
            # Sample
            z = model.sample_training(mu, logvar)
            # Decoding sample z
            x_recon = model.decoder(z)
        
        # Calculate scalar loss
        loss, recon, kl = loss_function(x_recon, data, mu, logvar, cnn)
        loss.backward()
        optimizer.step()
        
        # Store losses in dictionary
        if 'loss' not in losses:
            losses['loss'] = [] # Start a list
        losses['loss'].append(loss.item())
        if 'recon' not in losses:
            losses['recon'] = [] # Start a list
        losses['recon'].append(recon.item())
        if 'kl' not in losses:
            losses['kl'] = [] # Start a list
        losses['kl'].append(kl.item())    
        
        description = ''
        
        # This is just to report training information 
        for k,v in losses.items():
            avg_loss = np.mean(losses[k][-50:]) # report avg of last 50 losses
            description += '{}: {} '.format(k, avg_loss)
        
        if batch_idx % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader)))
            print(description)

    print('\n ====> After training the Epoch: {}'.format(epoch))
    print('Losses: ', description)
    
    return losses # Dict containing a list with losses per batch 

def test(model, test_loader, cnn, epoch, folder_path_reconstructions=None):
    """
    Returns dictionary containing losses 
    """
    model.eval()
    test_loss = 0
    kl_loss = 0
    recon_loss = 0
    losses = OrderedDict()

    with torch.no_grad():
        for i, (data, _) in enumerate(test_loader):
            
            batch_size = data.shape[0]
            data = data.to(device)
            
            # Pass image to the model
            if cnn != True:
                x_recon, mu, logvar = model(data) # MLP
            
            else:
                mu, logvar = model.encoder(data) # shape of each (batch_size, 16)
                # Sample
                z = model.sample_training(mu, logvar)
                # Decoding sample z
                x_recon = model.decoder(z)
            
            loss, recon, kl = loss_function(x_recon, data, mu, logvar, cnn)
            
            # Multiply by batch size to later on divide by the whole set of images in the dataset 
            test_loss += loss.item() * batch_size
            recon_loss += recon.item() * batch_size
            kl_loss += kl.item() * batch_size 
            
            # Visualize the reconstructions once per epoch
            if i == 0:
                n = min(batch_size, 10) 
                
                # x_recon = torch.sigmoid(x_recon) if no sigmoid in decoder's last layer
                comparison = torch.cat([data[:n],
                                        x_recon.view(batch_size, 1, 32, 32)[:n]])
                # Show and save samples
                show_samples_(comparison.data.cpu(), folder_path_reconstructions + '/reconstructions_per_epoch/{}'.format(epoch), nrow=10)

        # Divide by the whole set of images to have a normalized loss 
        test_loss /= len(test_loader.dataset)
        recon_loss /= len(test_loader.dataset)
        kl_loss /= len(test_loader.dataset)
        
        print('====> Test set loss: {:.4f}'.format(test_loss))
        print('====> Recon loss: {:.4f}'.format(recon_loss))
        print('====> KL loss: {:.4f}'.format(kl_loss))
        
        losses['loss'] = test_loss
        losses['recon'] = recon_loss
        losses['kl'] = kl_loss
        return losses
    
def train_epochs(model, train_loader, test_loader, train_args, save_model_path, cnn):
    """
    model: class including encoder and decoder 
    train_loader test_loader: 
    train_args: dictionary with number of epochs and learning rate and gradient clips option
    save_path: path to save model 
    quiet (opt): print logs 
    ------------------------
    Returns: train losses and test losses 
    """
    epochs, lr = train_args['epochs'], train_args['lr']
    grad_clip = train_args.get('grad_clip', None)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Prepare dictionary to store losses  
    train_losses, test_losses = OrderedDict(), OrderedDict()
    best_loss = 1e30
    
    for epoch in range(1, epochs + 1):
        train_loss = train(model, train_loader, optimizer, epoch, cnn)
        test_loss = test(model, test_loader, cnn, epoch, save_model_path)

        # Store losses 
        for k in train_loss.keys():
            if k not in train_losses:
                train_losses[k] = [] # list of lists 
                test_losses[k] = []  # list of scalaras 
            train_losses[k].extend(train_loss[k])
            test_losses[k].append(test_loss[k])

        # Save model
        val_loss = test_loss['loss']
        if val_loss < best_loss:
            best_loss = val_loss
            print("Saving model, predictions and generated output for epoch " + str(epoch) + " with Loss: " + str(
                best_loss))
            
            # if encoder and decoder separately: 
            save_path_encoder = save_model_path + '/encoder_epoch_{}.pt'.format(epoch)
            save_path_decoder = save_model_path + '/decoder_epoch_{}.pt'.format(epoch)
            save_path_vae = save_model_path + '/vae_epoch_{}.pt'.format(epoch)
            
            torch.save(model.encoder, save_path_encoder)
            torch.save(model.decoder, save_path_decoder)
            torch.save(model, save_path_vae)

    return train_losses, test_losses

