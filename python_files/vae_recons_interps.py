import torch
import numpy as np
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

def reconstruct(model, loader, quantity, cnn, image_size=32):
    # get some images from test loader 
    x, _ = next(iter(loader))
    x = x[:quantity] # shape (quantity, 1, 28, 28)
    x = x.to(device)

    with torch.no_grad():
        n = x.shape[0]
        if cnn != True:
            x = x.view(n, -1) # Flatten if using MLP
        # encode 
        z, _ = model.encoder(x) 

        # decode
        x_recon = model.decoder(z)

        # x_recon = torch.sigmoid(x_recon)
        x_recon = torch.clamp(x_recon, 0, 1) # CLAMP

    reconstructions = torch.stack((x, x_recon), dim=1).view(-1, 1, image_size, image_size)
    reconstructions = reconstructions.cpu().numpy()
    
    return reconstructions

def interpolate(model, loader, quantity, cnn, image_size=32):
    # get some samples from loader
    x, _ = next(iter(loader))
    x = x[:quantity]
    x = x.to(device)

    with torch.no_grad():
        n = x.shape[0]
        if cnn != True:
            x = x.view(n, -1) # Flatten if using MLP
        
        # encode
        z, _ = model.encoder(x)    # (num_samples, latent dim)
        
        # divide by 2 to have both ends
        z1, z2 = z.chunk(2, dim=0) # (num_samples / 2, latent dim)

        # np linspace (0,1,10) gives you [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        interps = [model.decoder(z1 * (1 - alpha) + z2 * alpha) for alpha in np.linspace(0, 1, 10)]
        interps = torch.stack(interps, dim=1).view(-1, 1, image_size, image_size)

        interps = torch.clamp(interps, 0, 1) # CLAMP
        # interps = torch.sigmoid(interps) # SIGMOID
    
    interps = interps.cpu().numpy()
    return interps