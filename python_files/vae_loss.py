import torch

def loss_function(recon_x, x, mu, logvar, cnn, pixelwise=False):
    """
    Calculates reconstruction loss and KL loss. 
    If pixelwise = True, reduction = mean, and KL is normalized by the  total number of pixels (pixels_image * batch). 
    If pixelwise = False, reduction = sum, and reconstruction loss and KL loss are normalized by batch size.

    """
    batch_size = x.shape[0]
    
    ################################# Reconstruction loss #####################################
    if pixelwise:
        BCE = torch.nn.BCEWithLogitsLoss(reduction='mean') # reduction over pixels. Sums each pixel's error and divides by total number of pixels. 
        if cnn == False:
            BCE = BCE(input=recon_x, target=x.view(batch_size, 1024)) # flatten when MLP
        else:
            BCE = BCE(input=recon_x, target=x) # for CNN
    else:
        BCE = torch.nn.BCEWithLogitsLoss(reduction='sum') # reduction over image. Sums each pixel's error.
        if cnn == False:
            BCE = BCE(input=recon_x, target=x.view(batch_size, 1024))  # flatten when MLP
        else:
            BCE = BCE(input=recon_x, target=x) # for CNN

        BCE = BCE / batch_size # average over batch size


    def calc_kl(mu,logsigmasq, axis=-1):
        return 0.5 * torch.sum(torch.exp(logsigmasq) + mu.pow(2) - 1. - logsigmasq, axis=axis)

    KLD = calc_kl(mu, logvar)
    KLD = torch.mean(KLD)
    
    # BCE tries to make our reconstruction as accurate as possible
    # KLD tries to push the distributions as close as possible to unit Gaussian
    loss = BCE + KLD
    return loss, BCE, KLD
