from vaes_encoder import Encoder, Encoder_MLP, Net
from vaes_generators import Generator, Generator_MLP
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

    
class ConvVAE(Net):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim)
        self.decoder = Generator(latent_dim)
    
    def sample_training(self, mu, log_var):
        ##### Reparameterization trick 

        # STANDARD DEVIATION: Multiplying log_var by 0.5, then in-place exponent to yield std dev
        std_dev = (0.5 * log_var).exp()

        # Random noise 
        epsilon = torch.randn_like(mu) # noise from N~(0,1)

        # Sampling from learned zdims normal distribution 
        z = epsilon * std_dev + mu # z shape (batch_size, 16)

        return z

    def sample(self, n): # this is for inference
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim) # torch.randn outputs N~(0,1) ---> (number_samples, latent dim)
            z = z.to(device)
            samples = self.decoder(z)        
        return samples.cpu().numpy() # (number_samples, 32, 32, 3)


class MLP_VAE(Net):
    def __init__(self, latent_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.encoder = Encoder_MLP(latent_dim)
        self.decoder = Generator_MLP(latent_dim)

    def sample_training(self, mu, log_var):
        ##### Reparameterization trick

        # STANDARD DEVIATION: Multiplying log_var by 0.5, then in-place exponent to yield std dev
        std_dev = (0.5 * log_var).exp()

        # Random noise
        epsilon = torch.randn_like(mu)  # noise from N~(0,1)

        # Sampling from learned zdims normal distribution
        z = epsilon * std_dev + mu  # z shape (batch_size, 16)

        return z

    def sample(self, n):  # this is for inference
        with torch.no_grad():
            z = torch.randn(n, self.latent_dim)  # torch.randn outputs N~(0,1) ---> (number_samples, latent dim)
            z = z.to(device)
            samples = self.decoder(z)
            samples = torch.clamp(samples, 0, 1)
        return samples.cpu().numpy()