from torch import nn
import torch
cuda = torch.cuda.is_available()
device = torch.device("cuda" if cuda else "cpu")

class Reshape(nn.Module):
    def __init__(self, shape, **kwargs):
        self.shape = shape
        super(Reshape, self).__init__(**kwargs)

    def forward(self, input):
        return input.view(self.shape)


class Net(nn.Module):
    """ A base class for both generator and the discriminator.
    Provides a common weight initialization scheme.

    """
    def weights_init(self):
        for m in self.modules():
            classname = m.__class__.__name__

            if "Conv" in classname:
                m.weight.data.normal_(0.0, 0.02)

            elif "BatchNorm" in classname:
                m.weight.data.normal_(1.0, 0.02)
                m.bias.data.fill_(0)

    def forward(self, x):
        return x

class MLP_VAE(nn.Module):
    def __init__(self, ZDIMS):
        super().__init__()
        self.z_dims = ZDIMS
        # ENCODER
        # 28x28 pixels = 748 input pixels, 400 outputs
        self.fc1 = nn.Linear(1024, 400)
        
        # rectified linear unit layer from 400 to 400
        # max (0, x)
        self.relu = nn.ReLU()
        self.fc21 = nn.Linear(400, ZDIMS) # mu layer
        self.fc22 = nn.Linear(400, ZDIMS) # logvariance layer
        # this last layer bottlenecks through ZDIMS connections
        
        # DECODER
        # from bottleneck to hidden 400
        self.fc3 = nn.Linear(ZDIMS, 400)
        self.fc4 = nn.Linear(400, 1024)

        
    def encoder(self, x):
        """
        Input vector x --> fully connected 1 --> RELU --> fully connected 21, fully connected 22
        
        Parameters
        ----------
        x: [batch size, 784], batch size number of digits of 28x28 pixels each
        
        Returns 
        -------
        (mu, logvar): ZDIMS mean units one for each latent dimension, ZDIMS variance units one for each 
        latent dimension
        """

        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h1 = self.relu(self.fc1(x)) 
        mu = self.fc21(h1)
        logvar = self.fc22(h1)
        
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std
    
    def decoder(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3)
        #return self.sigmoid(self.fc4(h3)) # because we are using bce with logits loss (already uses sigmoid)
    
    def forward(self, x):
        mu, logvar = self.encoder(x.view(-1, 1024))
        z = self.reparameterize(mu, logvar)
        reconstruction_x = self.decoder(z)
        return reconstruction_x, mu, logvar
    
    def sample(self, n):
        with torch.no_grad():
            z = torch.randn(n, self.z_dims) # torch.randn outputs N~(0,1) ---> (number_samples, latent dim)
            z = z.to(device)
            samples = self.decoder(z)
            samples = torch.clamp(samples, 0, 1) # (number_samples, 3, 32, 32)
        
        return samples.cpu().numpy()


class Encoder(Net):
    """
    Return mu and log variance
    """

    def __init__(self, z_dim):
        super(Encoder, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3,
                      stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3,
                      stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = nn.Linear(in_features=4096, out_features=512)
        self.r1 = nn.LeakyReLU(0.2, inplace=False)
        self.fc2 = nn.Linear(in_features=512, out_features=256)
        self.r2 = nn.LeakyReLU(0.2, inplace=False)

        self.mu_projection = nn.Linear(in_features=256, out_features=z_dim)
        self.logsigmasq_projection = nn.Linear(
            in_features=256, out_features=z_dim)

        self.weights_init()

    def forward(self, x):
        x = self.net(x)
        x = x.view(x.shape[0], -1)

        x = self.r1(self.fc1(x))
        x = self.r2(self.fc2(x))

        mu = self.mu_projection(x)
        logsigmasq = self.logsigmasq_projection(x)
        return mu, logsigmasq


class Generator(Net):

    def __init__(self, z_dim):
        super(Generator, self).__init__()

        self.net1 = nn.Sequential(
            nn.Linear(in_features=z_dim, out_features=256),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=256),

            nn.Linear(in_features=256, out_features=512),
            nn.LeakyReLU(negative_slope=0.2),
            nn.BatchNorm1d(num_features=512),

            nn.Linear(in_features=512, out_features=8 * 8 * 64),
            nn.LeakyReLU(negative_slope=0.2),
        )
        self.net2 = nn.Sequential(
            Reshape(shape=(-1, 64, 8, 8)),  # 8 pixels
            nn.BatchNorm2d(num_features=64),
            nn.ConvTranspose2d(in_channels=64, out_channels=32,
                               kernel_size=4, stride=2, padding=1),  # 16 pixels
            nn.LeakyReLU(negative_slope=0.2),

            nn.BatchNorm2d(num_features=32),
            nn.ConvTranspose2d(in_channels=32, out_channels=16,
                               kernel_size=4, stride=2, padding=1),  # 32 pixels
            nn.LeakyReLU(negative_slope=0.2),

            nn.BatchNorm2d(num_features=16),
            nn.Conv2d(in_channels=16, out_channels=1,
                      kernel_size=1, stride=1, padding=0),
        )
        self.weights_init()

    def forward(self, x):
        x = self.net1(x)
        x = self.net2(x)
        return x

    
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