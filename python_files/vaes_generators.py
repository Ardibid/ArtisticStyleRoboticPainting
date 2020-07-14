from torch import nn


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

class Generator_MLP(Net):

    def __init__(self, z_dim):
        super(Generator_MLP, self).__init__()

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(z_dim, 400)
        self.fc2 = nn.Linear(400, 1024)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)