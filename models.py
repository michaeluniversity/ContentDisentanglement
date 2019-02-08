import torch.nn as nn
from spectral import SpectralNorm

class E1(nn.Module):
    def __init__(self, sep, size):
        super(E1, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 32, 4, 2, 1)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(256, (512 - self.sep), 4, 2, 1)),
            nn.InstanceNorm2d(512 - self.sep),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d((512 - self.sep), (512 - 2 * self.sep), 4, 2, 1),
            nn.InstanceNorm2d(512 - 2 * self.sep),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, (512 - 2 * self.sep) * self.size * self.size)
        return net


class E2(nn.Module):
    def __init__(self, sep, size):
        super(E2, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 32, 4, 2, 1)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(128, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(128, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.sep, 4, 2, 1),
            nn.InstanceNorm2d(self.sep),
            nn.LeakyReLU(0.2),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class E3(nn.Module):
    def __init__(self, sep, size):
        super(E3, self).__init__()
        self.sep = sep
        self.size = size

        self.full = nn.Sequential(
            SpectralNorm(nn.Conv2d(3, 32, 4, 2, 1)),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(32, 64, 4, 2, 1)),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(128, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            SpectralNorm(nn.Conv2d(128, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, self.sep, 4, 2, 1),
            nn.InstanceNorm2d(self.sep),
            nn.LeakyReLU(0.2),
        )

    def forward(self, net):
        net = self.full(net)
        net = net.view(-1, self.sep * self.size * self.size)
        return net


class Decoder(nn.Module):
    def __init__(self, size):
        super(Decoder, self).__init__()
        self.size = size

        self.main = nn.Sequential(
            SpectralNorm(nn.ConvTranspose2d(512, 512, 4, 2, 1)),
            nn.InstanceNorm2d(512),
            nn.ReLU(inplace=True),
            SpectralNorm(nn.ConvTranspose2d(512, 256, 4, 2, 1)),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
            SpectralNorm(nn.ConvTranspose2d(256, 128, 4, 2, 1)),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            SpectralNorm(nn.ConvTranspose2d(128, 64, 4, 2, 1)),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            SpectralNorm(nn.ConvTranspose2d(64, 32, 4, 2, 1)),
            nn.InstanceNorm2d(32),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, net):
        net = net.view(-1, 512, self.size, self.size)
        net = self.main(net)
        return net


class Disc(nn.Module):
    def __init__(self, sep, size):
        super(Disc, self).__init__()
        self.sep = sep
        self.size = size

        self.classify = nn.Sequential(
            nn.Linear((512 - 2 * self.sep) * self.size * self.size, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, net):
        net = net.view(-1, (512 - 2 * self.sep) * self.size * self.size)
        net = self.classify(net)
        net = net.view(-1)
        return net


class PatchDiscriminator(nn.Module):
    def __init__(self, img_sz):
        super(PatchDiscriminator, self).__init__()

        self.img_sz = img_sz
        self.img_fm = 3
        self.init_fm = 32
        self.max_fm = 512
        self.n_patch_dis_layers = 3

        layers = []
        layers.append(nn.Conv2d(self.img_fm, self.init_fm, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, True))

        n_in = self.init_fm
        n_out = min(2 * n_in, self.max_fm)

        for n in range(self.n_patch_dis_layers):
            stride = 1 if n == self.n_patch_dis_layers - 1 else 2
            layers.append(nn.Conv2d(n_in, n_out, kernel_size=4, stride=stride, padding=1))
            layers.append(nn.BatchNorm2d(n_out))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if n < self.n_patch_dis_layers - 1:
                n_in = n_out
                n_out = min(2 * n_out, self.max_fm)

        layers.append(nn.Conv2d(n_out, 1, kernel_size=4, stride=1, padding=1))
        layers.append(nn.Sigmoid())

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        assert x.dim() == 4
        return self.layers(x).view(x.size(0), -1).mean(1).view(x.size(0))
