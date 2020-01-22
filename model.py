import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, latent_dim=118):
        super(Generator, self).__init__()
        d1 = 512
        self.latent_dim = latent_dim
        self.dense = nn.Linear(latent_dim + 10, d1)
        self.bn0 = nn.BatchNorm1d(d1)
        self.conv1 = nn.Conv2d(32, 128, 3, 1, 1)  # 128, 4, 4
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 256, 3, 2, 1, 1)  # 256, 8, 8
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 512, 3, 2, 1, 1)  # 512, 16, 16
        self.bn3 = nn.BatchNorm2d(512)
        self.deconv4 = nn.ConvTranspose2d(512, 1024, 3, 2, 1, 1)  # 1024, 32, 32
        self.bn4 = nn.BatchNorm2d(1024)
        self.conv5 = nn.Conv2d(1024, 256, 3, 1, 1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 64, 3, 1, 1)
        self.bn6 = nn.BatchNorm2d(64)
        self.conv7 = nn.Conv2d(64, 16, 3, 1, 1)
        self.bn7 = nn.BatchNorm2d(16)
        self.conv8 = nn.Conv2d(16, 1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, label, BATCH=16):
        z = torch.randn(BATCH, self.latent_dim)
        if torch.cuda.is_available():
            z = z.cuda()
        x = torch.cat([z, label], 1)
        x = self.bn0(self.relu(self.dense(x)))
        x = x.reshape(-1, 32, 4, 4)
        x = self.bn1(self.relu(self.conv1(x)))
        x = self.bn2(self.relu(self.deconv2(x)))
        x = self.bn3(self.relu(self.deconv3(x)))
        x = self.bn4(self.relu(self.deconv4(x)))
        x = self.bn5(self.relu(self.conv5(x)))
        x = self.bn6(self.relu(self.conv6(x)))
        x = self.bn7(self.relu(self.conv7(x)))
        x = self.sigmoid(self.conv8(x))
        out = x[:, :, 2:-2, 2:-2]
        return out


class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.linear0 = nn.Linear(794, 794)
        self.linear1 = nn.Linear(794, 794)
        self.linear2 = nn.Linear(794, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(74, 32) # add the skip connection here
        self.linear6 = nn.Linear(32, 16)
        self.linear7 = nn.Linear(16, 1)
        self.ln0 = nn.LayerNorm(794)
        self.ln1 = nn.LayerNorm(794)
        self.ln2 = nn.LayerNorm(256)
        self.ln3 = nn.LayerNorm(128)
        self.ln4 = nn.LayerNorm(64)
        self.ln5 = nn.LayerNorm(32)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x, z):
        x = x.reshape(-1, 784)
        y = torch.cat([z, x], 1)
        w = self.relu(self.ln0(self.linear0(y)))
        w = self.relu(self.ln1(self.linear1(w)))
        w = self.relu(self.ln2(self.linear2(w)))
        w = self.relu(self.ln3(self.linear3(w)))
        w = self.relu(self.ln4(self.linear4(w)))
        w = torch.cat([w, z], 1)
        w = self.relu(self.ln5(self.linear5(w)))
        w = self.relu(self.linear6(w))
        out = self.linear7(w)
        return out
