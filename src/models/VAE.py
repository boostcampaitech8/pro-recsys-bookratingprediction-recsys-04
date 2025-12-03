import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.hidden = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = self.hidden(x)
        x = self.relu(x)
        mu = self.mu(x)
        logvar = self.logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden = nn.Linear(latent_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        x = self.hidden(z)
        x = self.relu(x)
        x = self.fc(x)
        return x

class VAE(nn.Module):
    def __init__(self, args, data):
        super().__init__()
        self.encoder = Encoder(args.input_dim, args.hidden_dim, args.latent_dim)
        self.decoder = Decoder(args.latent_dim, args.hidden_dim, args.input_dim)    # output_dim 굳이 추가하기 싫어서 input_dim 재사용

    def reparemeterize(self, mu, logvar):
        eps = torch.randn_like(logvar)
        return mu + torch.exp(logvar / 2) * eps

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparemeterize(mu, logvar)
        x_hat = self.decoder(z)

        return x_hat, mu, log_var
