import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain='linear'):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class Repara(nn.Module):

    def __init__(self, dim_emb, hidden_sizes, latent_size):
        super(Repara,self).__init__()

        self.linearelu = nn.Sequential(
                nn.Linear(dim_emb, hidden_sizes),
                nn.ReLU(),
                nn.Linear(hidden_sizes, latent_size),
                nn.ReLU())

        self.linearelu_mu = nn.Linear(latent_size, latent_size)
        self.linearelu_logvar = nn.Linear(latent_size, latent_size)


    def gaussian_param_projection(self, x):
        return self.linearelu_mu(x), self.linearelu_logvar(x)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        x = self.linearelu(x)
        mu, logvar = self.gaussian_param_projection(x)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar