import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiVAE(nn.Module):
    def __init__(self, p_dims, dropout=0.5):
        super(MultiVAE, self).__init__()

        self.p_dims = p_dims                  # es: [600, 200, n_items]
        self.q_dims = p_dims[::-1]            # encoder reverse
        self.drop = dropout

        # ----- Encoder (q-network) -----
        self.encoder = nn.ModuleList()
        for i, (din, dout) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims) - 2:
                # ultimo layer → mean + logvar
                dout *= 2
            self.encoder.append(nn.Linear(din, dout))

        # ----- Decoder (p-network) -----
        self.decoder = nn.ModuleList()
        for din, dout in zip(self.p_dims[:-1], self.p_dims[1:]):
            self.decoder.append(nn.Linear(din, dout))

    def encode(self, x):
        h=x 
        h = F.dropout(h, p=self.drop, training=self.training)

        for layer in self.encoder[:-1]:
            h = torch.tanh(layer(h))

        h_last = self.encoder[-1](h)
        mu = h_last[:, :self.q_dims[-1]]
        logvar = h_last[:, self.q_dims[-1]:]

        return mu, logvar

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def decode(self, z):
        h = z
        for layer in self.decoder[:-1]:
            h = torch.tanh(layer(h))
        return self.decoder[-1](h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    def loss_function(self, logits, x, mu, logvar, beta=1.0):
        # multinomial log-likelihood
        log_softmax = F.log_softmax(logits, dim=1)
        neg_ll = -torch.mean(torch.sum(log_softmax * x, dim=1))

        # KL divergence
        KL = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))

        return neg_ll + beta * KL, neg_ll, KL
