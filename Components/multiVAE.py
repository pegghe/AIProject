import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiVAE(nn.Module):
    def __init__(self, p_dims, dropout=0.5, lam=0.0, random_seed=None):
        """
        PyTorch reimplementation of the MultiVAE model from:
        Variational Autoencoders for Collaborative Filtering (Liang et al., 2018)
        
        Args:
            p_dims: list like [600, 200, n_items]
            dropout: dropout probability (same as TF keep_prob=1-dropout)
            lam: L2 regularization coefficient (0 for no regularization, same as TF default)
        """
        super(MultiVAE, self).__init__()

        if random_seed is not None:
            torch.manual_seed(random_seed)

        self.p_dims = p_dims                         # Decoder dims
        self.q_dims = p_dims[::-1]                  # Encoder dims (reversed)
        self.drop = dropout
        self.lam = lam

        # ===========================
        # ENCODER (q-network)
        # ===========================
        self.encoder = nn.ModuleList()
        for i, (din, dout) in enumerate(zip(self.q_dims[:-1], self.q_dims[1:])):
            if i == len(self.q_dims) - 2:
                # last encoder layer → mean + logvar → size = 2 * latent_dim
                dout *= 2
            self.encoder.append(nn.Linear(din, dout))

        # ===========================
        # DECODER (p-network)
        # ===========================
        self.decoder = nn.ModuleList()
        for din, dout in zip(self.p_dims[:-1], self.p_dims[1:]):
            self.decoder.append(nn.Linear(din, dout))

    # ============================================================
    # ENCODER FORWARD PASS  (q-graph in the TF code)
    # ============================================================
    def encode(self, x):
        # L2 normalization at input (exactly like TF)
        h = F.normalize(x, p=2, dim=1)

        # dropout after normalization (TF uses keep_prob_ph)
        h = F.dropout(h, p=self.drop, training=self.training)

        # hidden layers
        for layer in self.encoder[:-1]:
            h = torch.tanh(layer(h))

        h_last = self.encoder[-1](h)

        # split into mu and logvar
        latent_dim = self.q_dims[-1]
        mu = h_last[:, :latent_dim]
        logvar = h_last[:, latent_dim:]

        return mu, logvar

    # ============================================================
    # Reparameterization  (same as TF sampled_z = mu + eps * std)
    # ============================================================
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            # same as TF: when scoring, sampled_z = mu
            return mu

    # ============================================================
    # DECODER (p-graph in TF code)
    # ============================================================
    def decode(self, z):
        h = z
        for layer in self.decoder[:-1]:
            h = torch.tanh(layer(h))
        return self.decoder[-1](h)

    # ============================================================
    # Full forward pass (same as TF forward_pass)
    # ============================================================
    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        logits = self.decode(z)
        return logits, mu, logvar

    # ============================================================
    # LOSS FUNCTION — EXACT TF NEG-ELBO
    # ============================================================
    def loss_function(self, logits, x, mu, logvar, beta):
        """
        Returns:
            neg_ELBO, neg_log_likelihood, KL
        """

        # multinomial log-likelihood (same formula as TF)
        log_softmax = F.log_softmax(logits, dim=1)
        neg_ll = -torch.mean(torch.sum(log_softmax * x, dim=1))

        # KL divergence EXACTLY like TF:
        # 0.5 * sum(-logvar + exp(logvar) + mu^2 - 1)
        KL = 0.5 * torch.mean(torch.sum(
            -logvar + torch.exp(logvar) + mu * mu - 1, dim=1))

        # weight decay (OPTIONAL — paper default = 0)
        l2_reg = 0
        if self.lam > 0:
            for p in self.parameters():
                l2_reg += torch.sum(p.pow(2))
            l2_reg = self.lam * l2_reg

        neg_ELBO = neg_ll + beta * KL + l2_reg

        return neg_ELBO, neg_ll, KL
