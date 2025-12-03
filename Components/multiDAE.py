import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiDAE(nn.Module):
    def __init__(self, p_dims, lam=0.01, dropout=0.5):
        super(MultiDAE, self).__init__()
        
        self.p_dims = p_dims
        self.q_dims = p_dims[::-1]  # reverse
        self.dims = self.q_dims + p_dims[1:]
        self.lam = lam
        self.drop = dropout

        # ----- Encoder -----
        self.encoder = nn.ModuleList()
        for din, dout in zip(self.q_dims[:-1], self.q_dims[1:]):
            layer = nn.Linear(din, dout)
            nn.init.xavier_normal_(layer.weight)   # same as TF xavier initializer
            nn.init.trunc_normal_(layer.bias, std=0.001)
            self.encoder.append(layer)

        # ----- Decoder -----
        self.decoder = nn.ModuleList()
        for din, dout in zip(self.p_dims[:-1], self.p_dims[1:]):
            layer = nn.Linear(din, dout)
            nn.init.xavier_normal_(layer.weight)
            nn.init.trunc_normal_(layer.bias, std=0.001)
            self.decoder.append(layer)

    def forward(self, x):
        # ----- L2 Normalization (TF: tf.nn.l2_normalize) -----
        h = x / (torch.norm(x, p=2, dim=1, keepdim=True) + 1e-8)

        # ----- Dropout (TF: tf.nn.dropout) -----
        h = F.dropout(h, p=self.drop, training=self.training)

        # ----- Encoder -----
        for layer in self.encoder:
            h = torch.tanh(layer(h))

        # ----- Decoder -----
        for layer in self.decoder[:-1]:
            h = torch.tanh(layer(h))

        logits = self.decoder[-1](h)
        return logits

    def loss_function(self, logits, x):
        # ---------- Negative Log-Likelihood (multinomial) ----------
        log_softmax = F.log_softmax(logits, dim=1)
        neg_ll = -torch.mean(torch.sum(log_softmax * x, dim=1))

        # ---------- L2 Weight Regularization SAME AS PAPER ----------
        reg = 0
        for layer in list(self.encoder) + list(self.decoder):
            reg += torch.norm(layer.weight)

        loss = neg_ll + self.lam * reg
        return loss, neg_ll, reg
