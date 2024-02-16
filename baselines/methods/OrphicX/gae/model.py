import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.methods.OrphicX.gae.layers import GraphConvolution 


class VGAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, output_dim, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, output_dim, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, output_dim, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class VGAE3(VGAE):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, output_dim, dropout):
        super(VGAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc1_1 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim2, output_dim, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim2, output_dim, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        hidden2 = self.gc1_1(hidden1, adj)
        return self.gc2(hidden2, adj), self.gc3(hidden2, adj)


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, torch.transpose(z, 1, 2)))
        return adj


class VGAE2MLP(VGAE): # added by sakkas. This one uses 2 GCN layers instead three.
    def __init__(self, input_feat_dim, hidden_dim1, output_dim, decoder_hidden_dim1, decoder_hidden_dim2, K, dropout):
        super(VGAE2MLP, self).__init__(input_feat_dim, hidden_dim1, output_dim, dropout)
        self.dc = InnerProductDecoderMLP(output_dim, decoder_hidden_dim1, decoder_hidden_dim2, dropout, act=lambda x: x)


class VGAE3MLP(VGAE3):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, output_dim, decoder_hidden_dim1, decoder_hidden_dim2, K, dropout):
        super(VGAE3MLP, self).__init__(input_feat_dim, hidden_dim1, hidden_dim2, output_dim, dropout)
        self.dc = InnerProductDecoderMLP(output_dim, decoder_hidden_dim1, decoder_hidden_dim2, dropout, act=lambda x: x)


class InnerProductDecoderMLP(nn.Module):
    """Decoder for using inner product for prediction."""
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, dropout, act=torch.sigmoid):
        super(InnerProductDecoderMLP, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_dim1)
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.dropout = dropout
        self.act = act
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.fc.weight)
        torch.nn.init.zeros_(self.fc.bias)
        torch.nn.init.xavier_uniform_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)

    def forward(self, z):
        z = F.relu(self.fc(z))
        z = torch.sigmoid(self.fc2(z))
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.bmm(z, torch.transpose(z, 1, 2)))
        return adj