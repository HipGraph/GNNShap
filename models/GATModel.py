import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.conv import GATConv
from tqdm.auto import tqdm


class GATModel(torch.nn.Module):
    def __init__(self, hidden_channels,
                 num_features, num_classes, num_layers=2,
                 add_self_loops=True, 
                 dropout = 0.0,
                 normalize = True,
                 log_softmax_return=False,
                 heads=1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        assert num_layers >= 2, "Number of layers should be two or larger"
        self.heads = heads
        self.convs = nn.ModuleList(
            [GATConv(num_features, hidden_channels, normalize=normalize,
                          add_self_loops=add_self_loops, dropout=0.6, heads=heads)] +
            [GATConv(hidden_channels * heads, hidden_channels, normalize=normalize,
                           add_self_loops=add_self_loops, dropout=0.6, heads=heads
                           ) for i in range(num_layers - 2)] +
            [GATConv(hidden_channels * heads, num_classes, normalize=normalize,
                      add_self_loops=add_self_loops, dropout=0.6, heads=1)])
        self.softmax_fn = nn.LogSoftmax(dim=-1) if log_softmax_return else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        x = self.softmax_fn(x) # applied based on parameter. Default: not applied
        return x
