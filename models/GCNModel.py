import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.utils import add_remaining_self_loops, degree
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter_add
from tqdm.auto import tqdm


class GCNModel(torch.nn.Module):
    r"""GCN model
        Args:
        hidden_channels (int): hidden layer dimensions
        num_features (int): number of input features
        num_classes (int): number of output classes
        num_layers (int, optional): number of layers. Defaults to 2.
        add_self_loops (bool, optional): whether to add self loops. Defaults to True.
        dropout (float, optional): dropout rate. Defaults to 0.0.
        normalize (bool, optional): whether to normalize. Defaults to True.
        log_softmax_return (bool, optional): whether to return raw output or log softmax. 
            Defaults to False.
    """
    
    def __init__(self, hidden_channels, 
                 num_features, num_classes, num_layers=2,
                 add_self_loops=True, 
                 dropout = 0.0,
                 normalize = True,
                 log_softmax_return=False):
        
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        assert num_layers >= 2, "Number of layers should be two or larger"
        self.convs = nn.ModuleList(
            [GCNConv(num_features, hidden_channels, normalize=normalize, 
                          add_self_loops=add_self_loops)] +
            [GCNConv(hidden_channels, hidden_channels, normalize=normalize,
                           add_self_loops=add_self_loops) for i in range(num_layers - 2)] +
            [GCNConv(hidden_channels, num_classes, normalize=normalize, 
                      add_self_loops=add_self_loops)])
        self.softmax_fn = nn.LogSoftmax(dim=-1) if log_softmax_return else nn.Identity()

    def forward(self, x, edge_index, edge_weight=None):
        for i in range(self.num_layers - 1):
            x = F.relu(self.convs[i](x, edge_index, edge_weight))
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index, edge_weight)
        x = self.softmax_fn(x) # applied based on parameter. Default: not applied
        return x
    
    # faster inference for reddit dataset
    # assumes there are two gcn layers. It won't work if there are more than two layers
    @torch.no_grad()
    def inference(self, x_all, subgraph_loader, device):
        pbar = tqdm(total=len(subgraph_loader.dataset) * 2)
        pbar.set_description('Evaluating')

        # Compute representations of nodes layer by layer, using *all*
        # available edges. This leads to faster computation in contrast to
        # immediately computing the final representations of each batch:
        for i, conv in enumerate(self.convs):
            xs = []
            for batch in subgraph_loader:
                x = x_all[batch.n_id.to(x_all.device)].to(device)
                x = conv(x, batch.edge_index.to(device))
                if i == 0: # first layer
                    x = x.relu_()
                xs.append(x[:batch.batch_size].cpu())
                pbar.update(batch.batch_size)
            x_all = torch.cat(xs, dim=0)
        pbar.close()
        return x_all