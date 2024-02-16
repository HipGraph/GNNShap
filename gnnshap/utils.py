
import logging
import sys
from typing import List, Optional, Tuple, Union

import colorlog
import numpy as np
import torch
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils.num_nodes import maybe_num_nodes


def get_logger(name: str) -> logging.Logger:
    """Returns a logger

    Args:
        name (str): Logger name

    Returns:
        logging.Logger: logger
    """
    logger = colorlog.getLogger(name)
    handler = colorlog.StreamHandler(stream=sys.stdout)

    formatter = colorlog.ColoredFormatter(
        #"%(name)s: %(asctime)s : %(levelname)s : %(filename)s:%(lineno)s : %(message)s"
        "%(log_color)s%(filename)s:%(lineno)s:%(levelname)s: %(message)s",
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'red,bg_white',
            },
    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.setLevel(logging.WARNING)

    return logger




def get_coalition_counts(mask_matrix: Union[np.array, Tensor]) -> np.array:
    """Finds counts of each coalition size for a given mask matrix.

    Args:
        mask_matrix (Union[np.array, Tensor]): mask matrix obtained from a sampler

    Returns:
        np.array: coalition counts
    """
    if torch.is_tensor(mask_matrix):
        mask_matrix = mask_matrix.cpu().numpy()
    coal_sizes = mask_matrix.sum(1).astype(int)
    unique, counts = np.unique(coal_sizes, return_counts=True)
    return counts


def get_coalition_size_weights(mask_matrix: Union[np.array, Tensor],
                               weights: Union[np.array, Tensor]) -> np.array:
    """Finds sum of total weights for each coalition size.

    Args:
        mask_matrix (Union[np.array, Tensor]): mask matrix obtained from a sampler
        weights (Union[np.array, Tensor]): weights vector obtained from a sampler

    Returns:
        np.array: coalition size weights
    """
    if torch.is_tensor(mask_matrix):
        mask_matrix = mask_matrix.cpu().numpy()

    if torch.is_tensor(weights):
        weights = weights.cpu().numpy()

    counts = mask_matrix.sum(1)
    nplayers = mask_matrix.shape[1]

    weight_sums = np.zeros(nplayers -1)
    for i in range(1, nplayers):
        weight_sums[i-1] = weights[counts == i].sum()
    return weight_sums


def get_gnn_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Finds and returns GNN layers.

    Args:
        model (torch.nn.Module): pyg model.

    Returns:
        List[torch.nn.Module]: GNN layers as a list
    """
    gnn_layers = []
    for module in model.modules():
        if isinstance(module, MessagePassing):
            gnn_layers.append(module)
    return gnn_layers

def switch_add_self_loops(model: torch.nn.Module):
    """Switches each layers add_self_loops value to True or False.

    Args:
        model (torch.nn.Module): pyg model.
    """
    layers = get_gnn_layers(model)
    for layer in layers:
        layer.add_self_loops = not layer.add_self_loops

def switch_normalize(model: torch.nn.Module):
    """Switches each layers normalize value to True or False.

    Args:
        model (torch.nn.Module): pyg model.
    """
    layers = get_gnn_layers(model)
    for layer in layers:
        layer.normalize = not layer.normalize

def has_normalization(model: torch.nn.Module) -> bool:
    """Checks if gnn layers have normalization. It controls whether all layers
    have same configuration.

    Args:
        model (torch.nn.Module): pyg model.

    Raises:
        AssertionError: Raises assertion error if different layers have different configurations.
        AssertionError: Raises assertion error if there is no gnn layers.

    Returns:
        bool: boolean value whether gnn layers have normalization
    """
    layers = get_gnn_layers(model)
    if len(layers) > 0:
        try: # some GNN types has no normalize attribute
            normalize = layers[0].normalize
        except:
            return False
        if len(layers) > 1:
            for layer in layers[1:]:
                if layer.normalize != normalize:
                    raise AssertionError(("Layers have different normalization settings."
                                         " This is not supported!"))
        return normalize
    raise AssertionError("No GNN layers found!")


def has_add_self_loops(model: torch.nn.Module) -> bool:
    """Checks if model adds self loops. It controls whether all layers have same configuration.

    Args:
        model (torch.nn.Module): pyg model.

    Raises:
        AssertionError: Raises assertion error if different layers have different configurations.
        AssertionError: Raises assertion error if there is no gnn layers.

    Returns:
        bool: boolean value whether model adds self loops or not.
    """

    layers = get_gnn_layers(model)
    if len(layers) > 0:
        try:
            self_loop = layers[0].add_self_loops
        except:
            return False

        if len(layers) > 1:
            for layer in layers[1:]:
                if layer.add_self_loops != self_loop:
                    raise AssertionError(("Layers have different add_self_loops settings."
                                         " This is not supported!"))
        return self_loop
    raise AssertionError("No GNN layers found!")


@torch.no_grad()
def pruned_comp_graph(node_idx: Union[int, List[int], Tensor],
    num_hops: int,
    edge_index: Tensor,
    relabel_nodes: bool = False,
    num_nodes: Optional[int] = None,
    flow: str = 'source_to_target',
    directed: bool = False) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """Finds the pruned computational graph for a given node index. Similar to k_hop_subgraph, but
    k_hop_subgraph returns all edges between k-hop nodes. We are only interested in edges that
    carries message to target node in k_hops."""

    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    assert flow in ['source_to_target', 'target_to_source']
    if flow == 'target_to_source':
        row, col = edge_index
    else:
        col, row = edge_index

    my_edge_mask = row.new_empty(row.size(0), dtype=torch.bool) # added by sakkas
    my_edge_mask.fill_(False) # added by sakkas

    node_mask = row.new_empty(num_nodes, dtype=torch.bool)
    edge_mask = row.new_empty(row.size(0), dtype=torch.bool)

    if isinstance(node_idx, (int, list, tuple)):
        node_idx = torch.tensor([node_idx], device=row.device).flatten()
    else:
        node_idx = node_idx.to(row.device)

    subsets = [node_idx]

    for _ in range(num_hops):
        node_mask.fill_(False)
        node_mask[subsets[-1]] = True
        torch.index_select(node_mask, 0, row, out=edge_mask)# input, dimension, index
        my_edge_mask[edge_mask] = True
        subsets.append(col[edge_mask])

    subset, inv = torch.cat(subsets).unique(return_inverse=True)
    inv = inv[:node_idx.numel()]

    edge_index = edge_index[:, my_edge_mask]

    if relabel_nodes:
        node_idx = row.new_full((num_nodes, ), -1)
        node_idx[subset] = torch.arange(subset.size(0), device=row.device)
        edge_index = node_idx[edge_index]

    return subset, edge_index, inv, my_edge_mask
