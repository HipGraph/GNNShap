import math

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data

from gnnshap.utils import pruned_comp_graph


def node2edge_score(edge_index: torch.Tensor, node_scores: np.array):
    """Converts node scores to edge scores: an edge score is equal to average of connected nodes. 
    Needed for some baselines that only provide node scores.

    Args:
        edge_index (torch.Tensor): PyG edge index.
        node_scores (np.array[float]): node scores

    Returns:
        np.array: edge scores
    """

    edge_scores = np.zeros(edge_index.size(1))
    np_node_scores = np.array(node_scores)
    edge_scores += np_node_scores[edge_index[0].cpu().numpy()]
    edge_scores += np_node_scores[edge_index[1].cpu().numpy()]
    return edge_scores/2



def fidelity(node_data: dict, data: Data, model: torch.nn.Module, sparsity: float = 0.3,
               fid_type: str = 'neg', topk: int = 0, target_class: int = None,
               apply_abs: bool=True) -> tuple:
    """Computes fidelity+ and fidelity- score of a node. It supports both topk and sparsity. 
    If sparsity set to 0.3, it drops 30% of the edges. Based on the neg or pos, it drops 
    unimportant or important edges. It applies topk based keep if topk is set to a positive 
    integer other than zero.

    Note that it computes fidelity scores for the predicted class if target class is not provided.

    Args:
        node_data (dict): a node's explanation data with node_id, num_players, scores keys.
        data (Data): pyG Data.
        model (torch.nn.Module): a PyTorch model.
        sparsity (float, optional): target sparsity value. Defaults to 0.3.
        fid_type (str, optional): Fidelity type: neg or pos. Defaults to 'neg'.
        topk (int, optional): Topk edges to keep. Defaults to 0.
        target_class (int, optional): Target class to compute fidelity score. Defaults to None.
        apply_abs (bool, optional): applies absolute to scores. Some methods can find negative and 
            positive contributing nodes/edges. Fidelity-wise, we only care the change amount. We can 
            use this to get rid of negative contributing edges to improve accuracy. Defaults to 
            True.

    Returns:
        tuple: node_id, nplayers, fidelity score, current sparsity, correct_class, init_pred_class, 
            and sparse_pred_class.
    """
    assert topk >= 0, "topk cannot be a negative number"
    assert 0 <= sparsity <= 1, "Sparsity should be between zero and one."

    node_id = int(node_data['node_id'])
    correct_class = data.y[node_id].item()

    model.eval()

    

    # find khop computational graph
    (subset, sub_edge_index, new_node_id,
     _) = pruned_comp_graph(node_id, model.num_layers, data.edge_index, relabel_nodes=True)
    # new node id due to relabeling
    new_node_id = int(new_node_id[0].cpu().numpy())
    num_initial_edges = sub_edge_index.size(1)  # number of players


    subset = subset.cpu().numpy()

    # initial prediction
    init_pred = F.softmax(model(data.x[subset], sub_edge_index), dim=1)[new_node_id]
    init_pred_class = init_pred.argmax(dim=-1).item()
    if target_class is None:
        target_class = init_pred_class
    init_prob = init_pred[target_class].item()


    if node_data['num_players'] == num_initial_edges:
        edge_scores = np.array(node_data['scores'])

    # convert node scores to edge scores if node score is provided
    elif node_data['num_players'] == subset.shape[0]:
        edge_scores = node2edge_score(sub_edge_index, node_data['scores'])

    else:
        raise ValueError("Number of players should be equal to either"
                        " number of edges or number of nodes!")


    edge_scores = np.abs(edge_scores) if apply_abs else edge_scores


    # less important edge at first index
    edge_importance_sorted = edge_scores.argsort()

    if topk == 0:  # sparsity based
        if fid_type == 'pos':  # reverse the list: most important edge at first index
            edge_importance_sorted = edge_scores.argsort()[::-1].copy()
            # copy required for bug fixing. pytorch doesn't support negative index

        # how many edges to drop
        drop_len = num_initial_edges - math.ceil(num_initial_edges * (1 - sparsity))
        keep_edges = edge_importance_sorted[drop_len:]

    else:  # topk based
        if fid_type == 'neg':
            keep_edges = edge_importance_sorted[topk:]  # drop least important topk edges
        else:  # fid+
            keep_edges = edge_importance_sorted[:-topk] # keep edges except topk

        drop_len = num_initial_edges - len(keep_edges)


    keep_edges.sort()

    sparse_pred = F.softmax(model(data.x[subset], sub_edge_index[:, keep_edges]),
                            dim=-1)[new_node_id]
    
    sparse_pred_class = sparse_pred.argmax(dim=-1).item()
    sparse_prob = sparse_pred[target_class].item()

    prob_score = sparse_prob - init_prob
    prob_score = np.abs(prob_score) if apply_abs else prob_score


    current_sparsity = drop_len / num_initial_edges
    return (node_id, num_initial_edges, prob_score, current_sparsity,
            correct_class, init_pred_class, sparse_pred_class)