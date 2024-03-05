import math
import time
from typing import Callable, List, Tuple, Union

import torch
import torch_geometric
from torch import BoolTensor, Tensor
from torch.nn.functional import softmax
from torch_geometric.utils import add_remaining_self_loops
from tqdm import tqdm

from gnnshap.samplers import get_sampler
from gnnshap.solvers import get_solver
from gnnshap.utils import *
from gnnshap.explanation import GNNShapExplanation

log = get_logger(__name__)


def default_predict_fn(model: torch.nn.Module,
                       node_features: Tensor,
                       edge_index: Tensor,
                       node_idx: Union[int, List[int]],
                       edge_weight: Tensor = None) -> Tensor:
    r"""Model prediction function for prediction. A custom predict function can be provided for
    different tasks.

    Args:
        model (torch.nn.Module): a PyG model.
        node_features (Tensor): node feature tensor.
        edge_index (Tensor): edge_index tensor.
        node_idx (Union[int, List[int]]): node index. Can be an integer or list (list
            for batched data).
        edge_weight (Tensor, optional): edge weights. Defaults to None.

    Returns:
        Tensor: model prediction for the node being explained.
    """

    model.eval()

    # [node_idx] will only work for non-batched. [node_idx, :] works for both
    pred = model.forward(node_features, edge_index, edge_weight=edge_weight)

    # this is for 3d predictions tensors
    # pred = pred[node_idx, :] if len(pred.size()) == 2 else pred[:, node_idx, :]

    pred = softmax(pred[node_idx, :], dim=-1)
    return pred


class GNNShapExplainer:
    """GNNShap main Explainer class.

        Args:
            model (torch.nn.Module): a pyg model.
            data (torch_geometric.data.Data): a pyg data.
            nhops (int, optional): number of hops. It will be computed if not provided.
                Defaults to None.
            device (Tuple[str, torch.device], optional): torch device. Defaults to 'cpu'.
            forward_fn (Callable, optional): A forward function. It can be customized for custom
                needs. Defaults to default_predict_fn.
            progress_hide (bool, optional): Hides tqdm progress if set to True. Defaults to False.
            verbose (int, optional): Shows some information if set to a positive number.
                Defaults to 0.
    """

    def __init__(self, model: torch.nn.Module, data: torch_geometric.data.Data,
                 nhops: int = None, device: Tuple[str, torch.device] = 'cpu',
                 forward_fn: Callable = default_predict_fn,
                 progress_hide: bool = False,
                 verbose: int = 0):

        self.model = model
        self.num_hops = nhops if nhops is not None else len(get_gnn_layers(self.model))
        self.data = data
        self.forward_fn = forward_fn  # prediction function
        self.progress_hide = progress_hide  # tqdm progress bar show or hide
        self.verbose = verbose  # to show or hide extra info
        self.device = device

        self.has_self_loops = data.has_self_loops()

        self.sampler = None  # will be set in explain.
        self.preds = None # will be set in compute_model_predictions.


    def __compute_preds_no_batching(self, node_features: Tensor,
                                    edge_index: Tensor, mask_matrix: BoolTensor,
                                    node_idx: int, target_class: int) -> Tensor:
        """Computes predictions by iterating each coalitions one by one.

        Args:
            node_features (Tensor): node features.
            edge_index (Tensor): edge index.
            mask_matrix (BoolTensor): boolean 2d mask matrix.
            node_idx (int): node index (it should be the relabeled node index in the subgraph).
            target_class (int): Target class.

        Returns:
            Tensor: Returns predictions tensor.
        """
        preds = torch.zeros((mask_matrix.size(0)),
                                 dtype=torch.double, device=mask_matrix.device,
                                 requires_grad=False)

        for i in tqdm(range(mask_matrix.shape[0]), desc="Coalition scores",
                      disable=self.progress_hide):
            mask = mask_matrix[i]
            masked_edges = edge_index[:, mask]
            # masked_edges = self.filter_fn(masked_edges, node_idx, self.num_hops)
            y_hat = self.forward_fn(model=self.model, node_features=node_features,
                                    edge_index=masked_edges, node_idx=node_idx, edge_weight=None)
            preds[i] = y_hat[target_class]
        return preds

    def __compute_preds_batched(self, node_features: Tensor, edge_index: Tensor,
                                     mask_matrix: torch.BoolTensor, node_idx: int,
                                     batch_size: int, target_class: int) -> Tensor:
        """Computes model predictions by mini-batching. Creates a large graph by stacking edge
          indices. Note that individual subgraphs are disconnected. So, for 5 nodes subgraphs,
          first subgraph node numbers are 0, 1, 2, 3, and 4. Second subgraph node numbers are 5, 6,
          7, and 8. Three key tensors are created like below:
          batch_edge_index = [edge_index1, edge_index2, ...] : (2, nplayers * batch_size)
          node_features = node_features.repeat(batch_size, 1) : (num_nodes * batch_size, F)
          batch_mask = mask_matrix[batch_start:batch_end].flatten() : (nplayers * batch_size,)


        Args:
            node_features (Tensor): node features.
            edge_index (Tensor): edge index.
            mask_matrix (torch.BoolTensor): boolean 2d mask matrix.
            node_idx (int): node index (it should be the relabeled node index in the subgraph).
            batch_size (int): batch size
            target_class (int): Target class.

        Returns:
            Tensor: Returns predictions tensor.
        """
        preds = torch.zeros((mask_matrix.size(0)),
                                 dtype=torch.double, device=mask_matrix.device,
                                 requires_grad=False)

        num_batches = math.ceil(mask_matrix.shape[0] / batch_size)
        num_nodes = node_features.size(0)

        # pyg creates a large graph by combining our masked subgraphs.
        # We need to get predictions of the same node for each subgraph.
        # [node0, node1, node2, node3 ... node0, node1, node2]
        node_indices = torch.arange(node_idx, batch_size * num_nodes, num_nodes, device=self.device)
        current_ind = 0

        edge_size = edge_index.size(1)

        # Creating batched_data using PyG minibatching approach. It has a small overhead.
        # tmp_data = Data(x=node_features, edge_index=edge_index).to(node_features.device)
        # data_list = [tmp_data for _ in range(batch_size)]
        # loader = DataLoader(data_list, batch_size=len(data_list))
        # batched_data = next(iter(loader))
        # batch_edge_index = batched_data.x
        # batch_node_features = batched_data.edge_index


        # Alternative approach to PyG minibatch since we don't need many features of PyG minibatch
        batch_node_features = node_features.repeat(batch_size, 1)
        #create large batched edge indices
        batch_edge_index = torch.zeros((2, edge_index.size(1) * batch_size), device=self.device,
                                            dtype=torch.long)
        for k, n_ind  in enumerate(range(0, batch_size * edge_size, edge_size)):
            batch_edge_index[:, n_ind:n_ind + edge_size] = edge_index + k * num_nodes


        # predictions for batches
        for i in tqdm(range(num_batches), desc="Batched coalition scores",
                        disable=self.progress_hide, leave=False):

            batch_start = batch_size * i
            batch_end = min(batch_size * (i + 1), mask_matrix.shape[0])

            tmp_batch_size = batch_end - batch_start
            if tmp_batch_size < batch_size: # for the last batch
                batch_edge_index=batch_edge_index[:, :edge_index.size(1) * tmp_batch_size]
                node_features=batch_node_features[: tmp_batch_size * num_nodes]

            tmp_node_indices = node_indices[:tmp_batch_size] # to make sure for the last batch
            y_hat = self.forward_fn(model=self.model,
                                    node_features=batch_node_features,
                                    edge_index=batch_edge_index[:,mask_matrix[
                                        batch_start:batch_end].flatten()],
                                    edge_weight=None,
                                    node_idx=tmp_node_indices)
            preds[current_ind: current_ind + tmp_batch_size] = y_hat[:, target_class]
            current_ind += tmp_batch_size
        return preds

    def compute_model_predictions(self, node_features: Tensor, edge_index: Tensor,
                                     mask_matrix: torch.BoolTensor, node_idx: int,
                                     batch_size: int, target_class: int) -> Tuple[Tensor, int]:
        """Computes model predictions and writes results to self.preds variable.

        Args:
            node_features (Tensor): node features.
            edge_index (Tensor): edge index.
            mask_matrix (torch.BoolTensor): boolean 2d mask (coalition) matrix.
            node_idx (int): node index (it should be the relabeled node index in the subgraph).
            batch_size (int): batch size. No batching if set to zero.
            target_class (int): Target class.

        Returns:
            Tuple[Tensor, int]: Returns predictions tensor and number of computed samples.

        """
        assert batch_size >= 0, "Batch size can not be a negative number"


        # empty coalition prediction, use only self loop edges
        self.fnull = self.forward_fn(self.model, node_features,
                                    edge_index[:, self.sampler.nplayers:],
                                node_idx)[target_class].item()

        # full coalition prediction, use all edges in the subgraph
        self.fx = self.forward_fn(self.model, node_features, edge_index,
                             node_idx)[target_class].item()

        # s_time = time.time()

        # there could be added self loops. Limit with nplayers
        one_hop_incoming_idx = (edge_index[1, :self.sampler.nplayers] == node_idx).nonzero()[:,0]

        # only compute indices when the target node is not isolated
        compute_indices =  0 != mask_matrix[:, one_hop_incoming_idx].count_nonzero(dim=-1)

        preds = torch.zeros((mask_matrix.size(0)),
                                 dtype=torch.double, device=mask_matrix.device,
                                 requires_grad=False).fill_(self.fnull)

        # no batch
        if batch_size == 0:
            tmp_preds = self.__compute_preds_no_batching(
                node_features, edge_index, mask_matrix[compute_indices], node_idx, target_class)
        # batch
        else:
            tmp_preds = self.__compute_preds_batched(
                node_features, edge_index, mask_matrix[compute_indices], node_idx,
                batch_size,target_class)
        preds[compute_indices] = tmp_preds

        return preds, compute_indices.count_nonzero().item()

    @torch.no_grad()
    def explain(self, node_idx: int, nsamples: int,
                    batch_size: int = 512, sampler_name: str = 'GNNShapSampler',
                    target_class: Union[int, None]=None, l1_reg: bool= False,
                    solver_name: str = "WLSSolver", **kwargs):
        r"""Computes shapley scores. It has four steps:

            | 1. finds computational graph and players
            | 2. samples coalitions
            | 3. runs model and get predictions for sampled graphs.
            | 4. solves linear regression problem to compute shapley scores.

        Args:
            node_idx (int): Node index to explain
            nsamples (int, optional): number of samples.
            batch_size (int, optional): batch size. Defaults to 512.
            sampler_name (str, optional): Sampler class name for sampling.
                Defaults to 'SHAPSampler'.
            target_class (int, optional): Computes Shapley scores for the target class.
                Predicted class is used if target_class is not provided. Defaults to None.
            l1_reg (bool, optional): use l1 reg or not. l1 reg will not be used if
                fraction_evaluated > 0.2
            solver_name (str, optional): Solver name. Defaults to 'TorchSolver'.
            kwargs: Additional sampler & solver args if needed.

        Returns:
            GNNSHAPExplanation: Returns GNNSHAPExplanation objects that contain many information.
        """

        device = self.device

        # use the predicted class if no target class is provided.
        if target_class is None:
            # target_class = self.data.y[node_idx].item() # for ground truth
            target_class = torch.argmax(self.forward_fn(self.model, self.data.x,
                                                        self.data.edge_index, node_idx)).item()
        start_time = time.time()
        # we only need k-hop neighbors for explanation
        (subset, sub_edge_index, sub_mapping,
         sub_edge_mask) = pruned_comp_graph(node_idx, self.num_hops,
                                                   self.data.edge_index,
                                                   relabel_nodes=True)

        nplayers = sub_edge_index.size(1) # number of players

        compgraph_time = time.time()
        log.info(f"Computational graph finding time(s):\t{compgraph_time -  start_time:.4f}")

        # get samples
        self.sampler = get_sampler(
            sampler_name=sampler_name, nplayers=nplayers, nsamples=nsamples,
            edge_index=sub_edge_index, nhops=self.num_hops,
            target_node=sub_mapping[0].item(), **kwargs)
        mask_matrix, kernel_weights = self.sampler.sample()

        mask_matrix = mask_matrix.to(device)
        kernel_weights = kernel_weights.to(device)

        nsamples = self.sampler.nsamples  # samplers may update nsamples

        sampling_time = time.time()

        log.info(f"Sampling time(s):\t\t{sampling_time - compgraph_time:.4f}")


        # temporarily switch add_self_loops to False if it is enabled.
        # we will add self loops manually.
        use_add_self_loops = has_add_self_loops(self.model)
        add_self_loops_swithced = False
        if use_add_self_loops:
            switch_add_self_loops(self.model)
            add_self_loops_swithced = True


        self.model.eval()

        if self.verbose == 1:
            print(f"Number of samples: {self.sampler.nsamples}, "
                  f"sampler:{self.sampler.__class__.__name__}, "
                  "batch size: {batch_size}")

        # new node_idx after relabeling in k hop subgraph.
        new_node_idx = sub_mapping[0].item()

        node_features = self.data.x[subset].to(device)
        sub_edge_index = sub_edge_index.to(device)

        # add remaining self loops if GNN layers' add_self_loops param set to True
        self_loop_sub_edge_index = add_remaining_self_loops(
            edge_index=sub_edge_index)[0] if use_add_self_loops else sub_edge_index

        if use_add_self_loops:
            torch_self_loop_mask_matrix_bool = torch.ones((mask_matrix.size(0),
                                                        self_loop_sub_edge_index.size(1)),
                                                        dtype=torch.bool).to(device)
            # Self loop indices are always True, rest is based on coalition matrix
            torch_self_loop_mask_matrix_bool[:, :mask_matrix.size(1)] = mask_matrix
        else:
            torch_self_loop_mask_matrix_bool = mask_matrix

        preds, comp_samp = self.compute_model_predictions(node_features, self_loop_sub_edge_index,
                                               torch_self_loop_mask_matrix_bool,
                                               new_node_idx, batch_size, target_class)

        # revert back if add_self_loops are disabled
        if add_self_loops_swithced:
            switch_add_self_loops(self.model)
            del torch_self_loop_mask_matrix_bool


        del self_loop_sub_edge_index
        pred_time = time.time()
        log.info(f"Model predictions time(s):{pred_time - sampling_time:.4f}")

        fraction_evaluated = nsamples / self.sampler.max_samples

        # We'll most likely get OOM error if we use WLSSolver on GPU for large number of players.
        # This can be disabled if GPU has enough memory. Our GPU has 24GB memory.
        if nplayers > 5000 and solver_name == "WLSSolver" and device != 'cpu':
            solver_name = "WLRSolver"
            log.warning(f"Switching to WLRSolver. Reason: large number of players: {nplayers}")

        solver = get_solver(solver_name=solver_name,
                            mask_matrix=mask_matrix,
                            kernel_weights=kernel_weights,
                            yhat=preds, fnull=self.fnull,
                            ffull=self.fx, device=device, fraction_evaluated=fraction_evaluated,
                            l1_reg=l1_reg)

        shap_vals = solver.solve()

        solve_time = time.time()

        log.info(f"Solve time(s):\t{solve_time -  pred_time:.4f}")



        # non-relabeled computional edge index
        sub_edge_index = self.data.edge_index[:, sub_edge_mask]

        total_time = time.time() - start_time


        explanation = GNNShapExplanation(node_idx, nplayers, float(self.fnull), shap_vals, nsamples,
                                         self.fx, target_class, sub_edge_index, subset,
                                         self.data.y[subset],
                                         time_total_comp=total_time,
                                         time_comp_graph=compgraph_time -  start_time,
                                         time_sampling=sampling_time - compgraph_time,
                                         time_predictions=pred_time - sampling_time,
                                         time_solver=solve_time -  pred_time,
                                         computed_samples=comp_samp)

        return explanation
