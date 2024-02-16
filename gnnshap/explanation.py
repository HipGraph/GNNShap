import warnings
from typing import Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D
from numba.core.errors import (NumbaDeprecationWarning,
                               NumbaPendingDeprecationWarning)
from shap._explanation import Explanation as SHAPExplanation
from shap.plots._bar import bar
from shap.plots._force import force as shap_force_plt
from torch import Tensor

from gnnshap.eval_utils import fidelity
from gnnshap.utils import get_logger

warnings.simplefilter('ignore', category=NumbaDeprecationWarning)
warnings.simplefilter('ignore', category=NumbaPendingDeprecationWarning)


log = get_logger(__name__)


class GNNShapExplanation:
    """This class is used to return explanation results. Tensor values are converted and stored
        as numpy array for convention. Time parts are used for benchmarking and they are optional.
        The results can be visualized, or fidelity scores can be computed via methods.

    Args:
        node_idx (int): Explained node idx.
        nplayers (int): Number of players.
        base_value (int): Base value.
        shap_values (np.array): Shapley values.
        nsamples (int): Number of samples.
        fx (float): Model's prediction with the subgraph.
        target_class (int): Target class for the explanation.
        sub_edge_index (Tensor): computational graph's edge index.
        sub_nodes (Tensor): nodes in the computational graph.
        sub_node_labels (Tensor): node labels.
        time_total_comp (float, optional): Total SHAP computation time. Defaults to None.
        time_comp_graph (float, optional): computational graph extraction time. Defaults to None.
        time_sampling (float, optional): Sampling time. Defaults to None.
        time_predictions (float, optional): Total coalition predictions time. Defaults to None.
        time_solver (float, optional): Solver time. Defaults to None.
        computed_samples (int, optional): Number of computed samples. Some samples doesn't need
            computing when the target node is isolated. Defaults to None.
        **kwargs (dict): Other arguments.
    """

    def __init__(self, node_idx: int, nplayers: int, base_value: int,
                 shap_values: np.array, nsamples: int,  fx: float, target_class: int,
                 sub_edge_index: np.array, sub_nodes: np.array, sub_node_labels: np.array,
                 time_total_comp: float = None, time_comp_graph: float = None,
                 time_sampling: float = None, time_predictions: float = None,
                 time_solver: float = None, **kwargs: dict):
        self.node_idx = node_idx
        self.nplayers = nplayers
        self.base_value = base_value
        self.shap_values = shap_values
        self.nsamples = nsamples
        self.fx = fx
        self.target_class = target_class
        self.sub_edge_index = sub_edge_index.detach().cpu().numpy()
        self.sub_nodes = sub_nodes.detach().cpu().numpy()
        self.sub_node_labels = sub_node_labels.detach().cpu().numpy()

        self.time_total_comp = time_total_comp
        self.time_comp_graph = time_comp_graph
        self.time_sampling = time_sampling
        self.time_predictions = time_predictions
        self.time_solver = time_solver
        self.computed_samples = kwargs.get('computed_samples', None)
        self.kwargs = kwargs

    def __get_edge_names(self) -> list:
        """Gets edge names as list of strings in `source -> target` format.

        Returns:
            list: edge names as list of strings.
        """
        labels = []
        for src, trgt in self.sub_edge_index.T:
            lbl = f'{src}\u2192{trgt}'
            labels.append(lbl)

        return labels

    def __get_edge_names_series(self) -> pd.Series:
        """Gets edge names and shapley values as pandas series. This can be used if shapley values
        wants to be seen together with edge names in the visualizations.

        Returns:
            pd.Series: Pandas series
        """
        tmp_dict = {}
        for i, (src, trgt) in enumerate(self.sub_edge_index.T):
            lbl = f'{src}\u2192{trgt}'
            tmp_dict[lbl] = self.shap_values[i]
        return pd.Series(tmp_dict)

    def plot_force(self, contrib_threshold: float = 0.005,
                   show_values: bool = False) -> None:
        """Plots force plot using SHAP package's force plot.

        Args:
            contrib_threshold (float, optional): A threshold value to discard some edges.
                Defaults to 0.005.
            show_values (bool, optional): Shows Shapley values along with edge names.
                Defaults to False.
        """
        features = self.__get_edge_names_series() if show_values else None
        feature_names = self.__get_edge_names()
        shap_force_plt(self.base_value, self.shap_values, features, feature_names, matplotlib=True,
                       contribution_threshold=contrib_threshold)

    def plot_bar(self, max_display: int = 10, show=True) -> None:
        """Plots force plot using SHAP package's force plot.

        Args:
            max_display (int, optional): A threshold to show top number of edges/players.
                Defaults to 10.
            show (bool, optional): Shows the plot. Defaults to True. Set to False if you want more
                costumization.
        """
        feature_names = self.__get_edge_names()

        shap_explanation = SHAPExplanation(self.shap_values, np.array([self.base_value]),
                                           feature_names=feature_names)
        bar(shap_explanation, max_display=max_display, show=show)

    def plot_graph(self, topk: int = 25, save_path: str = None, pos=None, show_scores: bool = False,
                   return_pos: bool = False, show: bool = True) -> None:
        """Plots computational computational graph with topk edges. Since it uses topk edges, some 
            nodes will not be visible if connecting edges to target node not in the topk.

        Args:
            topk (int, optional): maximum number of topk edges in the plot. Defaults to 25.
            save_path (str, optional): Save path. The plot will be saved if provided.
                Defaults to None.
            pos (dict, optional): Position dictionary. If not provided, it will be computed.
                Defaults to None.
            show_scores (bool, optional): Shows Shapley values along with edges.
                Defaults to False.
            return_pos (bool, optional): Returns position dictionary. Defaults to False.
            show (bool, optional): Shows the plot. Defaults to True. Set to False if you want more
                costumization.

        Returns:
            dict: position dictionary.

        """

        # maximum one at the first index
        top_edges = np.argsort(-np.abs(self.shap_values))
        topk_edges = top_edges[:topk]
        max_shap_val = np.abs(self.shap_values).max()
        color_list = ['orange', 'blue', 'red', 'green',
                      '#D3B98C', "lightblue", '#D3A38C', 'red', 'yellow',
                      'pink', 'grey', 'purple', 'gold']

        # predefined colors are not enough for dataset. Remaining class backgrounds are white.
        if self.sub_node_labels.max() > len(color_list):
            color_list += ['white' for i in range(
                self.sub_node_labels.max() - len(color_list))]
            log.warning("Predefined colors are not enough for each class. Classes from %d to %d"
                        "are colored as white", len(color_list), self.sub_node_labels.max())

        fig, ax = plt.subplots()

        G = nx.DiGraph()
        for i in topk_edges:
            G.add_edge(int(self.sub_edge_index[0, i]), int(self.sub_edge_index[1, i]),
                       label=self.shap_values[i])

        edge_labels = nx.get_edge_attributes(G, 'label')
        formatted_edge_labels = {
            (elem[0], elem[1]): f"{edge_labels[elem]:.4f}" for elem in edge_labels}

        edge_colors = ['blue' if edge_labels[elem] < 0 else 'red' for elem in edge_labels]

        nodes = list(G.nodes)
        node_sizes = [600 if n == self.node_idx else 500 for n in nodes]

        node_colors = [color_list[self.sub_node_labels
                                  [np.where(self.sub_nodes == n)[0][0]]] for n in nodes]
        node_border_colors = ['black' if n == self.node_idx
                              else node_colors[i] for i, n in enumerate(nodes)]


        #pos = nx.kamada_kawai_layout(G, scale=5)
        if pos is None:
            pos = nx.spring_layout(G, scale=5)
        edge_transparency = np.array(
            [np.abs(v/max_shap_val) for v in self.shap_values[topk_edges]])
        tmp_min, tmp_max = edge_transparency.min(), edge_transparency.max()
        # scale transparencies between 0.2 and 1.0
        edge_transparency = (
            1-0.2) * ((edge_transparency - tmp_min)/(tmp_max - tmp_min)) + 0.2

        nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                               edgecolors=node_border_colors)
        nx.draw_networkx_labels(
            G, pos, labels={n: f'{n}'for n in nodes}, font_size=7)
        nx.draw_networkx_edges(G, pos, width=2, connectionstyle="arc3,rad=0.1", arrows=True,
                               arrowsize=14, arrowstyle='-|>', node_size=500,
                               edge_color=edge_colors, alpha=edge_transparency)

        legend_labels = ["lower", "higher"]
        handles = [Line2D([0], [0], color='b', lw=2, label='Line'),
                   Line2D([0], [0], color='r', lw=2, label='Line2')]
        ax.legend(handles, legend_labels, loc='best', fontsize='small',
                  fancybox=True, framealpha=0.7)

        if show_scores:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=formatted_edge_labels,
                                         rotate=True, label_pos=0.75, font_size=6, ax=ax,
                                         bbox=dict(alpha=0))

        if save_path is not None:
            plt.savefig(save_path)

        if show:
            plt.show()

        if return_pos:
            return pos

    def fidelity_prob(self, model: torch.nn.Module, data: Tensor, sparsity: float=0.1,
                      score_type: str = 'neg', topk: int = 0,
                      apply_abs: bool = True) -> Union[float, float]:
        """Computes fidelity probabilty score. Score type 'neg' computes fidelity- score, and 'pos'
            computes fidelity+ score. If topk is other than 0, then fidelity score is computed by
            droping k edges.

        Args:
            model (torch.nn.Module): A PyG model.
            data (Tensor): A PyG data.
            sparsity (float): Target sparsity value. It should be a value in range (0, 1). Defaults
            to 0.1.
            score_type (str, optional): 'neg' for fidelity- or 'pos' for fidelity+.
                Defaults to 'neg'.
            topk (int, optional): It is used to drop k edges if not 0. Sparsity value will not be
                used if topk is used (a dummy value can be provided for sparsity). Defaults to 0.
            apply_abs (bool, optional): Applies absolute to scores and fidelity. Defaults to True.

        Returns:
            Union[float, float]: fidelity score and sparsity value.
        """
        assert 0 <= sparsity <= 1, "Sparsity should be between zero and one."
        assert topk >= 0, "topk cannot be a negative number"

        res = fidelity(self.result2dict(), data, model, sparsity, score_type, topk,
                     self.target_class, apply_abs)

        return res[2], res[3]

    def result2dict(self) -> dict:
        """Converts an explanation result to a dictionary.

        Args:
            node_id (int): node id
            scores (np.array): importance scores

        Returns:
            dict: result as dictionary
        """
        return {'node_id': self.node_idx, 'scores': self.shap_values,
                'num_players': self.nplayers, 'num_samples': self.nsamples,
                'base_val': self.base_value,
                'time': self.time_total_comp}
