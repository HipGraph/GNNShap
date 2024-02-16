import argparse
import pickle
import time
import numpy as np

import torch
from tqdm.auto import tqdm

from baselines.methods.OrphicX.orphicx import OrphicXExplainer
from baselines.utils import result2dict
from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph
from torch_geometric.utils import dense_to_sparse
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, k_hop_subgraph

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)
parser.add_argument('--epoch', default=50, type=int)

# reduced the number of samples for alpha and beta due to gpu memory limitations
parser.add_argument('--Nalpha', type=int, default=15, help='Number of samples of alpha.')
parser.add_argument('--Nbeta', type=int, default=50, help='Number of samples of beta.')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model, data, config = get_model_data_config(args.dataset, load_pretrained=True, device=device)

target = torch.argmax(model(data.x, data.edge_index), dim=-1)
num_classes = data.y.max().item() + 1
num_features = data.x.shape[1]

class WrapperModel(torch.nn.Module):
    """We need to wrap the model in a class to match input and output dimensions of OrphicXExplainer.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x, adj):
        if x.size(0) == 1:
            edge_index, weight = dense_to_sparse(adj)
            return self.model(x[0], edge_index, weight).unsqueeze(0).unsqueeze(0)
        else:
            out = torch.zeros((x.shape[0], x.shape[1], num_classes)).to(device)
            # for i in range(x.shape[1]):
            #     edge_index, weight = dense_to_sparse(adj[i])
            #     out[i] = self.model(x[i], edge_index, weight)
            # return out.unsqueeze(0)
        
            # batched inference. this speeds up the process by a lot.
            # without batched inference, the process takes a lot of time.
            batch_size = 128
            num_nodes = x.shape[1]
            for i in range(0, x.shape[0], batch_size):
                data_list = []
                for j in range(batch_size):
                    if i+j >= x.shape[0]:
                        break
                    edge_index, weight = dense_to_sparse(adj[i+j])
                    data_list.append(Data(x=x[i+j], edge_index=edge_index, edge_weight=weight))
                
                loader = DataLoader(data_list, batch_size=len(data_list))
                batched_data = next(iter(loader))
                out[i:i+batch_size] = self.model(batched_data.x, batched_data.edge_index,
                                                 batched_data.edge_weight
                                                 ).reshape(-1, num_nodes, num_classes)

            return out.unsqueeze(0)

model = WrapperModel(model).to(device)

result_file = f'{config["results_path"]}/{args.dataset}_OrphicX.txt'


model.eval()

test_nodes = config['test_nodes']


for r in range(args.repeat):
    results = []
    explainer = OrphicXExplainer(data, model, config['num_hops'], device=device)
    start = time.time()
    explainer.train(args.epoch)
    train_time = time.time() - start
    for ind in tqdm(test_nodes, desc=f"OrphicX Individual explanations - run{r+1}"):
        try:
            start_time = time.time()
            explanation = explainer.explain(ind)


            (subset, sub_edge_index, sub_mapping,
            sub_edge_mask) = pruned_comp_graph(ind, config['num_hops'], data.edge_index,
                                                        relabel_nodes=True)

            exp_results = np.zeros(sub_edge_index.size(1))
            for i in range(sub_edge_index.size(1)):
                exp_results[i] = explanation[sub_edge_index[0, i], sub_edge_index[1, i]].item()
            
            results.append(result2dict(ind, exp_results, time.time() - start_time))
        
        except Exception as e:
            print(f"Node {ind} has failed. General error: {e}")
        

    rfile = f'{config["results_path"]}/{args.dataset}_OrphicX_{args.epoch}_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, train_time], pkl_file)



