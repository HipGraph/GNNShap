import argparse
import pickle
import time

import torch
from tqdm.auto import tqdm

from baselines.methods.subgraphx import SubgraphX
from baselines.utils import result2dict
from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model, data, config = get_model_data_config(args.dataset, load_pretrained=True, device=device,
                                            log_softmax_return=True)


result_file = f'{config["results_path"]}/{args.dataset}_SubgraphX.txt'


model.eval()


target = torch.argmax(model(data.x, data.edge_index), dim=-1)

test_nodes = config['test_nodes']

max_nodes = config['subgraphx_args']['max_nodes']
print(f"max nodes: {max_nodes}")

#TODO: It's is working but number of nodes in the explanation is a parameter. Can't control edge sparsity. 
# It also requires a new run for each node based sparsity.

for r in range(args.repeat):
    results = []
    explainer = SubgraphX(model, num_classes=config['num_classes'],
                num_hops=config['num_hops'], explain_graph=False, device=device, high2low=True,
                reward_method='nc_mc_l_shapley', rollout=20, min_atoms=4, expand_atoms=14,
                sample_num=50, local_radius=4, subgraph_building_method='zero_filling')
    
    

    for ind in tqdm(test_nodes, desc=f"SubgraphX Individual explanations - run{r+1}"):
        start_time = time.time()
        explanation = explainer.explain(data.x, data.edge_index, edge_weight=None,
                                        label=target[ind].item(), node_idx=ind, max_nodes=max_nodes)
        (subset, sub_edge_index, sub_mapping,
            sub_edge_mask) = pruned_comp_graph(ind, config['num_hops'], data.edge_index,
                                                    relabel_nodes=False)
        edge_importance = explanation[sub_edge_mask.detach().cpu().numpy()]
        results.append(result2dict(ind, edge_importance, time.time() - start_time))
    rfile = f'{config["results_path"]}/{args.dataset}_SubgraphX_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, 0], pkl_file)
