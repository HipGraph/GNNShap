import argparse
import pickle
import time

import torch
from captum.attr import Saliency
from tqdm.auto import tqdm

from baselines.utils import result2dict
from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=5, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model, data, config = get_model_data_config(args.dataset, load_pretrained=True,
                                            device=device)


target = torch.argmax(model(data.x, data.edge_index), dim=-1)


#model.eval()
test_nodes = config['test_nodes']

def model_forward_node(x, model, edge_index, node_idx):
    out = model(x, edge_index).softmax(dim=-1)
    return out[[node_idx]]

for r in range(args.repeat):
    results = []
    for i, ind in tqdm(enumerate(test_nodes), desc=f"SA explanations - run{r+1}"):
        start_time = time.time()
        explainer = Saliency(model_forward_node)

        (subset, sub_edge_index, sub_mapping,
        sub_edge_mask) = pruned_comp_graph(ind, config['num_hops'], data.edge_index,
                                           relabel_nodes=True)
        x_mask = data.x[subset].clone().requires_grad_(True).to(device)
        saliency_mask = explainer.attribute(
            x_mask, target=target[i].item(),
            additional_forward_args=(model, sub_edge_index, sub_mapping.item()), abs=False)
        
        node_importance = saliency_mask.cpu().numpy().sum(axis=1)
        results.append(result2dict(ind, node_importance, time.time() - start_time))
    
    
    rfile = f'{config["results_path"]}/{args.dataset}_SA_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, 0], pkl_file)
