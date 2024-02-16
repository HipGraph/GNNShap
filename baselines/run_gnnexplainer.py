import argparse
import pickle
import time

import torch
from torch_geometric.explain import Explainer, GNNExplainer
from tqdm.auto import tqdm

from baselines.utils import result2dict
from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph
from torch_geometric.utils import k_hop_subgraph

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model, data, config = get_model_data_config(args.dataset, load_pretrained=True,
                                            device=device)


model.eval()

# target = data.y
target = torch.argmax(model(data.x, data.edge_index), dim=-1)

test_nodes = config['test_nodes']


for r in range(args.repeat):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type="phenomenon",
        node_mask_type= None,
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node', #node level prediction.
            return_type='raw',
        ),
    )
    results = []
    
    for ind in tqdm(test_nodes, desc=f"GNNExplainer explanations - run{r+1}"):
        try:
            start_time = time.time()

            # explain just using the k-hop subgraph: original paper uses this,
            # but pyg implementation does not.
            (subset, sub_edge_index, sub_mapping,
                sub_edge_mask) = k_hop_subgraph(ind, config['num_hops'],
                                                        data.edge_index, relabel_nodes=True)
            target2 = target[subset]
            explanation = explainer(data.x[subset], sub_edge_index, index=sub_mapping,
                                    target=target2)
            

            # save in our format: pruned edges
            (_, _, mapping2, mask2) = pruned_comp_graph(sub_mapping, config['num_hops'],
                                                        sub_edge_index, relabel_nodes=False)
            edge_importance = explanation.edge_mask[mask2].detach().cpu().numpy()
            results.append(result2dict(ind, edge_importance, time.time() - start_time))
        except Exception as e:
            print(f"Node {ind} failed!")
    rfile = f'{config["results_path"]}/{args.dataset}_GNNExplainer_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, 0], pkl_file)
