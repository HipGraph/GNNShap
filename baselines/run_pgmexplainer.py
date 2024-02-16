import torch
from baselines.methods.pgm_explainer import PGM_Node_Explainer
import numpy as np
import time
from tqdm.auto import tqdm
from baselines.utils import result2dict
import pickle
from gnnshap.utils import pruned_comp_graph
from dataset.utils import get_model_data_config
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model, data, config = get_model_data_config(args.dataset, load_pretrained=True, device=device,
                                            log_softmax_return=True)


model.eval()

# use the predictions as the target
target = torch.argmax(model(data.x, data.edge_index), dim=-1)
test_nodes = config['test_nodes']

for r in range(args.repeat):
    results = []
    pgm_explainer = PGM_Node_Explainer(model, data.edge_index, None, data.x, 
                                       num_layers=config['num_hops'], device=device, mode=0,
                                       print_result=1)

    for ind in tqdm(test_nodes, desc=f"PGMExp explanations - run{r+1}"):
        start_time = time.time()
        explanation = pgm_explainer.explain(ind, target=target[ind], num_samples=100,
                                            top_node=None)
        subset, e_index, _, _ = pruned_comp_graph(ind, config['num_hops'],
                                                            data.edge_index)
        
        results.append(result2dict(ind, np.array(explanation[subset.cpu().numpy()]),
                                   time.time() - start_time))
    rfile = f'{config["results_path"]}/{args.dataset}_PGMExplainer_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, 0], pkl_file)
