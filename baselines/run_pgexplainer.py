import argparse
import pickle
import time
import torch

from torch_geometric.explain import Explainer, PGExplainer
from tqdm.auto import tqdm

from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph
from baselines.utils import result2dict

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', default='Cora', type=str)
parser.add_argument('--repeat', default=1, type=int)

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model, data, config = get_model_data_config(args.dataset, load_pretrained=True, device=device)


model.eval()

train_nodes = data.train_mask.nonzero(as_tuple=False).cpu().numpy().flatten().tolist()
test_nodes = config['test_nodes']

# use the predictions as the target
target = torch.argmax(model(data.x, data.edge_index), dim=-1)

for r in range(args.repeat):
    explainer = Explainer(
        model=model,
        algorithm=PGExplainer(epochs=20, lr=0.005, device=device),
        explanation_type='phenomenon', # it only supports this. no model option
        edge_mask_type='object',
        model_config=dict(
            mode='multiclass_classification',
            task_level='node', #node level prediction.
            return_type='raw',),)

    train_start = time.time()
    # Train the explainer.
    for epoch in tqdm(range(20), desc="PGExplainer Model Training"):
        if len(train_nodes) > 500:
            # Randomly sample 500 nodes to train against.
            tr_nodes = torch.randperm(data.num_nodes, device=device)[:500]
        else:
            tr_nodes = train_nodes
        for index in tr_nodes:  # train on a subset of the training nodes
            loss = explainer.algorithm.train(epoch, model, data.x, data.edge_index,
                                            target=target, index=int(index))
    train_time = time.time() - train_start

    results = []
    for ind in tqdm(test_nodes, desc=f"PGExplainer explanations - run{r+1}"):
        start_time = time.time()
        explanation = explainer(data.x, data.edge_index, index=ind,
                                edge_weight=data.edge_weight, target=target)
        (subset, sub_edge_index, sub_mapping,
            sub_edge_mask) = pruned_comp_graph(ind, config['num_hops'],
                                                data.edge_index, relabel_nodes=False)
        edge_importance = explanation.edge_mask[sub_edge_mask].detach().cpu().numpy()
        results.append(result2dict(ind, edge_importance, time.time() - start_time))

    rfile = f'{config["results_path"]}/{args.dataset}_PGExplainer_run{r+1}.pkl'
    with open(rfile, 'wb') as pkl_file:
        pickle.dump([results, train_time], pkl_file)
