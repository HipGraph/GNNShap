# Description: This file is used to train GNN models on large datasets. The trained model is saved 
# in the pretrained folder. The model is trained using NeighborLoader, which is not supported
# in train.py.

# The trained model is used for benchmarking explanation methods.


import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm

from dataset.utils import get_model_data_config
from gnnshap.utils import pruned_comp_graph

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Reddit', type=str)
args = parser.parse_args()
dataset_name = args.dataset

def train(epoch):
    model.train()

    pbar = tqdm(total=int(len(train_loader.dataset)))
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = total_examples = 0
    for batch in train_loader:
        optimizer.zero_grad()
        y = batch.y[:batch.batch_size]
        y_hat = model(batch.x, batch.edge_index.to(device))[:batch.batch_size]
        loss = F.cross_entropy(y_hat, y)
        loss.backward()
        optimizer.step()

        total_loss += float(loss) * batch.batch_size
        total_correct += int((y_hat.argmax(dim=-1) == y).sum())
        total_examples += batch.batch_size
        pbar.update(batch.batch_size)
    pbar.close()

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def test():
    model.eval()
    y_hat = model.inference(data.x, subgraph_loader, device=device).argmax(dim=-1)
    y = data.y.to(y_hat.device)

    accs = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:
        accs.append(int((y_hat[mask] == y[mask]).sum()) / int(mask.sum()))
    return accs


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # don't load data to GPU, as it will be loaded to GPU during sampling.
    model, data, config = get_model_data_config(dataset_name, load_pretrained=False, device='cpu',
                                                full_data=True)

    model = model.to(device)


    pretrained_file = f"{config['root_path']}/pretrained/{dataset_name}_pretrained.pt"

    if os.path.exists(pretrained_file):
        user_input = input('A pretrained file exist. Do you want to retrain? (y/n):')
        if user_input.lower() != 'y':
            print("Skipping training!")
            sys.exit(0)


    # Already send node features/labels to GPU for faster access during sampling:
    data = data.to(device, 'x', 'y')
    neig_args = config['nei_sampler_args']
    kwargs = {'batch_size': neig_args['batch_size'], 'num_workers': 6, 'persistent_workers': True}
    train_loader = NeighborLoader(data, input_nodes=data.train_mask,
                                num_neighbors=neig_args['sizes'], shuffle=True, **kwargs)

    subgraph_loader = NeighborLoader(copy.copy(data), input_nodes=None,
                                    num_neighbors=[-1], shuffle=False, **kwargs)

    # No need to maintain these features during evaluation:
    del subgraph_loader.data.x, subgraph_loader.data.y
    # Add global node index information.
    subgraph_loader.data.num_nodes = data.num_nodes
    subgraph_loader.data.n_id = torch.arange(data.num_nodes)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 11):
        loss, acc = train(epoch)
        print(f'Epoch {epoch:02d}, Loss: {loss:.4f}, Approx. Train: {acc:.4f}')
        train_acc, val_acc, test_acc = test()
        print(f'Epoch: {epoch:02d}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, '
            f'Test: {test_acc:.4f}')
        if test_acc > best_test:
            best_test = test_acc
            torch.save(model.state_dict(), pretrained_file)
    torch.save(model.state_dict(), pretrained_file)
    print(f"Model saved to {pretrained_file}.")



    # Sample explain data and save. This is used for benchmarking explanation methods.
    # This makes sure that the explain data is the same for all methods.

    num_test_nodes = 100
    explain_loader = NeighborLoader(data, input_nodes=data.test_mask.nonzero()[:num_test_nodes,0],
                                    num_neighbors=[200, 50], batch_size=num_test_nodes,
                                    num_workers=8, persistent_workers=True)

    max_size = 0
    max_ind = 0
    avg_size = 0
    batch = next(iter(explain_loader))
    for i in range(batch.batch_size):
        m = pruned_comp_graph(i, 2, batch.edge_index)[1].size(1)
        if m > max_size:
            max_size = m
            max_ind = i
        avg_size += m

    del batch.x, batch.y # reduce saved file size in disk. Can be reloaded from the original data.
    torch.save(batch, f"{config['root_path']}/pretrained/{dataset_name}_explain_data.pt")
    print(f"Explain data saved to {config['root_path']}/pretrained.")
    print("Maximum size: ", max_size, "max index: ", max_ind, "avg size: ",
          avg_size / num_test_nodes)