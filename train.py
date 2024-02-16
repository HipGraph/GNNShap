# Train models and save the model. The model will be saved in the pretrained folder.
# Can't train large models with this script. Use train_large.py instead.
# Large models are trained with NeighborLoader, which is not supported in this script.

# The trained model is used for benchmarking explanation methods.

import argparse
import os
import sys

import torch

from dataset.configs import get_config
from dataset.utils import get_model_data_config

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='Cora', type=str)
args = parser.parse_args()


config = get_config(args.dataset)
pretrained_file = f"{config['root_path']}/pretrained/{args.dataset}_pretrained.pt"

if os.path.exists(pretrained_file):
    user_input = input('A pretrained file exist. Do you want to retrain? (y/n):')
    if user_input.lower() != 'y':
        print("Skipping training!")
        sys.exit(0)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

model, data, config = get_model_data_config(args.dataset, load_pretrained=False, device=device)


criterion = torch.nn.CrossEntropyLoss()  # Define loss criterion.
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'],
                             weight_decay=config['weight_decay'])  # Define optimizer.

if 'grad_clip' in config:
    grad_clip = config['grad_clip']
else:
    grad_clip = False

def train():
    model.train()
    optimizer.zero_grad()  # Clear gradients.
    out = model(data.x, data.edge_index)  # Perform a single forward pass.
    loss = criterion(out[data.train_mask],data.y[data.train_mask])
    if grad_clip:
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
    loss.backward()  # Derive gradients.
    optimizer.step()  # Update parameters based on gradients.
    return loss


def acc_test():
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    train_correct = pred[data.train_mask] == data.y[data.train_mask]
    train_acc = int(train_correct.sum()) / int(data.train_mask.sum())

    val_correct = pred[data.val_mask] == data.y[data.val_mask]
    val_acc = int(val_correct.sum()) / int(data.val_mask.sum())


    return train_acc, val_acc, test_acc


if __name__ == '__main__':
    best_val_acc = 0
    best_test_acc = 0
    for epoch in range(0, config['epoch']):
        loss = train()
        train_acc, val_acc, test_acc = acc_test()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc
            torch.save(model.state_dict(), pretrained_file)
        
        print(f'Epoch: {epoch+1:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, '
              f'Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f"Best Val Accuracy: {best_val_acc:.4f} ,Best Test Accuracy: {best_test_acc:.4f}")