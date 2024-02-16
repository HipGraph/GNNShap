import os

import numpy as np
import torch
from torch import Tensor
from torch_geometric.datasets import (Coauthor, FacebookPagePage, Planetoid,
                                      Reddit)
from torch_geometric.transforms import NormalizeFeatures

from dataset.configs import get_config
from models.GATModel import GATModel
from models.GCNModel import GCNModel


def get_model(model_name, config, num_features, num_classes, log_softmax_return=False):
    """Gets the model for a given model name.

    Args:
        model_name (str): model name like 'GCNModel', 'GATModel'
        config (dict): configuration dictionary
        num_features (int): number of input features
        num_classes (int): number of output classes
        log_softmax_return (bool, optional): whether to return raw output or log softmax.
            Defaults to False.

    Returns:
        model: PyTorch model
    """

    hidden_channels = config['hidden_dim']
    num_layers = config['num_layers']
    dropout = config['dropout']
    normalize = config.get('normalize', None)
    add_self_loops = config.get('add_self_loops', None)


    if model_name == 'GCNModel':
        model = GCNModel(num_layers=num_layers,
                         hidden_channels=hidden_channels,
                         num_features=num_features,
                         num_classes=num_classes,
                         dropout=dropout,
                         normalize=normalize,
                         add_self_loops=add_self_loops,
                         log_softmax_return=log_softmax_return)
    elif model_name == 'GATModel':
        model = GATModel(hidden_channels=hidden_channels, num_features=num_features,
                                num_classes=num_classes,
                                num_layers=num_layers,
                                add_self_loops=add_self_loops, dropout=dropout, normalize=normalize,
                                log_softmax_return=log_softmax_return,
                                heads=config.get('heads', 1))
    else:
        raise ValueError(f"Model name {model_name} is not supported")
    return model

def get_model_data_config(dataset_name, device='cpu', load_pretrained=True,
                          log_softmax_return=False, full_data=False):
    """Gets model, data and configuration for a given dataset name. Check the dataset.configs.py
        for the configuration details.
    Args:
        dataset_name (str): name of the dataset
        device (str, optional): 'cpu' or 'cuda'. Defaults to 'cpu'.
        load_pretrained (bool, optional): whether to load pretrained model. Defaults to True.
        log_softmax_return (bool, optional): whether to return raw output or log softmax.
            Defaults to False.
        full_data (bool, optional): full data or explanation data. If True for large datasets, only
            loads the subset of the data used in explanation that created by neighbor sampling.
            It doesn't make any difference for others. Defaults to False.

    Returns:
        model, data, config
    """

    config = get_config(dataset_name)
    root_path = config['root_path']

    if 'Cora' in dataset_name:
        dataset = Planetoid(root=f'{root_path}/data/Planetoid', name='Cora',
                            transform=NormalizeFeatures())
    elif 'CiteSeer' in dataset_name:
        dataset = Planetoid(root=f'{root_path}/data/Planetoid', name='CiteSeer',
                            transform=NormalizeFeatures())
    elif 'PubMed' in dataset_name:
        dataset = Planetoid(root=f'{root_path}/data/Planetoid', name='PubMed',
                            transform=NormalizeFeatures())
    elif 'Coauthor-CS' in dataset_name:
        dataset = Coauthor(root=f'{root_path}/data/Coauthor', name='CS',
                         transform=NormalizeFeatures())
    elif 'Coauthor-Physics' in dataset_name:
        dataset = Coauthor(root=f'{root_path}/data/Coauthor', name='Physics',
                         transform=NormalizeFeatures())
    elif dataset_name == 'Facebook':
        dataset = FacebookPagePage(root=f'{root_path}/data/FacebookPagePage',
                                   transform=NormalizeFeatures())
    elif dataset_name == 'Reddit':
        dataset = Reddit(f'{root_path}/data/Reddit')
    elif dataset_name == 'ogbn-products':
        from ogb.nodeproppred import PygNodePropPredDataset
        dataset = PygNodePropPredDataset(name='ogbn-products',
                                         root=f'{root_path}/data/ogbn-products')
    else:
        raise ValueError(f"Dataset name {dataset_name} is not supported")

    model_name = config['model']

    model = get_model(model_name, config, dataset.num_features,
                       dataset.num_classes, log_softmax_return=log_softmax_return).to(device)
    
    if load_pretrained:
        try:
            state_dict = torch.load(f"{root_path}/pretrained/{dataset_name}_pretrained.pt")
            model.load_state_dict(state_dict)
            model.eval()
        except FileNotFoundError:
            print(f"No pretrained model found for {dataset_name}. Don't forget to train the model.")

    data = dataset[0]
    config['num_classes'] = dataset.num_classes

    if dataset_name == 'ogbn-products':
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True
        data.y = data.y.squeeze(1)

    elif not hasattr(data, 'train_mask'):
        split = get_split(f'{root_path}/pretrained/', dataset_name, data)
        data.train_mask, data.val_mask, data.test_mask = split[0], split[1], split[2]

    if dataset_name in ['Reddit', 'ogbn-products'] and not full_data:
        data = torch.load(f"{root_path}/pretrained/{dataset_name}_explain_data.pt")
        # these are not saved in the file. Load them from the original data
        data.x = dataset[0].x[data.n_id]
        data.y = dataset[0].y[data.n_id]
        del dataset # free up some memory
        if dataset_name == 'ogbn-products':
            data.y = data.y.squeeze(1)
        data = data.to(device)
    else:
        data = data.to(device)
        
    config['test_nodes'] = data.test_mask.nonzero().cpu().numpy()[:,0].tolist()[:100]

    return model, data, config


def generate_balanced_split(data, num_train_per_class, num_val_per_class):
    """Generates a balanced train, validation, test split for a given dataset. This is used for
    datasets that do not have built-in splits.

    Args:
        data (torch.Tensor): PyTorch Geometric Data object
        num_train_per_class (int): number of training samples per class
        num_val_per_class (_type_): number of validation samples per class

    Returns:
        list: a list contains splits.
    """

    labels = data.y.cpu()
    num_classes = labels.max().item() + 1

    train_mask = np.zeros(len(labels), dtype=np.bool)
    val_mask = np.zeros(len(labels), dtype=np.bool)
    test_mask = np.zeros(len(labels), dtype=np.bool)

    for c in range(num_classes):
        idx = np.where(labels == c)[0]
        tr_idx = idx[np.random.choice(len(idx), size=num_train_per_class, replace=False)]
        train_mask[tr_idx] = True

        val_idx = np.array([i for i in idx if i not in tr_idx])
        val_idx = val_idx[np.random.choice(len(val_idx), size=num_val_per_class, replace=False)]
        val_mask[val_idx] = True
    #     print(f"class: {c+1}, tr count: {len(tr_idx)}, val count: {len(val_idx)}")
    # print(f'train mask count: {np.count_nonzero(train_mask)}')
    # print(f'val mask count: {np.count_nonzero()}')
    remaining = np.where((train_mask == False) & (val_mask == False))[0]
    test_mask[remaining] = True
    return [torch.from_numpy(train_mask), torch.from_numpy(val_mask), torch.from_numpy(test_mask)]

def get_split(pretrained_dir, dataset_name, data):
    """Gets the train, validation, test split for a given dataset. If the split is not found,
    creates a new split and saves it to the disk. This is used for datasets that do not have
    built-in splits.

    Args:
        pretrained_dir (str): directory to save the split
        dataset_name (str): name of the dataset
        data: PyTorch Geometric Data object
    
    Returns:
        split: a list contains splits
    """
    f_name = f'{pretrained_dir}/split_{dataset_name}.pt'
    if os.path.exists(f_name):
        split = torch.load(f_name)
    else:
        split = generate_balanced_split(data, num_train_per_class=30, num_val_per_class=30)
        torch.save(split, f_name)
        print(f'No previous split found for {dataset_name}.',
              f'New split was created and saved to {f_name}.')
    return split
