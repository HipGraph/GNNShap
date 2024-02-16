def get_config(conf_name):
    """ Required  configs for each dataset

    Args:
        conf_name (str): Dataset name

    Returns:
        dict: configuration dictionary
    """

    root_path = './' # root path. Can be changed to the path where the dataset is stored.
    results_path = f'./results' # path to store the explanation results

    dataset_configs = {
        'Cora': {
            'hidden_dim': 16,
            'model': 'GCNModel',
            'num_layers': 2,
            'epoch': 200,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'graphSVX_args': {'num_samples': 1000, 'S': 3, 'coal': 'SmarterSeparate',
                              'feat':'Expectation'},
        },
        'Cora_GAT': {
            'hidden_dim': 16,
            'num_layers': 2,
            'epoch': 200,
            'lr': 0.005,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'heads': 8,
            'model': 'GATModel',
        },
        'CiteSeer': {
            'hidden_dim': 16,
            'model': 'GCNModel',
            'num_layers': 2,
            'epoch': 200,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'graphSVX_args': {'num_samples': 1000, 'S':3, 'coal': 'SmarterSeparate',
                    'feat':'Expectation'},

        },
        'PubMed': {
            'hidden_dim': 16,
            'model': 'GCNModel',
            'num_layers': 2,
            'epoch': 200,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'graphSVX_args': {'num_samples': 1000, 'S':3, 'coal': 'SmarterSeparate',
                    'feat':'Expectation'},
        },
        'Facebook': {
            'hidden_dim': 16,
            'model': 'GCNModel',
            'num_layers': 2,
            'epoch': 200,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'graphSVX_args': {'num_samples': 1000, 'S': 3, 'coal': 'SmarterSeparate',
                              'feat':'Expectation'},
        },
        'Coauthor-CS': {
            'hidden_dim': 64,
            'model': 'GCNModel',
            'num_layers': 2,
            'epoch': 200,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'graphSVX_args': {'num_samples': 1000, 'S':3, 'coal': 'SmarterSeparate',
                    'feat':'Expectation'},
        },
        'Coauthor-Physics': {
            'hidden_dim': 64,
            'model': 'GCNModel',
            'num_layers': 2,
            'epoch': 200,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'graphSVX_args': {'num_samples': 1000, 'S':3, 'coal': 'SmarterSeparate',
                    'feat':'Expectation'},
        },
        'Reddit': {
            'hidden_dim': 128,
            'model': 'GCNModel',
            'num_layers': 2,
            'epoch': 11,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'graphSVX_args': {'num_samples': 1000, 'S':3, 'coal': 'SmarterSeparate',
                    'feat':'Expectation'},
            'nei_sampler_args': {'sizes': [25, 10], 'batch_size': 1024},
        },
        'ogbn-products': {
            'hidden_dim': 128,
            'model': 'GCNModel',
            'num_layers': 2,
            'failed_test_nodes': [],
            'epoch': 11,
            'lr': 0.01,
            'weight_decay': 5e-4,
            'dropout': 0.5,
            'normalize': True,
            'add_self_loops': True,
            'graphSVX_args': {'num_samples': 1000, 'S':3, 'coal': 'SmarterSeparate',
                    'feat':'Expectation'},
            'nei_sampler_args': {'sizes': [25, 10], 'batch_size': 2048},
        }
    }
    if conf_name is None:
        return dataset_configs

    conf = dataset_configs[conf_name]
    conf['root_path'] = root_path
    conf['results_path'] = results_path

    conf['num_hops'] = conf['num_layers']
    return conf
