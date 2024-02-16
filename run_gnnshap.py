import argparse
import pickle
import time

import torch
from tqdm.auto import tqdm

from dataset.utils import get_model_data_config
from gnnshap.explainer import GNNShapExplainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--result_path', type=str, default=None,
                        help=('Path to save the results. It will be saved in the config results '
                              'path if not provided.'))
    parser.add_argument('--num_samples', type=int, default=10000,
                        help='Number of samples to use for GNNShap')
    parser.add_argument('--repeat', default=1, type=int)
    parser.add_argument('--batch_size', type=int, default=1024)
    parser.add_argument('--sampler', type=str, default='GNNShapSampler',
                        help='Sampler to use for sampling coalitions',
                        choices=['GNNShapSampler', 'SVXSampler', 'SHAPSampler',
                                'SHAPUniqueSampler'],
                        help='Sampler to use for sampling coalitions.')
    parser.add_argument('--solver', type=str, default='WLSSolver',
                        help='Solver to use for solving SVX', choices=['WLSSolver', 'WLRSolver'])
    
    # SVXSampler maximum size of coalitions to sample from
    parser.add_argument('--size_lim', type=int, default=3)

    args = parser.parse_args()

    dataset_name = args.dataset
    num_samples = args.num_samples
    batch_size = args.batch_size
    sampler_name = args.sampler
    solver_name = args.solver


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, data, config = get_model_data_config(dataset_name, load_pretrained=True, device=device)

    test_nodes = config['test_nodes']

    result_path = args.result_path if args.result_path is not None else config["results_path"]




    if sampler_name == "SVXSampler":
        extra_param_suffixes = f"_{args.size_lim}"
    else:
        extra_param_suffixes = ""

    #explain_node_idx = 0
    for r in range(args.repeat):
        results = []
        
        shap = GNNShapExplainer(model, data, nhops=config['num_hops'], verbose=0, device=device,
                           progress_hide=True)
        start_time = time.time()

        failed_indices = []
        for ind in tqdm(test_nodes, desc=f"GNNShap explanations - run{r+1}"):
            try:
                explanation = shap.explain(ind, nsamples=num_samples,
                                            sampler_name=sampler_name, batch_size=batch_size,
                                            solver_name=solver_name, size_lim=args.size_lim)
                results.append(explanation.result2dict())
            except RuntimeError as e:
                failed_indices.append(ind)
                if 'out of memory' in str(e):
                    print(f"Node {ind} has failed: out of memory")
                else:
                    print(f"Node {ind} has failed: {e}")
            except Exception as e:
                print(f"Node {ind} has failed. General error: {e}")
                failed_indices.append(ind)

        rfile = (f'{result_path}/{dataset_name}_GNNShap_{sampler_name}_{solver_name}_'
                   f'{num_samples}_{batch_size}{extra_param_suffixes}_run{r+1}.pkl')
        with open(rfile, 'wb') as pkl_file:
            pickle.dump([results, 0], pkl_file)
        
        if len(failed_indices) > 0:
            print(f"Failed indices: {failed_indices}")
