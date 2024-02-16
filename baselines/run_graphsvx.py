import warnings

import torch

warnings.filterwarnings("ignore")

import pickle
import time

from tqdm.auto import tqdm

from baselines.methods.graphsvx import GraphSVX, arg_parse
from baselines.utils import result2dict
from dataset.utils import get_model_data_config


def main():

    args = arg_parse()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model, data, config = get_model_data_config(args.dataset, load_pretrained=True, device=device,
                                                log_softmax_return=True)
    args.gpu = True
    args.num_samples= config['graphSVX_args']['num_samples']
    args.hops=config['num_hops']
    args.hv='compute_pred'
    args.coal = "SmarterSeparate"
    args.S = config['graphSVX_args']['S']
    args.regu=0 # only cosidering graph structure. Node features are not considered.
    args.feat = config['graphSVX_args']['feat']
    data.name = args.dataset
    data.num_classes = (max(data.y)+1).item()
    
    num_samples = args.num_samples
    

    test_nodes = config['test_nodes']


    for r in range(args.repeat):
        # Explain it with GraphSVX
        explainer = GraphSVX(data, model, args.gpu)
        results = []

        
        for ind in tqdm(test_nodes,
                        desc=f"GraphSVX explanations - run{r+1} - nsamp:{num_samples}"):
            try:
                start_time = time.time()
                explanations = explainer.explain([ind], args.hops, num_samples, args.info,
                                                args.multiclass, args.fullempty, args.S,
                                                args.hv, args.feat, args.coal, args.g,
                                                args.regu, False)
                results.append(result2dict(ind, explanations[0], time.time() - start_time))
            except Exception as e:
                print(f"Node {ind} failed!")
                print(e)

        rfile = (f'{config["results_path"]}/{data.name}_GraphSVX_{args.coal}_{args.S }_'
                    f'{num_samples}_run{r+1}.pkl')
        with open(rfile, 'wb') as pkl_file:
            pickle.dump([results, 0], pkl_file)

if __name__ == "__main__":
    main()
