
import pickle

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from tqdm import tqdm

from dataset.utils import get_model_data_config
from gnnshap.eval_utils import fidelity


def read_results(file_path: str) -> tuple[list, float]:
    """Reads a pickle results file and returns scores, total time and training time.
    Note that only some methods like PGExplainer requires separate training. Other methods will have
    0.0 as the training time.

    Args:
        file_path (str): file path

    Returns:
        tuple[list, float]: results as list, and training time.
    """
    try:
        res = pickle.load(open(file_path, 'rb'))
        if len(res) == 2:
            return res[0], res[1]
        else:
            return res, 0.0
    except Exception as e:
        return [], 0.0

def compute_fidelity_score(results: list, data: Data, model: torch.nn.Module,
                                sparsity: float = 0.3, fid_type: str = 'neg', topk: int = 0,
                                target_class: int = None, testing_pred: str = 'mix',
                                apply_abs: bool=True) -> tuple:
    """Computes fidelity+ and fidelity- scores. It supports both topk and sparsity. If sparsity set
    to 0.3, it drops 30% of the edges. Based on the neg or pos, it drops unimportant or important
    edges. It applies topk based keep if topk is set to a positive integer other than zero.

    `testing_pred` helps to further analyze fidelity scores for correct and wrong classified nodes.

    Note that it computes fidelity scores for the predicted class if target class is not provided.
    Args:
        results (list): List of dictionaries. Each dictionary should have node_id, num_players,
            scores keys.
        data (Data): pyG Data.
        model (torch.nn.Module): a PyTorch model.
        sparsity (float, optional): target sparsity value. Defaults to 0.3.
        fid_type (str, optional): Fidelity type: neg or pos. Defaults to 'neg'.
        topk (int, optional): Topk edges to keep. Defaults to 0.
        target_class (int, optional): Target class to compute fidelity score. If None, it computes
            fidelity score for the predicted class. Defaults to None.
        testing_pred (str, optional): Testing prediction filter. Options are 'mix', 'wrong',
            and 'correct' Defaults to mix.
        apply_abs (bool, optional): applies absolute to scores. Some methods can find negative and
            positive contributing nodes/edges. Fidelity-wise, we only care the change amount. We can
            use this to get rid of negative contributing edges to improve accuracy. Defaults to
            True.

    Returns:
        tuple: average score, list of individual scores[node_id, nplayers, fidelity  prob score,
            current sparsity, correct_class, init_pred_class, sparse_pred_class, fidelity acc]
    """
    assert testing_pred in [
        'mix', 'wrong', 'correct'], "Testing prediction option is not correct!"


    fid_scores = []
    sum_prob = 0
    for res in results:
        if res['num_players'] < 2:
            continue
        try:
            (node_id, num_players, prob_score, current_sparsity, correct_class, init_pred_class,
            sparse_pred_class) = fidelity(res, data, model, sparsity, fid_type, topk,
                                                   target_class, apply_abs)

            if testing_pred == 'wrong' and correct_class == init_pred_class:  # skip correct preds
                continue
            if testing_pred == 'correct' and correct_class != init_pred_class:  # skip wrong preds
                continue

            sum_prob += prob_score
            fid_scores.append([node_id, num_players, prob_score, current_sparsity, correct_class,
                            init_pred_class, sparse_pred_class])
        except:
            print(f"Error in: {res}")
            return -1.0, []
        
    overall_prob = sum_prob / len(fid_scores)
    return overall_prob, fid_scores


def run_times_table(path_gen_fn, dataset_names, num_repeats: int=5) -> pd.DataFrame:
    """Reads pickle files and extract times. When there is multiple run, it computes the average.

    Args:
        path_gen_fn: function that returns the paths of the results
        dataset_names (list): list of dataset names
        num_repeats (int, optional): Number of repeats. Defaults to 5.

    Returns:
        explanation_times: pd.DataFrame: table with the average times for each method and dataset
        exp_model_train_times: pd.DataFrame: table with the average training times for
            each method and dataset. Valid for PGExplainer like methods.

    """
    method_names = np.array(path_gen_fn(dataset_names[0]))[:,0].tolist()

    res_table = np.zeros((len(method_names), len(dataset_names)), dtype=object)
    res_tr_table = np.zeros((len(method_names), len(dataset_names)), dtype=object)

    for i, dataset_name in enumerate(tqdm(dataset_names)):
        total_times = []
        tr_total_times = []

        for rep_num in range(1, num_repeats+1):
            result_file_paths = path_gen_fn(dataset_name, rep_num)
            total_times.append([])
            tr_total_times.append([])
            for res_file in result_file_paths:
                res, tr_time = read_results(res_file[1])
                tmp_time = 0
                for r in res:
                    tmp_time += r['time']
                total_times[-1].append(tmp_time)
                tr_total_times[-1].append(tr_time)
        total_times = np.array(total_times)
        tr_total_times = np.array(tr_total_times)


        # compute the mean and std
        mean_res = np.mean(total_times, axis=0)
        std_res = np.std(total_times, axis=0)
        mean_tr_res = np.mean(tr_total_times, axis=0)
        std_tr_res = np.std(tr_total_times, axis=0)


        # fill the table with the results
        for k in range(len(method_names)):
            res_table[k,i] = f"{mean_res[k]:.2f}\u00B1{std_res[k]:.2f}"
            res_tr_table[k,i] = f"{mean_tr_res[k]:.2f}\u00B1{std_tr_res[k]:.2f}"

    expl_df = pd.DataFrame(res_table, columns=dataset_names, index=method_names)
    expl_tr_df = pd.DataFrame(res_tr_table, columns=dataset_names, index=method_names)

    return expl_df, expl_tr_df

def fidelity_table(path_gen_fn, dataset_names, sparsity=0.1, score_type='neg', topk=0,
                   num_repeats=1, device='cpu', testing_pred='mix', apply_abs=True) -> pd.DataFrame:
    """Create a table with the fidelity scores of the methods. It applies topk based edge keep if
        topk is set to a positive integer other than zero.

    Args:
        path_gen_fn: function that returns the paths of the results
        dataset_names (list): list of dataset names
        sparsity (float, optional): sparsity of the explanations. Defaults to 0.1.
        score_type (str, optional): score_type (str, optional): Fidelity type: 'neg' or 'pos'.
            Defaults to 'neg'.
        topk (int, optional): Topk edges to keep. Defaults to 0.
        num_repeats (int, optional): number of experiment repeats. Defaults to 1.
        device (str, optional): device. Defaults to 'cpu'.
        testing_pred (str, optional): Testing prediction filter. Options are 'mix', 'wrong',
            and 'correct' Defaults to mix.
        apply_abs (bool, optional): applies absolute to scores. Some methods can find negative and
            positive contributing nodes/edges. Fidelity-wise, we only care the change amount. We can
            use this to get rid of negative contributing edges to improve accuracy. Defaults to
            True.

    Returns:
        pd.DataFrame: Fidelity table
    """

    method_names = np.array(path_gen_fn(dataset_names[0]))[:,0].tolist()
    # create the table
    res_table = np.zeros((len(method_names), len(dataset_names)), dtype=object)

    # iterate over the datasets
    for i, dataset_name in enumerate(tqdm(dataset_names)):
        model, data, _ = get_model_data_config(dataset_name, load_pretrained=True, device=device)
        res_runs = []
        not_founds = {name: [] for name in method_names}
        # iterate over the repeats
        for rep_num in range(1, num_repeats+1):
            f_paths = path_gen_fn(dataset_name, rep_num)
            res_runs.append([])
            for name, path in f_paths:
                res_data, _ = read_results(path)
                if len(res_data) == 0:
                    not_founds[name].append(path)
                    res_runs[-1].append(-1.0)
                else:
                    res_runs[-1].append(compute_fidelity_score(res_data, data, model, sparsity,
                                                               score_type, topk,
                                                               testing_pred=testing_pred,
                                                               apply_abs=apply_abs)[0])

        res_runs = np.array(res_runs)

        # compute the mean and std
        mean_res = np.mean(res_runs, axis=0)
        std_res = np.std(res_runs, axis=0)
        # fill the table with the results
        for k, n in enumerate(method_names):
            if len(not_founds[n]) > 0:
                res_table[k,i] = "N/A"
                print(f"Results not found for {not_founds[n]}. Check the paths.")
            else:
                res_table[k,i] = f"{mean_res[k]:.3f}\u00B1{std_res[k]:.3f}"
    res_table = pd.DataFrame(res_table, columns=dataset_names, index=method_names)
    return res_table
