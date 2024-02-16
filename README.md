# GNNShap: Scalable and Accurate GNN Explanation using Shapley Values
This repository contains the source code of
`GNNShap: Scalable and Accurate GNN Explanation using Shapley Values` paper accepted in The
Web Conference 2024.

### Setup
Our implementation is based on PyTorch and PYG. Also, Our Shapley sampling strategy is implemented 
in Cuda. Therefore, GNNShap requires a GPU with Cuda support.
Required packages and versions are provided in the `requirements.txt` file.

You can install the requirements by running:
```bash
pip install -r requirements.txt
```

### Dataset Configs

Dataset and dataset-specific model configurations are in the `dataset/configs.py` file.


### Model training  

We provided pretrained models in the `pretrained` folder.

To train Cora, CiteSeer, PubMed, Coauthor-CS, Coauthor-Physics, and Facebook datasets: 
```bash
python train.py --dataset Cora
```

Reddit and ogbn-products require `NeighborLoader` for training. To train them:
```bash
python train_large.py --dataset Reddit
```

### Experiments

We provided scripts for baselines and GNNShap experiments. Scripts will save explanation results to
the results folder. Note that scripts repeat each experiment five times. This can be changed in the 
scripts.

For baselines, you can use the following script. For individual baseline, you can refer to 
the script file content.

```bash
./run_baseline_experiments.sh
```

For GNNShap experiments, you can use the following script:
```
./run_gnnshap_experiments.sh
```

For individual dataset experiments, an example is provided below:
```bash
python run_gnnshap.py --dataset Cora --num_samples 25000 --sampler GNNShapSampler 
--batch_size 1024 --repeat 1
```

The results will be saved to the `results` folder. The default result folder can be changed 
in `dataset/configs.py`

**We ran our experiments on a GPU with 24GB of memory. You may need to adjust `batch_size` 
and `num_samples` parameters.**


### Evaluation
We used the `BaselinesEvaluation.ipynb`  notebook under the `examples` folder for explanation times
and fidelity results.

### Visualization
We provided explanation visualization examples in the `Visualization.ipynb`  notebook 
under `examples.`

### Custom Model & Data Explanations
We provided an example in the `CustomModelData.ipynb`  notebook under `examples`.

### Citation
Please cite our work if you find it useful.

```
@article{akkas2024gnnshap,
  title={GNNShap: Scalable and Accurate GNN Explanations using Shapley Values},
  author={Akkas, Selahattin and Azad, Ariful},
  journal={arXiv preprint arXiv:2401.04829},
  year={2024}
}
```
