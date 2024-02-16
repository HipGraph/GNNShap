#!/bin/bash

# since baseline scripts are not in the root directory, they can't find modules.
# so, we need to add the root directory to the PYTHONPATH
export PYTHONPATH="$PYTHONPATH:$(pwd)"

datasets=("Cora" "CiteSeer" "PubMed" "Coauthor-CS" "Coauthor-Physics" "Facebook")

nrepeat=5 # number of repeats for each experiment

for d in "${datasets[@]}"; do
    python baselines/run_gnnexplainer.py --dataset $d --repeat $nrepeat
    echo "${d} -  GNNExplainer done!!!!"
    python baselines/run_pgexplainer.py --dataset $d --repeat $nrepeat
    echo "${d} -  PGExplainer done!!!!"
    python baselines/run_pgmexplainer.py --dataset $d --repeat $nrepeat
    echo "${d} -  PGMExplainer done!!!!"
    python baselines/run_graphsvx.py --dataset $d --repeat $nrepeat
    echo "${d} -  GraphSVX done!!!!"
    python baselines/run_sa.py --dataset $d --repeat $nrepeat
    echo "${d} -  Saliency done!!!!"

    # OrphicX gives OOM error for some datasets
    # python baselines/run_orphicx.py --dataset $d --repeat $nrepeat --epoch 50
    # echo "${d} -  OrphicX done!!!!"
done
