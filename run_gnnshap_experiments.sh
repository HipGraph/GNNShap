#!/bin/bash

# num_samples=(10000 25000 50000) # number of samples for GNNShap
num_samples=(10000)

datasets=("Cora" "CiteSeer" "PubMed" "Coauthor-CS" "Coauthor-Physics" "Facebook")
solvers=("WLSSolver") # WLRSolver is suggested for Reddit and ogbn-products

# samplers=("GNNShapSampler" "SVXSampler")
samplers=("GNNShapSampler")

for n in "${num_samples[@]}"; do
    for d in "${datasets[@]}"; do
        
        # batch size is set to 1024 for all datasets except Coauthor-CS and Coauthor-Physics.
        if [[ "$d" == "Coauthor"* ]]; then
            if [[ "$d" == "Coauthor-CS" ]]; then
                batch_size=512
            else
                batch_size=128 # coauthor-physics
        fi
        else
            batch_size=1024
        fi

        for solv in "${solvers[@]}"; do
            for samp in "${samplers[@]}"; do
                python run_gnnshap.py --dataset $d --num_samples $n \
                    --batch_size $batch_size --repeat 5 --sampler $samp --solver $solv
                echo "${solv} ${samp} ${d} -  ${n} done!!!!"
            done
        done
    done
done
