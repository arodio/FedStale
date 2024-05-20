#!/bin/bash

cd ../..

echo "=> generate data"

cd data/cifar10 || exit 1
rm -rf all_data
python generate_data.py \
    --n_tasks 24 \
    --s_frac 0.2 \
    --test_tasks_frac 0.0 \
    --seed 12345

cd ../..

participation="0.5"
heterogeneities="0.0 0.2 0.4 0.6 0.8 1.0"
weights="0.5"
seeds="12"
lrs="5e-3"
device="cuda"
      
for heterogeneity in $heterogeneities; do
    for lr in $lrs; do
        for seed in $seeds; do
            echo "Run FedVARP: p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
            python train.py \
                cifar10 \
                --n_rounds 4000 \
                --participation_probs 1.0 ${participation} \
                --unknown_participation_probs \
                --bz 128 \
                --lr ${lr} \
                --log_freq 1 \
                --device ${device} \
                --optimizer sgd \
                --server_optimizer history \
                --swap_labels \
                --swap_proportion ${heterogeneity} \
                --logs_dir logs/cifar10/p_${participation}/h_${heterogeneity}/fedvarp/lr_${lr}/seed_${seed} \
                --seed ${seed} \
                --verbose 0
        done
    done
done

for heterogeneity in $heterogeneities; do
    for lr in $lrs; do
        for seed in $seeds; do
            echo "Run FedAvg : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"    
            python train.py \
                cifar10 \
                --n_rounds 4000 \
                --participation_probs 1.0 ${participation} \
                --unknown_participation_probs \
                --bz 128 \
                --lr ${lr} \
                --log_freq 1 \
                --device ${device} \
                --optimizer sgd \
                --server_optimizer sgd \
                --swap_labels \
                --swap_proportion ${heterogeneity} \
                --logs_dir logs/cifar10/p_${participation}/h_${heterogeneity}/fedavg/lr_${lr}/seed_${seed} \
                --seed ${seed} \
                --verbose 0
        done
    done
done

for heterogeneity in $heterogeneities; do
    for lr in $lrs; do
        for weight in $weights; do
            for seed in $seeds ; do
                echo "Run FedStale : p ${participation}, h ${heterogeneity}, beta ${weight}, lr ${lr}, seed ${seed}"
                python train.py \
                    cifar10 \
                    --n_rounds 4000 \
                    --participation_probs 1.0 ${participation} \
                    --unknown_participation_probs \
                    --bz 128 \
                    --lr ${lr} \
                    --log_freq 1 \
                    --device ${device} \
                    --optimizer sgd \
                    --server_optimizer history \
                    --history_coefficient ${weight} \
                    --swap_labels \
                    --swap_proportion ${heterogeneity} \
                    --logs_dir logs/cifar10/p_${participation}/h_${heterogeneity}/fedstale/b_${weight}/lr_${lr}/seed_${seed} \
                    --seed ${seed} \
                    --verbose 0
            done
        done
    done
done
