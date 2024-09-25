#!/bin/bash

cd ../..

###########################
### DATASETS GENERATION ###
###########################

echo "=> generate data"

### - Parameters to choose for dataset generation - ###
### - Only change here - ###
alpha="100000" # 0.1:non-iid, 100000:iid, 0: true iid
############################

n_tasks="7" # 7 clients, one client per country
#######################################################


# cd data/mnist || exit 1
cd data/mnist

# --- dataset creation --- #
rm -rf all_data
(
python generate_data.py \
--n_tasks ${n_tasks} \
--s_frac 0.2 \
--test_tasks_frac 0.0 \
--seed 12345 \
--by_labels_split \
--alpha ${alpha}
) # /!\ the two last lines are for non-iid
# --------------------------- #

cd ../..


################
### TRAINING ###
################

echo "=> training"

### - Parameters to choose for training - ###
### - Only change here - ###
# availabilities="opt-pb3-stage2"
availabilities="opt-pb3-stage2 uniform-carbon-budget uniform-time-budget uniform-CI-threshold uniform-carbon-budget-fine-tuning nonlinear-optimization-cvxpy_w-no-w_a-0.5 nonlinear-optimization-cvxpy_w-no-w_a-0.1 all-available" # list of availability matrices
fl_algo="fedavg fedvarp fedstale" # list of FL algorithms
biased="0" # 0:unbiased, 1:biased, 2:hybrid (unbiased except when all clients available)
############################

participation="1.0"
heterogeneities="0.0"
weights="0.5" # is the beta parameter in the FedStale paper
seeds="12"
lrs="5e-3" # list of learning rates
device="cuda"
n_rounds="100" # number of fl rounds
#############################################

### other availability matrices' names ###
# opt-new-problem-cvxpy_a-1
# opt-new-problem-cvxpy_a-10
# opt-new-problem-cvxpy_a-21
# uniform-CI-threshold 
# uniform-carbon-budget 
# uniform-carbon-budget-fine-tuning 
# uniform-time-budget
# nonlinear-optimization-cvxpy_w-no-w_a-1 
# nonlinear-optimization-cvxpy_w-no-w_a-0.5 
# random4_uniform-carbon-budget-fine-tuning
# nonlinear-optimization-cvxpy_w-no-w_a-0.1


# ------------------------------ #
# --- Experiments for FedAvg --- #
# ------------------------------ #
# known participation probs:
if echo "$fl_algo" | grep -q "fedavg"; then
# if [[ "fedavg" == *"$fl_algo"* ]]; then
for availability in $availabilities; do
availability_matrix_path="availability_matrices/av-mat_${availability}.csv"
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Run FedAvg : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
mnist \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer sgd \
--logs_dir logs/mnist/${availability}/biased_${biased}/fedavg/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose 0 \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased}
)
done
done
done
done
fi

# # # unknown participation probs:
# # for availability in $availabilities; do
# # availability_matrix_path="availability_matrices/availability_matrix_${availability}.csv"
# # for heterogeneity in $heterogeneities; do
# # for lr in $lrs; do
# # for seed in $seeds; do
# # echo "Run FedAvg : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
# # (
# # python train.py \
# # mnist \
# # --n_rounds ${n_rounds} \
# # --participation_probs 1.0 ${participation} \
# # --unknown_participation_probs \
# # --bz 128 \
# # --lr ${lr} \
# # --log_freq 1 \
# # --device ${device} \
# # --optimizer sgd \
# # --server_optimizer sgd \
# # --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedavg/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# # --seed ${seed} \
# # --verbose 0 \
# # --availability_matrix_path ${availability_matrix_path} \
# # --biased ${biased}
# # )
# # done
# # done
# # done
# # done

# ------------------------------- #
# --- Experiments for FedVARP --- #
# ------------------------------- #
# known participation probs:
if echo "$fl_algo" | grep -q "fedvarp"; then
for availability in $availabilities; do
availability_matrix_path="availability_matrices/av-mat_${availability}.csv"
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for seed in $seeds; do
echo "Run FedVARP : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
(
python train.py \
mnist \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--logs_dir logs/mnist/${availability}/biased_${biased}/fedvarp/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose 0 \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased}
)
done
done
done
done
fi

# # # unknown participation probs:
# # for availability in $availabilities; do
# # availability_matrix_path="availability_matrices/availability_matrix_${availability}.csv"
# # for heterogeneity in $heterogeneities; do
# # for lr in $lrs; do
# # for seed in $seeds; do
# # echo "Run FedVARP : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
# # (
# # python train.py \
# # mnist \
# # --n_rounds ${n_rounds} \
# # --participation_probs 1.0 ${participation} \
# # --unknown_participation_probs \
# # --bz 128 \
# # --lr ${lr} \
# # --log_freq 1 \
# # --device ${device} \
# # --optimizer sgd \
# # --server_optimizer history \
# # --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedvarp/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# # --seed ${seed} \
# # --verbose 0 \
# # --availability_matrix_path ${availability_matrix_path} \
# # --biased ${biased}
# # )
# # done
# # done
# # done
# # done

# -------------------------------- #
# --- Experiments for FedStale --- #
# -------------------------------- #
# known participation probs:
if echo "$fl_algo" | grep -q "fedstale"; then
for availability in $availabilities; do
availability_matrix_path="availability_matrices/av-mat_${availability}.csv"
for heterogeneity in $heterogeneities; do
for lr in $lrs; do
for weight in $weights; do
for seed in $seeds ; do
echo "Run FedStale : p ${participation}, h ${heterogeneity}, beta ${weight}, lr ${lr}, seed ${seed}"
(
python train.py \
mnist \
--n_rounds ${n_rounds} \
--participation_probs 1.0 ${participation} \
--bz 128 \
--lr ${lr} \
--log_freq 1 \
--device ${device} \
--optimizer sgd \
--server_optimizer history \
--history_coefficient ${weight} \
--logs_dir logs/mnist/${availability}/biased_${biased}/fedstale/alpha_${alpha}/lr_${lr}/seed_${seed} \
--seed ${seed} \
--verbose 0 \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased}
)
done
done
done
done
done
fi

# # # unknown participation probs:
# # for availability in $availabilities; do
# # availability_matrix_path="availability_matrices/availability_matrix_${availability}.csv"
# # for heterogeneity in $heterogeneities; do
# # for lr in $lrs; do
# # for weight in $weights; do
# # for seed in $seeds ; do
# # echo "Run FedStale : p ${participation}, h ${heterogeneity}, beta ${weight}, lr ${lr}, seed ${seed}"
# # (
# # python train.py \
# # mnist \
# # --n_rounds ${n_rounds} \
# # --participation_probs 1.0 ${participation} \
# # --unknown_participation_probs \
# # --bz 128 \
# # --lr ${lr} \
# # --log_freq 1 \
# # --device ${device} \
# # --optimizer sgd \
# # --server_optimizer history \
# # --history_coefficient ${weight} \
# # --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedstale/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# # --seed ${seed} \
# # --verbose 0 \
# # --availability_matrix_path ${availability_matrix_path} \
# # --biased ${biased}
# # )
# # done
# # done
# # done
# # done
# # done