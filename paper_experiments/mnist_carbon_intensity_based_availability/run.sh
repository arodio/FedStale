cd ../..

echo "=> generate data"

n_tasks="7"
# alpha="100000"
alpha="-1" # use this when true iid

# cd data/mnist || exit 1
cd data/mnist
rm -rf all_data
(
python generate_data.py \
--n_tasks ${n_tasks} \
--s_frac 0.2 \
--test_tasks_frac 0.0 \
--seed 12345
)
# --by_labels_split \
# --alpha ${alpha}

cd ../..

participation="1.0"
heterogeneities="0.0"
weights="0.5" # is the beta parameter in the FedStale paper
seeds="12"
lrs="5e-3"
device="cuda"
n_rounds="100"
availabilities="carbon-budget-fine-tuning random_for_carbon-budget-fine-tuning"

# CI-threshold-local-mean CI-threshold-global-mean 
# CI-threshold-median CI-threshold-penalized-local-mean
# carbon-budget 
# random_for_CI-threshold-local-mean 
# random_for_CI-threshold-global-mean random_for_CI-threshold-median 
# random_for_CI-threshold-penalized-local-mean 
# random_for_carbon-budget 
# carbon-budget-fine-tuning random_for_carbon-budget-fine-tuning

# ------------------------------ #
# --- Experiments for FedAvg --- #
# ------------------------------ #
# known participation probs:
for availability in $availabilities; do
availability_matrix_path="data_availability/availability_matrix_${availability}.csv"
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
--logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/fedavg/known_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
--seed ${seed} \
--verbose 0 \
--availability_matrix_path ${availability_matrix_path}
)
done
done
done
done

# # unknown participation probs:
# for availability in $availabilities; do
# availability_matrix_path="data_availability/availability_matrix_${availability}.csv"
# for heterogeneity in $heterogeneities; do
# for lr in $lrs; do
# for seed in $seeds; do
# echo "Run FedAvg : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
# (
# python train.py \
# mnist \
# --n_rounds ${n_rounds} \
# --participation_probs 1.0 ${participation} \
# --unknown_participation_probs \
# --bz 128 \
# --lr ${lr} \
# --log_freq 1 \
# --device ${device} \
# --optimizer sgd \
# --server_optimizer sgd \
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/fedavg/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0 \
# --availability_matrix_path ${availability_matrix_path}
# )
# done
# done
# done
# done

# ------------------------------- #
# --- Experiments for FedVARP --- #
# ------------------------------- #
# known participation probs:
for availability in $availabilities; do
availability_matrix_path="data_availability/availability_matrix_${availability}.csv"
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
--logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/fedvarp/known_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
--seed ${seed} \
--verbose 0 \
--availability_matrix_path ${availability_matrix_path}
)
done
done
done
done

# # unknown participation probs:
# for availability in $availabilities; do
# availability_matrix_path="data_availability/availability_matrix_${availability}.csv"
# for heterogeneity in $heterogeneities; do
# for lr in $lrs; do
# for seed in $seeds; do
# echo "Run FedVARP : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
# (
# python train.py \
# mnist \
# --n_rounds ${n_rounds} \
# --participation_probs 1.0 ${participation} \
# --unknown_participation_probs \
# --bz 128 \
# --lr ${lr} \
# --log_freq 1 \
# --device ${device} \
# --optimizer sgd \
# --server_optimizer history \
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/fedvarp/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0 \
# --availability_matrix_path ${availability_matrix_path}
# )
# done
# done
# done
# done

# ------------------------------- #
# --- Experiments for FedStale --- #
# ------------------------------- #
# known participation probs:
for availability in $availabilities; do
availability_matrix_path="data_availability/availability_matrix_${availability}.csv"
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
--logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/fedstale/known_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
--seed ${seed} \
--verbose 0 \
--availability_matrix_path ${availability_matrix_path}
)
done
done
done
done
done

# # unknown participation probs:
# for availability in $availabilities; do
# availability_matrix_path="data_availability/availability_matrix_${availability}.csv"
# for heterogeneity in $heterogeneities; do
# for lr in $lrs; do
# for weight in $weights; do
# for seed in $seeds ; do
# echo "Run FedStale : p ${participation}, h ${heterogeneity}, beta ${weight}, lr ${lr}, seed ${seed}"
# (
# python train.py \
# mnist \
# --n_rounds ${n_rounds} \
# --participation_probs 1.0 ${participation} \
# --unknown_participation_probs \
# --bz 128 \
# --lr ${lr} \
# --log_freq 1 \
# --device ${device} \
# --optimizer sgd \
# --server_optimizer history \
# --history_coefficient ${weight} \
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/fedstale/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0 \
# --availability_matrix_path ${availability_matrix_path}
# )
# done
# done
# done
# done
# done