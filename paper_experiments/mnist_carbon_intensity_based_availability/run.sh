cd ../..

echo "=> generate data"

alpha="100000"
biased="2"
# alpha="-1" # use this when true iid

n_tasks="7"
# cd data/mnist || exit 1
cd data/mnist

# --- if dataset creation --- #
rm -rf all_data
(
python generate_data.py \
--n_tasks ${n_tasks} \
--s_frac 0.2 \
--test_tasks_frac 0.0 \
--seed 12345 \
--by_labels_split \
--alpha ${alpha}    
) # the two last lines are for non-iid
# --------------------------- #

cd ../..

participation="1.0"
heterogeneities="0.0"
weights="0.5" # is the beta parameter in the FedStale paper
seeds="12"
lrs="5e-3"
device="cuda"
n_rounds="100"
availabilities="uniform-carbon-budget-fine-tuning"

# done:
# uniform-CI-threshold uniform-carbon-budget uniform-carbon-budget-fine-tuning uniform-time-budget
# nonlinear-optimization-cvxpy_w-no-w_a-1 
# nonlinear-optimization-cvxpy_w-no-w_a-0.5 

# nonlinear-optimization-cvxpy_w-no-w_a-0.1

# ------------------------------ #
# --- Experiments for FedAvg --- #
# ------------------------------ #
# known participation probs:
for availability in $availabilities; do
availability_matrix_path="data_availability/av-mat_${availability}.csv"
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
--logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedavg/known_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
--seed ${seed} \
--verbose 0 \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased}
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
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedavg/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0 \
# --availability_matrix_path ${availability_matrix_path} \
# --biased ${biased}
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
availability_matrix_path="data_availability/av-mat_${availability}.csv"
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
--logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedvarp/known_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
--seed ${seed} \
--verbose 0 \
--availability_matrix_path ${availability_matrix_path} \
--biased ${biased}
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
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedvarp/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0 \
# --availability_matrix_path ${availability_matrix_path} \
# --biased ${biased}
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
availability_matrix_path="data_availability/av-mat_${availability}.csv"
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
--logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedstale/known_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
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
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/${availability}/biased_${biased}/fedstale/unknown_participation_probs/alpha_${alpha}/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0 \
# --availability_matrix_path ${availability_matrix_path} \
# --biased ${biased}
# )
# done
# done
# done
# done
# done