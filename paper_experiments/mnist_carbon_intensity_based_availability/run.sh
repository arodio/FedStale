cd ../..

echo "=> generate data"

n_tasks="7"

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

cd ../..

participation="1.0"
heterogeneities="0.0"
weights="0.5" # is the beta parameter in the FedStale paper
seeds="12"
lrs="5e-3"
device="cuda"
n_rounds="100"

# # ------------------------------ #
# # --- Experiments for FedAvg --- #
# # ------------------------------ #
# # known participation probs:
# for heterogeneity in $heterogeneities; do
# for lr in $lrs; do
# for seed in $seeds; do
# echo "Run FedAvg : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
# (
# python train.py \
# mnist \
# --n_rounds ${n_rounds} \
# --participation_probs 1.0 ${participation} \
# --bz 128 \
# --lr ${lr} \
# --log_freq 1 \
# --device ${device} \
# --optimizer sgd \
# --server_optimizer sgd \
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/fedavg/known_participation_probs/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0
# )
# done
# done
# done
# # unknown participation probs:
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
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/fedavg/unknown_participation_probs/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0
# )
# done
# done
# done

# # ------------------------------- #
# # --- Experiments for FedVARP --- #
# # ------------------------------- #
# # known participation probs:
# for heterogeneity in $heterogeneities; do
# for lr in $lrs; do
# for seed in $seeds; do
# echo "Run FedVARP : p ${participation}, h ${heterogeneity}, lr ${lr}, seed ${seed}"
# (
# python train.py \
# mnist \
# --n_rounds ${n_rounds} \
# --participation_probs 1.0 ${participation} \
# --bz 128 \
# --lr ${lr} \
# --log_freq 1 \
# --device ${device} \
# --optimizer sgd \
# --server_optimizer history \
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/fedvarp/unknown_participation_probs/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0
# )
# done
# done
# done

# # unknown participation probs:
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
# --logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/fedvarp/unknown_participation_probs/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
# --seed ${seed} \
# --verbose 0
# )
# done
# done
# done

# ------------------------------- #
# --- Experiments for FedStale --- #
# ------------------------------- #
# known participation probs:
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
--logs_dir logs/mnist_CI_based_availability/clients_${n_tasks}/fedstale/unknown_participation_probs/lr_${lr}/seed_${seed}/rounds_${n_rounds} \
--seed ${seed} \
--verbose 0
)
done
done
done
done

# known participation probs: