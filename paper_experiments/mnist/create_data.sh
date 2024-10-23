#!/bin/bash

cd ../..

###########################
### DATASETS GENERATION ###
###########################

echo "=> generate data"

### - Parameters to choose for dataset generation - ###
### - Only change here - ###
alpha="0.1" # 0.1:non-iid, 100000:iid, 0: true iid
############################

n_tasks="7" # 7 clients, one client per country
#######################################################


cd fl_training/data/mnist || exit 1
# cd data/mnist

# --- dataset creation --- #
rm -rf all_data
(
python generate_data.py \
--n_tasks ${n_tasks} \
--s_frac 0.6 \
--test_tasks_frac 0.0 \
--seed 12345 \
--by_labels_split \
--alpha ${alpha}
) # /!\ the two last lines are for non-iid
# --------------------------- #