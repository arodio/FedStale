# Tmp

## New experiments

*Warnings:* any folder of the logs should contain at most one log file

**First experiment:**
- train with FedAvg, 2 clients, --participation_probs 1.0 1.0, --n_rounds 100
- do not use the parameters --swap_labels --swap_proportion 0.0 (we don't want to swap labels at the moment).
- do not use the activity estimator but rather provide the participation probabilities. However, we cannot rely for this on the argument --participation_probs because it entails the presence of 2 groups of clients with a participation probability for each of the two groups. In our case we want to provide the exact participation probability for each client (not groups of clients).
- create a participation_matrix (rows are clients and columns are participation 0 or 1 for each round of the federated training). Save this matrix (e.g., with numpy.save). **Done: the participation matrix is originally a pandas.DataFrame saved in a csv file. It is loaded and then transformed into a 2d-array. The number of rows (countries) must match the number of clients (n_tasks). The number of columns must larger of equal to than the number of FL rounds.** 
- need to change the class ActivitySimulator. This class should just use the participation_matrix. **Done: but still need to change the inputs of the ``__init__`` of this class.**
- change the function get_participation_dict. **Done.**

*For after:*
- completely remove the parameter participation_probs from the code since we won't use it other than with --participation_probs 1.0 1.0.
- use the activity estimator and compare the estimated participation probabilities with the exact participation probability of each client. This will permit to evaluate the performance of the activity estimator.

**For the new experiement (cross silo with CI based availability):**

*Test 1:*
```bash
cd data/mnist
python generate_data.py --n_tasks 2 --s_frac 0.2 --test_tasks_frac 0.0 --seed 12345
cd ../..
python train.py mnist --n_rounds 100 --participation_probs 1.0 1.0 --unknown_participation_probs --bz 128 --lr 5e-3 --log_freq 1 --device cuda --optimizer sgd --server_optimizer history --swap_labels --swap_proportion 0.0 --logs_dir logs/mnist/p_0.5/h_0.0/fedvarp/lr_5e-3/seed_12 --seed 12 --verbose 0
tensorboard --logdir logs/mnist/p_0.5/h_0.0/fedvarp/lr_5e-3/seed_12
```

*Test 2:*
```bash
cd data/mnist
python generate_data.py --n_tasks 7 --s_frac 0.2 --test_tasks_frac 0.0 --seed 12345
cd ../..
python train.py mnist --n_rounds 100 --participation_probs 1.0 1.0 --bz 128 --lr 5e-3 --log_freq 1 --device cuda --optimizer sgd --server_optimizer sgd --logs_dir logs/CI_based_availability/7clients/fedavg/100rounds --seed 12 --verbose 0
tensorboard --logdir logs/CI_based_availability/7clients/100rounds
```



## Other

**First output of generate_data:**

(venv_fedstale) C:\Users\charlotte.rodriguez\Downloads\gitlab-REPOSITORIES\FedStale\data\mnist>python generate_data.py --n_tasks 24 --s_frac 0.2 --test_tasks_frac 0.0 --seed 12345
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz to raw_data/MNIST\raw\train-images-idx3-ubyte.gz
100.0%
Extracting raw_data/MNIST\raw\train-images-idx3-ubyte.gz to raw_data/MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz to raw_data/MNIST\raw\train-labels-idx1-ubyte.gz
100.0%
Extracting raw_data/MNIST\raw\train-labels-idx1-ubyte.gz to raw_data/MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz to raw_data/MNIST\raw\t10k-images-idx3-ubyte.gz
100.0%
Extracting raw_data/MNIST\raw\t10k-images-idx3-ubyte.gz to raw_data/MNIST\raw

Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Failed to download (trying next):
HTTP Error 403: Forbidden

Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz
Downloading https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz to raw_data/MNIST\raw\t10k-labels-idx1-ubyte.gz
100.0%
Extracting raw_data/MNIST\raw\t10k-labels-idx1-ubyte.gz to raw_data/MNIST\raw

## Requirements

**For requirements.txt file:**
```
numpy
--extra-index-url https://download.pytorch.org/whl/cu121
torch torchvision
scikit-learn
tqdm
tensorboard
tensorflow
pandas
```