# Readme

## Requirements

Install the follwoing in a virtual environment:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn tqdm tensorboard tensorflow pandas
```

## Quickstart

Clone the repository and go inside the created folder. Then, the following commands permit to prepare the data and train a deep learning model with this data:

```bash
cd data/mnist
python generate_data.py --n_tasks 24 --s_frac 0.2 --test_tasks_frac 0.0 --seed 12345
cd ../..
python train.py mnist --n_rounds 4 --participation_probs 1.0 0.5 --unknown_participation_probs --bz 128 --lr 5e-3 --log_freq 1 --device cuda --optimizer sgd --server_optimizer history --swap_labels --swap_proportion 0.0 --logs_dir logs/mnist/p_0.5/h_0.0/fedvarp/lr_5e-3/seed_12 --seed 12 --verbose 0
tensorboard --logdir logs/mnist/p_0.5/h_0.0/fedvarp/lr_5e-3/seed_12
```

## Organization of this repository

This repository is organized as follows:
```bash
./
¦   activity_estimator.py        # Class ActivityEstimator: Computes aggregation weights based on the previous participation history
¦   activity_simulator.py        # Class ActivitySimulator: The activity of each client follows a Bernoulli random variable
¦   aggregator.py                # Class Aggregator: Aggregator dictates communications between clients (also NoCommunicationAggregator and CentralizedAggregator classes)
¦   client.py                    # Class Client: Implements a client
¦   client_sampler.py            # Class ClientSampler: Base class for clients sampler
¦   datasets.py                  # Class TabularDataset: Constructs a torch.utils.Dataset object from a pickle file; Class SubMNIST: Constructs a subset of MNIST dataset from a pickle file (also SubCIFAR10, SubCIFAR100, SubFEMNIST); Class CharacterDataset: Dataset for next character prediction; Function get_mnist: gets full (both train and test) MNIST dataset inputs and labels (also get_cifar10, get_cifar100 functions)
¦   history_tracker.py           # Class HistoryTracker: Class for tracking historical gradients. Designed for implementing FedVARP
¦   models.py                    # Classes for different ML models; Function get_mobilenet: creates MobileNet model with `num_classes` outputs
¦   train.py                     # Runs the function run_experiment; Also contains the function init_clients contained in this script
¦   
+---data/
¦   +---cifar10/
¦   ¦       generate_data.py     # Runs a function that downloads the data and splits the dataset among n_clients (three methods are available); see README for more information
¦   ¦       README.md
¦   ¦       utils.py
¦   ¦       
¦   +---mnist/
¦           generate_data.py     # Same as for cifar10 above
¦           README.md
¦           utils.py
¦           
+---learners/
¦       learner.py               # Class Learner: Responsible for training and evaluating a (deep-)learning model (also LanguageModelingLearner class)
¦       __init__.py
¦       
+---paper_experiments/
¦   +---cifar10/
¦   ¦       run.sh               # Runs the scripts generate_data.py and train.py with appropriate arguments
¦   ¦       
¦   +---mnist/
¦           run.sh               # Same as for cifar10 above
¦           
+---utils/
        args.py                  # Class ArgumentsManager: Defines options used during training and test time, also implements several helper functions such as parsing, printing, and saving the options (also TrainArgumentsManager class)
        constants.py             # Various constants
        metrics.py               # Various metrics
        optim.py                 # Class HistorySGD: Implements FedVARP; Class ProxSGD: Adaptation of torch.optim.SGD to proximal SGD
        torch_utils.py           # Function average_learners: computes the average of learners and store it into target_learner; Function copy_model: Copy learners_weights from target to source; Function copy_gradient: Copy param.grad.data from source to target; Function partial_average: performs a step towards aggregation for learners; Function differentiate_learner: Set the gradient of the model to be the difference between `target` and `reference` multiplied by `coeff`; Function simplex_projection: Compute the Euclidean projection on a positive simplex
        utils.py
        __init__.py
```