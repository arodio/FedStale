# Readme

## Requirements

Create and activate a virtual environment. For example, with linux:
```bash
python -m venv <your_venv_name>
source activate <path_to_your_venv_folder>/bin/activate
```

Then, install the following in the virtual environment: 
```bash
pip install ipykernel
ipython kernel install --user --name=<your_venv_name>
pip install pandas matplotlib gekko cvxpy gurobipy PySCIPOpt                      # for CI data analysis
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # for experiments
pip install numpy scikit-learn tqdm tensorboard tensorflow pandas                 # for experiments
```
Then, in VS Code: click on 'Select kernel': choose your virtual environment.

## Quickstart

Clone the repository and go inside the created folder. 
The file paper_experiments/mnist/run.sh permits to run the experiments.
First choose the parameters of you expeiments by modifying the following variables in run.sh:
```python
alpha="0.1"                     # 0.1:non-iid, 100000:iid, 0: true iid
availabilities="opt-pb3-stage2" # list of availability matrices names (separeted by a space)
fl_algo="fedavg"                # list of FL algorithms (separeted by a space)
biased="0"                      # 0:unbiased, 1:biased, 2:hybrid (=unbiased except when all clients available)
```
*Remark:* In run.sh, the argument --by_labels_split is given to generate_data.py, which makes the distributions non-iid accross clients. Then, one chooses the level of non-iid ness with the argument --alpha, where 0.1 is strongly non-iid and 100000 is similar to iid.

Then, run the sh file as follows:

```bash
cd paper_experiments/mnist
sh run.sh
```


## Displaying the results

### Displaying the results with by hand

TODO

### Displaying the results with tensorboard

Below change the path to the logs folder if needed.
```bash
set TF_ENABLE_ONEDNN_OPTS=0
tensorboard --logdir tensorboard --logdir logs/mnist_CI_based_availability/clients_7/
```

Go to *SCALARS* and use regular expression to filter results. For instance, 
`.*train\\global$`, `^local_mean\\.*train\\global$`, or `^local_mean\\.*alpha_0\.1.*train\\global$`.


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
+---availability_matrices/           # Contains different availability matrices in csv files
¦           
+---learners/
¦       learner.py               # Class Learner: Responsible for training and evaluating a (deep-)learning model (also LanguageModelingLearner class)
¦       __init__.py
¦       
+---paper_experiments/
¦   ¦       
¦   +---mnist/
¦           run.sh               # Runs the scripts generate_data.py and train.py with mnist dataset
¦
+---plots/
¦           analyze_logs_v1.ipynb # Plot accuracy of experiments
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