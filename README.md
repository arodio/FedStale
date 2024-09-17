# Readme

## Introduction

Our aim is to do **energy-mix aware** cross-silo Federated Learning (FL): 
- each client is in a different location, with its own energy-mix
- the availability of the clients depends on the energy-mix of their location, and its evolution through time.

The FL training is a succession of training rounds during which a certain group of clients (possibly all of them) is available to train collaboratively a global model based on their data.
An availability matrix specifying the availability of all clients for each training round is determined a priori, depending on the energy-mix data of the clients. Then, a specific FL algorithm is used to do the training constrained by this availablity matrix.

Various FL training algorithm exist. They have been built to answer to different settings in terms of clients availability, data heterogeneity, participation heterogeneity, etc. Each algorithm has specific advantages and drawbacks in terms of bias and convergence of the algorithm.

**Our approach:** On the one hand, we will design appropriate availability sequences for all clients, based on the estimated future Carbon Intensity (CI) data of the clients' locations. 
Each availability sequence is a sequences of 0 and 1, where each number refers to one round of the FL training, and 0 means that the client is not available for this round while 1 means that the client is available.
On the other hand, we will choose an appropriate FL algortihm to train the ML model based on these availability sequences.
The FL training plan (choice of clients at specific times) will need to make a balance between environmental cost, bias, catastrophic forgetting, etc.

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

Clone the repository and go inside the created folder. Checkout to the branch ``feat/CI_based_availability``.

**For building availability matrices:**

All files related to building availability matrices are lcoated in the folder ``build_availability_mat/``. The latest jupyter notebook used to build these matrices is ``build_availability_mat_3.piynb``. 

Availability matrices to be used for experiments should be pasted in the folder ``availability_matrices/``.

*/!\ Warning:* the file ``build_availability_mat_2.piynb`` takes a few minutes to run.

**For experiments:**

The file paper_experiments/mnist/run.sh permits to run the experiments.
First choose the parameters of you experiments by modifying the following variables in run.sh:
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


## Displaying the results of experiments

**Displaying the results with by hand:**

Run the jupyter notebook ``analyze_logs_v1.ipynb`` in the folder ``plots/``, figures will be saved in ``plots/figures/``.

**Displaying the results with tensorboard:**

Below change the path to the logs folder if needed.
```bash
set TF_ENABLE_ONEDNN_OPTS=0
tensorboard --logdir tensorboard --logdir logs/mnist_CI_based_availability/clients_7/
```

Go to *SCALARS* and use regular expression to filter results. For instance, 
`.*train\\global$`, `^local_mean\\.*train\\global$`, or `^local_mean\\.*alpha_0\.1.*train\\global$`.


## Appendix

### CI data

The 2022 Cabon Intensity (CI) data comes from *Electricity Maps*: csv files for different countries can be freely downloaded (https://www.electricitymaps.com/data-portal).
Electricity maps also proposes a paid plan providing access, through an API, to historical, real-time and **forecasted (over the next 24 hours)** data.

Description of the data:
- The granularity at which this data is available is one value per hour.
- The CI is expressed in gram of CO2 equivalents per Watt-hour, or gCO2eq/kWh.


### Organization of this repository

This repository is organized as follows:
```bash
./
¦   activity_estimator.py  # Class ActivityEstimator: Computes aggregation weights based on the previous participation history
¦   activity_simulator.py  # Class ActivitySimulator: The activity of each client follows a Bernoulli random variable
¦   aggregator.py          # Class Aggregator: Aggregator dictates communications between clients (also NoCommunicationAggregator and CentralizedAggregator classes)
¦   client.py              # Class Client: Implements a client
¦   client_sampler.py      # Class ClientSampler: Base class for clients sampler
¦   datasets.py            # Class TabularDataset: Constructs a torch.utils.Dataset object from a pickle file; Class SubMNIST: Constructs a subset of MNIST dataset from a pickle file (also SubCIFAR10, SubCIFAR100, SubFEMNIST); Class CharacterDataset: Dataset for next character prediction; Function get_mnist: gets full (both train and test) MNIST dataset inputs and labels (also get_cifar10, get_cifar100 functions)
¦   history_tracker.py     # Class HistoryTracker: Class for tracking historical gradients. Designed for implementing FedVARP
¦   models.py              # Classes for different ML models; Function get_mobilenet: creates MobileNet model with `num_classes` outputs
¦   train.py               # Runs the function run_experiment; Also contains the function init_clients contained in this script
¦   
+---data/
¦   +---cifar10/
¦   ¦       generate_data.py  # Runs a function that downloads the data and splits the dataset among n_clients (three methods are available); see README for more information
¦   ¦       README.md
¦   ¦       utils.py
¦   ¦       
¦   +---mnist/
¦           generate_data.py  # Same as for cifar10 above
¦           README.md
¦           utils.py
¦   
+---availability_matrices/    # Contains different availability matrices in csv files
¦           
+---learners/
¦       learner.py            # Class Learner: Responsible for training and evaluating a (deep-)learning model (also LanguageModelingLearner class)
¦       __init__.py
¦       
+---paper_experiments/
¦   ¦       
¦   +---mnist/
¦           run.sh               # Runs the scripts generate_data.py and train.py with mnist dataset
¦
+---plots/
¦       analyze_logs_v0.ipynb # Plot accuracy of experiments
¦       analyze_logs_v1.ipynb # Plot accuracy of experiments
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