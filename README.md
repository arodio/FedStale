# Green Federated Learning via Carbon-Aware Client and Time Slot Scheduling


This repository supports research on **carbon-aware federated learning (FL)**, where client participation and scheduling are optimized based on carbon intensity (CI) data. It includes tools for generating availability matrices, running FL experiments, and analyzing results.

**Table of Contents:**

1. Requirements
2. Quickstart
3. Running Experiments
4. Displaying Results
5. Appendix: Carbon Intensity Data, Repository Structure


## 1. Introduction

Training large-scale machine learning models incurs substantial carbon emissions. Federated Learning (FL), by distributing computation across geographically dispersed clients, offers a natural framework to leverage regional and temporal variations in Carbon Intensity (CI). This paper investigates how to reduce emissions in FL through carbon-aware client selection and training scheduling. 

We first quantify the emission savings of a carbon-aware scheduling policy that leverages slack time---permitting a modest extension of the training duration so that clients can defer local training rounds to lower-carbon periods.
We then examine the performance trade-offs of such scheduling which stem from  statistical heterogeneity among clients, selection bias in participation, and temporal correlation in model updates.
To leverage these trade-offs, we construct a carbon-aware scheduler that integrates slack time, $\alpha$-fair carbon allocation, and a global fine-tuning phase. Experiments on real-world CI data show that our scheduler outperforms slack-agnostic baselines, achieving higher model accuracy across a wide range of carbon budgets, with especially strong gains under tight carbon constraints.

## 2. Requirements

Create and activate a virtual environment. Example for linux:
```bash
python -m venv <your_venv_name>
source activate <path_to_your_venv_folder>/bin/activate
```

Install the required packages in the virtual environment: 
```bash
# For jupyter notebooks
pip install ipykernel 
ipython kernel install --user --name=<your_venv_name> 

# For training experiments
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 
pip install numpy scikit-learn tqdm tensorboard tensorflow pandas

# For CI data based availability matrices
pip install gekko cvxpy gurobipy Mosek

# For gaussian processes-based synthetic availability matrices creation
pip install tf-keras tensorflow-probability

# for plots and results analysis
pip install pandas matplotlib seaborn scipy scikit-learn

# Miscellaneous
pip install dataframe-image fpdf scikit-learn ipywidgets
```
In **VS Code**, click on 'Select kernel' and choose your virtual environment.

_Notes:_
- To install pytorch please refer to the official pytorch webpage
- Solver conflicts may occur with cvxpy. The notebooks `10_av_mat_analysis` run successfully with `cvxpy` and `Mosek`.


## 3. Quickstart

**3.1. Repository and Requirements**

Clone the repository and go inside the created folder. Checkout to the branch ``feat/model_quality``. Create and activate your virtual environment as specified in the previous section.

**3.2. Building Availability Matrices**

All files related to building availability matrices are located in the folder ``building_availability_matrices/``. Availability matrices to be used for experiments should be **pasted** in the folder ``availability_matrices/``.

**3.3. Experiments**

The file `paper_experiments/mnist/run.sh` can be used to run the experiments.
First select values for experiments' parameters by modifying variables in run.sh.

In the section "Parameters to choose for dataset generation" please set the following variable values:
```python
alpha="0.1" # distribution of data among clients: 0.1:non-iid, 100000:iid, 0: true iid
generate_data=true #true/false true will regenerate the clients' datasets
```

In the section "Parameters to choose for training" please set the following variable values:
- for the availability matrix:
```python
# Which availability matrix/matrices are you using?
availabilities="alphaF-0.7cb-10ft" # space separated names of availability matrices

# Does the av. mat. include a fine-tuning phase?
fine_tuning=10 # number of finetuning steps

# How many training rounds does it include?
n_rounds="100" # number of training rounds
```
- for the federated learning algorithm choice:
```python
# Which FL algorithm are you using?
fl_algo="fedavg" # space separated names of FL algorithms

# Is the algorithm unbiased?
biased="2" # 0:unbiased, 1:biased, 2:hybrid (unbiased except when all clients available)
```
- for the training algorithm:
```python
grad_clip_threshold="1.0" # Change this to None if you don't want to clip
verbose=2 # 0,1,2
seeds="42 78 84"
lrs="5e-2" # list of learning rates
```

Then, run the sh script as follows:

```bash
cd paper_experiments/mnist
./run.sh
```

_Notes:_
- the argument --by_labels_split is given to generate_data.py, and makes the distributions non-iid accross clients
- one chooses the level of non-iid ness with the argument --alpha, where 0.1 is strongly non-iid and 100000 is similar to iid
- hybrid means that we used an unbiased algorithm except for training rounds with all clients available

**3.4. Displaying Experiments Results**

**Displaying the results with the jupyter notebooks:**

Several jupyter notebooks within the folder ``plots/`` permit to display results. Graphs are saved in the folder ``plots/figures/``.

**Displaying the results with tensorboard:**

Below change the path to the logs folder if needed.
```bash
set TF_ENABLE_ONEDNN_OPTS=0
tensorboard --logdir tensorboard --logdir logs/mnist_CI_based_availability/clients_7/
```

Go to *SCALARS* and use regular expression to filter results. For instance, 
`.*train\\global$`, `^local_mean\\.*train\\global$`, or `^local_mean\\.*alpha_0\.1.*train\\global$`.


## 5. Appendix

### 5.1. CI Data

The 2022 Carbon Intensity (CI) data comes from *Electricity Maps*: csv files for different countries can be freely downloaded (https://www.electricitymaps.com/data-portal).
Electricity maps also proposes a paid plan providing access, through an API, to historical, real-time and **forecasted (over the next 24 hours)** data.

Description of the data:
- The granularity at which this data is available is one value per hour.
- The CI is expressed in gram of CO2 equivalents per Watt-hour, or gCO2eq/kWh.


### 5.2. Organization of This Repository

This repository is divided into 6 main folders:

- `availability_matrices` contains FL training schedules, also called 'availability matrices` here, that are to be used for training.
- `building_availability_matrices` contrains various Jupyter Notebooks to create availability matrices either based on Carbon Intensity data from Electricity maps, or artificially using Markov Chains or Gaussian processes. 
- `fl_training` contains the scripts for the federated learning training simulation. They will run the scripts generate_data.py and train.py with the mnist or cifar10 datasets.
- `logs` contains training experiments logs.
- `paper_experiments` contains sh script for running series of FL training experiments.
- `plots` contains various jupyter notebooks to analyse training experiments results in terms of accuracy.


**Organization of the `fl_training` folder:**
```bash
fl_training/
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
+---learners/
¦       learner.py            # Class Learner: Responsible for training and evaluating a (deep-)learning model (also LanguageModelingLearner class)
¦       __init__.py
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