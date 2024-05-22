
Organization of this repository:
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
¦   tree.txt
¦   
+---data/
¦   +---cifar10/
¦   ¦       generate_data.py     # Runs a function that downloads the data and splits the dataset among n_clients (three methods are available); see README for more information
¦   ¦       README.md
¦   ¦       utils.py
¦   ¦       
¦   +---mnist/
¦           generate_data.py
¦           README.md
¦           utils.py
¦           
+---learners/
¦       learner.py
¦       __init__.py
¦       
+---paper_experiments/
¦   +---cifar10/
¦   ¦       run.sh
¦   ¦       
¦   +---mnist/
¦           run.sh
¦           
+---utils/
        args.py
        constants.py
        metrics.py
        optim.py
        torch_utils.py
        utils.py
        __init__.py
```