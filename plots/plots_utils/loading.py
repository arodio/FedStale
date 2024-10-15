from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

def parse_tf_events_file(events_path, tag, time_horizon=None):
    ea = EventAccumulator(events_path).Reload()
    # print(ea.Tags())
    tag_values, steps = [], []
    for event in ea.Scalars(tag):
        if time_horizon is None or event.step <= time_horizon:
            tag_values.append(event.value)
            steps.append(event.step)
    return steps, tag_values

class ExperimentConfig:
    def __init__(self, base_path, experiment, seeds, algorithms, events, lr_list,
                 alphas, n_clients_list, availabilities, n_rounds, participations, biased_list, train_test):
        """
        base_path: path to the folder logs/
        experiment: name of exp (str: mnist_CI_based_availability)
        seeds: list of seeds
        algorithms: list of algorithms
        events: list of events (e.g. ["global"])
        # b_values:
        lr_list: list of learning rates
        alphas: list of alpha values (for the iid - non iid -ness)
        n_clients_list: list of number of clients
        availabilities: list of availability types
        n_rounds: number of FL rounds
        participations: list with "unknown" and/or "known" (whether we work with the participation probabilities estimator or the true values)
        biased: list of elements of {0, 1} (0=unbiased, 1=biased)
        train_test: "train" or "test"
        """
        self.base_path = base_path
        self.experiment = experiment
        self.seeds = seeds
        self.algorithms = algorithms
        self.events = events
        # self.b_values = b_values
        self.lr_list = lr_list
        self.alphas = alphas
        self.n_clients_list = n_clients_list
        self.availabilities = availabilities
        self.n_rounds = n_rounds
        self.participations = participations
        self.biased_list = biased_list
        self.train_test = train_test

    def get_event_dir(self, algo, lr, seed, event, alpha, n_clients, availability, n_rounds, participation, biased, train_test):
        """
        Returns the path to the saved data corresponding to the parameters given as inputs to this function.
        Intputs:
        algorithm: algorithm (str:fedavg, defvarp or fedstale)
        lr: learning rate (str)
        b: beta (str)
        seed:
        event:
        alpha:
        n_clients:
        availability:
        n_rounds:
        participation:
        biased: 0 or 1 (str)
        train_test: "train" or "test"
        """
        path = f"{self.base_path}/{self.experiment}/{availability}"
        # path += f"/{algo}/b_{b}" if algo == "mixture" else f"/{algo}" # in case we vary beta
        path += f"/biased_{biased}/{algo}"
        path += f"/alpha_{alpha}/lr_{lr}/seed_{seed}/{train_test}/{event}"

        path = os.path.join(self.base_path, self.experiment, availability,
                            "biased_"+biased, algo, "alpha_"+alpha,
                            "lr_"+lr, "seed_"+seed, train_test, event)

        # path = '/'.join([self.base_path, self.experiment, availability,
        #                     "biased_"+biased, algo, "alpha_"+alpha,
        #                     "lr_"+lr, "seed_"+seed, "train", event])

        return path

# Load and Process Experiment Results
def load_experiment_results(config):
    results = [] 
    for lr in tqdm(config.lr_list, desc="Processing experiments"):
        # time_horizon = time_horizons[p]  
        for algorithm in config.algorithms:
            # b_loop = config.b_values if algorithm == 'mixture' else [None] # in case we vary beta
            # for b in b_loop:
            for event in config.events:
                for seed in config.seeds:
                    for a in config.alphas:
                        for n_c in config.n_clients_list:
                            for av in config.availabilities:
                                for part in config.participations:
                                    for biased in config.biased_list:

                                        event_dir = config.get_event_dir(algorithm, lr, seed, 
                                                                            event, a, n_c, av, 
                                                                            config.n_rounds, part, biased, config.train_test) 

                                        # print(event_dir)                                   
                                        files = os.listdir(event_dir)
                                        # print(files)

                                        if os.path.exists(event_dir):
                                            # print('x')
                                            # _, values = parse_tf_events_file(event_dir, tag="Test/Metric", time_horizon=time_horizon)
                                            _, test_accuracy_values = parse_tf_events_file(event_dir, tag="Test/Metric")
                                            _, test_loss_values = parse_tf_events_file(event_dir, tag="Test/Loss")
                                            _, train_accuracy_values = parse_tf_events_file(event_dir, tag="Train/Metric")
                                            _, train_loss_values = parse_tf_events_file(event_dir, tag="Train/Loss")
                                            ### tag can be: 'Train/Loss', 'Train/Metric', 'Test/Loss', 'Test/Metric'
                                            max_accuracy = np.array(test_accuracy_values).max() * 100
                                            results.append({
                                                "algorithm": algorithm, "availability": av,
                                                "alpha": a, "participation": part,
                                                "max_test_accuracy": float(max_accuracy),
                                                "test_accuracy": "Test/Metric",
                                                "test_loss": "Test/Loss",
                                                "train_accuracy": "Train/Metric",
                                                "train_loss": "Train/Loss",
                                                "seed": seed,
                                                "lr": lr, "event": event, "n_clients": n_c,
                                                "biased": biased
                                                # "event_dir": event_dir
                                            })

                                            # "b": float(b) if b else np.nan # in case we vary beta
    return pd.DataFrame(results)