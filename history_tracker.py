import torch

import numpy as np

from utils.torch_utils import *


class HistoryTracker:
    r""" Class for tracking historical gradients. Designed for implementing FedVARP.

    Attributes
    ----------
    history_dict : Dict[int: Learner]
        A dictionary that maps client IDs to their corresponding historical learners.

    Methods
    -------
    __init__
    average
    update_history

    """

    def __init__(self, clients_ids, history_learners, averaged_history_learner):
        """

        Parameters
        ----------
        clients_ids : list
        history_learners : list
        averaged_history_learner : Learner

        """
        self.history_dict = \
            {client_id: history_learner for client_id, history_learner in zip(clients_ids, history_learners)}

        for client_id in self.history_dict:
            self.history_dict[client_id].zero_grad()

        self.averaged_history_learner = averaged_history_learner
        self.averaged_history_learner.zero_grad()

    def average(self, sampled_clients_ids, clients_ids, sampled_clients_weights, clients_weights):
        """
        Computes the historical term used in FedVARP's aggregation step.

        Parameters
        ----------
        sampled_clients_ids : list
        clients_ids : list
        sampled_clients_weights : torch.tensor
        clients_weights : torch.tensor

        Returns
        -------
        Learner

        """
        inactive_clients_ids = set(clients_ids) - set(sampled_clients_ids)

        active_clients_weights = clients_weights[sampled_clients_ids] - sampled_clients_weights
        inactive_clients_weights = clients_weights[list(inactive_clients_ids)]

        active_history_learners = [self.history_dict[client_id] for client_id in sampled_clients_ids]
        inactive_history_learners = [self.history_dict[client_id] for client_id in inactive_clients_ids]

        weights = torch.cat([inactive_clients_weights, active_clients_weights])
        learners = inactive_history_learners + active_history_learners

        average_learners(
            learners=learners,
            target_learner=self.averaged_history_learner,
            weights=weights,
            average_params=False,
            average_gradients=True
        )

        return self.averaged_history_learner

    def update_history(self, new_history_dict):
        """
        Updates the history dictionary with new learners for the sampled clients.

        Parameters
        ----------
        new_history_dict : dict
            A dictionary containing client IDs and their corresponding new historical learners.

        """
        for client_id in new_history_dict:

            copy_gradient(
                target=self.history_dict[client_id].model,
                source=new_history_dict[client_id].model
            )
