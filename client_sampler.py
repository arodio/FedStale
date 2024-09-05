import time
import numpy as np

from abc import ABC, abstractmethod


class ClientsSampler(ABC):
    r"""Base class for clients sampler

    Attributes
    ----------

    clients_weights_dict: Dict[int: float]
        maps clients ids to their corresponding weight/importance in the objective function

    participation_dict: Dict[int: float]
        maps clients ids to their corresponding participation probabilities

    activity_simulator: ActivitySimulator

    activity_estimator: ActivityEstimator

    unknown_participation_probs: Bool
        if True, participation probabilities are estimated through ActivityEstimator

    _time_step: int
        tracks the number of steps

    Methods
    ----------
    __init__

    sample_clients

    step

    """

    def __init__(
            self,
            clients,
            participation_probs,
            activity_simulator,
            activity_estimator,
            unknown_participation_probs,
            *args,
            **kwargs
    ):
        """

        Parameters
        ----------
        activity_simulator: ActivitySimulator

        activity_estimator: ActivityEstimator

        clients_weights_dict: Dict[int: float]

        """

        n_clients = len(clients)

        self.clients_weights_dict = self.get_client_weights_dict(clients)

        # self.participation_dict = self.get_participation_dict(n_clients, participation_probs)
        self.participation_dict = self.get_participation_dict(activity_simulator.participation_matrix)

        self.activity_simulator = activity_simulator

        self.activity_estimator = activity_estimator

        self.unknown_participation_probs = unknown_participation_probs

        if self.unknown_participation_probs:
            print("Estimate participation probabilities")

        self._time_step = -1

    @staticmethod
    def get_client_weights_dict(clients):
        """compute client weights as a proportion of training samples

        Parameters
        ----------
        clients : list

        Returns
        -------
        dict : key is client_id and value is client_weight.
        """
        clients_weights = np.array([client.n_train_samples for client in clients])
        clients_weights = clients_weights / clients_weights.sum()

        return {client.id: weight for client, weight in zip(clients, clients_weights)}

    # --- CHANGE HERE --- #

    @staticmethod
    def get_participation_dict(participation_matrix):
        """return a dictionary mapping client_id to participation_prob

        Parameters
        ----------
        participation_matrix : pandas.DataFrame (rows are clients-countries and columns are FL rounds)

        Returns
        -------
        dict : key is client_id and value is participation_prob
        """
        # client_probs = np.tile(participation_probs, n_clients // len(participation_probs))
        # return dict(enumerate(client_probs))

        row_means = participation_matrix.mean(axis=1)
        return dict(enumerate(row_means))

    def get_active_clients(self, c_round):
        """receive the list of active clients

        Parameters
        ----------

        c_round:

        Returns
        -------
            * List[int]
        """
        return self.activity_simulator.get_active_clients(c_round)

    def estimate_participation_probs(self, c_round):
        """receive the list of estimated client participations from the ActivityEstimator

        Parameters
        ----------

        c_round: int

        Returns
        -------
            * Dict[int]: a dictionary with client_id as keys and estimated participation probabilities as values
        """
        return self.activity_estimator.estimate_participation_probs(c_round)

    def step(self):
        """update the internal step of the clients sampler

        Parameters
        ----------

        Returns
        -------
            None
        """

        self._time_step += 1

    @abstractmethod
    def sample(self, c_round):
        """sample clients

        Parameters
        ----------
        c_round: int

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """
        pass


class UnbiasedClientsSampler(ClientsSampler):
    """
    Samples all active clients with aggregation weight inversely proportional to their participation prob
    """

    def sample(self, c_round):
        """implementation of the abstract method ClientSampler.sample for the UnbiasedClientSampler

        Samples all active clients with aggregation weight inversely proportional to their participation probability.

        Parameters
        ----------
        c_round: int

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """
        sampled_clients_ids, sampled_clients_weights = [], []

        active_clients = self.get_active_clients(c_round)
 
        if self.unknown_participation_probs:
            participation_probs = self.estimate_participation_probs(c_round)
            print('x --- x')
            print('---> Estimated participation probs', {key : round(participation_probs[key], 2) for key in participation_probs})
            print('---> Participation dict', {key : round(self.participation_dict[key], 2) for key in self.participation_dict})
            print('x --- x')
        else:
            participation_probs = self.participation_dict
            print('x --- x')
            print('---> Participation dict', {key : round(self.participation_dict[key], 2) for key in self.participation_dict})
            print('x --- x')

        for client_id in active_clients:
            sampled_clients_ids.append(client_id)
            # sampled_clients_weights.append(self.clients_weights_dict[client_id] / participation_probs[client_id]/7) #test
            sampled_clients_weights.append(self.clients_weights_dict[client_id] / participation_probs[client_id])
            # sampled_clients_weights.append(1 / participation_probs[client_id]) #test

        self.step()

        return sampled_clients_ids, sampled_clients_weights
    


class BiasedClientsSampler(ClientsSampler):
    """
    Samples all active clients with aggregation weight 1/nb_clients
    """

    def sample(self, c_round):
        """implementation of the abstract method ClientSampler.sample for the UnbiasedClientSampler

        Samples all active clients with aggregation weight 1/nb_clients.

        Parameters
        ----------
        c_round: int

        Returns
        -------
            * List[int]: indices of the sampled clients_dict
            * List[float]: weights to be associated to the sampled clients_dict
        """
        sampled_clients_ids, sampled_clients_weights = [], []

        active_clients = self.get_active_clients(c_round)

        # print('---> ', len(active_clients))
        # biased_weight = 1/len(active_clients)

        for client_id in active_clients:
            sampled_clients_ids.append(client_id)
            sampled_clients_weights.append(self.clients_weights_dict[client_id])

        self.step()

        return sampled_clients_ids, sampled_clients_weights