import numpy as np


class ActivityEstimator:
    def __init__(self, participation_matrix, cutoff=1e8):
        """
        Computes aggregation weights based on the previous participation history.

        Implements
         `A Lightweight Method for Tackling Unknown Participation Statistics in Federated Averaging`.
            (https://arxiv.org/abs/2306.03401)

            Follows implementation from https://github.com/IBM/fedau/tree/main

        Parameters:
        - participation_matrix: 2-D array, participation history of clients,
          where rows are clients and columns are rounds of participation (0 or 1).
        - cutoff: float, cutoff interval length beyond which participation is assumed.
        """
        n_clients, n_rounds = participation_matrix.shape  # Number of clients and rounds
        self.weights_matrix = np.ones((n_clients, n_rounds))  # Initialize weights to 1 for t=0

        for n in range(n_clients):  # For each client
            n_intervals = 0  # Number of (possibly cutoff) participation intervals collected
            length_current_interval = 0  # Length of the last interval being computed

            for t in range(1, n_rounds):  # Starting from t=1
                length_current_interval += 1  # Increment the interval length

                # Check if the client participated in the previous round or if the cutoff is reached
                if participation_matrix[n, t - 1] == 1 or length_current_interval == cutoff:
                    length_interval = length_current_interval  # Final interval computed

                    # Compute the weight
                    if n_intervals == 0:
                        self.weights_matrix[n, t] = length_interval
                    else:
                        self.weights_matrix[n, t] = (n_intervals * self.weights_matrix[n, t - 1] + length_interval) / (
                                    n_intervals + 1)

                    n_intervals += 1  # Increment the count of intervals
                    length_current_interval = 0  # Reset the interval length
                else:
                    # If the client didn't participate, carry over the previous weight
                    self.weights_matrix[n, t] = self.weights_matrix[n, t - 1]

    def estimate_clients_weights(self, c_round):
        """returns estimated aggregation weights of clients at current round

        Parameters
        ----------

        c_round:

        Returns
        -------
            * List[int]
        """
        return self.weights_matrix[:, c_round].tolist()
