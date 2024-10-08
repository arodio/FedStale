from pandas import Series, crosstab
import numpy as np

"""
Considering a markov chain with only two states 0, 1
Constructing the transition matrix T as
| \\alpha    1 - \\alpha |
| 1- \\beta  \\beta      |
"""


def cnstrct_trans_mat(alpha, beta):
    """
    Given alpha and beta, construct the transition matrix
    """
    return np.array([[alpha, 1 - alpha], [1 - beta, beta]])


def second_eigen_val(alpha, beta):
    """
    Second largest eigenvalue in magnitude. Only for specific case when there are only two states
    """
    return alpha + beta - 1


def _gen_state(cur_state, trans_mat):
    """
    Generate the next state given the current state and transition matrix
    """
    return np.random.choice([0, 1], 1, p=trans_mat[cur_state, :])[0]


def gen_seq(init_state, trans_mat, seq_len):
    """
    Generate sequence of events given initial state
    """

    assert init_state in [0, 1]
    assert np.sum(trans_mat) == trans_mat.shape[0]

    seq = [init_state]
    for ix in range(seq_len - 1):
        _ = _gen_state(cur_state=seq[-1], trans_mat=trans_mat)
        seq.append(_)

    return np.squeeze(np.array(seq))


def emp_trans_mat(seq):
    """
    Estimate the transition matrix given a realisation
    """
    return crosstab(
        Series(seq[:-1], name="from"), Series(seq[1:], name="to"), normalize=0
    ).to_numpy()


# TODO: delete this
def gen_eigv_stny_dist(eps, no_samples):
    """
    Given a stationary distribution, generate the second largest eigenvalue of the transition matrix.
    Second largest eigenvalue is given by \lambda_2 = \\alpha + \\\beta - 1. From perron-ferron's formula
    |\lambda_2| <= 1.

    \pi_c = [\eps 1-\eps]^T which is also given by

    \pi_c = [\\frac{1-\\beta}{2-\\alpha-\\beta} \\frac{1-\\alpha}{2-\\alpha-\\beta}]^T

    Thus, \\alpha = \eps + (1 - \eps)*\lambda_2
    or \\beta = 1 - \eps + \eps*\lambda_2

    :param eps: first component of stationarity distribution
    :param no_samples: # of eigenvalues to return

    :return: list of tuple of eigenvalue, \\alpha
    :rtype: list
    """

    lst_lambda2 = np.linspace(max(-1.0, eps / (eps - 1)) + 0.02, 1.0 - 0.01, no_samples)
    lst_alpha = np.vectorize(lambda x: eps + (1 - eps) * x)(lst_lambda2)

    return list(zip(lst_lambda2, lst_alpha))
