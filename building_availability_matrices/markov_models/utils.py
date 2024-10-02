from pandas import Series, crosstab
import numpy as np

"""
Considering a markov chain with only two states 0, 1
Constructing the transition matrix T as
| \alpha    1 - \alpha |
| 1- \beta  \beta      |
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
    return np.random.choice([0, 1], 1, p=trans_mat[cur_state,:][0])


def _gen_seq(init_state, trans_mat, seq_len):
    """
    Generate sequence of events given initial state
    """

    assert init_state in [0, 1]
    assert np.sum(trans_mat) == trans_mat.shape[0]

    _tmp = [init_state]
    for ix in range(seq_len-1):
        _ = _gen_state(cur_state=_tmp[-1], trans_mat=trans_mat)
        _tmp.append(_)

    return np.squeeze(np.array(_tmp))


def emp_trans_mat(seq):
    """
    Estimate the transition matrix given a realisation
    """
    return crosstab(
        Series(seq[:-1], name="from"), Series(seq[1:], name="to"), normalize=0
    ).to_numpy()


def gen_avail_mat(client_alpha: dict, seq_len: int):
    return np.array(
        [
            _gen_seq(
                init_state=np.random.binomial(size=1, n=1, p=0.5),
                trans_mat=cnstrct_trans_mat(*v),
                seq_len=seq_len,
            )
            for k, v in client_alpha.items()
        ]
    )
