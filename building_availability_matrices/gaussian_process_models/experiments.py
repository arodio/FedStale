from gaussian_process_models.utils import get_cov_mat
from utils import (
    NO_CLIENTS,
    CORR,
    UNCORR,
    CORR_FT,
    UNCORR_FT,
)
import numpy as np


def _get_avail_mat(
    cov,
    seq_len,
    n_clients,
    freq1,
):

    y = np.random.multivariate_normal(mean=np.zeros(seq_len), cov=cov, size=n_clients)
    thresh = np.quantile(y, [1 - freq1], method="higher", axis=1).T
    y_bin = (y >= thresh).astype(np.int8)

    return y_bin


# TODO: Make changes to function definition to take in arguments @daniel
def exp1(freq1, k=10, seq_len=100, n_clients=NO_CLIENTS):
    """
    Fetch covariance matrix as numpy array basis the kernel name
    """

    assert 0 <= freq1 <= 1

    xlim = (-3, 3)
    x = np.expand_dims(np.linspace(*xlim, seq_len), 1)
    corr_cov = get_cov_mat(
        x=x,
        periodic_length_scale=1.0,
        period=1.0,
        amplitude=1.0,
        local_length_scale=1.0,
    )
    uncorr_cov = get_cov_mat(
        x=x,
        periodic_length_scale=1.0,
        period=1.0,
        amplitude=1.0,
        local_length_scale=0.01,
    )

    avail_mat = {
        CORR: _get_avail_mat(
            cov=corr_cov,
            seq_len=seq_len,
            n_clients=n_clients,
            freq1=freq1,
        ),
        UNCORR: _get_avail_mat(
            cov=uncorr_cov,
            seq_len=seq_len,
            n_clients=n_clients,
            freq1=freq1,
        ),
        CORR_FT: np.hstack(
            (
                _get_avail_mat(
                    cov=corr_cov[:-k, :-k],
                    seq_len=seq_len - k,
                    n_clients=n_clients,
                    freq1=(freq1 * seq_len - k) / (seq_len - k),
                ),
                np.ones((n_clients, k)),
            )
        ),
        UNCORR_FT: np.hstack(
            (
                _get_avail_mat(
                    cov=uncorr_cov[:-k, :-k],
                    seq_len=seq_len - k,
                    n_clients=n_clients,
                    freq1=(freq1 * seq_len - k) / (seq_len - k),
                ),
                np.ones((n_clients, k)),
            )
        ),
    }

    return avail_mat
