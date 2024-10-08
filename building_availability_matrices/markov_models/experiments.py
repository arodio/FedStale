import pickle
import numpy as np
from building_availability_matrices.markov_models.utils import (
    CORR,
    UNCORR,
    CORR_FT,
    UNCORR_FT,
    NO_CLIENTS,
    gen_seq,
    cnstrct_trans_mat,
)
from matplotlib import pyplot as plt


EPS = 5e-1


def _get_alpha_beta(t, T, eps=1):
    """
    Given freq1 denotes first component of the stationary, return alpha, beta
    From the document, freq1 = b/(a+b)
    """
    assert type(t) == int
    assert type(T) == int
    assert t < T

    alpha = 1 - (T - t) * eps / T
    beta = 1 - t * eps / T

    return (alpha, beta)


def exp1(t, k=10, T=100, _eps=EPS):
    """
    Homogenous
    """

    exp_params = {
        CORR: (_get_alpha_beta(t=t, T=T, eps=_eps), float(t) / T, T),
        UNCORR: (_get_alpha_beta(t=t, T=T), float(t) / T, T),
        CORR_FT: (
            _get_alpha_beta(t=t - k, T=T - k, eps=_eps),
            float(t - k) / (T - k),
            T - k,
        ),
        UNCORR_FT: (
            _get_alpha_beta(t=t - k, T=T - k),
            float(t-k) / (T - k),
            T - k,
        ),
    }

    res = dict()
    for exp_type, val in exp_params.items():
        trans_mat=cnstrct_trans_mat(*val[0])
        _res = [gen_seq(init_state=np.random.binomial(n=1, p=1-val[1]), trans_mat=trans_mat, seq_len=val[2])
                     for _ in range(NO_CLIENTS)]
        if exp_type in [CORR_FT, UNCORR_FT]:
            _res = np.hstack((_res, np.ones((NO_CLIENTS, k))))
        res[exp_type] = np.array(_res)

    return res[CORR]
    # with open("avail_mat.pkl", "wb") as f:
    #     pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)

    # fig, axs = plt.subplots(4, 1, figsize=(6, 12))

    # # Plot each array using imshow
    # for i, ty in enumerate(res.keys()):
    #     axs[i].imshow(res[ty], cmap="binary")
    #     axs[i].set_title(ty)

    # # Adjust layout
    # plt.tight_layout()
    # plt.savefig("Avail_mat_markov.png")
