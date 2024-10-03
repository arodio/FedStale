import pandas as pd
from markov_models.utils import gen_avail_mat

CLIENTS = {
    "C0": (0.4, 0.6),
    "C1": (0.4, 0.6),
    "C2": (0.4, 0.6),
    "C3": (0.4, 0.6),
    "C4": (0.4, 0.6),
    "C5": (0.4, 0.6),
    "C6": (0.4, 0.6),
}

# CLIENTS_CORR = {
#     'C0' : (0.04, 0.06),
#     'C1' : (0.04, 0.06),
#     'C2' : (0.04, 0.06),
#     'C3' : (0.04, 0.06),
#     'C4' : (0.04, 0.06),
#     'C5' : (0.04, 0.06),
#     'C6' : (0.04, 0.06)
# }

CLIENTS_CORR = {
    "C0": (0.94, 0.96),
    "C1": (0.94, 0.96),
    "C2": (0.94, 0.96),
    "C3": (0.94, 0.96),
    "C4": (0.94, 0.96),
    "C5": (0.94, 0.96),
    "C6": (0.94, 0.96),
}


def save_avail_mat(avail_mat, key_word):
    pd.DataFrame(
        data=avail_mat,
        columns=[f"t{i}" for i in range(100)],
        index=[f"C_{i}" for i in range(7)],
    ).to_csv(
        f"building_availability_matrices/availability_matrices/av-mat-3/av-mat_markov_spots_{key_word}.csv"
    )


avail_mat = gen_avail_mat(CLIENTS_CORR, 100)
save_avail_mat(avail_mat=avail_mat, key_word="corr")

avail_mat = gen_avail_mat(CLIENTS, 100)
save_avail_mat(avail_mat=avail_mat, key_word="uncorr")
