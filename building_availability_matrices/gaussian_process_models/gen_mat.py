from gaussian_process_models.experiments import exp1, exp2
from utils import NO_CLIENTS



def hom_participation(freq, time_corr, spatial_corr, folder, k=10, seq_len=100, n_clients=NO_CLIENTS):
    if time_corr:
        if spatial_corr:
            av_mat = exp1(freq1=freq,k=k, seq_len=seq_len, n_clients=n_clients)
        else:
            pass
    elif spatial_corr:
        pass
    else:
        pass