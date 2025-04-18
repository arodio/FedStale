import pandas as pd
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from gaussian_process_models.experiments import exp1, exp2, exp3
from building_availability_matrices.utils import (
    NO_CLIENTS,
    CORR,
    UNCORR,
    CORR_FT,
    UNCORR_FT,
)

COUNTRIES = [i+1 for i in range(NO_CLIENTS)]
MAIN_FOLDER = 'availability_matrices/av-mat-new'
N_ROUNDS=50

class GP():

    def __init__(self, freq=0.5, k=10, n_rounds=N_ROUNDS, countries=COUNTRIES, 
                 n_clients=NO_CLIENTS, out_folder=MAIN_FOLDER):
        self.n_rounds=n_rounds
        self.countries=countries
        self.freq=freq
        self.n_clients=n_clients
        self.k=k 
        self.out_folder=out_folder

        self.formatted_array = list(range(self.n_rounds))

        if type(self.freq)==float:
            self.res = exp1(freq1=self.freq, k=self.k, seq_len=self.n_rounds, n_clients=self.n_clients)
        else:
            self.res = exp2(freq_seq=self.freq, k=self.k, seq_len=self.n_rounds, n_clients=self.n_clients)

        self.trad_C2c = dict(zip([CORR, UNCORR, CORR_FT, UNCORR_FT], ['tcsu', 'tusu', 'tcsu-ft', 'tusu-ft']))
        self.trad_c2C = dict(zip(['tcsu', 'tusu', 'tcsu-ft', 'tusu-ft'], [CORR, UNCORR, CORR_FT, UNCORR_FT]))

    def create_av_mat(self, corr_ft_type):
        """
        Create an availablity matrix for the correlation type and with or 
        without fine-tuning, according to the input corr_ft_type.
        corr_ft_type must be one of: 'tcsu', 'tusu', 'tcsu-ft', 'tusu-ft'.
        """

        av_mat=self.res[self.trad_c2C[corr_ft_type]]
        av_df = pd.DataFrame(av_mat, index = self.countries, columns = self.formatted_array)

        if type(self.freq)==float:
            freq_str = str(int(self.freq*100))
        else:
            freq_str_list = [str(int(f*100)) for f in self.freq]
            freq_str = ''.join(freq_str_list)
        key_word = 'gp'+str(N_ROUNDS)+'-'+freq_str+'-'+corr_ft_type.split('-')[0]
        if len(corr_ft_type.split('-'))>1:
            key_word+='-'+str(self.k)+'ft'
        self.plot_availability_heatmap(av_df, key_word)
        av_df.to_csv(self.out_folder+'/av-mat_'+key_word+'.csv', columns=self.formatted_array)

    def create_all_av_mat(self):
        """
        Create all possible availablity matrices: with/without time correlation
        and fine-tuning.
        """
        for corr_ft_type in ['tcsu', 'tusu', 'tcsu-ft', 'tusu-ft']:
            self.create_av_mat(corr_ft_type)

    def plot_availability_heatmap(self, similarity_matrix, key_word):
        """
        Plot heatmap of availability matrix (countries x datetime list).
        Green: available, Red: not available.
        """
        plt.figure(figsize=(7, 2))
        ax = plt.subplot()
        cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
        sns.heatmap(similarity_matrix.astype(int), annot=False, fmt='d', cbar=False, cmap=cmap, 
                    linewidths=0.5, linecolor='white', ax=ax) # create heatmap
        if not isinstance(similarity_matrix.columns[0], np.int64):
            plt.xticks(rotation=45, ha='right')  # rotate x-axis labels to diagonal
        xticks = ax.get_xticks()
        xticks = xticks[::2]
        ax.set_xticks(xticks) # set new xticks
        plt.title(key_word)
        plt.savefig(self.out_folder+'/'+key_word+'.png', bbox_inches='tight')
        plt.show()
