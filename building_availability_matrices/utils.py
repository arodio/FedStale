import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.dates as mdates
from scipy.spatial.distance import hamming

import sys
sys.path.append(os.path.abspath('..'))

list_colors = ['blue', 'green', 'orange', 'red', 'purple', 'pink', 'yellow']

NO_CLIENTS = 7
CORR = "corr"
UNCORR = "uncorr"
CORR_FT = "corr_fine_tuning"
UNCORR_FT = "uncorr_fine_tuning"

from pandas import Series, crosstab
from sklearn.metrics import mutual_info_score


def hamming_sim(seq1, seq2=[], lag=1):
    """
    Computes one minus the normalized hamming similarity between two binary sequences.
    Hamming is a measure of similarity, not statistical correlation.
    The output is between 0 and 1.
    Close to 0 means that the sequences are not similar.
    Close to 1 means that the sequences are very similar.
    E.g., Two constant sequences with the same values will have the output 1.
    """
    # return 1 - 2*hamming(seq[:-lag], seq[lag:]) # rescaled hammming bwt -1 and 1

    if len(seq2) == 0:
        return 1 - hamming(seq1[:-lag], seq1[lag:])
    else:
        return 1 - hamming(seq1, seq2)

def pearson_corr(seq1, seq2=[], lag=1):
    """
    Computes the pearson correlation for two binary sequences.
    Is not well defined if any of the sequences is constant,
    it needs to have sequences with variations
    """
    if len(seq2) == 0:
        return np.corrcoef(seq1[:-lag], seq1[lag:])[0, 1]
    else:
        return np.corrcoef(seq1, seq2)[0, 1]

def mis_corr(seq1, seq2=[], lag=1):
    """
    Computes the Mutual Information Score between two binary.
    Is well defined even if any of the sequences is constant.
    However, for the constant and equal sequences, the output will be 0.
    """
    if len(seq2) == 0:
        return mutual_info_score(seq1[:-lag], seq1[lag:])
    else:
        return mutual_info_score(seq1, seq2)

# def av_mat_p_corr(availability_matrix):
#     """
#     Returns the list of p_corr of the rows of an availability matrix, 
#     and the mean of this list
#     """
#     countries = availability_matrix.index
#     p_corr_list = np.zeros(len(countries))
#     for i, country in enumerate(countries):
#         seq = availability_matrix.loc[country, :].values
#         p_corr_list[i] = pearson_corr(seq)
#     return p_corr_list, np.mean(p_corr_list)

def av_mat_p_corr(availability_matrix):
    """
    Returns the list of ...
    """
    countries = availability_matrix.index

    t_corr_list = np.zeros(len(countries))
    for i, country in enumerate(countries):
        seq = availability_matrix.loc[country, :].values
        if sum(seq)==0 or sum(seq)==len(seq): # set value 1 for constant sequences
            t_corr_list[i] = 1
        else:
            t_corr_list[i] = pearson_corr(seq)
    t_corr_mean = np.mean(t_corr_list)

    sp_corr_dict = {}
    for idx, country_a in enumerate(countries): 
        for country_b in countries[idx + 1:]:
            seq_a = availability_matrix.loc[country_a, :].values
            seq_b = availability_matrix.loc[country_b, :].values
            # if sum(seq)==0 or sum(seq)==len(seq):
            sp_corr_dict[str(country_a)+'-'+str(country_b)] = pearson_corr(seq_a, seq_b)
    sp_corr_list = np.array(list(sp_corr_dict.values()))
    sp_corr_mean = np.mean(sp_corr_list)
    return t_corr_list, t_corr_mean, sp_corr_dict, sp_corr_mean

def hamming_similarity(availability_matrix):
    """
    Returns the list of ...
    """
    countries = availability_matrix.index

    t_corr_list = np.zeros(len(countries))
    for i, country in enumerate(countries):
        seq = availability_matrix.loc[country, :].values
        t_corr_list[i] = hamming_sim(seq)
    t_corr_mean = np.mean(t_corr_list)

    sp_corr_dict = {}
    for idx, country_a in enumerate(countries): 
        for country_b in countries[idx + 1:]:
            seq_a = availability_matrix.loc[country_a, :].values
            seq_b = availability_matrix.loc[country_b, :].values
            # sp_corr_dict[str(country_a)+'-'+str(country_b)] = 1 - 2*hamming(seq_a, seq_b) # rescaled hamming btw -1 and 1
            sp_corr_dict[str(country_a)+'-'+str(country_b)] = hamming_sim(seq_a, seq_b)
    sp_corr_list = np.array(list(sp_corr_dict.values()))
    sp_corr_mean = np.mean(sp_corr_list)
    return t_corr_list, t_corr_mean, sp_corr_dict, sp_corr_mean

def mis(availability_matrix):
    """
    Returns the list of ...
    """
    countries = availability_matrix.index

    t_corr_list = np.zeros(len(countries))
    for i, country in enumerate(countries):
        seq = availability_matrix.loc[country, :].values
        t_corr_list[i] = mis_corr(seq)
    t_corr_mean = np.mean(t_corr_list)

    sp_corr_dict = {}
    for idx, country_a in enumerate(countries): 
        for country_b in countries[idx + 1:]:
            seq_a = availability_matrix.loc[country_a, :].values
            seq_b = availability_matrix.loc[country_b, :].values
            # p_corr_dict[str(country_a)+'-'+str(country_b)] = np.corrcoef(seq_a, seq_b)[0,1]
            sp_corr_dict[str(country_a)+'-'+str(country_b)] = mis_corr(seq_a, seq_b)
    sp_corr_list = np.array(list(sp_corr_dict.values()))
    sp_corr_mean = np.mean(sp_corr_list)
    return t_corr_list, t_corr_mean, sp_corr_dict, sp_corr_mean

def emp_trans_mat(seq):
    """
    Estimate the transition matrix given a realisation
    """
    return crosstab(
        Series(seq[:-1], name="from"), Series(seq[1:], name="to"), normalize=0
    ).to_numpy()

# def lambda_2(trans_mat):
#     """
#     Estimates the second (largest) eigenvalue of a given transition matrix
#     """
#     return trans_mat[0,0] + trans_mat[1, 1] - 1

def lambda_2(seq):
    """
    Estimates the second (largest) eigenvalue of the transition matrix
    associeted with the sequence of states seq
    """
    trans_mat = emp_trans_mat(seq)
    if len(trans_mat) < 2:
        trans_mat =  np.array([[1., 0.], [0., 1.]])
    return trans_mat[0,0] + trans_mat[1, 1] - 1

def av_mat_corr(availability_matrix):
    """
    Returns the list of second (largest) eigenvalues of the transition matrix
    associated with the rows of an availability matrix, and the mean of this list
    """
    countries = availability_matrix.index
    lambda_2_list = np.zeros(len(countries))
    for i, country in enumerate(countries):
        seq = availability_matrix.loc[country, :].values
        lambda_2_list[i] = lambda_2(seq)
    return lambda_2_list, np.mean(lambda_2_list)


def plot_availability_heatmap(title, similarity_matrix, key_word, folder, objective=None):
    """
    Plot heatmap of availability matrix (countries x datetime list).
    Green: available, Red: not available.
    """
    
    plt.figure(figsize=(7, 2))
    ax = plt.subplot()

    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
    sns.heatmap(similarity_matrix.astype(int), annot=False, fmt='d', cbar=False, cmap=cmap, linewidths=0.5, linecolor='white', ax=ax) # create heatmap

    if not isinstance(similarity_matrix.columns[0], np.int64):
        plt.xticks(rotation=45, ha='right')  # rotate x-axis labels to diagonal
    xticks = ax.get_xticks()
    xticks = xticks[::2]
    ax.set_xticks(xticks) # set new xticks

    if objective == None:
        plt.title(key_word)
    else:
        plt.title(key_word+'/obj='+str(np.round(objective, 2)))
    
    plt.savefig(folder+'/'+key_word+'.png', bbox_inches='tight')
    plt.show()

def get_CI_values(_dfs, country, start_date, end_date):
    """ 
    Returns array of CI values of country between start_date and end_date.
    Start_date and end_date are datetime objects.
    Country is a string.
    """
    df_to_plot = _dfs[country][_dfs[country]['datetime'].between(start_date,end_date)]
    return df_to_plot['CI_direct'].values

def get_datetime_values(_dfs, country, start_date, end_date):
    """ 
    Returns array of datetime values of country between start_date and end_date.
    Start_date and end_date are datetime objects.
    Country is a string.
    """
    df_to_plot = _dfs[country][_dfs[country]['datetime'].between(start_date,end_date)]
    return df_to_plot['datetime'].values

def load_data():
    """
    Loads the CI data in df_dict a dictionary where key=country, value=dataframe of CI data.
    Returns: df_dict
    The columns of each dataframe are datetime, CI_direct, CI_LA.
    The unit of the CI data is: gCO2eq/kWh.
    """

    # prepare links to the data csv files
    folder = 'historical_data'
    _paths = {}
    _paths['Germany'] = os.path.join(folder,'DE_2022_hourly.csv')
    _paths['Ireland'] = os.path.join(folder,'IE_2022_hourly.csv')
    _paths['Great Britain'] = os.path.join(folder,'GB_2022_hourly.csv')
    _paths['France'] = os.path.join(folder,'FR_2022_hourly.csv')
    _paths['Sweden'] = os.path.join(folder,'SE-SE3_2022_hourly.csv')
    _paths['Finland'] = os.path.join(folder,'FI_2022_hourly.csv')
    _paths['Belgium'] = os.path.join(folder,'BE_2022_hourly.csv')

    # loading the data in a pandas dataframe
    df_dict = {}
    usecols=['Datetime (UTC)','Carbon Intensity gCO₂eq/kWh (direct)','Carbon Intensity gCO₂eq/kWh (LCA)']
    for key in _paths.keys():
        df_dict[key] = pd.DataFrame(pd.read_csv(_paths[key], usecols=usecols, parse_dates=['Datetime (UTC)']))
        df_dict[key] = df_dict[key].rename(columns={'Datetime (UTC)': 'datetime'})
        df_dict[key] = df_dict[key].rename(columns={'Carbon Intensity gCO₂eq/kWh (direct)': 'CI_direct'})
        df_dict[key] = df_dict[key].rename(columns={'Carbon Intensity gCO₂eq/kWh (LCA)': 'CI_LCA'})

    return df_dict