import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.dates as mdates

import sys
sys.path.append(os.path.abspath('..'))

list_colors = ['blue', 'green', 'orange', 'red', 'purple', 'pink', 'yellow']


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