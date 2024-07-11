import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from  matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import matplotlib.dates as mdates

list_colors = ['blue', 'green', 'orange', 'red', 'purple', 'pink', 'yellow']

def plot_fft(CI_values, pos, max_pos, legend):

    # Generate time points
    t = np.linspace(0, len(CI_values), len(CI_values), False)

    # Perform Fourier Transform:
    # This function computes the one-dimensional n-point discrete Fourier Transform (DFT) 
    # with the efficient Fast Fourier Transform (FFT) algorithm.
    fft = np.fft.fft(CI_values)
    T = t[1] - t[0]  # Sample time
    # print('T: ', T)
    N = CI_values.size
    # print('N: ', N)



    # Plot the time series
    plt.figure(figsize=(12.5, 20))

    plt.subplot(max_pos, 2, pos)
    plt.plot(t, CI_values)
    plt.title('Time Series - ' + legend)
    plt.xlabel('Sample index (corresponds to hours)')
    plt.ylabel('Amplitude')

    # Plot the Fourier Transform
    plt.subplot(max_pos, 2, pos+1)

        
    # f = np.linspace(0, 1 / T, N) # Get the frequency range
    # plt.plot(f[:N // 2], np.abs(fft)[:N // 2] * 1 / N, '+')  # Only plot the frequencies in the positive half of the spectrum
    
    f = np.linspace(0, N-1, N) # Get the frequency range
    plt.plot(f[:N // 2], np.abs(fft)[:N // 2], '+')  # Only plot the frequencies in the positive half of the spectrum
    

    # plt.plot(f[:N // 2], np.abs(fft) * 1 / N, '+')  # Only plot the frequencies in the positive half of the spectrum
    # plt.plot(f[:N // 2], np.abs(fft)[:N // 2], '+')  # Only plot the frequencies in the positive half of the spectrum
    plt.title('DFT - ' + legend)
    plt.xlabel('Frequency component')
    plt.ylabel('Magnitude')

    # # Perform Fourier Transform
    # frequencies = np.fft.fftfreq(len(CI_values))
    # transformed_data = np.fft.fft(CI_values)

    # # Plot the absolute value of the transformed data
    # plt.plot(frequencies, np.abs(transformed_data))
    # plt.show()

def plot_availability_heatmap(title, similarity_matrix, key_word, folder):
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

    plt.title(key_word)
    
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

def plot_raw_data(ax, countries, df_dict, start_date, end_date):
    # plot raw data for all countries:
    # fig = plt.figure()
    # ax = plt.subplot()
    for country_idx, country in enumerate(countries):
        df_to_plot = df_dict[country][df_dict[country]['datetime'].between(start_date,end_date)]
        plt.plot(df_to_plot['datetime'].values, df_to_plot['CI_direct'].values, label=country, color = list_colors[country_idx])
        _ = plt.xticks(rotation=90)
    # ax.set_xlim([df_to_plot['datetime'].values[0], df_to_plot['datetime'].values[-1]])
    plt.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %h, %H:%M"))
    ax.xaxis.set_minor_formatter(mdates.DateFormatter("%d %h, %H:%M"))
    plt.ylabel('CI (gCO2eq/kWh)')
    # _ = plt.xticks(rotation=90)
    plt.xticks(rotation=45, ha='right')  # rotate x-axis labels to diagonal
    plt.grid()

def plot_heatmap(ax, similarity_matrix):
    """
    Plots a heatmap for the data in similarity matrix.
    The similarity matrix contains differences of average CI data over a certain time period.
    If the value is positive, it is deemed beneficial to move from the row country to the column country.
    If the value is negative, it is not deemed beneficial to move from the row country to the column country.
    Positive values are in green, while negative values are in red.
    """
    
    mask = np.eye(similarity_matrix.shape[0], dtype=bool)
    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 

    sns.heatmap(similarity_matrix.astype(int), annot=True, mask=mask, cmap=cmap, fmt='d', cbar=False, ax=ax) # create heatmap

    plt.subplots_adjust(left=0.325)  # adjust the position of the heatmap: increase the left margin
    ax.xaxis.tick_top() # move tick labels for the x-axis to the top
    plt.xticks(rotation=45, ha='left')  # rotate x-axis labels to diagonal

    # plt.savefig('heatmap.pdf', bbox_inches='tight') 


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

def display_CI(start_date, end_date, countries, df_dict):
    """ Function for plotting the hourly CI between start_date and end_date, for each country.
        Inputs:
            start_date, end_date: datetime objects of the form datetime(year, month, day, hour, minutes, seconds)
            countries: list of country names
    """
    plt.figure()
    ax = plt.gca()

    for key in countries:
        df_country = df_dict[key]
        # CI (Carbon Intensity) unit is in gCO2eq/kWh
        df_country[df_country['datetime'].between(start_date,end_date)].plot(y='CI_direct', x='datetime', ax=ax, figsize=(7, 3), label=key, grid=True)

    plt.savefig('figures/'+'raw_CI'+'.png', bbox_inches='tight')
    plt.show()

def last_day(i):
    """ Function for finding the number of days in a given month.
        Input: 
            i (int): month number
        Output:
            last_day (int): number of days in the month i
    """
    if (i%2 == 0 and i<=7) or (i%2 != 0 and i>7):
        last_day = 30
    else:
        last_day = 31
    if i == 2:
        last_day=28
    return last_day