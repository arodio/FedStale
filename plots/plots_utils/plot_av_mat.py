import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from  matplotlib.colors import LinearSegmentedColormap

def plot_av_mat(av_mat_name, folder):
    """
    Plot heatmap of availability matrix (countries x datetime list).
    Green: available, Red: not available.
    """

    df = pd.DataFrame(pd.read_csv("../availability_matrices/av-mat_"+av_mat_name+".csv", index_col=[0]))
    
    plt.figure(figsize=(7, 2))
    ax = plt.subplot()

    cmap=LinearSegmentedColormap.from_list('rg',["r", "w", "g"], N=256) 
    sns.heatmap(df.astype(int), annot=False, fmt='d', cbar=False, cmap=cmap, linewidths=0.5, linecolor='white', ax=ax) # create heatmap

    if not isinstance(df.columns[0], np.int64):
        plt.xticks(rotation=45, ha='right')  # rotate x-axis labels to diagonal
    xticks = ax.get_xticks()
    xticks = xticks[::2]
    ax.set_xticks(xticks) # set new xticks

    # if objective == None:
    #     plt.title(av_mat_name)
    # else:
    #     plt.title(av_mat_name+'/obj='+str(np.round(objective, 2)))
    plt.title(av_mat_name)
    plt.savefig('figures/'+folder+'/'+av_mat_name+'.png', bbox_inches='tight')
    plt.show()