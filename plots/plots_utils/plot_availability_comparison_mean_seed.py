import matplotlib.pyplot as plt
from plots_utils.loading import parse_tf_events_file
import pandas as pd

def plot_availability_comparison_mean_seed(config, res, metric, folder):
    """
    Creates one plot for each setting (lr, algo, event, seet, a, n_c, biased, part).
    Each plot contains one curve per availability matrix.
    Each setting must have been run with all availability matrix of the list.
    """
    
    xvalues = [i for i in range(int(config.n_rounds)+1)]

    for lr in config.lr_list:
        for algo in config.algorithms:
            for event in config.events:
                for a in config.alphas:
                    for n_c in config.n_clients_list:
                        for biased in config.biased_list:
                            # for av in config.availabilities:
                            for part in config.participations:


                                    res_tmp = res[(res.lr == lr) & (res.algorithm == algo) & (res.event == event) &
                                                (res.alpha == a) & (res.n_clients == n_c) &
                                                (res.participation == part)]
                                    # display(res_tmp) # this is what we are going to compare

                                    fig, ax = plt.subplots(figsize=(6, 4))
                                    for av in config.availabilities:
                                        df = pd.DataFrame()
                                        for seed in config.seeds:
                                            
                                            res_plot = res_tmp[res_tmp.availability == av] 
                                            event_dir = config.get_event_dir(algo, lr, seed, 
                                                                            event, a, n_c, av, 
                                                                            config.n_rounds, part, biased, config.train_test) 
                                            # print('xxx')
                                            tag = res_plot[metric].values[0]
                                            # print('---------->',tag)
                                            # print('---------->',event_dir)
                                            _, test_accuracy_values = parse_tf_events_file(event_dir, tag=tag)
                                            df[seed] = test_accuracy_values
                                            # print(av)
                                            # yvalues = res_plot[(res_plot.availability == av)][metric]
                                            # print(res_plot[(res_plot.availability == av)][metric])

                                        df['mean'] = df[config.seeds].mean(axis=1)
                                        df['std'] = df[config.seeds].std(axis=1)
                                        
                                        y = df['mean']
                                        y_upper = [a+b for a,b in zip(df['mean'], df['std'])]
                                        y_lower = [a-b for a,b in zip(df['mean'], df['std'])]

                                        # plt.errorbar(xvalues, y, df['std'], fmt='-', label = av)
                                        plt.plot(xvalues, y, label= av)
                                        ax.fill_between(xvalues, y_upper, y_lower, alpha=0.2)

                                        # plt.plot(xvalues, test_accuracy_values, label= av)
                                        ax = plt.gca()
                                        ax.set_ylim([0, 1])
                                        title = ('_').join([algo, a.replace("100000", "iid").replace("0.1", "non-iid"), "biased-"+biased])
                                        plt.title(title)

                                    plt.legend(loc='lower left') 
                                    ax = plt.gca()
                                    ax.set_facecolor('#EBEBEB')
                                    ax.grid(which='major', color='white', linewidth=1.2)
                                    ax.grid(which='minor', color='white', linewidth=0.6)
                                    # Show the minor ticks and grid.
                                    ax.minorticks_on()
                                    # Now hide the minor ticks (but leave the gridlines).
                                    ax.tick_params(which='minor', bottom=False, left=False)
                            
                                    plt.savefig('figures/'+folder+'/'+title+'.png', bbox_inches='tight')
                                    plt.show()


# fig, ax = plt.subplots()

# cpt=0
# for country in countries:
#     # _dfs_country = _dfs[country]
#     # CI (Carbon Intensity) unit is in gCO2eq/kWh
#     # _dfs_plot = _dfs_country[_dfs_country['datetime'].between(start_date,end_date)]

#     # plt.plot([i+1 for i in range(12)], _stats_dfs[country]['mean'])

#     x = [i+1 for i in range(12)]
#     y = _stats_dfs[country]['mean']
    
#     y_upper = [a+b for a,b in zip(_stats_dfs[country]['mean'], _stats_dfs[country]['std'])]
#     y_lower = [a-b for a,b in zip(_stats_dfs[country]['mean'], _stats_dfs[country]['std'])]
    
#     plt.errorbar(x, y, _stats_dfs[country]['std'], fmt='-o', color = list_colors[cpt], label = country)
#     ax.fill_between(x, y_upper, y_lower, color=list_colors[cpt], alpha=0.2)

#     plt.title('Mean Carbon Intensity (CI) over each month, in 2022')
#     plt.xlabel('month')
#     plt.ylabel('CI (gCO2eq/kWh)')
#     plt.legend()
#     # _dfs_plot.plot(y='CI_direct', x='datetime', ax=axes[ind1, ind2], label=country, grid=True)
#     cpt+=1