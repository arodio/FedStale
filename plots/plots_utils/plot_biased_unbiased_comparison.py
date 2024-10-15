import matplotlib.pyplot as plt
from plots_utils.loading import parse_tf_events_file

def plot_biased_unbiased_comparison(config, res, metric, folder):

    xvalues = [i for i in range(int(config.n_rounds)+1)]

    for lr in config.lr_list:
        
        for event in config.events:
            for seed in config.seeds:
                for a in config.alphas:
                    for n_c in config.n_clients_list:
                        # for av in config.availabilities:
                        for part in config.participations:
                                    
                            for av in config.availabilities:

                                for algo in config.algorithms:

                                    res_tmp = res[(res.lr == lr) & (res.availability == av) & (res.event == event) &
                                                (res.seed == seed) & (res.alpha == a) & (res.n_clients == n_c) &
                                                (res.participation == part) & (res.algorithm == algo)]
                                    
                                    # display(res_tmp) # this is what we are going to compare
                                    
                                    fig = plt.figure(figsize=(6, 4))
                                    
                                    
                                    for biased in config.biased_list:
                                        
                                        res_plot = res_tmp[res_tmp.biased == biased]
                                        event_dir = config.get_event_dir(algo, lr, seed, 
                                                                        event, a, n_c, av, 
                                                                        config.n_rounds, part, biased, config.train_test)  

                                        tag = res_plot[metric].values[0]
                                        _, test_accuracy_values = parse_tf_events_file(event_dir, tag=tag)
                                        
                                        if biased == "0":
                                            plt.plot(xvalues, test_accuracy_values, label="unbiased", color='g')
                                        elif biased == "1":
                                            plt.plot(xvalues, test_accuracy_values, label="biased")
                                        else:
                                            plt.plot(xvalues, test_accuracy_values, label="hybrid", linestyle="--")
                                        ax = plt.gca()
                                        ax.set_ylim([0, 1])
                                        title = ('_').join([av.replace("random_for", "R").replace("carbon-budget-", "").replace("uniform", "unif").replace("nonlinear-optimization-cvxpy", "opt-pb-1"), algo, a.replace("100000", "iid").replace("0.1", "non-iid"), metric])
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
                                    
                                    # plt.grid()
                                    plt.savefig('figures/'+folder+'/'+title+'.png', bbox_inches='tight')
                                    plt.show()