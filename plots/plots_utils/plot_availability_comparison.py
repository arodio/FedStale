import matplotlib.pyplot as plt
from plots_utils.loading import parse_tf_events_file

def plot_availability_comparison(config, res, metric, folder):
    
    xvalues = [i for i in range(int(config.n_rounds)+1)]

    for lr in config.lr_list:
        for algo in config.algorithms:
            for event in config.events:
                for seed in config.seeds:
                    for a in config.alphas:
                        for n_c in config.n_clients_list:
                            for biased in config.biased_list:
                                # for av in config.availabilities:
                                for part in config.participations:


                                        res_tmp = res[(res.lr == lr) & (res.algorithm == algo) & (res.event == event) &
                                                    (res.seed == seed) & (res.alpha == a) & (res.n_clients == n_c) &
                                                    (res.participation == part)]
                                        
                                        # display(res_plot) # this is what we are going to compare
                                        
                                        ### for random equivalents:
                                        # fig = plt.figure(figsize=(6, 4))
                                        # for av in config.availabilities:
                                        #     if 'random' in av:
                                        #         res_plot = res_tmp[res_tmp.availability == av]
                                        #         event_dir = config.get_event_dir(algo, lr, seed, 
                                        #                                         event, a, n_c, av, 
                                        #                                         config.n_rounds, part)  
                                        #         # display(res_plot)
                                        #         # print('xxx')
                                        #         tag = res_plot[metric].values[0]
                                        #         _, test_accuracy_values = parse_tf_events_file(event_dir, tag=tag)
                                        #         # print(av)
                                        #         # yvalues = res_plot[(res_plot.availability == av)][metric]
                                        #         # print(res_plot[(res_plot.availability == av)][metric])
                                        #         plt.plot(xvalues, test_accuracy_values, label= av)
                                        #         title = ('_').join([algo, 'alpha'+a, 'random', metric])
                                        #         plt.title(title)
                                        # plt.legend()
                                        # plt.grid()
                                        # plt.savefig('figures/accross_availabilities/'+title+'.png', bbox_inches='tight')
                                        # plt.show()

                                        fig = plt.figure(figsize=(6, 4))
                                        for av in config.availabilities:
                                            if 'random' not in av:
                                                res_plot = res_tmp[res_tmp.availability == av]
                                                # event_dir = config.get_event_dir(algo, lr, seed, 
                                                #                                 event, a, n_c, av, 
                                                #                                 config.n_rounds, part)  
                                                event_dir = config.get_event_dir(algo, lr, seed, 
                                                                                event, a, n_c, av, 
                                                                                config.n_rounds, part, biased, config.train_test) 
                                                # display(res_plot)
                                                # print('xxx')
                                                tag = res_plot[metric].values[0]
                                                # print('---------->',tag)
                                                # print('---------->',event_dir)
                                                _, test_accuracy_values = parse_tf_events_file(event_dir, tag=tag)
                                                # print(av)
                                                # yvalues = res_plot[(res_plot.availability == av)][metric]
                                                # print(res_plot[(res_plot.availability == av)][metric])
                                                plt.plot(xvalues, test_accuracy_values, label= av)
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
