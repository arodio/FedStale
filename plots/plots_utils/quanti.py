import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def refine_results(raw_res):

    refined_res = raw_res.groupby(['algorithm', 'availability','lr']).test_accuracy.apply(np.vstack).to_frame().reset_index() # regroup seeds results

    refined_res['mean_test_acc'] = refined_res['test_accuracy'].apply(lambda x : x.mean(axis=0))
    refined_res['var_test_acc'] = refined_res['test_accuracy'].apply(lambda x : x.var(axis=0))
    refined_res['mean_var_test_acc'] = refined_res['var_test_acc'].apply(lambda x : x.mean())
    refined_res['min_test_acc'] = refined_res['test_accuracy'].apply(lambda x: x.min(axis=0)) #minimum across sseds
    refined_res['max_test_acc'] = refined_res['test_accuracy'].apply(lambda x: x.max(axis=0)) #maximum across sseds

    refined_res['final_mean_acc'] = refined_res['mean_test_acc'].apply(lambda x : x[-1])
    refined_res['final_min_acc'] = refined_res['min_test_acc'].apply(lambda x : x[-1])
    refined_res['final_max_acc'] = refined_res['max_test_acc'].apply(lambda x : x[-1])


    best_res = refined_res.loc[refined_res.groupby(['algorithm', 'availability'])['final_mean_acc'].idxmax()][['algorithm','availability','final_mean_acc','final_min_acc','final_max_acc']]
    best_res = best_res.rename(columns={'final_mean_acc': 'best_final_mean_acc', 'final_min_acc': 'best_final_min_acc', 'final_max_acc': 'best_final_max_acc'})
    worst_res = refined_res.loc[refined_res.groupby(['algorithm', 'availability'])['final_mean_acc'].idxmin()][['algorithm','availability','final_mean_acc']]
    worst_res=worst_res.rename(columns={'final_mean_acc': 'worst_final_mean_acc'})

    best_worst_res = pd.merge(best_res, worst_res, on=['algorithm', 'availability'], how='outer')

    return refined_res, best_worst_res


def plot_final_best_test_acc(res_, folder, keyword):

    x_values = [a.replace('gp-','').replace('-10ft', '') for a in res_[res_.algorithm=='fedavg']['availability']]
    plt.figure()
    ax=plt.gca()
    for algo in ['fedavg', 'fedvarp']:

        y = res_[res_.algorithm==algo]['best_final_mean_acc']
        y_upper = res_[res_.algorithm==algo]['best_final_max_acc']
        y_lower = res_[res_.algorithm==algo]['best_final_min_acc']
        plt.plot(x_values, y, '-o', label='hybrid '+algo)
        ax.fill_between(x_values, y_upper, y_lower, alpha=0.2)
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.title('Final best test accuracy (mean over seeds) - '+keyword)
    plt.grid()
    ax.set_ylim([0.8, 1])
    plt.savefig(folder+'/final_best_test_acc.png', bbox_inches='tight')


def plot_final_best_worst_test_acc(res_, folder, keyword):

    x_values = [a.replace('gp-','').replace('-10ft', '') for a in res_[res_.algorithm=='fedavg']['availability']]
    plt.figure()
    for algo in ['fedavg', 'fedvarp']:
        plt.plot(x_values, res_[res_.algorithm==algo]['best_final_mean_acc'], '-o', label='best hybrid '+algo+' & ft')
        plt.plot(x_values, res_[res_.algorithm==algo]['worst_final_mean_acc'], '-o', label='worst hybrid '+algo+' & ft')
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.title('Final worst and best test accuracy (mean over seeds) - '+keyword)
    plt.grid()
    ax=plt.gca()
    ax.set_ylim([0, 1])
    plt.savefig(folder+'/final_best_worst_test_acc.png', bbox_inches='tight')


def plot_mean_seeds_var_over_training(refined_res, lr_list, folder, keyword):

    tmp_lr = refined_res[['algorithm', 'availability', 'lr', 'mean_var_test_acc']]
    x_values = [a.replace('gp-','').replace('-10ft', '') for a in tmp_lr.loc[(tmp_lr.lr==lr_list[0])&(tmp_lr.algorithm=='fedavg'),'availability']]

    for algo in ['fedavg', 'fedvarp']:
        plt.figure()

        for lr in lr_list:
            plt.plot(x_values, tmp_lr[(tmp_lr.lr==lr)&(tmp_lr.algorithm==algo)]['mean_var_test_acc'], '-o', label='mean variance '+lr)

        plt.xticks(rotation=45, ha='right')
        plt.legend()
        plt.title(algo+' mean variance (among seeds) over the training - '+keyword)
        plt.grid()
        ax=plt.gca()
        ax.set_ylim([-0.005, 0.14])
        plt.savefig(folder+'/'+algo+'-mean_seeds_var_over_training.png', bbox_inches='tight')