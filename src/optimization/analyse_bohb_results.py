import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis
# from src.optimization.learna_worker import LearnaWorker
# from src.optimization.meta_learna_worker import MetaLearnaWorker
from fanova import fANOVA
import fanova.visualizer
from pathlib import Path
import ConfigSpace as CS
import itertools as it

import scipy.stats as sps

from src.optimization.write_config_space import get_fine_tuning_config, get_meta_freinet_config, get_freinet_config


def analyse_bohb_run(run):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(f"results/bohb/{run}")

    # get all executed runs
    all_runs = result.get_all_runs()
    # print(all_runs[20:25])
    # all_runs[0].info)
    # print(all_runs)

    # extracted_valid_results = []
    # for x in all_runs:
    #     if x.info:
    #         results = x.info
    #         results["loss"] = x.loss
    #         current = [list(x.config_id), x.budget, x.time_stamps, results]
    #         extracted_valid_results.append(current)
    #


    # get the 'dict' that translates config ids to the actual configurations
    id2conf = result.get_id2config_mapping()


    # Here is how you get he incumbent (best configuration)
    inc_id = result.get_incumbent_id()

    # let's grab the run on the highest budget
    inc_runs = result.get_runs_by_id(inc_id)
    inc_run = inc_runs[-1]


    # We have access to all information: the config, the loss observed during
    #optimization, and all the additional information
    inc_loss = inc_run.loss
    inc_config = id2conf[inc_id]['config']
    print('### Incumbent config:', inc_id, '\n', inc_config)
    inc_test_loss = inc_run.info['normalized_solved_sequences']

    # print(f"validation info of inc: {inc_test_loss}")

    # Let's plot the observed losses grouped by budget,
    hpvis.losses_over_time(all_runs)

    # the number of concurent runs,
    hpvis.concurrent_runs_over_time(all_runs)

    # and the number of finished runs.
    hpvis.finished_runs_over_time(all_runs)

    # This one visualizes the spearman rank correlation coefficients of the losses
    # between different budgets.
    hpvis.correlation_across_budgets(result)

    # For model based optimizers, one might wonder how much the model actually helped.
    # The next plot compares the performance of configs picked by the model vs. random ones
    hpvis.performance_histogram_model_vs_random(all_runs, id2conf)

    # plt.show()

    # df = result.get_pandas_dataframe()
    # print(df)

    min_solved = 0

    all_solved = [(x.info['solved_sequences'], x.loss, x.config_id, x.budget, id2conf[x.config_id]['config']) for x in all_runs if x.info and int(x.info['solved_sequences']) >= min_solved]

    all_solved_dict = [
        {'num_solved': x.info['solved_sequences'], 'sum_of_min_distances': x.info['solved_sequences'], 'config_id': x.config_id, 'budget': x.budget, 'start_time': x['time_stamps']['started'], 'finishing_time': x['time_stamps']['finished'], 'config': id2conf[x.config_id]['config']} for x in all_runs if x.info and int(x.info['solved_sequences']) >= min_solved
    ]

    all_solved_df = pd.DataFrame(all_solved_dict)
    print(all_solved_df)
    print(all_solved_df['finishing_time'].min(), all_solved_df['finishing_time'].max())
    
    Path(f'processed_data/bohb/pgfplot/{run}').mkdir(parents=True, exist_ok=True)
    write_pgf_plot_data(all_solved_df, f'processed_data/bohb/pgfplot/{run}')

    # # print(all_solved)
    # training_info = [x.info["train_info"] for x in all_runs if x.info]
    # print(training_info[:2])
    # all_failed_encodings = [id2conf[x.config_id]['config']['encoding'] for x in all_runs if not x.info]
    all_failed_convs_1 = [(id2conf[x.config_id]['config']['conv_size1'], id2conf[x.config_id]['config']['conv_size2']) for x in all_runs if not x.info]
    all_failed_convs_2 = [id2conf[x.config_id]['config']['conv_size2'] for x in all_runs if not x.info]
#     all_failed_embeddings = [id2conf[x.config_id]['config']['embedding_size'] for x in all_runs if not x.info]
    # all_failed_training_set = [id2conf[x.config_id]['config']['trainingset'] for x in all_runs if not x.info]
    all_failed_batch_size = [id2conf[x.config_id]['config']['batch_size'] for x in all_runs if not x.info]
    # embdding_size_seq_bias = [id2conf[x.config_id]['config']['embedding_size_seq_bias'] for x in all_runs if not x.info and id2conf[x.config_id]['config']['encoding'] == 'sequence_bias']
    # embdding_size_tuple = [id2conf[x.config_id]['config']['embedding_size_tuple'] for x in all_runs if not x.info and id2conf[x.config_id]['config']['encoding'] == 'tuple']
    all_failed = [(x.config_id, id2conf[x.config_id]['config']) for x in all_runs if not x.info]
    # all_failed_data_type = [id2conf[x.config_id]['config']['data_type'] for x in all_runs if not x.info]

    from collections import Counter
    # print(Counter(all_failed_encodings))
    print(Counter(all_failed_convs_1))
    # print(Counter(all_failed_convs_2))
#     print(Counter(all_failed_embeddings))
    # print(Counter(all_failed_training_set))
    # print(Counter(embdding_size_seq_bias))
    # print(Counter(embdding_size_tuple))
    print(Counter(all_failed_batch_size))
    all_failed_config_ids = [x[0][0] for x in all_failed]
    print(all_failed[0:20])
    # print(Counter(all_failed_data_type))

    print(Counter(all_failed_config_ids))
    # print(any(all_failed_convs_1))
    # print(any(all_failed_convs_2))


    # print(all_failed[:20])
    print(len(all_failed))

    all_solved_sorted = sorted(all_solved, key=lambda x: x[0], reverse=True)

    print(f"number of configurations evaluated: {len(all_solved)}")

    print(f"{len(all_solved_sorted)} configurations solved at least {min_solved} targets:")

    print(f"Most solving config: {all_solved_sorted[0][1]}")
    # print('batch\tc_channels1\tc_channels2\tc_radius1\tc_radius2\tembedding\tentropy\tfc_units\tlearning_rate\tlstm_units\tfc_layers\tlstm_layers\tpairs\talpha\treward_f\ts_radius')
    # for index, i in enumerate(all_solved_sorted[:10]):
    for index, i in enumerate(all_solved_sorted[:10]):
        print(f"[{index + 1}]")
        print(str(i) + '\n')
        # print(f"{i[2]['batch_size']}\t{i[2]['conv_channels1']}\t{i[2]['conv_channels2']}\t{i[2]['conv_radius1']}\t{i[2]['conv_radius2']}\t{i[2]['embedding_size']}\t{i[2]['entropy_regularization']}\t{i[2]['fc_units']}\t{i[2]['learning_rate']}\t{i[2]['lstm_units']}\t{i[2]['num_fc_layers']}\t{i[2]['num_lstm_layers']}\t{i[2]['predict_pairs']}\t{i[2]['reward_exponent']}\t{i[2]['reward_function']}\t{i[2]['state_radius_relative']}")

    # print(all_solved)
    # print('\n')
    # print('Incumbent:')
    # print(inc_config)
    # print('It achieved accuracies of %f (validation) and %f (test).'%(1-inc_loss, inc_test_loss))


def write_pgf_plot_data(df, out_dir):
    for budget, group in df.groupby('budget'):
        with open(f"{out_dir}/num_solved_budget_{int(budget)}.tsv", "w+") as f:
            for i, row in group.iterrows():
                unsolvedsolved = 100 - row['num_solved']
                start_time = row['finishing_time']
                f.write(f"{start_time}\t{unsolvedsolved}\n")
                # budget = row['budget']
        
        with open(f"{out_dir}/sum_of_min_distances_budget_{int(budget)}.tsv", "w+") as f:
            for i, row in group.iterrows():
                sum_of_min_distances = row['sum_of_min_distances']
                start_time = row['finishing_time']
                f.write(f"{start_time}\t{sum_of_min_distances}\n")
                # budget = row['budget']

    with open(f"{out_dir}/num_solved_vs_sum_of_min_distances.tsv", "w+") as f:
        for i, row in df.iterrows():
            solved = row['num_solved']
            loss = row['sum_of_min_distances']
            f.write(f"{loss}\t{solved}\n")
            # budget = row['budget']



def get_spearman_correlation_between_budgets(results_object, show=False):
    runs = results_object.get_all_runs()
    id2conf = results_object.get_id2config_mapping()

    budgets = list(set([r.budget for r in runs]))
    budgets.sort()

    import itertools

    loss_pairs = {}
    for b in budgets[:-1]:
        loss_pairs[b] = {}
    
    for b1,b2 in itertools.combinations(budgets, 2):
        loss_pairs[b1][b2]= []

    for cid in id2conf.keys():
        runs = results_object.get_runs_by_id(cid)
        if len(runs) < 2: continue
		
        for r1,r2 in itertools.combinations(runs,2):
            if r1.loss is None or r2.loss is None: continue
            if not np.isfinite(r1.loss) or not np.isfinite(r2.loss): continue
            loss_pairs[float(r1.budget)][float(r2.budget)].append((r1.loss, r2.loss))
		
		
    rhos = np.eye(len(budgets)-1)
    rhos.fill(np.nan)

    ps = np.eye(len(budgets)-1)
    ps.fill(np.nan)

    for i in range(len(budgets)-1):
        for j in range(i+1,len(budgets)):
            spr = sps.spearmanr(loss_pairs[budgets[i]][budgets[j]])
            rhos[i][j-1] = spr.correlation
            ps[i][j-1] = spr.pvalue
    
    return rhos, ps
    


def generate_fanova_plots(path, run, out_dir, mode, n, param):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(f"{path}/{run}")

    # param = "reward_function"

    if mode == 'autoLEARNA':
        cs = get_freinet_config()
    elif mode == 'autoMeta-LEARNA':
        cs = get_meta_freinet_config()
    else:
        raise
    print('generate fanova data')
    a, b, _ = result.get_fANOVA_data(cs)
    b = np.array([100 - np.float64(x) for x in b])
    print('create fanova object')
    f = fANOVA(a, b, cs)
    print('create visualizer')
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    vis = fanova.visualizer.Visualizer(f, cs, directory=path)
    hp_importances = []
    print('computing importance of parameters')
    for i in range(1, len(cs.get_hyperparameter_names())):
        print(f"Computing importance of {cs.get_hyperparameter_by_idx(i)}")
        importance = f.quantify_importance(dims=[cs.get_hyperparameter_by_idx(i)])[(cs.get_hyperparameter_by_idx(i), )]
        print(f"total importance: {importance['total importance']}")
        hp_name = cs.get_hyperparameter_by_idx(i)
        # print(hp_name)
        total_importance = importance['total importance']
        # print(total_importance)
        individual_importance = importance['individual importance']
        # print(individual_importance)
        individual_std = importance['individual std']
        # print(individual_std)
        total_std = importance['total std']
        # print(total_importance)
        # print('\n')
        hp_importances.append((hp_name, total_importance, total_std, individual_importance, individual_std))
        # print(hp_importances)
        print(f"generate marginal plot for hyperparameter: {cs.get_hyperparameter_by_idx(i)}")
        try:
            log = cs.get_hyperparameter(cs.get_hyperparameter_by_idx(i)).log
        except:
            log = False
        # print(f"log_scale is {log}")
        fig = vis.plot_marginal(param=i, log_scale=log, show=False)
        fig.savefig(Path('results', 'fanova', run, f"{cs.get_hyperparameter_by_idx(i)}"))
        fig.close()
    hp_importances = sorted(hp_importances, key=lambda x: x[1], reverse=True)
    # print(hp_importances)
    print('Writing importance files')
    for hp_importance in hp_importances:
        # print(hp_importance)
        imps_path = Path('results', 'fanova', run, 'importances.txt')
        imp_path = Path('results', 'fanova', run, f"importance_{hp_importance[0]}.txt")
        with open(imp_path, 'w+') as imp:
            # print(str(hp_importance[0])
            imp.write(str(hp_importance[0]) + '\n')
            # print(str(hp_importance[1])
            imp.write('total importance: ' + str(hp_importance[1]) + '\n')
            # print(str(hp_importance[2])
            imp.write('total_std: ' + str(hp_importance[2]) + '\n')
            # print(str(hp_importance[3])
            imp.write('individual importance: ' + str(hp_importance[3]) + '\n')
            # print(str(hp_importance[4])
            imp.write('individual std: ' + str(hp_importance[4]) + '\n')
        with open(imps_path, 'a+') as imps:
            imps.write(str(hp_importance) + '\n')
                # imps.write('\n')
    # print(importance)

        # print('generate plots')
        # vis.create_most_important_pairwise_marginal_plots(n=n)
        # for i in range(1, len(cs.get_hyperparameter_names())):
        # try:
        # except Exception as e:
        #     print(e)
    # for i in range(1, len(cs.get_hyperparameter_names())):
    #     for j in range(1, len(cs.get_hyperparameter_names())):
    #         try:
    #             if i != j:
    #                 print(f"generate pairwise marginal plot for hyperparameters: {cs.get_hyperparameter_by_idx(i)} and {cs.get_hyperparameter_by_idx(j)}")
    #                 fig = vis.plot_pairwise_marginal(param_list=(i, j), show=False)
    #                 fig.savefig(Path('results', 'fanova', run, f"pairwiswe_marginal_{cs.get_hyperparameter_by_idx(i)}_{cs.get_hyperparameter_by_idx(j)}"))
    #                 fig.close()
    #         except Exception as e:
    #             print(e)

    # # # getting the 10 most important pairwise marginals sorted by importance
    # # best_margs = f.get_most_important_pairwise_marginals(n=10)
    # # print(best_margs)
    # # # creating the plot of pairwise marginal:
    # # vis.plot_pairwise_marginal((0,2), resolution=20)
    # # creating all plots in the directory


def create_pairwise_marginals(path, run, out_dir, mode, params):
    # load the example run from the log files
    result = hpres.logged_results_to_HBS_result(f"{path}/{run}")

    # param = "reward_function"

    if mode == 'autoLEARNA':
        cs = get_freinet_config()
    elif mode == 'autoMeta-LEARNA':
        cs = get_meta_freinet_config()
    else:
        raise
    print('generate fanova data')
    a, b, _ = result.get_fANOVA_data(cs)
    b = np.array([100 - np.float64(x) for x in b])
    print('create fanova object')
    f = fANOVA(a, b, cs)
    print('create visualizer')
    path = Path(out_dir)
    path.mkdir(parents=True, exist_ok=True)
    vis = fanova.visualizer.Visualizer(f, cs, directory=path)
    pairwise_parameters = it.permutations(params, 2)
    for item in pairwise_parameters:
        p1 = item[0]
        p2 = item[1]
        print(f"generate pairwise marginal plot for hyperparameters: {p1} and {p2}")
        fig = vis.plot_pairwise_marginal(param_list=(cs.get_idx_by_hyperparameter_name(p1), cs.get_idx_by_hyperparameter_name(p2)), show=False)
        fig.savefig(Path('results', 'fanova', run, f"pairwiswe_marginal_{p1}_{p2}"))
        fig.close()
    # print(f"Creating {n} most important pairwise marginal plots")
    # vis.create_most_important_pairwise_marginal_plots(f"results/fanova/{run}/", n)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--path", type=str, default="results/bohb", help="Path to results of run"
    )

    parser.add_argument(
        "--run", type=str, help="The id of the run"
    )

    parser.add_argument(
        "--out_dir", type=str, help="The id of the run"
    )

    parser.add_argument(
        "--mode", type=str, help="Choose between autoLEARNA, autoMeta-LEARNA"
    )

    parser.add_argument(
        "--n", type=int, help="The number of most important marginal plots fanova should generate"
    )

    parser.add_argument(
        "--parameter", type=str, help="The parameter to analyse", default=""
    )

    args = parser.parse_args()

    # params = ['state_radius_relative', 'learning_rate', 'num_lstm_layers']
    params = [
        "state_radius",
        "reward_exponent",
        "learning_rate",
        "batch_size",
        "entropy_regularization",
        "likelihood_ratio_clipping",
        "conv_size1",
        "conv_channel1",
        "conv_size2",
        "conv_channel2",
        "conv_size3",
        "conv_channel3",
        "conv_size4",
        "conv_channel4",
        "num_lstm_layers",
        "lstm_units",
        "lstm_horizon",
        "fc_activation",
        "fc_units1",
        "fc_units2",
        "embedding_size",
        "embedding_activation",
        # "padding"
    ]

    analyse_bohb_run(args.run)

    generate_fanova_plots(args.path, args.run, args.out_dir, args.mode, args.n, params)

    # create_pairwise_marginals(args.path, args.run, args.out_dir, args.mode, params)
