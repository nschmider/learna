import argparse

import matplotlib.pyplot as plt
import hpbandster.core.result as hpres
import hpbandster.visualization as hpvis


parser = argparse.ArgumentParser()
parser.add_argument('--result_directory')

args = parser.parse_args()


# load the example run from the log files
result = hpres.logged_results_to_HBS_result(args.result_directory)

# get all executed runs
all_runs = result.get_all_runs()

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
inc_test_loss = inc_run.info

print('Best found configuration:')
print(inc_config)
print('It achieved accuracies of %f (validation) and solved %f of sequences.'%(1-inc_loss, inc_test_loss))


# Let's plot the observed losses grouped by budget,
hpvis.losses_over_time(all_runs) 
plt.tight_layout()
plt.savefig('learna/plots/losses_over_time.png', bbox_inches='tight')

# the number of concurent runs,
hpvis.concurrent_runs_over_time(all_runs) 
plt.tight_layout()
plt.savefig('learna/plots/concurrent_runs_over_time.png', bbox_inches='tight')

# and the number of finished runs.
hpvis.finished_runs_over_time(all_runs)
plt.tight_layout()
plt.savefig('learna/plots/finished_runs_over_time.png', bbox_inches='tight')

# This one visualizes the spearman rank correlation coefficients of the losses
# between different budgets.
hpvis.correlation_across_budgets(result)
plt.tight_layout()
plt.savefig('learna/plots/correlation_across_budgets.png', bbox_inches='tight')

# For model based optimizers, one might wonder how much the model actually helped.
# The next plot compares the performance of configs picked by the model vs. random ones
hpvis.performance_histogram_model_vs_random(all_runs, id2conf) 

plt.tight_layout()
plt.savefig('learna/plots/histogram_model_vs_random.png', bbox_inches='tight')
