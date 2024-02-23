import logging
logging.basicConfig(level=logging.DEBUG)

import argparse

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
import numpy as np
from tensorforce.agents import Agent

from src.data.read_data import read_test_data, read_train_data, read_validation_data, filter_data
from src.learna.environment import RnaDesignEnvironment
from src.learna.agent import get_agent, get_network, ppo_agent_kwargs
from src.optimization.learna_worker import LearnaWorker
from src.optimization.training import evaluate, get_configs


# Step 1: Start a nameserver
# Every run needs a nameserver. It could be a 'static' server with a
# permanent address, but here it will be started for the local machine with the default port.
# The nameserver manages the concurrent running workers across all possible threads or clusternodes.
# Note the run_id argument. This uniquely identifies a run of any HpBandSter optimizer.
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Step 2: Start a worker
# Now we can instantiate a worker, providing the mandatory information
# Besides the sleep_interval, we need to define the nameserver information and
# the same run_id as above. After that, we can start the worker in the background,
# where it will wait for incoming configurations to evaluate.
train_sequences = filter_data(read_train_data(), 32)
validation_sequences = filter_data(read_validation_data(), 32)
test_sequences = filter_data(read_test_data(), 32)
w = LearnaWorker(
    num_cores=1,
    train_sequences=train_sequences,
    validation_sequences=validation_sequences,
    nameserver='127.0.0.1',
    run_id='example1'
)
w.run(background=True)

# Step 3: Run an optimizer
# Now we can create an optimizer object and start the run.
# Here, we run BOHB, but that is not essential.
# The run method will return the `Result` that contains all runs performed.
bohb = BOHB(
    configspace=w.get_configspace(),
    run_id='example1', nameserver='127.0.0.1',
    min_budget=1, max_budget=1
)
res = bohb.run(n_iterations=1, min_n_workers=1)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Step 5: Analysis
# Each optimizer returns a hpbandster.core.result.Result object.
# It holds informations about the optimization run like the incumbent (=best) configuration.
# For further details about the Result object, see its documentation.
# Here we simply print out the best config and some statistics about the performed runs.
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()
best_config = id2config[incumbent]['config']

env_config, agent_config, network_config = get_configs(best_config)
environment = RnaDesignEnvironment(dot_brackets=test_sequences, env_config=env_config)

best_agent = Agent.load(
    directory="%i_%i_%i/last_model" % (incumbent[0], incumbent[1], incumbent[2]),
    agent="ppo",
    environment=environment,
    network=get_network(network_config),
    **ppo_agent_kwargs(agent_config)
)

rewards = evaluate(
    env_config=env_config,
    agent=best_agent,
    dot_brackets=test_sequences,
    tries=5
)

max_rewards = np.max(rewards, axis=1)
print('Best found configuration:', best_config)
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print(f'Mean test rewards: {np.mean(max_rewards)}')
print(f'Solved test sequences: {sum(max_rewards == 1) / len(test_sequences)}')
