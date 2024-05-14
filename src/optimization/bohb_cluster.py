import logging
logging.basicConfig(level=logging.DEBUG)
import argparse
import os
from pathlib import Path
import pickle
import time

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB as BOHB
import numpy as np
from tensorforce.agents import Agent

from src.data.read_data import read_test_data, read_train_data, read_validation_data, filter_data
from src.learna.environment import RnaDesignEnvironment
from src.learna.agent import get_agent, get_network, ppo_agent_kwargs
from src.optimization.learna_worker import LearnaWorker
from src.optimization.training import evaluate, get_configs


parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget',   type=float, help='Minimum budget used during the optimization.',    default=9)
parser.add_argument('--max_budget',   type=float, help='Maximum budget used during the optimization.',    default=243)
parser.add_argument('--n_iterations', type=int,   help='Number of iterations performed by the optimizer', default=4)
parser.add_argument('--n_workers', type=int,   help='Number of workers to run in parallel.', default=2)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--n_cores', default=1)
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.')
parser.add_argument('--shared_directory', type=str, help='A directory that is accessible for all processes, e.g. a NFS share.')
parser.add_argument('--data_directory', type=str, help='A directory that contains the data.')
parser.add_argument('--masked', type=bool, default=False)

args=parser.parse_args()

result_logger = hpres.json_result_logger(directory=args.shared_directory, overwrite=True)

# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)


train_sequences = read_train_data(args.data_directory)
validation_sequences = read_validation_data(args.data_directory)
test_sequences = read_test_data(args.data_directory)

if args.worker:
    time.sleep(5)	# short artificial delay to make sure the nameserver is already running
    w = LearnaWorker(
        num_cores=args.n_cores,
        train_sequences=train_sequences,
        validation_sequences=validation_sequences,
        run_id=args.run_id,
        host=host
    )
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)

# Start a nameserver:
# We now start the nameserver with the host name from above and a random open port (by setting the port to 0)
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Most optimizers are so computationally inexpensive that we can afford to run a
# worker in parallel to it. Note that this one has to run in the background to
# not plock!
w = LearnaWorker(
    num_cores=args.n_cores,
    train_sequences=train_sequences,
    validation_sequences=validation_sequences,
    run_id=args.run_id,
    host=host,
    nameserver=ns_host,
    nameserver_port=ns_port
)
w.run(background=True)

# Run an optimizer
# We now have to specify the host, and the nameserver information
bohb = BOHB(
    configspace=LearnaWorker.get_configspace(),
    run_id=args.run_id,
    host=host,
    nameserver=ns_host,
    nameserver_port=ns_port,
    result_logger=result_logger,
    min_budget=args.min_budget, max_budget=args.max_budget
)
res = bohb.run(n_iterations=args.n_iterations, min_n_workers=args.n_workers)


# In a cluster environment, you usually want to store the results for later analysis.
# One option is to simply pickle the Result object 
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
	pickle.dump(res, fh)

# Step 4: Shutdown
# After the optimizer run, we must shutdown the master and the nameserver.
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

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
    tries=5,
    max=1000
)

max_rewards = np.max(rewards, axis=1)
print('Best found configuration:', best_config)
print('A total of %i unique configurations where sampled.' % len(id2config.keys()))
print(f'Mean test rewards: {np.mean(max_rewards)}')
print(f'Solved test sequences: {sum(max_rewards == 1) / len(test_sequences)}')
