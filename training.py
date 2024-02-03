import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig
from agent import AgentConfig, ppo_agent_kwargs, get_agent, NetworkConfig, get_network
# from read_eterna import read_eterna


def make_synthetic_data(amount, length):
    """
    Returns a training set with dot-bracket notations.

    Args:
        amount: The amount of dot-bracket notations.
        length: The length of the dot-bracket notations.
    Returns:
        A training set of dot-bracket notations.
    """
    dot_brackets = []
    for _ in range(amount):
        dot_bracket = "." * length
        for i in range(length):
            if dot_bracket[i] == ")":
                continue
            if not any([dot_bracket[j] == "." for j in range(i+1, length)]): # Checks if you can still make 2 brackets
                continue
            make_bracket = np.random.choice([True, False], p=[0.2, 0.8])
            if make_bracket:
                # Returns an index that is not assigned yet, every index has uniform probabilities
                possible_indices = [j for j in range(i+1, length) if dot_bracket[j] == "."]
                second_bracket = np.random.choice(possible_indices)
                dot_bracket = dot_bracket[:i] + "(" + dot_bracket[i+1:]
                dot_bracket = dot_bracket[:second_bracket] + ")" + dot_bracket[second_bracket+1:]
        dot_brackets.append(dot_bracket)
    return dot_brackets


def training(env_config, agent_config, network_config, dot_brackets, budget):
    """
    Training with one configuration of the environment and the agent.

    Args:
        env_config: The configuration of the environment.
        agent_config: The configuration of the agent.
        network_config: The configuration of the agent's network.
        dot_brackets: The learning targets.
        budget: The budget for the configuration, here epochs.

    Returns:
        Rewards
    """
    environment = RnaDesignEnvironment(dot_brackets=dot_brackets, env_config=env_config)
    agent = get_agent(environment, agent_config, network_config)
    rewards = []

    for _ in tqdm(range(budget)):
        # Initialize episode
        states = environment.reset()
        terminal = False

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
        rewards.append(reward)
    return rewards

if __name__ == "__main__":
    dot_bracket = "(((((......)))))"
    # dot_brackets = make_synthetic_data(10000, 8)
    dot_brackets = [dot_bracket]
    # dot_brackets = read_eterna()
    # dot_brackets = [dot_bracket for dot_bracket in dot_brackets if len(dot_bracket) == 16]

    env_config = RnaDesignEnvironmentConfig(matrix_size=21, reward_exponent=1.0, padding_mode="wrap")
    agent_config = AgentConfig(learning_rate=1e-4,
                               batch_size=1,
                               likelihood_ratio_clipping=0.3,
                               entropy_regularization=1.5e-3)
    network_config = NetworkConfig(conv_sizes=(3, 3, 5, 5, 7, 7, 9), #9), #9),
                                   conv_channels=(2, 4, 8, 16, 32, 64, 128), #256, #512),
                                   lstm_units=1,
                                   num_lstm_layers=1,
                                   lstm_horizon=12,
                                   fc_activation="relu",
                                   fc_layer_units=(50, 20))

    rewards = training(env_config, agent_config, network_config, dot_brackets, 20000)
    rewards = [np.mean(rewards[i:i+100]) for i in range(0, len(rewards), 10)]
    plt.plot(np.arange(len(rewards)), rewards)
    plt.show()
