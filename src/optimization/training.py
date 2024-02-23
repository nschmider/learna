import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.learna.environment import RnaDesignEnvironment, RnaDesignEnvironmentConfig
from src.learna.agent import AgentConfig, ppo_agent_kwargs, get_agent, NetworkConfig, get_network


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


def evaluate(env_config, agent, dot_brackets, tries):
    """
    Evaluating the agent on the validation data.

    Args:
        env_config: The configuration of the environment.
        agent: The agent to be evaluated.
        dot_brackets: The validation data.
        budget: The budget for the configuration, here epochs.
        tries: The amount of tries the agent gets to solve the sequence.

    Returns:
        The rewards.
    """
    data_len = len(dot_brackets)
    environment = RnaDesignEnvironment(dot_brackets=dot_brackets, env_config=env_config)
    rewards = np.zeros((data_len, tries))

    for i in tqdm(range(data_len * tries)):
        # Initialize episode
        states = environment.reset()
        terminal = False

        while not terminal:
            # Episode timestep
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)
        bracket = i % data_len
        epoch = i // data_len
        rewards[bracket, epoch] = reward
    return rewards


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
        The trained agent
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
    return agent


def get_configs(config):
    # used_conv_layers = [
    #     layer
    #     for layer in range(1, 17)
    #     if config[f"conv_size{layer}"] > 0
    #     and config[f"conv_channel{layer}"] > 0
    # ]
    # conv_sizes = tuple(
    #     config[f"conv_size{layer}"] for layer in used_conv_layers
    # )
    # conv_channels = tuple(
    #     config[f"conv_channel{layer}"] for layer in used_conv_layers
    # )
    
    # network_config = NetworkConfig(
    #     conv_sizes=conv_sizes,
    #     conv_channels=conv_channels,
    #     fc_layer_units=(config["fc_units1"], config["fc_units2"]),
    #     fc_activation=config["fc_activation"],
    #     num_lstm_layers=config["num_lstm_layers"],
    #     lstm_units=config["lstm_units"],
    #     lstm_horizon=config["lstm_horizon"]
    # )

    # agent_config = AgentConfig(
    #     learning_rate=config["learning_rate"],
    #     batch_size=config["batch_size"],
    #     entropy_regularization=config["entropy_regularization"],
    #     likelihood_ratio_clipping=config["likelihood_ratio_clipping"],
    # )
    
    # env_config = RnaDesignEnvironmentConfig(
    #     reward_exponent=config["reward_exponent"],
    #     padding_mode=config["padding_mode"],
    #     pad_lower=config["pad_lower"],
    #     matrix_size=config["matrix_size"]
    # )

    network_config = NetworkConfig()
    agent_config = AgentConfig(
        learning_rate=config["learning_rate"]
    )
    env_config = RnaDesignEnvironmentConfig()

    return env_config, agent_config, network_config


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
    # rewards = [np.mean(rewards[i:i+100]) for i in range(0, len(rewards), 10)]
    # plt.plot(np.arange(len(rewards)), rewards)
    # plt.show()
