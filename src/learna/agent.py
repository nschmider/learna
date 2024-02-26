from dataclasses import dataclass
from typing import Tuple

from tensorforce.agents import Agent


@dataclass
class NetworkConfig:
    """
    Dataclass providing the network configuration.

    Default values describe:
        conv_sizes: The number of filters.
        conv_channels: The convolution window size.
        lstm_units: The number of lstm units in a single layer.
        num_lstm_layers: The number of lstm layers.
        lstm_horizon: The horizon of the lstm layer.
        fc_activation: The activation function.
        fc_layer_units: The list of output units of the dense layers.
    """
    conv_sizes: Tuple[int] = (7, 9)
    conv_channels: Tuple[int] = (8, 16)
    num_pooling_layers: int = 1
    lstm_units: int = 0
    num_lstm_layers: int = 0
    lstm_horizon: int = 12
    fc_activation: str = "relu"
    fc_layer_units: Tuple[int] = (50, 20)


def get_network(network_config):
    """
    Get a specific policy network as specified in the network configuration.

    Args:
        network_config: The configuration for the network.

    Returns:
        The policy network of the agent.
    """
    # convolution1 = []
    # for size, window in zip(network_config.conv_channels, network_config.conv_sizes):
    #     convolution1.append(dict(
    #         type="conv2d",
    #         size=size,
    #         window=window,
    #         stride=1,
    #         padding="valid",
    #         bias=True,
    #         activation="relu",
    #         l2_regularization=0.0
    #     ))
    #     convolution1.append(dict(
    #         type="conv2d",
    #         size=size,
    #         window=window,
    #         stride=1,
    #         padding="same",
    #         bias=True,
    #         activation="relu",
    #         l2_regularization=0.0
    #     ))

    layers_before_pool = round(len(network_config.conv_sizes) /
                          network_config.num_pooling_layers)

    convolution = [
        dict(
            type="conv2d",
            size=size,
            window=window,
            stride=1,
            padding="same",
            bias=True,
            activation="relu",
            l2_regularization=0.0
        )
        for size, window in
        zip(
            network_config.conv_channels,
            network_config.conv_sizes
        )
    ]
    
    pooling = [dict(type="pool2d", reduction="max")]

    flatten = [dict(type="flatten")]

    lstm = [
        dict(
        type="lstm",
        size=network_config.lstm_units,
        horizon=network_config.lstm_horizon
        )
    ]

    dense = [
        dict(
            type="dense",
            size=units,
            bias=True,
            activation=network_config.fc_activation
        )
        for units in network_config.fc_layer_units if units >= 4
    ]

    network = []
    
    for conv_block in range(network_config.num_pooling_layers):
        index = conv_block * layers_before_pool
        network += convolution[index : index + layers_before_pool]
        network += pooling
    network += flatten
    network += lstm * network_config.num_lstm_layers
    network += dense
    
    return network


@dataclass
class AgentConfig:
    """
    Dataclass providing the agent configuration.

    Default values describe:
        learning_rate: The learning rate to use by PPO.
        batch_size: Integer of the batch size.
        likelihood_ratio_clipping: Likelihood ratio clipping for policy gradient.
        entropy_regularization: Entropy regularization weight.
    """
    learning_rate: float = 5e-4
    batch_size: int = 8
    likelihood_ratio_clipping: float = 0.3
    entropy_regularization: float = 1.5e-3


def ppo_agent_kwargs(agent_config):
    """
    Get keyword arguments for initializing a PPO agent.

    Args:
        agent_config: The configuration of the agent.

    Returns:
        Dictionary of arguments for initialization of a PPO agent.
    """
    optimizer = dict(type="adam", learning_rate=agent_config.learning_rate)
    return dict(
        batch_size=agent_config.batch_size,
        learning_rate=agent_config.learning_rate,
        likelihood_ratio_clipping=agent_config.likelihood_ratio_clipping,
        entropy_regularization=agent_config.entropy_regularization,
        max_episode_timesteps=501
    )


def get_agent(environment, agent_config, network_config):
    """
    Builds an agent.

    Args:
        environment: The environment the agent acts in.
        agent_config: The configuration of the agent.
    
    Returns:
        The agent.
    """
    agent = Agent.create(
        agent="ppo",
        environment=environment,
        network=get_network(network_config),
        **ppo_agent_kwargs(agent_config)
    )
    return agent
