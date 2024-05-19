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
    embedding_size: int = 3
    embedding_activation: str = "none"
    conv_sizes: Tuple[int] = (17, 5)
    conv_channels: Tuple[int] = (7, 18)
    num_pooling_layers: int = 1
    lstm_units: int = 28
    num_lstm_layers: int = 1
    lstm_horizon: int = 5
    fc_activation: str = "relu"
    fc_layer_units: Tuple[int] = (57,)
    padding: str = "valid"
    state_radius: int = 32


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





    # layers_before_pool = round(len(network_config.conv_sizes) /
    #                       network_config.num_pooling_layers)

    # convolution = [
    #     dict(
    #         type="conv2d",
    #         size=size,
    #         window=window,
    #         stride=1,
    #         padding="same",
    #         bias=True,
    #         activation="relu",
    #         l2_regularization=0.0
    #     )
    #     for size, window in
    #     zip(
    #         network_config.conv_channels,
    #         network_config.conv_sizes
    #     )
    # ]
    
    # pooling = [dict(type="pool2d", reduction="max")]

    embedding = [
        dict(
            type="embedding",
            size=network_config.embedding_size,
            num_embeddings=4,
            activation=network_config.embedding_activation
        )
    ]
    valid_padding = False
    valid_padding = (network_config.padding == "valid") and (network_config.state_radius - sum(network_config.conv_sizes) - len(network_config.conv_sizes) >= 1)
    padding = "valid" if valid_padding else "same"
    convolution = [
        dict(
            type="conv1d",
            size=size,
            window=window,
            stride=1,
            padding=padding,
            activation="relu"
        )
        for size, window in zip(network_config.conv_channels, network_config.conv_sizes)
        if window > 1
    ]

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

    use_conv = any(map(lambda x: x > 1, network_config.conv_sizes))
    network = []
    
    # for conv_block in range(network_config.num_pooling_layers):
    #     index = conv_block * layers_before_pool
    #     network += convolution[index : index + layers_before_pool]
    #     network += pooling
    if network_config.embedding_size:
        network += embedding
    if use_conv:
        network += convolution
    if use_conv or network_config.embedding_size:
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
    learning_rate: float = 5.99e-4
    batch_size: int = 126
    likelihood_ratio_clipping: float = 0.3
    entropy_regularization: float = 6.76e-5


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
        discount=1.0,
        learning_rate=agent_config.learning_rate,
        likelihood_ratio_clipping=agent_config.likelihood_ratio_clipping,
        entropy_regularization=agent_config.entropy_regularization,
        max_episode_timesteps=500,
        multi_step=1,
        config=dict(buffer_observe=1000, device=None),
        parallel_interactions=128
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
