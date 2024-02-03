from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import tensorflow as tf
from RNA import fold
import numpy as np
from distance import hamming
import time
from dataclasses import dataclass


def _random_epoch_gen(data):
    """
    Generator to get epoch data.

    Args:
        data: The targets of the epoch
    """
    while True:
        for i in np.random.permutation(len(data)):
            yield data[i]


@dataclass
class RnaDesignEnvironmentConfig:
    """
    Dataclass for the configuration of the environment.

    Default values describe:
        matrix_size: The size of the matrix for the internal state.
        reward_exponent: A parameter to shape the reward function.
    """

    matrix_size: int = 12
    padding_mode: str = "reflect"
    pad_lower: bool = False
    reward_exponent: float = 1.0


class RnaDesignEnvironment(Environment):
    """
    The environment for RNA design using deep reinforcement learning.
    """

    action_to_base = {
        0: "A",
        1: "U",
        2: "C",
        3: "G"
    }

    design_to_int = {
        ".": 0,
        "(": 1,
        ")": 2
    }

    def __init__(self, dot_brackets, env_config):
        """
        Initialize an environment.

        Args:
            dot_brackets: The training targets.
            env_config: The configuration of the environment.
        """
        self._reward_exponent = env_config.reward_exponent
        self._current_site = None
        self._target_gen = _random_epoch_gen(dot_brackets)
        self._target = None
        self._matrix = None
        self._matrix_size = env_config.matrix_size
        self._padding_mode = env_config.padding_mode
        self._pad_lower = env_config.pad_lower
        self._rna_seq = None

    def states(self):
        return dict(type='float', shape=(self._matrix_size, self._matrix_size, 2))

    def actions(self):
        return dict(type='int', num_values=4)

    def __str__(self):
        return "RnaDesignEnvironment"

    def reset(self):
        """
        Reset the environment. First function called by runner. Returns first state.

        Returns:
            The first state.
        """
        self._current_site = 0
        self._target = next(self._target_gen)
        self._rna_seq = ""
        design_channel = self._dot_bracket_to_matrix(self._target)
        design_channel = np.expand_dims(design_channel, axis=-1)
        gene_channel = np.zeros(design_channel.shape)
        self._matrix = np.concatenate((gene_channel, design_channel), axis=-1)
        return self._get_state()

    def execute(self, actions):
        """
        Execute one interaction of the environment with the agent.

        Args:
            actions: Current action of the agent.

        Returns:
            state: The next state for the agent.
            terminal: The signal for end of an episode.
            reward: The reward if at terminal timestep, else 0.
        """
        # First matrix channel contains the actions taken
        self._matrix[self._current_site, self._current_site, 0] += (actions + 1)
        self._current_site += 1
        self._rna_seq += self.action_to_base[actions]

        next_state = self._get_state()
        terminal = self._current_site == len(self._target)
        reward = self._get_reward(terminal)
        return next_state, terminal, reward

    def _dot_bracket_to_matrix(self, dot_bracket):
        """
        Computes an adjacency matrix from the dot-bracket notation.

        Args:
            dot_bracket: String representing the dot-bracket notation.

        Returns:
            The adjacency matrix representing the design.
        """
        matrix = np.zeros((len(dot_bracket), len(dot_bracket)))
        i = 0
        while i < len(dot_bracket):
            if dot_bracket[i] == ')':
                j = i
                while dot_bracket[j] != '(':
                    j -= 1
                matrix[i][j] = 1
                matrix[j][i] = 1
                dot_bracket = dot_bracket[:i] + '.' + dot_bracket[i+1:]
                dot_bracket = dot_bracket[:j] + '.' + dot_bracket[j+1:]
            i += 1
        return matrix

    def _get_state(self):
        """
        Get a state dependend on the padded encoding of the target structure.

        Returns:
            The next state.
        """
        state = self._matrix
        pad_width =  self._matrix_size - self._matrix.shape[0]
        if pad_width > 0:
            pad_width_upper = int(np.ceil(pad_width / 2))
            pad_width_lower = int(np.floor(pad_width / 2))
            if self._pad_lower:
                pad_width_upper = 0
                pad_width_lower = pad_width
            # Pad the image if the desired matrix is bigger than the matrix
            state = np.pad(
                self._matrix,
                pad_width=(
                    (pad_width_upper, pad_width_lower),
                    (pad_width_upper, pad_width_lower),
                    (0, 0)),
                mode=self._padding_mode)
        if pad_width < 0:
            # Crop the image if the desired matrix is smaller than the matrix
            half_matrix_plus = int(np.ceil(self._matrix_size / 2))
            half_matrix_minus = int(np.floor(self._matrix_size / 2))
            min_index = max(self._current_site - half_matrix_plus, 0)
            max_index = min(
                min_index + self._matrix_size,
                self._matrix.shape[0]
            )
            min_index = max_index - self._matrix_size
            state = self._matrix[
                min_index : max_index,
                min_index : max_index,
                :
            ]
        return state
        return self._matrix.reshape(self._matrix_size, self._matrix_size, 2)

    def _get_reward(self, terminal):
        """
        Compute the reward after assignment of all nucleotides.

        Args:
            terminal: Bool defining if final timestep is reached yet.

        Returns:
            The reward at the terminal timestep or 0 if not at the terminal timestep.
        """
        if not terminal:
            return 0

        pred_fold = fold(self._rna_seq)[0]
        reward = (1 - hamming(pred_fold, self._target) / len(self._target)) ** self._reward_exponent
        print()
        print(f"RNA sequence: {self._rna_seq}")
        print(f"Prediction: {pred_fold}")
        print(f"Target: {self._target}")
        print(f"Reward: {reward}")
        return reward
