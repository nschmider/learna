from itertools import combinations, permutations, product

from axial_attention import AxialAttention
from tensorforce.environments import Environment
from tensorforce.agents import Agent
from tensorforce.execution import Runner
import tensorflow as tf
import torch
from RNA import fold
import numpy as np
from distance import hamming
from dataclasses import dataclass

from src.learna.utils.encodings import encode_dot_bracket, encode_pairing, probabilistic_pairing
from src.learna.utils.helper_functions import custom_hamming, dot_bracket_to_matrix, mask, replace_x


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

    matrix_size: int = 32
    padding_mode: str = "constant"
    pad_lower: bool = False
    reward_exponent: float = 1.0
    state_radius: int = 32


class RnaDesignEnvironment(Environment):
    """
    The environment for RNA design using deep reinforcement learning.
    """

    action_to_base = {
        0: "C",
        1: "A",
        2: "U",
        3: "G"
    }

    action_to_pair = {
        0: "CG",
        1: "AU",
        2: "UA",
        3: "GC"
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
        self._pairs = None
        self._attention = AxialAttention(
            dim=2,
            dim_index=-1,
            dim_heads=2,
            num_dimensions=2,
            heads=1
        )
        self._state_radius = env_config.state_radius
        self._padded_encoding = None

    def states(self):
        return dict(type='int', shape=(2 * self._state_radius + 1,))
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
        self._input_seq = "".join(np.random.choice(["A", "C", "G", "U"], size=50))
        # self._target = fold(self._input_seq)[0]
        self._target = next(self._target_gen)
        self._rna_seq = "-" * len(self._target)

        design_channel = dot_bracket_to_matrix(self._target)
        design_channel = np.expand_dims(design_channel, axis=-1)
        gene_channel = np.zeros(design_channel.shape)
        self._matrix = np.concatenate((gene_channel, design_channel), axis=-1)

        # self._input_seq = self._input_seq[:3] + "X" + self._input_seq[3:20] + "X" + self._input_seq[20:33] + "X" + self._input_seq[33:]
        # self._input_seq = replace_x(self._input_seq, min_length=10, max_length=20)
        self._target = mask(self._target)
        self._input_seq = mask(self._input_seq)
        # self._rna_seq = self._input_seq.replace("N", "-")

        self._padded_encoding = encode_dot_bracket(self._target, self._input_seq, self._state_radius)
        # self._pairing_encoding = encode_pairing(self._target)
        self._pairing_encoding, self._pairs = probabilistic_pairing(self._target)
        
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
        pair_index = self._pairing_encoding[self._current_site]
        if pair_index:
            cur_base, pair_base = self.action_to_pair[actions]
            self._rna_seq = (
                self._rna_seq[:self._current_site] +
                cur_base +
                self._rna_seq[self._current_site + 1:]
            )
            self._rna_seq = (
                self._rna_seq[:pair_index] +
                pair_base +
                self._rna_seq[pair_index + 1:]
            )
            self._matrix[self._current_site, self._current_site, 0] += (actions + 1)
            self._matrix[pair_index, pair_index, 0] += (4 - actions)
        else:
            # First matrix channel contains the actions taken
            self._matrix[self._current_site, self._current_site, 0] += (actions + 1)
            self._rna_seq = (
                self._rna_seq[:self._current_site] +
                self.action_to_base[actions] +
                self._rna_seq[self._current_site + 1:]
            )
        self._current_site = self._first_unassigned_site()

        terminal = self._current_site == -1
        if terminal:
            next_state = None
        else:
            next_state = self._get_state()
        reward = self._get_reward(terminal)
        return next_state, terminal, reward

    def _first_unassigned_site(self):
        """
        Finds the first unassigned site.

        Returns:
            int: The first unassigned site.
        """
        return self._rna_seq.find("-")

    def _local_improvement(self, folded_design):
        """
        Compute Hamming distance of locally improved candidate solutions.

        Returns:
            The minimum Hamming distance of all improved candidate solutions.
        """
        min_distance = float('inf')
        differing_sites = [i for i in range(len(self._target))
                           if self._target[i] != folded_design[i]
                           and self._target[i] != "N"]
        for mutation in product('ACGU', repeat=len(differing_sites)):
            mutated_sequence = self._get_mutated(differing_sites, mutation)
            folded_mutation = fold(mutated_sequence)[0]
            hamming_distance = custom_hamming(self._target, folded_mutation)
            min_distance = min(hamming_distance, min_distance)
        return min_distance

    def _local_improvement_pairs(self, folded_design):
        # differing_sites = [i for i in range(len(self._target))
        #                    if self._input_seq[i] == "N"
        #                    and self._target[i] == "N"]
        # bases = [self._rna_seq[i] for i in differing_sites]
        # if len(differing_sites) >= 40 or len(differing_sites) == 0:
        #     return
        # min_distance = float('inf')
        # for mutation in permutations(bases):
        #     mutated_sequence = self._get_mutated(differing_sites, mutation)
        #     folded_mutation = fold(mutated_sequence)[0]
        #     hamming_distance = custom_hamming(self._target, folded_mutation)
        #     min_distance = min(hamming_distance, min_distance)
        if len(self._pairs) >= 10 or len(self._pairs) == 0:
            return
        best_mutated = self._rna_seq
        for chosen_indices, pair_candidates in self._pairs:
            min_distance = float('inf')
            for changed_indices in permutations(pair_candidates, len(chosen_indices)):
                mutated_sequence = self._switch(best_mutated, chosen_indices, changed_indices)
                folded_mutation = fold(mutated_sequence)[0]
                hamming_distance = custom_hamming(self._target, folded_mutation)
                if hamming_distance < min_distance:
                    best_mutated = mutated_sequence
                    min_distance = hamming_distance
        return min_distance

    def _local_improvement_without_unknowns(self, folded_design):
        """
        Compute Hamming distance of locally improved candidate solutions.

        Returns:
            The minimum Hamming distance of all improved candidate solutions.
        """
        min_distance = float('inf')
        differing_sites = [i for i in range(len(self._target)) if self._target[i] != folded_design[i]]
        for mutation in product('ACGU', repeat=len(differing_sites)):
            mutated_sequence = self._get_mutated(differing_sites, mutation)
            folded_mutation = fold(mutated_sequence)[0]
            hamming_distance = hamming(folded_mutation, self._target)
            min_distance = min(hamming_distance, min_distance)
        return min_distance

    def _get_mutated(self, differing_sites, mutation):
        """
        Fills the RNA with mutated sites that differ from the RNA sequence.

        Args:
            differing_sites (list): The sites that differ from the target.
            mutation (list): The bases to set.

        Returns:
            string: The mutated sequence.
        """
        seq = ""
        mutated_set = 0
        for i in range(len(self._rna_seq)):
            if i in differing_sites:
                seq += mutation[mutated_set]
                mutated_set += 1
                continue
            seq += self._rna_seq[i]
        return seq

    def _switch(self, seq, indices1, indices2):
        rna_seq_list = list(seq)
        for index1, index2 in zip(indices1, indices2):
            tmp = rna_seq_list[index1]
            rna_seq_list[index1] = rna_seq_list[index2]
            rna_seq_list[index2] = tmp
            rna_seq_list = "".join(rna_seq_list)
        return rna_seq_list

    def _get_state(self):
        """
        Get a state dependend on the padded encoding of the target structure.

        Returns:
            The next state.
        """
        return self._padded_encoding[
            self._current_site :
            self._current_site + 2 * self._state_radius + 1
        ]

    def _get_state_matrix(self):
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
        state = state[np.newaxis, ...]
        state = torch.tensor(state, dtype=torch.float32)
        state = self._attention(state)
        state = state.detach().numpy()
        state = np.squeeze(state, axis=0)
        return state

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
        hamming_distance = custom_hamming(self._target, pred_fold)
        # hamming_distance = hamming(pred_fold, self._target)
        print(hamming_distance)

        # if 0 < hamming_distance < 5:
        #     hamming_distance = self._local_improvement(pred_fold)
        # else:
        changed_distance = self._local_improvement_pairs(pred_fold)
        if changed_distance is not None:
            hamming_distance = changed_distance

        print(hamming_distance)
        hamming_distance /= sum([site != "N" for site in self._target])
        reward = (1 - hamming_distance) #** self._reward_exponent
        print()
        print(f"RNA sequence: {self._rna_seq}")
        print(f"Prediction: {pred_fold}")
        print(f"Target: {self._target}")
        print(f"Reward: {reward}")
        return reward
