import os
from pathlib import Path
import shutil

import ConfigSpace as CS
from hpbandster.core.worker import Worker
import numpy as np

from src.learna.agent import AgentConfig, NetworkConfig
from src.learna.environment import RnaDesignEnvironmentConfig
from src.optimization.training import training, evaluate, get_configs


class LearnaWorker(Worker):
    def __init__(self, num_cores, train_sequences, validation_sequences, **kwargs):
        super().__init__(**kwargs)
        self.num_cores = num_cores
        self.train_sequences = train_sequences
        self.validation_sequences = validation_sequences

    def compute(self, config, budget, working_directory, config_id, **kwargs):
        """
        Computes the loss for the given configuration

        Args:
            config: The configuration.
            budget: The budget for a single config.

        Returns:
            The loss for the configuration.
        """
        # config = self._fill_config(config)
        tmp_dir = os.path.join(
            working_directory, "%i_%i_%i" % (config_id[0], config_id[1], config_id[2])
        )
        shutil.rmtree(tmp_dir, ignore_errors=True)
        os.makedirs(tmp_dir, exist_ok=True)

        env_config, agent_config, network_config = get_configs(config)

        agent = training(env_config, agent_config, network_config, self.train_sequences, int(budget))
        rewards = evaluate(env_config, agent, self.validation_sequences, 5)

        save_path = Path(tmp_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        agent.save(directory=save_path.joinpath("last_model"))

        min_distances = 1 - np.max(rewards, 1)
        normalized_solved_sequences = sum(min_distances == 0) / len(self.validation_sequences)
        mean_distance = np.mean(min_distances)

        return {"loss": mean_distance, "info": normalized_solved_sequences}

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "matrix_size", lower=1, upper=128, default_value=16
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.CategoricalHyperparameter(
        #         "padding_mode",
        #         choices=["constant", "maximum", "mean", "reflect", "wrap"],
        #         default_value="constant"
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.CategoricalHyperparameter(
        #         "pad_lower",
        #         choices=[True, False],
        #         default_value=False
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.UniformFloatHyperparameter(
        #         "reward_exponent", lower=1, upper=10, default_value=1
        #     )
        # )
        config_space.add_hyperparameter(
            CS.UniformFloatHyperparameter(
                "learning_rate", lower=1e-6, upper=5e-3, log=True, default_value=1e-4
            )
        )
        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "batch_size", lower=32, upper=128, default_value=32
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.UniformFloatHyperparameter(
        #         "entropy_regularization",
        #         lower=1e-5,
        #         upper=1e-2,
        #         log=True,
        #         default_value=1.5e-3,
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.UniformFloatHyperparameter(
        #         "likelihood_ratio_clipping", lower=0, upper=0.5, default_value=0.3
        #     )
        # )
        # for layer_index in range(1, 17):
        #     config_space.add_hyperparameter(
        #         CS.UniformIntegerHyperparameter(
        #             f"conv_size{layer_index}",
        #             lower=0,
        #             upper=9,
        #             default_value=5
        #         )
        #     )
        #     config_space.add_hyperparameter(
        #         CS.UniformIntegerHyperparameter(
        #             f"conv_channel{layer_index}",
        #             lower=0,
        #             upper=128,
        #             default_value=64
        #         )
        #     )
        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "num_pooling_layers", lower=0, upper=4, default_value=4
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "num_lstm_layers", lower=0, upper=1, default_value=0
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "lstm_units", lower=1, upper=64, default_value=1
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "lstm_horizon", lower=1, upper=64, default_value=12
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.CategoricalHyperparameter(
        #         "fc_activation", choices=["relu", "sigmoid", "tanh"], default_value="relu"
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "fc_units1", lower=0, upper=64, default_value=50
        #     )
        # )
        # config_space.add_hyperparameter(
        #     CS.UniformIntegerHyperparameter(
        #         "fc_units2", lower=0, upper=64, default_value=20
        #     )
        # )

        return config_space
    
    @staticmethod
    def _fill_config(config):
        # Prevent dimension mismatch
        config["num_pooling_layers"] = min(
            config["num_pooling_layers"],
            config["matrix_size"] // 2
        )
        return config
