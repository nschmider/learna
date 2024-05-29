import ConfigSpace as CS
from ConfigSpace.read_and_write import json
from pathlib import Path

def get_meta_freinet_config():
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "state_radius", lower=0, upper=64, default_value=32
        )
    )
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
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "reward_exponent", lower=1, upper=10, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-6, upper=5e-3, log=True, default_value=1e-4
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "batch_size", lower=32, upper=128, log=True, default_value=32
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "entropy_regularization",
            lower=1e-7,
            upper=1e-2,
            log=True,
            default_value=1.5e-3,
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "likelihood_ratio_clipping", lower=0.01, upper=0.5, log=True, default_value=0.3
        )
    )
    for layer_index in range(1, 5):
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                f"conv_size{layer_index}",
                lower=0,
                upper=19,
                default_value=5
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                f"conv_channel{layer_index}",
                lower=1,
                upper=32,
                log=True,
                default_value=16
            )
        )
    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter(
    #         "num_pooling_layers", lower=0, upper=4, default_value=4
    #     )
    # )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_lstm_layers", lower=0, upper=1, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_units", lower=1, upper=64, log=True, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_horizon", lower=1, upper=64, default_value=5
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "fc_activation", choices=["relu", "sigmoid", "tanh"], default_value="relu"
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units1", lower=1, upper=64, log=True, default_value=50
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units2", lower=1, upper=64, log=True, default_value=20
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "embedding_size", lower=0, upper=4, default_value=3
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "embedding_activation", choices=["relu", "sigmoid", "tanh", "none"], default_value="none"
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "padding", choices=["same", "valid"], default_value="valid"
        )
    )

    return config_space


def get_freinet_config():
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "state_radius", lower=0, upper=64, default_value=32
        )
    )
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
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "reward_exponent", lower=1, upper=10, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-6, upper=5e-3, log=True, default_value=1e-4
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "batch_size", lower=32, upper=128, log=True, default_value=32
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "entropy_regularization",
            lower=1e-7,
            upper=1e-2,
            log=True,
            default_value=1.5e-3,
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "likelihood_ratio_clipping", lower=0.01, upper=0.5, log=True, default_value=0.3
        )
    )
    for layer_index in range(1, 5):
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                f"conv_size{layer_index}",
                lower=0,
                upper=19,
                default_value=5
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                f"conv_channel{layer_index}",
                lower=1,
                upper=32,
                log=True,
                default_value=16
            )
        )
    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter(
    #         "num_pooling_layers", lower=0, upper=4, default_value=4
    #     )
    # )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_lstm_layers", lower=0, upper=1, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_units", lower=1, upper=64, log=True, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_horizon", lower=1, upper=64, default_value=5
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "fc_activation", choices=["relu", "sigmoid", "tanh"], default_value="relu"
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units1", lower=1, upper=64, log=True, default_value=50
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units2", lower=1, upper=64, log=True, default_value=20
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "embedding_size", lower=0, upper=4, default_value=3
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "embedding_activation", choices=["relu", "sigmoid", "tanh", "none"], default_value="none"
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "padding", choices=["same", "valid"], default_value="valid"
        )
    )

    return config_space



def get_fine_tuning_config():
    config_space = CS.ConfigurationSpace()
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "state_radius", lower=0, upper=64, default_value=32
        )
    )
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
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "reward_exponent", lower=1, upper=10, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "learning_rate", lower=1e-6, upper=5e-3, log=True, default_value=1e-4
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "batch_size", lower=32, upper=128, log=True, default_value=32
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "entropy_regularization",
            lower=1e-7,
            upper=1e-2,
            log=True,
            default_value=1.5e-3,
        )
    )
    config_space.add_hyperparameter(
        CS.UniformFloatHyperparameter(
            "likelihood_ratio_clipping", lower=0.01, upper=0.5, log=True, default_value=0.3
        )
    )
    for layer_index in range(1, 5):
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                f"conv_size{layer_index}",
                lower=0,
                upper=19,
                default_value=5
            )
        )
        config_space.add_hyperparameter(
            CS.UniformIntegerHyperparameter(
                f"conv_channel{layer_index}",
                lower=1,
                upper=32,
                log=True,
                default_value=16
            )
        )
    # config_space.add_hyperparameter(
    #     CS.UniformIntegerHyperparameter(
    #         "num_pooling_layers", lower=0, upper=4, default_value=4
    #     )
    # )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "num_lstm_layers", lower=0, upper=1, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_units", lower=1, upper=64, log=True, default_value=1
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "lstm_horizon", lower=1, upper=64, default_value=5
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "fc_activation", choices=["relu", "sigmoid", "tanh"], default_value="relu"
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units1", lower=1, upper=64, log=True, default_value=50
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "fc_units2", lower=1, upper=64, log=True, default_value=20
        )
    )
    config_space.add_hyperparameter(
        CS.UniformIntegerHyperparameter(
            "embedding_size", lower=0, upper=4, default_value=3
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "embedding_activation", choices=["relu", "sigmoid", "tanh", "none"], default_value="none"
        )
    )
    config_space.add_hyperparameter(
        CS.CategoricalHyperparameter(
            "padding", choices=["same", "valid"], default_value="valid"
        )
    )

    return config_space


if __name__ ==  '__main__':
    output_dir = 'results/bohb/6826895/'
    config_space = get_fine_tuning_config()
    print(config_space)
    out_file = Path(output_dir, 'configspace.pcs')

    with open(out_file, 'w') as fh:
        fh.write(json.write(config_space))
