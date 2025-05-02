
    # if config["optimizer"] == "adam":
    #     optimizer = optax.adam(config["learning_rate"])
    # elif config["optimizer"] == "adamw":
    #     optimizer = optax.adamw(config["learning_rate"])
    # elif config["optimizer"] == "rmsprop":
    #     optimizer = optax.rmsprop(config["learning_rate"])
    # elif config["optimizer"] == "muon":
    #     optimizer = optax.contrib.muon(config["learning_rate"])
    # else:
    #     optimizer = optax.sgd(config["learning_rate"])

from typing import TypedDict
import optax


class OptimizerBankConfig(TypedDict):
    optimizer: str
    learning_rate: float



def get_optimizer(config: OptimizerBankConfig) -> optax.GradientTransformation:
    if config["optimizer"] == "adam":
        return optax.adam(config["learning_rate"])
    elif config["optimizer"] == "adamw":
        return optax.adamw(config["learning_rate"])
    elif config["optimizer"] == "rmsprop":
        return optax.rmsprop(config["learning_rate"])
    elif config["optimizer"] == "muon":
        return optax.contrib.muon(config["learning_rate"])
    else:
        return optax.sgd(config["learning_rate"])
