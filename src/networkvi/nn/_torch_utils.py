import logging
import torch

def get_activation(name: str):
    activation_name = name.lower()
    act_mapping = {
        "relu": torch.nn.ReLU,
        "lrelu": torch.nn.LeakyReLU,
        "sigmoid": torch.nn.Sigmoid,
        "swish": torch.nn.Hardswish,
        "prelu": torch.nn.PReLU,
        "id": torch.nn.Identity,
        "tanh": torch.nn.Tanh,
        "tanhshrink": torch.nn.Tanhshrink,
        "softsign": torch.nn.Softsign,
        "softshrink": torch.nn.Softshrink,
        "selu": torch.nn.SELU,
        "gelu": torch.nn.GELU,
    }
    return act_mapping[activation_name]()


def get_optimizer(name, lr, weight_decay, params) -> torch.optim.Optimizer:
    """Given a nested config and model parameters, returns a matching optimizer"""
    opt_mapping = {
        "sgd": torch.optim.SGD,
        "adam": torch.optim.Adam,
        "adamw": torch.optim.AdamW,
    }

    if name not in opt_mapping.keys():
        raise ValueError(f"Not a valid optimizer: {name}")
    return opt_mapping[name](params, lr=lr, weight_decay=weight_decay)


def get_schedule(
    name,
    # page_size,
    step_size,
    gamma,
    learning_rate,
    epochs,
    steps_per_epoch: int,
    optimizer: torch.optim.Optimizer,
):
    # TODO: steps per epoch seems to be samples per epoch!
    """Given a nested config, an optimizer and additional parameters, returns a matching learning rate scheduler.
        Currently supports to scheduler types: OneCycleLR and StepLR

    Parameters:
    -----------
    ogmdef: DictConfig
        A nested configuration object
    steps_per_epoch: int
        The number of training steps per epoch
    optimizer: torch.optim.Optimizer
        An Optimizer instance to be passed to the scheduler at initialization
    num_gpus: int
        The number of GPUs to use for the scheduler
    """
    if name == "onecycle":
        logger.info(
            f"Using schedule with {epochs} epochs and {steps_per_epoch} steps per epoch"
        )
        # if num_gpus > 0:
        #     steps_per_epoch = steps_per_epoch // num_gpus # TODO For ddp https://pytorch-lightning.readthedocs.io/en/0.8.3/multi_gpu.html ('Each GPU gets visibility into a subset of the overall dataset. It will only ever see that subset.')
        # steps_per_epoch = (
        #     steps_per_epoch // page_size
        # ) + 2  # +2 because epoch end also does an optimizer step

        schedule = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=learning_rate,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=0.8,
        )
    elif name == "steplr":
        logger.info(
            f"Using StepLR schedule with step size {step_size} and learning rate {gamma}"
        )
        schedule = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer, step_size=step_size, gamma=gamma, last_epoch=-1
        )

    else:
        schedule = None
    return schedule
