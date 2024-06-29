import numpy as np
import pandas as pd

import torch

from omegaconf import OmegaConf


def no_param(
    model: torch.nn.Module,
) -> int:
    """
    Calculates the number of parameters of a torch.nn.Module.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def flatten_configdict(
    cfg: OmegaConf,
    separation_mark: str = ".",
):
    """
    Returns a nested OmecaConf dict as a flattened dict, merged with the separation mark.
    Example: With separation_mark == '.', {'data':{'this': 1, 'that': 2} is returned as
    {'data.this': 1, 'data.that': 2}.

    :param cfg:
    :param sep:
    :return:
    """
    cfgdict = OmegaConf.to_container(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=separation_mark)

    return cfgdict.to_dict(orient="records")[0]


def verify_config(cfg: OmegaConf):
    if cfg.train.distributed and cfg.train.avail_gpus < 2:
        raise ValueError(
            f"Distributed only available with more than 1 GPU. Avail={cfg.train.avail_gpus}"
        )
    if cfg.train.batch_size % cfg.train.accumulate_grad_steps:
        raise ValueError(
            f"Batch size must be divisible by the number of grad accumulation steps.\n"
            f"Values: batch_size:{cfg.train.batch_size}, "
            f"accumulate_grad_steps:{cfg.train.accumulate_grad_steps}",
        )
