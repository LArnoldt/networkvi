import logging
import warnings
from collections.abc import Iterable as IterableClass
from collections.abc import Sequence
from typing import Literal, Optional, Union

import jax
import numpy as np
import scipy.sparse as sp_sparse
import torch
from lightning.pytorch.strategies import DDPStrategy, Strategy
from lightning.pytorch.trainer.connectors.accelerator_connector import (
    _AcceleratorConnector,
)

from networkvi import REGISTRY_KEYS, settings
from networkvi._types import Number
from networkvi.data import AnnDataManager
from networkvi.utils._docstrings import devices_dsp

logger = logging.getLogger(__name__)


def use_distributed_sampler(strategy: Union[str, Strategy]) -> bool:
    """``EXPERIMENTAL`` Return whether to use a distributed sampler.

    Currently only supports DDP.
    """
    if isinstance(strategy, str):
        # ["ddp", "ddp_spawn", "ddp_find_unused_parameters_true"]
        return "ddp" in strategy
    return isinstance(strategy, DDPStrategy)


def get_max_epochs_heuristic(
    n_obs: int, epochs_cap: int = 400, decay_at_n_obs: int = 20_000
) -> int:
    """Compute a heuristic for the default number of maximum epochs.

    If ``n_obs <= decay_at_n_obs``, the number of maximum epochs is set to
    ``epochs_cap``. Otherwise, the number of maximum epochs decays according to
    ``(decay_at_n_obs / n_obs) * epochs_cap``, with a minimum of 1. Raises a
    warning if the number of maximum epochs is set to 1.

    Parameters
    ----------
    n_obs
        The number of observations in the dataset.
    epochs_cap
        The maximum number of epochs for the heuristic.
    decay_at_n_obs
        The number of observations at which the heuristic starts decaying.

    Returns
    -------
    A heuristic for the number of maximum training epochs.
    """
    max_epochs = min(round((decay_at_n_obs / n_obs) * epochs_cap), epochs_cap)
    max_epochs = max(max_epochs, 1)

    if max_epochs == 1:
        warnings.warn(
            "The default number of maximum epochs has been set to 1 due to the large"
            "number of observations. Pass in `max_epochs` to the `train` function in "
            "order to override this behavior.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )

    return max_epochs


@devices_dsp.dedent
def parse_device_args(
    accelerator: str = "auto",
    devices: Union[int, list[int], str] = "auto",
    return_device: Optional[Literal["torch", "jax"]] = None,
    validate_single_device: bool = False,
):
    """Parses device-related arguments.

    Parameters
    ----------
    %(param_accelerator)s
    %(param_devices)s
    %(param_return_device)s
    %(param_validate_single_device)s
    """
    valid = [None, "torch", "jax"]
    if return_device not in valid:
        return ValueError(f"`return_device` must be one of {valid}")

    _validate_single_device = validate_single_device and devices != "auto"
    cond1 = isinstance(devices, list) and len(devices) > 1
    cond2 = isinstance(devices, str) and "," in devices
    cond3 = devices == -1
    if _validate_single_device and (cond1 or cond2 or cond3):
        raise ValueError("Only a single device can be specified for `device`.")

    connector = _AcceleratorConnector(accelerator=accelerator, devices=devices)
    _accelerator = connector._accelerator_flag
    _devices = connector._devices_flag

    if _accelerator in ["tpu", "ipu", "hpu"]:
        warnings.warn(
            f"The selected accelerator `{_accelerator}` has not been extensively "
            "tested in scvi-tools. Please report any issues in the GitHub repo.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )
    elif _accelerator == "mps" and accelerator == "auto":
        # auto accelerator should not default to mps
        connector = _AcceleratorConnector(accelerator="cpu", devices=devices)
        _accelerator = connector._accelerator_flag
        _devices = connector._devices_flag
    elif _accelerator == "mps" and accelerator != "auto":
        warnings.warn(
            "`accelerator` has been set to `mps`. Please note that not all PyTorch "
            "operations are supported with this backend. Refer to "
            "https://github.com/pytorch/pytorch/issues/77764 for more details.",
            UserWarning,
            stacklevel=settings.warnings_stacklevel,
        )

    # get the first device index
    if isinstance(_devices, list):
        device_idx = _devices[0]
    elif isinstance(_devices, str) and "," in _devices:
        device_idx = _devices.split(",")[0]
    else:
        device_idx = _devices

    if devices == "auto" and _accelerator != "cpu":
        # auto device should not use multiple devices for non-cpu accelerators
        _devices = [device_idx]

    if return_device == "torch":
        device = torch.device("cpu")
        if _accelerator != "cpu":
            device = torch.device(f"{_accelerator}:{device_idx}")
        return _accelerator, _devices, device
    elif return_device == "jax":
        device = jax.devices("cpu")[0]
        if _accelerator != "cpu":
            device = jax.devices(_accelerator)[device_idx]
        return _accelerator, _devices, device

    return _accelerator, _devices


def scrna_raw_counts_properties(
    adata_manager: AnnDataManager,
    idx1: Union[list[int], np.ndarray],
    idx2: Union[list[int], np.ndarray],
    var_idx: Optional[Union[list[int], np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    """Computes and returns some statistics on the raw counts of two sub-populations.

    Parameters
    ----------
    adata_manager
        :class:`~networkvi.data.AnnDataManager` object setup with :class:`~networkvi.model.SCVI`.
    idx1
        subset of indices describing the first population.
    idx2
        subset of indices describing the second population.
    var_idx
        subset of variables to extract properties from. if None, all variables are used.

    Returns
    -------
    type
        Dict of ``np.ndarray`` containing, by pair (one for each sub-population),
        mean expression per gene, proportion of non-zero expression per gene, mean of normalized
        expression.
    """
    adata = adata_manager.adata
    data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
    data1 = data[idx1]
    data2 = data[idx2]
    if var_idx is not None:
        data1 = data1[:, var_idx]
        data2 = data2[:, var_idx]

    mean1 = np.asarray((data1).mean(axis=0)).ravel()
    mean2 = np.asarray((data2).mean(axis=0)).ravel()
    nonz1 = np.asarray((data1 != 0).mean(axis=0)).ravel()
    nonz2 = np.asarray((data2 != 0).mean(axis=0)).ravel()

    key = "_scvi_raw_norm_scaling"
    if key not in adata.obs.keys():
        scaling_factor = 1 / np.asarray(data.sum(axis=1)).ravel().reshape(-1, 1)
        scaling_factor *= 1e4
        adata.obs[key] = scaling_factor.ravel()
    else:
        scaling_factor = adata.obs[key].to_numpy().ravel().reshape(-1, 1)

    if issubclass(type(data), sp_sparse.spmatrix):
        norm_data1 = data1.multiply(scaling_factor[idx1])
        norm_data2 = data2.multiply(scaling_factor[idx2])
    else:
        norm_data1 = data1 * scaling_factor[idx1] #(data1.T * scaling_factor[idx1]).T #data1 * scaling_factor[idx1]
        norm_data2 = data2 * scaling_factor[idx2] #(data2.T * scaling_factor[idx2]).T #data2 * scaling_factor[idx2]

    norm_mean1 = np.asarray(norm_data1.mean(axis=0)).ravel()
    norm_mean2 = np.asarray(norm_data2.mean(axis=0)).ravel()

    properties = {
        "raw_mean1": mean1,
        "raw_mean2": mean2,
        "non_zeros_proportion1": nonz1,
        "non_zeros_proportion2": nonz2,
        "raw_normalized_mean1": norm_mean1,
        "raw_normalized_mean2": norm_mean2,
    }
    return properties

def scatac_raw_counts_properties(
    adata_manager: AnnDataManager,
    idx1: Union[list[int], np.ndarray],
    idx2: Union[list[int], np.ndarray],
    var_idx: Optional[Union[list[int], np.ndarray]] = None,
) -> dict[str, np.ndarray]:
    """Computes and returns some statistics on the raw counts of two sub-populations.

    Parameters
    ----------
    adata_manager
        :class:`~networkvi.data.AnnDataManager` object setup with :class:`~networkvi.model.SCVI`.
    idx1
        subset of indices describing the first population.
    idx2
        subset of indices describing the second population.
    var_idx
        subset of variables to extract properties from. if None, all variables are used.

    Returns
    -------
    type
        Dict of ``np.ndarray`` containing, by pair (one for each sub-population).
    """
    data = adata_manager.get_from_registry(REGISTRY_KEYS.X_KEY)
    data1 = data[idx1]
    data2 = data[idx2]
    if var_idx is not None:
        data1 = data1[:, var_idx]
        data2 = data2[:, var_idx]
    mean1 = np.asarray((data1 > 0).mean(axis=0)).ravel()
    mean2 = np.asarray((data2 > 0).mean(axis=0)).ravel()
    properties = {"emp_mean1": mean1, "emp_mean2": mean2, "emp_effect": (mean1 - mean2)}
    return properties

def _get_batch_code_from_category(
    adata_manager: AnnDataManager, category: Sequence[Union[Number, str]]
):
    if not isinstance(category, IterableClass) or isinstance(category, str):
        category = [category]

    batch_mappings = adata_manager.get_state_registry(REGISTRY_KEYS.BATCH_KEY).categorical_mapping
    batch_code = []
    for cat in category:
        if cat is None:
            batch_code.append(None)
        elif cat not in batch_mappings:
            raise ValueError(f'"{cat}" not a valid batch category.')
        else:
            batch_loc = np.where(batch_mappings == cat)[0][0]
            batch_code.append(batch_loc)
    return batch_code

