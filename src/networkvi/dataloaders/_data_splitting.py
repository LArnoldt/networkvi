from math import ceil, floor
from typing import Optional, Union

import lightning.pytorch as pl
import numpy as np
import torch
from torch.utils.data import (
    BatchSampler,
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
)

from networkvi import REGISTRY_KEYS, settings
from networkvi.data import AnnDataManager
from networkvi.data._utils import get_anndata_attribute
from networkvi.dataloaders._ann_dataloader import AnnDataLoader
from networkvi.model._utils import parse_device_args
from networkvi.utils._docstrings import devices_dsp


def validate_data_split(
    n_samples: int, train_size: float, validation_size: Optional[float] = None
):
    """Check data splitting parameters and return n_train and n_val.

    Parameters
    ----------
    n_samples
        Number of samples to split
    train_size
        Size of train set. Need to be: 0 < train_size <= 1.
    validation_size
        Size of validation set. Need to be 0 <= validation_size < 1
    """
    if train_size > 1.0 or train_size <= 0.0:
        raise ValueError("Invalid train_size. Must be: 0 < train_size <= 1")

    n_train = ceil(train_size * n_samples)

    if validation_size is None:
        n_val = n_samples - n_train
    elif validation_size >= 1.0 or validation_size < 0.0:
        raise ValueError("Invalid validation_size. Must be 0 <= validation_size < 1")
    elif (train_size + validation_size) > 1:
        raise ValueError("train_size + validation_size must be between 0 and 1")
    else:
        n_val = floor(n_samples * validation_size)

    if n_train == 0:
        raise ValueError(
            f"With n_samples={n_samples}, train_size={train_size} and "
            f"validation_size={validation_size}, the resulting train set will be empty. Adjust "
            "any of the aforementioned parameters."
        )

    return n_train, n_val


class DataSplitterIndex(pl.LightningDataModule):
    """Creates data loaders ``train_set``, ``validation_set``, ``test_set``.

    If ``train_size + validation_set < 1`` then ``test_set`` is non-empty.

    Parameters
    ----------
    adata_manager
        :class:`~networkvi.data.AnnDataManager` object that has been created via ``setup_anndata``.
    train_size
        float, or None (default is 0.9)
    validation_size
        float, or None (default is None)
    train_idx
        list, ndarray, or None (default is None)
    validation_idx
        list, ndarray, or None (default is None)
    test_idx
        list, ndarray, or None (default is None)
    shuffle_set_split
        Whether to shuffle indices before splitting. If `False`, the val, train, and test set are
        split in the sequential order of the data according to `validation_size` and `train_size`
        percentages.
    load_sparse_tensor
        ``EXPERIMENTAL`` If `True`, loads sparse CSR or CSC arrays in the input dataset as sparse
        :class:`~torch.Tensor` with the same layout. Can lead to significant speedups in
        transferring data to GPUs, depending on the sparsity of the data.
    pin_memory
        Whether to copy tensors into device-pinned memory before returning them. Passed
        into :class:`~networkvi.data.AnnDataLoader`.
    **kwargs
        Keyword args for data loader. If adata has labeled data, data loader
        class is :class:`~networkvi.dataloaders.SemiSupervisedDataLoader`,
        else data loader class is :class:`~networkvi.dataloaders.AnnDataLoader`.

    Examples
    --------
    >>> adata = networkvi.data.synthetic_iid()
    >>> networkvi.model.networkvi.setup_anndata(adata)
    >>> adata_manager = networkvi.model.SCVI(adata).adata_manager
    >>> splitter = DataSplitter(adata)
    >>> splitter.setup()
    >>> train_dl = splitter.train_dataloader()
    """

    data_loader_cls = AnnDataLoader

    def __init__(
        self,
        adata_manager: AnnDataManager,
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        train_idx: np.ndarray | list | None = None,
        validation_idx: np.ndarray | list | None = None,
        test_idx: np.ndarray | list | None = None,
        shuffle_set_split: bool = True,
        load_sparse_tensor: bool = False,
        pin_memory: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.adata_manager = adata_manager
        self.train_size = float(train_size)
        self.validation_size = validation_size
        self.train_idx = train_idx
        self.val_idx = validation_idx
        self.test_idx = test_idx
        self.shuffle_set_split = shuffle_set_split
        self.load_sparse_tensor = load_sparse_tensor
        self.data_loader_kwargs = kwargs
        self.pin_memory = pin_memory

        self.n_train, self.n_val = validate_data_split(
            self.adata_manager.adata.n_obs, self.train_size, self.validation_size
        )

    def setup(self, stage: Optional[str] = None):
        """Split indices in train/test/val sets."""

        n_train = self.n_train
        n_val = self.n_val
        indices = np.arange(self.adata_manager.adata.n_obs)

        if self.shuffle_set_split:
            random_state = np.random.RandomState(seed=settings.seed)
            indices = random_state.permutation(indices)

        if self.train_idx is None:
            self.val_idx = indices[:n_val]
            self.train_idx = indices[n_val : (n_val + n_train)]
            self.test_idx = indices[(n_val + n_train) :]
        else:
            indices = np.setdiff1d(indices, self.train_idx)
            if self.val_idx is None:
                self.val_idx = indices[:n_val]
            if self.test_idx is None:
                self.test_idx = indices[len(self.val_idx) + len(self.train_idx):]

    def train_dataloader(self):
        """Create train data loader."""
        return self.data_loader_cls(
            self.adata_manager,
            indices=self.train_idx,
            shuffle=True,
            drop_last=False,
            load_sparse_tensor=self.load_sparse_tensor,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def val_dataloader(self):
        """Create validation data loader."""
        if len(self.val_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=False,
                drop_last=False,
                load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        """Create test data loader."""
        if len(self.test_idx) > 0:
            return self.data_loader_cls(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=False,
                drop_last=False,
                load_sparse_tensor=self.load_sparse_tensor,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def on_after_batch_transfer(self, batch, dataloader_idx):
        """Converts sparse tensors to dense if necessary."""
        if self.load_sparse_tensor:
            for key, val in batch.items():
                layout = val.layout if isinstance(val, torch.Tensor) else None
                if layout is torch.sparse_csr or layout is torch.sparse_csc:
                    batch[key] = val.to_dense()

        return batch

