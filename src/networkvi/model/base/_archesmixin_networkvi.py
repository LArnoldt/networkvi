import logging
import warnings
from copy import deepcopy
from typing import Optional, Union

import anndata
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
from scipy.sparse import csr_matrix

from networkvi import REGISTRY_KEYS, settings
from networkvi.data import _constants
from networkvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from networkvi.model._utils import parse_device_args
from networkvi.model.base._save_load import _initialize_model, _load_saved_files, _validate_var_names
from networkvi.nn import FCLayers
from networkvi.utils._docstrings import devices_dsp

from ._base_model import BaseModelClass

logger = logging.getLogger(__name__)

MIN_VAR_NAME_RATIO = 0.8


class ArchesMixinNetworkVI:
    """Universal scArches implementation."""

    @classmethod
    @devices_dsp.dedent
    def load_query_data(
        cls,
        adata: AnnData,
        reference_model: Union[str, BaseModelClass],
        inplace_subset_query_vars: bool = False,
        accelerator: str = "auto",
        device: Union[int, str] = "auto",
        freeze: bool = True,
        unfreeze_first_layers: bool = False,
    ):
        """Online update of a reference model with scArches algorithm :cite:p:`Lotfollahi2022`.

        Parameters
        ----------
        adata
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
        reference_model
            Either an already instantiated model of the same class, or a path to
            saved outputs for reference model.
        inplace_subset_query_vars
            Whether to subset and rearrange query vars inplace based on vars used to
            train reference model.
        %(param_accelerator)s
        %(param_device)s
        freeze
            freeze model
        """
        _, _, device = parse_device_args(
            accelerator=accelerator,
            devices=device,
            return_device="torch",
            validate_single_device=True,
        )

        attr_dict, var_names, load_state_dict = _get_loaded_data(reference_model, device=device)

        if inplace_subset_query_vars:
            logger.debug("Subsetting query vars to reference vars.")
            adata._inplace_subset_var(var_names)
        _validate_var_names(adata, var_names)

        registry = attr_dict.pop("registry_")
        if _MODEL_NAME_KEY in registry and registry[_MODEL_NAME_KEY] != cls.__name__:
            raise ValueError("It appears you are loading a model from a different class.")

        if _SETUP_ARGS_KEY not in registry:
            raise ValueError(
                "Saved model does not contain original setup inputs. "
                "Cannot load the original setup."
            )

        cls.setup_anndata(
            adata,
            source_registry=registry,
            extend_categories=True,
            allow_missing_labels=True,
            **registry[_SETUP_ARGS_KEY],
        )

        n_batch = registry['field_registries']["batch"]["summary_stats"]["n_batch"]
        if "n_cats_per_key" in registry['field_registries']["extra_categorical_covs"]["state_registry"].keys():
            n_extra_categorical_covs = sum(registry['field_registries']["extra_categorical_covs"]["state_registry"]["n_cats_per_key"])
        else:
            n_extra_categorical_covs = 0
        n_batch_query = len(np.unique(adata.obs['_scvi_batch']))
        if not unfreeze_first_layers:
            attr_dict["init_params_"]["non_kwargs"]["n_batch_query"] = n_batch_query
        else:
            attr_dict["init_params_"]["non_kwargs"]["n_batch_query"] = 0
            if attr_dict["init_params_"]["non_kwargs"]["decoder_deeply_inject_covariates"] is not False:
                raise NotImplementedError("")
        model = _initialize_model(cls, adata, attr_dict)
        adata_manager = model.get_anndata_manager(adata, required=True)


        """
        if REGISTRY_KEYS.CAT_COVS_KEY in adata_manager.data_registry:
            raise NotImplementedError(
                "scArches currently does not support models with extra categorical covariates."
            )

        version_split = adata_manager.registry[_constants._SCVI_VERSION_KEY].split(".")
        if int(version_split[1]) < 8 and int(version_split[0]) == 0:
            warnings.warn(
                "Query integration should be performed using models trained with "
                "version >= 0.8",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        """

        model.to_device(device)

        # model tweaking
        new_state_dict = model.module.state_dict()
        for key, load_ten in load_state_dict.items():
            new_ten = new_state_dict[key]
            if new_ten.size() == load_ten.size():
                if "fc_layers" in key and "indices" in key:
                    load_state_dict[key] = new_ten
                continue
            # new categoricals changed size
            else:
                if "AttentionLayer" in key:
                    if "Wk0" in key:
                        fixed_ten = torch.cat([load_ten[:n_batch, :], new_ten[n_batch:n_batch+n_batch_query, :], load_ten[n_batch:, :]], dim=0)
                    elif "Wv0" in key:
                        #dim_diff = new_ten.size()[-2] - load_ten.size()[-2]
                        fixed_ten = new_ten.detach().clone()
                        #fixed_ten[-dim_diff:, -dim_diff:] = load_ten.detach().clone()
                        fixed_ten[:n_batch, :n_batch] = load_ten[:n_batch, :n_batch]
                        fixed_ten[:n_batch, (n_batch + n_batch_query):] = load_ten[:n_batch, n_batch:]
                        fixed_ten[(n_batch + n_batch_query):, :n_batch] = load_ten[n_batch:, :n_batch]
                        fixed_ten[(n_batch + n_batch_query):, (n_batch + n_batch_query):] = load_ten[n_batch:, n_batch:]
                    else:
                        raise NotImplementedError("")
                else:
                    #dim_diff = new_ten.size()[-1] - load_ten.size()[-1]

                    if "fc_layers" in key and "indices" in key:
                        dim_diff = new_ten.shape[1] - load_ten.shape[1]
                        #if dim_diff > 0:
                        new_input_indices_0 = torch.tensor(np.random.choice(max(load_ten.cpu().numpy()[0]), dim_diff, replace=False), device=load_ten.device)
                        new_input_indices_1 = torch.full(new_input_indices_0.shape, max(new_ten.cpu().numpy()[1]), device=new_ten.device)
                        new_input_indices = torch.stack([new_input_indices_0, new_input_indices_1], axis=0)
                        fixed_ten = torch.concat([load_ten, new_input_indices], axis=1)
                        #else:
                        #    fixed_ten = load_ten
                    elif len(load_ten.shape) > 1:
                        len_input = load_ten.shape[1]-(n_batch+n_extra_categorical_covs)
                        fixed_ten = torch.cat([load_ten[..., :len_input+n_batch], new_ten[..., len_input+n_batch:len_input+n_batch+n_batch_query], load_ten[..., len_input+n_batch:]], dim=-1)
                    else:
                        fixed_ten = torch.cat([load_ten, new_ten[load_ten.shape[0]:]], axis=0)
                load_state_dict[key] = fixed_ten

            """
                encoder.encoder.fc_layers[index_encoder][0].indices = reference_encoder.encoder.fc_layers[index_encoder][0].indices.clone()
            else:
                dim_diff = encoder.encoder.fc_layers[index_encoder][0].indices.shape[1] - reference_encoder.encoder.fc_layers[index_encoder][0].indices.shape[1]
                new_input_indices_0 = torch.tensor(np.random.choice(max(reference_encoder.encoder.fc_layers[index_encoder][0].indices.cpu().numpy()[0]), dim_diff, replace=False), device=reference_encoder.encoder.fc_layers[index_encoder][0].indices.device)
                new_input_indices_1 = torch.full(new_input_indices_0.shape, max(encoder.encoder.fc_layers[index_encoder][0].indices.cpu().numpy()[1]), device=reference_encoder.encoder.fc_layers[index_encoder][0].indices.device)
                new_input_indices = torch.stack([new_input_indices_0, new_input_indices_1], axis=0)
                encoder.encoder.fc_layers[index_encoder][0].indices = torch.concat([reference_encoder.encoder.fc_layers[index_encoder][0].indices.clone(), new_input_indices], axis=1)
            """

        ###

        model.module.load_state_dict(load_state_dict)
        model.module.eval()

        if freeze is True:
            for key, par in model.module.named_parameters():
                if not ("AttentionLayer" in key or (unfreeze_first_layers and "fc_layers.Layer 0.0" in key)):
                    par.requires_grad = False

        model.is_trained_ = False

        return model

    @staticmethod
    def prepare_query_anndata(
        adata: AnnData,
        reference_model: Union[str, BaseModelClass],
        return_reference_var_names: bool = False,
        inplace: bool = True,
    ) -> Optional[Union[AnnData, pd.Index]]:
        """Prepare data for query integration.

        This function will return a new AnnData object with padded zeros
        for missing features, as well as correctly sorted features.

        Parameters
        ----------
        adata
            AnnData organized in the same way as data used to train model.
            It is not necessary to run setup_anndata,
            as AnnData is validated against the ``registry``.
        reference_model
            Either an already instantiated model of the same class, or a path to
            saved outputs for reference model.
        return_reference_var_names
            Only load and return reference var names if True.
        inplace
            Whether to subset and rearrange query vars inplace or return new AnnData.

        Returns
        -------
        Query adata ready to use in `load_query_data` unless `return_reference_var_names`
        in which case a pd.Index of reference var names is returned.
        """
        _, var_names, _ = _get_loaded_data(reference_model, device="cpu")
        var_names = pd.Index(var_names)

        if return_reference_var_names:
            return var_names

        intersection = adata.var_names.intersection(var_names)
        inter_len = len(intersection)
        if inter_len == 0:
            raise ValueError(
                "No reference var names found in query data. "
                "Please rerun with return_reference_var_names=True "
                "to see reference var names."
            )

        ratio = inter_len / len(var_names)
        logger.info(f"Found {ratio * 100}% reference vars in query data.")
        if ratio < MIN_VAR_NAME_RATIO:
            warnings.warn(
                f"Query data contains less than {MIN_VAR_NAME_RATIO:.0%} of reference "
                "var names. This may result in poor performance.",
                UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
        genes_to_add = var_names.difference(adata.var_names)
        needs_padding = len(genes_to_add) > 0
        if needs_padding:
            padding_mtx = csr_matrix(np.zeros((adata.n_obs, len(genes_to_add))))
            adata_padding = AnnData(
                X=padding_mtx.copy(),
                layers={layer: padding_mtx.copy() for layer in adata.layers},
            )
            adata_padding.var_names = genes_to_add
            adata_padding.obs_names = adata.obs_names
            # Concatenate object
            adata_out = anndata.concat(
                [adata, adata_padding],
                axis=1,
                join="outer",
                index_unique=None,
                merge="unique",
            )
        else:
            adata_out = adata

        # also covers the case when new adata has more var names than old
        if not var_names.equals(adata_out.var_names):
            adata_out._inplace_subset_var(var_names)

        if inplace:
            if adata_out is not adata:
                adata._init_as_actual(adata_out)
        else:
            return adata_out



def _get_loaded_data(reference_model, device=None):
    if isinstance(reference_model, str):
        attr_dict, var_names, load_state_dict, _ = _load_saved_files(
            reference_model, load_adata=False, map_location=device
        )
    else:
        attr_dict = reference_model._get_user_attributes()
        attr_dict = {a[0]: a[1] for a in attr_dict if a[0][-1] == "_"}
        var_names = reference_model.adata.var_names
        load_state_dict = deepcopy(reference_model.module.state_dict())

    return attr_dict, var_names, load_state_dict
