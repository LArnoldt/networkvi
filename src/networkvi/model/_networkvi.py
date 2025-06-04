from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable, Sequence
from collections.abc import Iterable as IterableClass
from functools import partial
from typing import Literal, Optional, Union, List

import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from anndata import AnnData
from scipy.sparse import csr_matrix, vstack
from torch.distributions import Normal
from tqdm import tqdm
import pickle
from collections import defaultdict

from networkvi import REGISTRY_KEYS, settings
from networkvi._types import Number
from networkvi.data import AnnDataManager
from networkvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
    ProteinObsmField,
)
from networkvi.model._utils import (
    _get_batch_code_from_category,
    scatac_raw_counts_properties,
    scrna_raw_counts_properties,
)
from networkvi.model.base import (
    ArchesMixinNetworkVI,
    BaseModelClass,
    UnsupervisedTrainingMixin,
    VAEMixin,
)
from networkvi.model.base._de_core import _de_core
from networkvi.module import NETWORKVAE
from networkvi.train import AdversarialTrainingPlan
from networkvi.train._callbacks import SaveBestState
from networkvi.utils._docstrings import de_dsp, devices_dsp, setup_anndata_dsp
from networkvi.dataloaders._data_splitting import validate_data_split

from networkvi.nn._introspections import calculate_interpretations, goobj_to_csv
from networkvi.nn._evaluate_funcs import goobj_to_graphml, make_json_str, create_html

logger = logging.getLogger(__name__)

AllowedStr = Literal["protein", "expression", "accessibility"]
CalculatePerturbationsModality = Union[bool, AllowedStr, List[AllowedStr], np.ndarray]

class NETWORKVI(VAEMixin, UnsupervisedTrainingMixin, BaseModelClass, ArchesMixinNetworkVI):
    """Integration of multi-moda data employing domain knowledge-driven neural networks :cite:p:`Arnoldt2024`.

    NetworkVI performs paired and mosaic integration of multiomic datasets using sparse encoders.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~networkvi.model.MULTIVI.setup_anndata`.
    n_regions
        The number of accessibility features (genomic regions).
    ensembl_ids_regions
        ENSEMBL-IDs of accessibility features (genomic regions).
    n_genes
        The number of gene expression features (genes).
    ensembl_ids_genes
        ENSEMBL-IDs of gene expression features (genes).
    ensembl_ids_proteins
        ENSEMBL-IDs of surface proteins (proteins).
    n_patient_covariates
        The number of patients.
    modality_weights
        Weighting scheme across modalities. One of the following:
        * ``"equal"``: Equal weight in each modality
        * ``"universal"``: Learn weights across modalities w_m.
        * ``"cell"``: Learn weights across modalities and cells. w_{m,c}
        * ``"moe"``: Learn weights with gating network for MoE.
    modality_penalty
        Training Penalty across modalities. One of the following:
        * ``"Jeffreys"``: Jeffreys penalty to align modalities
        * ``"MMD"``: MMD penalty to align modalities
        * ``"None"``: No penalty
    n_hidden
        Number of nodes per hidden layer. If `None`, defaults to square root
        of number of regions.
    n_latent
        Dimensionality of the latent space. If `None`, defaults to square root
        of `n_hidden`.
    n_layers_encoder
        Number of hidden layers used for encoder NNs.
    layers_encoder_type
        Type of hidden layers used for encoder NNs.
        Type of layer. One of the following
        * ``'linear'`` - Linear Layers
        * ``'go'`` - GO Layers
    n_layers_decoder
        Number of hidden layers used for decoder NNs.
    layers_decoder_type
        Type of hidden layers used for decoder NNs.
        * ``'linear'`` - Linear Layers
        * ``'go'`` - GO Layers
    expression_gene_layer_type
        Type of expression gene layer. One of the following
        * ``'none'`` - No gene layer
        * ``'standard'`` - Standard Gene Layer
        * ``'interaction'`` - Interaction Gene Layer
    accessibility_gene_layer_type
        Type of accessibility gene layer. One of the following
        * ``'none'`` - No gene layer
        * ``'standard'`` - Standard Gene Layer
        * ``'interaction'`` - Interaction Gene Layer
    protein_gene_layer_type
        Type of protein gene layer. One of the following
        * ``'none'`` - No gene layer
        * ``'standard'`` - Standard Gene Layer
        * ``'interaction'`` - Interaction Gene Layer
    gene_layer_interaction_source
        Gene layer interaction source. One of the following
        * ``'pp'`` - Protein-Protein
        * ``'tf'`` - Transcription Factor
        * ``'tad'`` - Topologically Associated Domains
    standard_gene_size
        Standard size of gene nodes in Gene Layers.
    standard_go_size
        Standard size of GO nodes in GO Layers.
    obo_file
        Path .obo file of GO.
    map_ensembl_go
        List of .gaf files with mappings of Ensembl IDs to GO.
    keep_activations
        Bool, whether keep activations in fully-connected encoder layers.
    use_mean_mixing
        Bool, whether perform mean modality mixing.
    use_product_of_experts
        Bool, whether perform modality mixing with PoE.
    use_mixture_of_experts
        Bool, whether perform modality mixing with MoE.
    dropout_rate
        Dropout rate for neural networks.
    model_depth
        Model sequencing depth / library size.
    region_factors
        Include region-specific factors in the model.
    gene_dispersion
        One of the following
        * ``'gene'`` - genes_dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - genes_dispersion can differ between different batches
        * ``'gene-label'`` - genes_dispersion can differ between different labels
    protein_dispersion
        One of the following
        * ``'protein'`` - protein_dispersion parameter is constant per protein across cells
        * ``'protein-batch'`` - protein_dispersion can differ between different batches NOT TESTED
        * ``'protein-label'`` - protein_dispersion can differ between different labels NOT TESTED
    latent_distribution
        One of
        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    deeply_inject_covariates
        Whether to deeply inject covariates into all layers of the endecoder. If False,
        covariates will only be included in the input layer.
    first_layer_inject_covariates
        Whether to deeply inject covariates into all layers of the decoder. If False,
        covariates will only be included in the input layer.
    last_layer_inject_covariates
        Whether to inject covariates into all layers of the decoder. If False,
        covariates will only be included in the input layer.
    fully_paired
        allows the simplification of the model if the data is fully paired. Currently ignored.
    **model_kwargs
        Keyword args for :class:`~networkvi.module.NETWORKVAE`

    Examples
    --------
    >>> adata_rna = anndata.read_h5ad(path_to_rna_anndata)
    >>> adata_atac = networkvi.data.read_10x_atac(path_to_atac_anndata)
    >>> adata_multi = networkvi.data.read_10x_multiome(path_to_multiomic_anndata)
    >>> adata_svi = networkvi.data.organize_multiome_anndatas(adata_multi, adata_rna, adata_atac)
    >>> networkvi.model.networkvi.setup_anndata(adata_svi, ensembl_ids_genes=adata_rna.var["ensembl_ids"], ensembl_ids_regions=adata_atac.var["ensembl_ids"], batch_key="modality")
    >>> vae = networkvi.model.NetworkVI(adata_svi)
    >>> vae.train()

    Notes
    -----
    * The model assumes that the features are organized so that all expression features are
       consecutive, followed by all accessibility features. For example, if the data has 100 genes
       and 250 genomic regions, the model assumes that the first 100 features are genes, and the
       next 250 are the regions.

    * The main batch annotation, specified in ``setup_anndata``, should correspond to
       the modality each cell originated from. This allows the model to focus mixing efforts, using
       an adversarial component, on mixing the modalities. Other covariates can be specified using
       the `categorical_covariate_keys` argument.
    """

    _module_cls = NETWORKVAE
    _training_plan_cls = AdversarialTrainingPlan

    def __init__(
        self,
        adata: AnnData,
        n_genes: int = 0,
        ensembl_ids_genes: np.ndarray | None = None,
        n_regions: int = 0,
        ensembl_ids_regions: np.ndarray | None = None,
        ensembl_ids_proteins: np.ndarray | None = None,
        n_patient_covariates: int = 0,
        modality_weights: Literal["equal", "cell", "universal", "moe"] = "moe",
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",
        n_hidden: int | None = None,
        n_latent: int | None = None,
        n_layers_encoder: int = 2,
        layers_encoder_type: Literal["linear", "go"] = "go",
        n_layers_decoder: int = 2,
        layers_decoder_type: Literal["linear"] = "linear",
        sparsities: list = [0.9, 0.9],
        dynamic: bool = True,
        dynamic_update_rate: Optional[int] = None,
        dynamic_end_update_rate: Optional[str] = None,
        gene_interaction_layer_dynamic: bool = False,
        gene_interaction_layer_pruning_frac: Optional[float] = None,
        gene_interaction_layer_dynamic_update_rate: Optional[int] = None,
        gene_interaction_layer_dynamic_end_update_rate: Optional[int] = None,
        gene_interaction_layer_dynamic_save_path: Optional[str] = None,
        keep_activations: bool = False,
        expression_gene_layer_type: Literal["none", "standard", "interaction"] = "interaction",
        accessibility_gene_layer_type: Literal["none", "standard", "interaction"] = "interaction",
        protein_gene_layer_type: Literal["none", "standard", "interaction"] = "interaction",
        gene_layer_interaction_source: Optional[str] = None,
        standard_gene_size: int = 4,
        standard_go_size: int = 6,
        obo_file: Optional[str] = None,
        map_ensembl_go: Optional[Union[list, np.ndarray]] = None,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        decoder_deeply_inject_covariates: bool = False,
        first_layer_inject_covariates: bool = False,
        last_layer_inject_covariates: bool = False,
        encode_covariates: bool = False,
        fully_paired: bool = False,
        protein_dispersion: Literal["protein", "protein-batch", "protein-label"] = "protein",
        activation_fn: nn.Module = nn.ReLU,
        use_mean_mixing: bool = False,
        use_product_of_experts: bool = False,
        use_mixture_of_experts: bool = True,
        **model_kwargs,
    ):
        super().__init__(adata)

        prior_mean, prior_scale = None, None
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else []
        )

        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry

        if "n_proteins" in self.summary_stats:
            n_proteins = self.summary_stats.n_proteins
        else:
            n_proteins = 0

        self.module = self._module_cls(
            n_input_genes=n_genes,
            ensembl_ids_genes=ensembl_ids_genes,
            n_input_regions=n_regions,
            ensembl_ids_regions=ensembl_ids_regions,
            n_input_proteins=n_proteins,
            ensembl_ids_proteins=ensembl_ids_proteins,
            n_patient_covariates=n_patient_covariates,
            modality_weights=modality_weights,
            modality_penalty=modality_penalty,
            n_batch=self.summary_stats.n_batch,
            n_obs=adata.n_obs,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers_encoder=n_layers_encoder,
            layers_encoder_type=layers_encoder_type,
            n_layers_decoder=n_layers_decoder,
            layers_decoder_type=layers_decoder_type,
            sparsities=sparsities,
            dynamic=dynamic,
            dynamic_update_rate=dynamic_update_rate,
            dynamic_end_update_rate=dynamic_end_update_rate,
            gene_interaction_layer_dynamic=gene_interaction_layer_dynamic,
            gene_interaction_layer_pruning_frac=gene_interaction_layer_pruning_frac,
            gene_interaction_layer_dynamic_update_rate=gene_interaction_layer_dynamic_update_rate,
            gene_interaction_layer_dynamic_end_update_rate=gene_interaction_layer_dynamic_end_update_rate,
            gene_interaction_layer_dynamic_save_path=gene_interaction_layer_dynamic_save_path,
            keep_activations=keep_activations,
            expression_gene_layer_type=expression_gene_layer_type,
            accessibility_gene_layer_type=accessibility_gene_layer_type,
            protein_gene_layer_type=protein_gene_layer_type,
            gene_layer_interaction_source=gene_layer_interaction_source,
            standard_gene_size=standard_gene_size,
            standard_go_size=standard_go_size,
            obo_file=obo_file,
            map_ensembl_go=map_ensembl_go,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            dropout_rate=dropout_rate,
            region_factors=region_factors,
            gene_likelihood=gene_likelihood,
            gene_dispersion=dispersion,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            use_size_factor_key=use_size_factor_key,
            latent_distribution=latent_distribution,
            deeply_inject_covariates=deeply_inject_covariates,
            decoder_deeply_inject_covariates=decoder_deeply_inject_covariates,
            first_layer_inject_covariates=first_layer_inject_covariates,
            last_layer_inject_covariates=last_layer_inject_covariates,
            encode_covariates=encode_covariates,
            protein_background_prior_mean=prior_mean,
            protein_background_prior_scale=prior_scale,
            protein_dispersion=protein_dispersion,
            activation_fn=activation_fn,
            use_mean_mixing=use_mean_mixing,
            use_product_of_experts=use_product_of_experts,
            use_mixture_of_experts=use_mixture_of_experts,
            **model_kwargs,
        )
        self._model_summary_string = (
            f"NetworkVI Model with the following params: \nn_genes: {n_genes}, "
            f"n_regions: {n_regions}, n_patient_covariates: {n_patient_covariates}, n_proteins: {n_proteins}, n_hidden: {self.module.n_hidden}, "
            f"n_latent: {self.module.n_latent}, n_layers_encoder: {n_layers_encoder}, "
            f"n_layers_decoder: {n_layers_decoder}, dropout_rate: {dropout_rate}, "
            f"layers_encoder_type: {layers_encoder_type}, layers_decoder_type: {layers_decoder_type}, "
            f"gene_layer_type: {expression_gene_layer_type}, accessibility_gene_layer_type: {accessibility_gene_layer_type}, "
            #f"library_size_layers_type: {library_size_layers_type}, accessibility_library_size_gene_layer_type: {accessibility_library_size_gene_layer_type}, "
            #f"expression_library_size_gene_layer_type: {expression_library_size_gene_layer_type},  protein_library_size_gene_layer_type: {protein_library_size_gene_layer_type}, "
            f"latent_distribution: {latent_distribution}, deeply_inject_covariates: {deeply_inject_covariates}, decoder_deeply_inject_covariates: {decoder_deeply_inject_covariates}"
            f"first_layer_inject_covariates: {first_layer_inject_covariates}, last_layer_inject_covariates: {last_layer_inject_covariates}, "
            f"gene_likelihood: {gene_likelihood}, "
            f"gene_dispersion:{dispersion}, Mod.Weights: {modality_weights}, "
            f"Mod.Penalty: {modality_penalty}, protein_dispersion: {protein_dispersion}"
        )

        self.fully_paired = fully_paired
        self.n_latent = n_latent
        self.init_params_ = self._get_init_params(locals())
        self.n_genes = n_genes
        self.n_regions = n_regions
        self.n_proteins = n_proteins

    @devices_dsp.dedent
    def train(
        self,
        max_epochs: int = 500,
        lr: float = 1e-4,
        accelerator: str = "auto",
        devices: int | list[int] | str = "auto",
        train_size: float = 0.9,
        validation_size: float | None = None,
        train_idx: np.ndarray | list | None = None,
        validation_idx: np.ndarray | list | None = None,
        test_idx: np.ndarray | list | None = None,
        shuffle_set_split: bool = True,
        batch_size: int = 128,
        weight_decay: float = 1e-3,
        eps: float = 1e-08,
        early_stopping: bool = True,
        save_best: bool = True,
        check_val_every_n_epoch: int | None = None,
        n_steps_kl_warmup: int | None = None,
        n_epochs_kl_warmup: int | None = 50,
        adversarial_mixing: bool = True,
        datasplitter_kwargs: dict | None = None,
        plan_kwargs: dict | None = None,
        **kwargs,
    ):
        """Trains the model using amortized variational inference.

        Parameters
        ----------
        max_epochs
            Number of passes through the dataset.
        lr
            Learning rate for optimization.
        %(param_accelerator)s
        %(param_devices)s
        train_size
            Size of training set in the range [0.0, 1.0].
        validation_size
            Size of the test set. If `None`, defaults to 1 - `train_size`. If
            `train_size + validation_size < 1`, the remaining cells belong to a test set.
        shuffle_set_split
            Whether to shuffle indices before splitting. If `False`, the val, train, and test set
            are split in the sequential order of the data according to `validation_size` and
            `train_size` percentages.
        batch_size
            Minibatch size to use during training.
        weight_decay
            weight decay regularization term for optimization
        eps
            Optimizer eps
        early_stopping
            Whether to perform early stopping with respect to the validation set.
        save_best
            ``DEPRECATED`` Save the best model state with respect to the validation loss, or use
            the final state in the training procedure.
        check_val_every_n_epoch
            Check val every n train epochs. By default, val is not checked, unless `early_stopping`
            is `True`. If so, val is checked every epoch.
        n_steps_kl_warmup
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. If `None`, defaults
            to `floor(0.75 * adata.n_obs)`.
        n_epochs_kl_warmup
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`.
        adversarial_mixing
            Whether to use adversarial training to penalize the model for umbalanced mixing of
            modalities.
        datasplitter_kwargs
            Additional keyword arguments passed into :class:`~networkvi.dataloaders.DataSplitter`.
        plan_kwargs
            Keyword args for :class:`~networkvi.train.TrainingPlan`. Keyword arguments passed to
            `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        **kwargs
            Other keyword args for :class:`~networkvi.train.Trainer`.

        Notes
        -----
        ``save_best`` is deprecated in v1.2 and will be removed in v1.3. Please use
        ``enable_checkpointing`` instead.
        """
        update_dict = {
            "lr": lr,
            "adversarial_classifier": adversarial_mixing,
            "weight_decay": weight_decay,
            "eps": eps,
            "n_epochs_kl_warmup": n_epochs_kl_warmup,
            "n_steps_kl_warmup": n_steps_kl_warmup,
            "optimizer": "AdamW",
            "scale_adversarial_loss": 1,
        }
        if plan_kwargs is not None:
            plan_kwargs.update(update_dict)
        else:
            plan_kwargs = update_dict

        datasplitter_kwargs = datasplitter_kwargs or {}

        if save_best:
            warnings.warn(
                "`save_best` is deprecated in v1.2 and will be removed in v1.3. Please use "
                "`enable_checkpointing` instead. See "
                "https://github.com/scverse/scvi-tools/issues/2568 for more details.",
                DeprecationWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            if "callbacks" not in kwargs.keys():
                kwargs["callbacks"] = []
            kwargs["callbacks"].append(SaveBestState(monitor="reconstruction_loss_validation"))

        data_splitter = self._data_splitter_cls(
            self.adata_manager,
            train_size=train_size,
            validation_size=validation_size,
            train_idx=train_idx,
            validation_idx=validation_idx,
            test_idx=test_idx,
            shuffle_set_split=shuffle_set_split,
            batch_size=batch_size,
            **datasplitter_kwargs,
        )
        training_plan = self._training_plan_cls(self.module, **plan_kwargs)
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            early_stopping=early_stopping,
            check_val_every_n_epoch=check_val_every_n_epoch,
            early_stopping_monitor="reconstruction_loss_validation",
            early_stopping_patience=50,
            #**kwargs,
        )
        return runner()

    @torch.inference_mode()
    def get_perturbation_go_gene_importances(
        self,
        labels_column: str,
        modality_key: str,
        gene_stable_id_key: str,
        perturbation_gene_stable_ids: list | np.ndarray,
        labels_selection: str | int | float | np.ndarray | list | set | bool = False,
        labels_selection_groups: np.ndarray | list | set | bool = False,
        adata: AnnData | None = None,
        indices: Sequence[int] = None,
        batch_size: int = 128,
        train_size: float = 0.9,
        validation_size: float | None = None,
        train_idx: np.ndarray | list | None = None,
        validation_idx: np.ndarray | list | None = None,
        shuffle_set_split: bool = True,
        restrict_unpaired_activations_mask: bool = False,
        restrict_unpaired_activations_mask_flip: bool = False,
        restrict_by_column_key_activations_mask: str | int | float | bool = False,
        restrict_by_column_values_activations_mask: str | int | float | np.ndarray | list | set | bool = False,
        restrict_by_column_values_alt_des_activations_mask: str | bool = False,
        comparison: str | int | float | np.ndarray | list | set = "rest",
        comparison_alt_des: str | bool = False,
        calc_go_terms: bool = True,
        calc_genes: bool = False,
        calc_gene_groups: bool | str | list | nd.array = False,
        restrict_modalities: bool | str | list | nd.array = True,
        load_save_fit: bool = False,
        save_fit: bool = True,
        overwrite_save_fit: bool = True,
        save_results: bool = True,
        results_dir: str = "",
        calculate_perturbations_modality: CalculatePerturbationsModality = ["protein"],
        calculate_all_modality: bool = False,
        keep_activations: bool = False,
    ) -> dict[str, dict[str, dict[str, np.ndarray]]]:

        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)

        peturbation_graph_csv_result_agg = {}

        ###

        if comparison != "rest":
            if type(comparison) == list or type(comparison) == set or type(comparison) == np.ndarray:
                if comparison_alt_des is not False:
                    comparison_str = f"_vs_{comparison_alt_des}"
                else:
                    comparison_str = f"_vs_{'_'.join(comparison)}"
            else:
                if comparison_alt_des is not False:
                    comparison_str = f"_vs_{comparison_alt_des}"
                else:
                    comparison_str = f"_vs_{comparison}"
        else:
            comparison_str = f"_vs_{comparison}"

        if restrict_by_column_key_activations_mask is not False and restrict_by_column_values_activations_mask is not False:
            if type(restrict_by_column_values_activations_mask) == list or type(
                restrict_by_column_values_activations_mask) == set or type(
                restrict_by_column_values_activations_mask) == np.ndarray:
                if restrict_by_column_values_alt_des_activations_mask is not False:
                    restrict_by_column_key_values_str = f"_res_{restrict_by_column_key_activations_mask}_{restrict_by_column_values_alt_des_activations_mask}"
                else:
                    restrict_by_column_key_values_str = f"_res_{restrict_by_column_key_activations_mask}_{'_'.join(restrict_by_column_values_activations_mask)}"
            else:
                restrict_by_column_key_values_str = f"_res_{restrict_by_column_key_activations_mask}_{restrict_by_column_values_activations_mask}"
        else:
            restrict_by_column_key_values_str = f""

        ###

        for peturbation_gene_stable_id in perturbation_gene_stable_ids:
            adata_perturbation = adata.copy()

            peturbation_graph_csv_result_agg[peturbation_gene_stable_id] = {}

            perturbation_des_all = []
            perturbation_indices_all = []

            for modality in np.unique(adata.var[modality_key]):
                if (calculate_perturbations_modality is True or modality in calculate_perturbations_modality):
                    modality_perturbation_indices = np.where((adata_perturbation.var[gene_stable_id_key] == peturbation_gene_stable_id) & (adata.var[modality_key] == modality))[0]
                    if len(modality_perturbation_indices) > 0:
                        perturbation_indices_all.append(modality_perturbation_indices)
                        perturbation_des_all.append(f"{modality}")
            if "protein_expression" in adata_perturbation.obsm.keys() and (calculate_perturbations_modality is True or "protein" in calculate_perturbations_modality):
                perturbation_indices_prot = np.where(adata_perturbation.uns['protein_expression']["var"]["gene_stable_id"] == peturbation_gene_stable_id)[0]
                if len(perturbation_indices_prot) > 0:
                    perturbation_indices_all.append(perturbation_indices_prot)
                    perturbation_des_all.append(f"protein")
            else:
                perturbation_indices_prot = []

            if len(perturbation_indices_all) == 0:
                raise ValueError(f"Perturbation gene {peturbation_gene_stable_id} not in adata.")

            if calculate_all_modality and len(perturbation_indices_all) > 1:
                perturbation_indices_all.append(np.array([index for modality_perturbation_indices in perturbation_indices_all for index in modality_perturbation_indices]))
                perturbation_des_all.append("all_modality")

            for perturbation_des, perturbation_indices in zip(perturbation_des_all, perturbation_indices_all):

                peturbation_graph_csv_result_agg[peturbation_gene_stable_id][perturbation_des] = {}

                if perturbation_des == "protein":
                    adata_perturbation.obsm["protein_expression"][:, perturbation_indices_prot] = np.zeros(shape=(adata_perturbation.obsm["protein_expression"].shape[0], len(perturbation_indices)))
                else:
                    if len(perturbation_indices) > 0:
                        adata_perturbation.X[:, perturbation_indices] = np.zeros(shape=(adata_perturbation.shape[0], len(perturbation_indices)))
                    if perturbation_des == "all_modality" and "protein_expression" in adata_perturbation.obsm.keys() and len(perturbation_indices_prot) > 0:
                        adata_perturbation.obsm["protein_expression"][:, perturbation_indices_prot] = np.zeros(shape=(adata_perturbation.obsm["protein_expression"].shape[0], len(perturbation_indices_prot)))

                if restrict_modalities is False or perturbation_des == "all_modality":
                    restrict_modalities_val = restrict_modalities
                else:
                    restrict_modalities_val = perturbation_des

                peturbation_graph_csv_result = self.get_go_gene_importances(
                    labels_column=labels_column,
                    labels_selection=labels_selection,
                    labels_selection_groups=labels_selection_groups,
                    adata=adata_perturbation,
                    indices=indices,
                    batch_size=batch_size,
                    train_size=train_size,
                    validation_size=validation_size,
                    train_idx=train_idx,
                    validation_idx=validation_idx,
                    shuffle_set_split=shuffle_set_split,
                    restrict_unpaired_activations_mask=restrict_unpaired_activations_mask,
                    restrict_unpaired_activations_mask_flip=restrict_unpaired_activations_mask_flip,
                    restrict_by_column_key_activations_mask=restrict_by_column_key_activations_mask,
                    restrict_by_column_values_activations_mask=restrict_by_column_values_activations_mask,
                    restrict_by_column_values_alt_des_activations_mask=restrict_by_column_values_alt_des_activations_mask,
                    comparison=comparison,
                    comparison_alt_des=comparison_alt_des,
                    calc_go_terms=calc_go_terms,
                    calc_genes=calc_genes,
                    calc_gene_groups=calc_gene_groups,
                    restrict_modalities=restrict_modalities_val,
                    load_save_fit=False,
                    save_fit=False,
                    overwrite_save_fit=False,
                    save_results=False,
                    results_dir=results_dir,
                    keep_activations=keep_activations,
                )

                for modality in peturbation_graph_csv_result.keys():
                    peturbation_graph_csv_result_agg[peturbation_gene_stable_id][perturbation_des][modality] = {}
                    for phenotype in peturbation_graph_csv_result[modality].keys():
                        peturbation_graph_csv_result_agg[peturbation_gene_stable_id][perturbation_des][modality][phenotype] = np.array(peturbation_graph_csv_result[modality][phenotype]['predictors_label_valid_roc'])

            if save_results:
                with open(os.path.join(results_dir, f'{labels_column}{comparison_str}_res_unp_act_mask_{restrict_unpaired_activations_mask}_flip_{restrict_unpaired_activations_mask_flip}{restrict_by_column_key_values_str}{"_go_terms" if calc_go_terms else ""}_perturbation_{peturbation_gene_stable_id}_evaluated_graph_csv.pkl'), 'wb') as f:
                    pickle.dump(peturbation_graph_csv_result_agg[peturbation_gene_stable_id], f)

        return peturbation_graph_csv_result_agg

    @torch.inference_mode()
    def get_go_gene_importances(
        self,
        labels_column: str | None = None,
        labels_selection: str | int | float | np.ndarray | list | set | bool = False,
        labels_selection_groups: np.ndarray | list | set | bool = False,
        adata: AnnData | None = None,
        indices: Sequence[int] = None,
        batch_size: int = 128,
        train_size: float = 0.9,
        validation_size: float | None = None,
        train_idx: np.ndarray | list | None = None,
        validation_idx: np.ndarray | list | None = None,
        shuffle_set_split: bool = True,
        restrict_unpaired_activations_mask: bool = False,
        restrict_unpaired_activations_mask_flip: bool = False,
        restrict_by_column_key_activations_mask: str | int | float | bool = False,
        restrict_by_column_values_activations_mask: str | int | float | np.ndarray | list | set | bool = False,
        restrict_by_column_values_alt_des_activations_mask: str | bool = False,
        comparison: str | int | float | np.ndarray | list | set = "rest",
        comparison_alt_des: str | bool = False,
        calc_go_terms: bool = True,
        calc_genes: bool = False,
        calc_gene_groups: bool | str | list | nd.array = False,
        restrict_modalities: bool | str | list | nd.array = False,
        load_save_fit: bool = False,
        save_fit: bool = True,
        overwrite_save_fit: bool = True,
        save_results: bool = True,
        results_dir: str = "",
        keep_activations: bool = False,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Return gene and GO importances.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `networkvi.settings.batch_size`.

        Returns
        -------
        Gene and GO importances for expression and accessibility
        """

        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        if labels_column is not None:
            labels = adata.obs[labels_column]

        if not calc_go_terms and not calc_genes and not calc_gene_groups:
            raise NotImplementedError("No RLIPP calculation entities selected.")

        self.train_idx = train_idx
        self.val_idx = validation_idx
        self.n_train, self.n_val = validate_data_split(
            self.adata_manager.adata.n_obs, train_size, validation_size
        )

        indices = np.arange(self.adata_manager.adata.n_obs)

        if shuffle_set_split:
            random_state = np.random.RandomState(seed=settings.seed)
            indices = random_state.permutation(indices)

        if self.train_idx is None:
            self.val_idx = indices[:self.n_val]
            self.train_idx = indices[self.n_val: (self.n_val + self.n_train)]
            #self.test_idx = indices[(self.n_val + self.n_train):]
        else:
            indices = np.setdiff1d(indices, self.train_idx)
            if self.val_idx is None:
                self.val_idx = indices[:n_val]
            #if not self.test_idx:
            #    self.test_idx = indices[len(self.val_idx) + len(self.train_idx):]

        graph_csv_result = {}

        modalities_names = ["expression", "accessibility", "protein"]
        modalities_encoders = [self.module.z_encoder_expression, self.module.z_encoder_accessibility, self.module.z_encoder_protein]
        #restrict_modalities
        if restrict_modalities is not False:
            if type(restrict_modalities) == str:
                restrict_modalities = [restrict_modalities]
            modalities_encoders = [modality_encoder for modality_name, modality_encoder in zip(modalities_names, modalities_encoders) if modality_name in restrict_modalities]
            modalities_names = [modality_name for modality_name in modalities_names if modality_name in restrict_modalities]

        for z_encoder_name, z_encoder in zip(modalities_names, modalities_encoders):

            if hasattr(z_encoder.encoder, "gomodel") or hasattr(z_encoder, "genemodel") or keep_activations:
                graph_csv_result[z_encoder_name] = {}

                if hasattr(z_encoder.encoder, "gomodel"):
                    z_encoder.encoder.gomodel.reset_accumulated_activations()
                    z_encoder.encoder.gomodel.keep_activations = "accumulate"
                elif keep_activations:
                    #z_encoder.encoder.reset_accumulated_activations()
                    z_encoder.encoder.keep_activations = True
                if hasattr(z_encoder, "genemodel"):
                    z_encoder.genemodel.genemodel.reset_accumulated_activations()
                    z_encoder.genemodel.genemodel.keep_activations = "accumulate"

                labels_registry = []
                mask_restrict_unpaired_registry = []

                for tensors in scdl:
                    # break
                    inputs = self.module._get_inference_input(tensors)
                    x = inputs["x"]
                    y = inputs["y"]
                    if self.module.n_input_genes == 0:
                        x_rna = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
                    else:
                        x_rna = x[:, : self.module.n_input_genes]
                    if self.module.n_input_regions == 0:
                        x_chr = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
                    else:
                        x_chr = x[:, self.module.n_input_genes: (self.module.n_input_genes + self.module.n_input_regions)]

                    if z_encoder_name == "expression":
                        mask = x_rna.sum(dim=1) > 0
                    elif z_encoder_name == "accessibility":
                        mask = x_chr.sum(dim=1) > 0
                    elif z_encoder_name == "protein":
                        mask = y.sum(dim=1) > 0

                    mask_restrict_unpaired_registry.append(mask.to("cpu").numpy().squeeze())

                    ###

                    _ = self.module.inference(**inputs)#(**self.module._get_inference_input(tensors))
                    labels_registry.append(tensors["labels"].to("cpu").numpy().squeeze())

                if labels_column is None:
                    labels = np.concatenate(labels_registry)
                    labels_column = "labels_registry"
                if len(np.unique(labels)) == 1:
                    raise ValueError("Only 1-type label.")
                if restrict_unpaired_activations_mask:
                    mask = np.concatenate(mask_restrict_unpaired_registry)
                    if restrict_unpaired_activations_mask_flip:
                        mask = ~mask
                else:
                    mask = np.full(labels.shape, True, dtype=bool)

                if restrict_by_column_key_activations_mask is not False and restrict_by_column_values_activations_mask is not False:
                    if type(restrict_by_column_values_activations_mask) == list or type(restrict_by_column_values_activations_mask) == set or type(restrict_by_column_values_activations_mask) == np.ndarray:
                        mask_restrict_by_column_key_values = np.array(adata.obs[restrict_by_column_key_activations_mask].isin(restrict_by_column_values_activations_mask))
                        if restrict_by_column_values_alt_des_activations_mask is not False:
                            restrict_by_column_key_values_str = f"_res_{restrict_by_column_key_activations_mask}_{restrict_by_column_values_alt_des_activations_mask}"
                        else:
                            restrict_by_column_key_values_str = f"_res_{restrict_by_column_key_activations_mask}_{'_'.join(restrict_by_column_values_activations_mask)}"
                    else:
                        mask_restrict_by_column_key_values = np.array(adata.obs[restrict_by_column_key_activations_mask] == restrict_by_column_values_activations_mask)
                        restrict_by_column_key_values_str = f"_res_{restrict_by_column_key_activations_mask}_{restrict_by_column_values_activations_mask}"
                    mask = mask & mask_restrict_by_column_key_values
                else:
                    restrict_by_column_key_values_str = f""

                if hasattr(z_encoder.encoder, "gomodel"):
                    activations = {
                        k: v.to("cpu").numpy()
                        for k, v in z_encoder.encoder.gomodel.merge_activations().items()
                    }
                    z_encoder.encoder.gomodel.reset_accumulated_activations()
                elif keep_activations:
                    activations_temp = {
                        k: v
                        for k, v in z_encoder.encoder.merge_activations().items()
                    }
                    activations = {}
                    activations[0] = np.concatenate([activations_temp[k] for k in activations_temp], axis=1)
                    z_encoder.encoder.reset_accumulated_activations()
                else:
                    activations = {}
                if hasattr(z_encoder, "genemodel"):
                    activations["genes"] = z_encoder.genemodel.genemodel.merge_activations().to("cpu").numpy()
                    z_encoder.genemodel.genemodel.reset_accumulated_activations()

                #with open(os.path.join(results_dir, f'activations_z_encoder_name.pkl'), 'wb') as f:
                #        pickle.dump(activations, f)
                ###

                if type(comparison) != str or (type(comparison) == str and comparison.lower() != "rest"):
                    if type(comparison) == list or type(comparison) == set or type(comparison) == np.ndarray:
                        labels_testing = set(labels[mask]) - set(comparison)
                        if comparison_alt_des is not False:
                            comparison_str = f"_vs_{comparison_alt_des}"
                        else:
                            comparison_str = f"_vs_{'_'.join(comparison)}"
                    else:
                        labels_testing = set(labels[mask]) - set([comparison])
                        if comparison_alt_des is not False:
                            comparison_str = f"_vs_{comparison_alt_des}"
                        else:
                            comparison_str = f"_vs_{comparison}"
                else:
                    labels_testing = set(labels[mask])
                    comparison_str = f"_vs_{comparison}"

                if labels_selection is not False:
                    if type(labels_selection) == set:
                        pass
                    elif type(labels_selection) == list or type(labels_selection) == np.ndarray:
                        labels_selection = set(labels_selection)
                    else:
                        labels_selection = set([labels_selection])
                    labels_testing = labels_testing & labels_selection
                elif labels_selection_groups is not False:
                    labels_testing = [sorted(labels_testing&set(label_selection_group)) for label_selection_group in labels_selection_groups]

                for phenotype in labels_testing:

                    mask_phenotype = mask.copy()

                    if type(comparison) != str or (type(comparison) == str and comparison.lower() != "rest"):
                        if type(phenotype) == str or type(phenotype) == int or type(phenotype) == float:
                            if type(comparison) == list or type(comparison) == np.ndarray:
                                mask_comparison = np.array(labels == phenotype) | np.isin(labels, comparison)
                            else:
                                mask_comparison = np.array(labels == phenotype) | np.array(labels == comparison)
                        else:
                            if type(comparison) == list or type(comparison) == np.ndarray:
                                mask_comparison = np.isin(labels, phenotype) | np.isin(labels, comparison)
                            else:
                                mask_comparison = np.isin(labels, phenotype) | np.array(labels == comparison)
                        mask_phenotype = mask_phenotype & mask_comparison

                    if type(phenotype) == str or type(phenotype) == int or type(phenotype) == float:
                        if len(np.unique((labels == phenotype)[self.train_idx[mask_phenotype[self.train_idx]]])) < 2 or len(np.unique((labels == phenotype)[self.val_idx[mask_phenotype[self.val_idx]]])) < 2:
                            continue
                    else:
                        if len(np.unique((np.isin(labels, phenotype))[self.train_idx[mask_phenotype[self.train_idx]]])) < 2 or len(np.unique((np.isin(labels, phenotype))[self.val_idx[mask_phenotype[self.val_idx]]])) < 2:
                            continue

                    ###

                    if type(phenotype) == str or type(phenotype) == int or type(phenotype) == float:
                        phenotype_str = str(phenotype).replace(" ", "_").replace("/", "_")
                    else:
                        phenotype_str = f"{'_'.join(phenotype)}"

                    os.makedirs(os.path.join(f"{results_dir}", f"{z_encoder_name}_{labels_column}{comparison_str}_{phenotype_str}_res_unp_act_mask_{restrict_unpaired_activations_mask}_flip_{restrict_unpaired_activations_mask_flip}{restrict_by_column_key_values_str}_save_fit", "go_term_clf"), exist_ok=True)

                    if type(phenotype) == str or type(phenotype) == int or type(phenotype) == float:
                        case_idces_train = (labels == phenotype)[self.train_idx[mask_phenotype[self.train_idx]]]
                        case_idces_eval = (labels == phenotype)[self.val_idx[mask_phenotype[self.val_idx]]]
                    else:
                        case_idces_train = (np.isin(labels, phenotype))[self.train_idx[mask_phenotype[self.train_idx]]]
                        case_idces_eval = (np.isin(labels, phenotype))[self.val_idx[mask_phenotype[self.val_idx]]]

                    if keep_activations:
                        levels_nodes = [level for level in activations.keys() for _ in range(activations[level].shape[1])]

                        class FakeNode():
                            def __init__(self, item_id, level):
                                self.id = item_id
                                self.item_id = item_id
                                self.level = level
                                self.depth = level
                                self.ogm_depth = level
                                self.block_index = item_id
                                self.out_slice = slice(item_id, item_id+1, None)

                        fake_order = [f"N:{item_id}" for item_id in range(len(levels_nodes))]
                        fake_goobj = {f"N:{item_id}": FakeNode(item_id, level_node) for item_id, level_node in enumerate(levels_nodes)}

                    goobj_phenotype = calculate_interpretations(
                        z_encoder.encoder.goobj if hasattr(z_encoder.encoder, "goobj") else (fake_goobj if keep_activations else None),
                        z_encoder.encoder.gomodel.order if hasattr(z_encoder.encoder, "goobj") else (fake_order if keep_activations else None),
                        activations_train={
                            k:  v[self.train_idx[mask_phenotype[self.train_idx]], :] #v[self.train_idx,:]
                            for k, v in activations.items()
                        },
                        activations_eval={
                            k: v[self.val_idx[mask_phenotype[self.val_idx]],:] #v[self.val_idx,:]
                            for k, v in activations.items()
                        },
                        case_idces_train=pd.DataFrame(data={"label": case_idces_train}),#pd.DataFrame(data={"label": (labels == phenotype)[self.train_idx]}),
                        case_idces_eval=pd.DataFrame(data={"label": case_idces_eval}),#pd.DataFrame(data={"label": (labels == phenotype)[self.val_idx]}),
                        geneobj=z_encoder.genemodel.geneobj if hasattr(z_encoder, "genemodel") and hasattr(z_encoder.genemodel, "geneobj") else None,
                        geneobj_genetic_effects=None,
                        geneobj_genetic_effects_input=None,
                        geneobj_snps_input=None,
                        load_go_terms=calc_go_terms,
                        load_genes=calc_genes,
                        load_gene_groups=calc_gene_groups,
                        expand_gene_nodes=False,
                        label_only=True,
                        load_save_fit=load_save_fit,
                        save_fit=save_fit,
                        overwrite_save_fit=overwrite_save_fit,
                        path_save_fit=os.path.join(f"{results_dir}", f"{z_encoder_name}_{phenotype_str}_save_fit")
                    )

                    graph_csv_phenotype = goobj_to_csv(
                        goobj_phenotype,
                        interpret_covariates=False,
                        label_only=True,
                    )

                    if save_results:
                        graph_csv_phenotype.to_pickle(
                            os.path.join(
                                results_dir,
                                f'{z_encoder_name}_{labels_column}{comparison_str}_{phenotype_str}_res_unp_act_mask_{restrict_unpaired_activations_mask}_flip_{restrict_unpaired_activations_mask_flip}{restrict_by_column_key_values_str}{"_go_terms" if calc_go_terms else ""}{"_genes" if calc_genes else ""}{f"_gene_groups_{Path(calc_gene_groups).stem}" if calc_gene_groups else ""}_evaluated_graph_csv.pkl',
                            )
                        )

                        """
                        # Save graph results
                        pickle.dump(
                            goobj_phenotype,
                            open(
                                os.path.join(
                                    results_dir,
                                    f'{z_encoder_name}_{labels_column}{comparison_str}_{phenotype_str}_res_unp_act_mask_{restrict_unpaired_activations_mask}_flip_{restrict_unpaired_activations_mask_flip}{restrict_by_column_key_values_str}{"_go_terms" if calc_go_terms else ""}{"_genes" if calc_genes else ""}{"_gene_groups" if calc_gene_groups else ""}_evaluated_graph.pkl',
                                ),
                                "wb",
                            ),
                        )
                        """

                        if calc_go_terms:
                            # Save graph results as graphml object
                            goobj_to_graphml(
                                goobj_phenotype,
                                os.path.join(
                                    results_dir,
                                    f"{z_encoder_name}_{labels_column}{comparison_str}_{phenotype_str}_res_unp_act_mask_{restrict_unpaired_activations_mask}_flip_{restrict_unpaired_activations_mask_flip}{restrict_by_column_key_values_str}_evaluated_graph.graphml",
                                ),
                                enrichment_names=[],
                                interpret_covariates=False,
                                label_only=True,
                            )
                            try:
                                json_str = make_json_str(goobj_phenotype, z_encoder.encoder.gomodel.order)
                                html_path = create_html(
                                    json_str, results_dir,
                                    f"{z_encoder_name}_{labels_column}{comparison_str}_{phenotype_str}_res_unp_act_mask_{restrict_unpaired_activations_mask}_flip_{restrict_unpaired_activations_mask_flip}{restrict_by_column_key_values_str}",
                                    ""
                                )
                                logger.info(f"Created html at {html_path}")
                            except Exception as e:
                                logger.warning(
                                    f"Encountered a problem for interpretability for {e}"
                                )
                                continue

                    del goobj_phenotype
                    graph_csv_result[z_encoder_name][phenotype_str] = graph_csv_phenotype

        return graph_csv_result

    @torch.inference_mode()
    def calculate_covariate_attention(
        self,
        labels_column: str | None = None,
        labels_selection: str | int | float | np.ndarray | list | set | bool = False,
        labels_selection_groups: np.ndarray | list | set | bool = False,
        adata: AnnData | None = None,
        indices: Sequence[int] = None,
        batch_size: int = 128,
        train_size: float = 0.9,
        validation_size: float | None = None,
        train_idx: np.ndarray | list | None = None,
        validation_idx: np.ndarray | list | None = None,
        modality_categorical_covariate_keys: np.ndarray | list | None = None,
        continuous_covariate_keys: np.ndarray | list | None = None,
        shuffle_set_split: bool = True,
        restrict_modalities: bool | str | list | nd.array = False,
        restrict_unpaired_activations_mask: bool = False,
        restrict_unpaired_activations_mask_flip: bool = False,
        save_results: bool = True,
        results_dir: str = "",
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """Return attention rollout.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `networkvi.settings.batch_size`.

        Returns
        -------
        Gene and GO importances for expression and accessibility
        """

        os.makedirs(results_dir, exist_ok=True)

        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)

        covariate_lengths = [0] + list(np.cumsum([len(np.unique(adata.obs[key])) for key in modality_categorical_covariate_keys] + [1 for key in continuous_covariate_keys]))
        if labels_column is not None:
            labels = adata.obs[labels_column]

        self.train_idx = train_idx
        self.val_idx = validation_idx
        self.n_train, self.n_val = validate_data_split(
            self.adata_manager.adata.n_obs, train_size, validation_size
        )

        indices = np.arange(self.adata_manager.adata.n_obs)

        if shuffle_set_split:
            random_state = np.random.RandomState(seed=settings.seed)
            indices = random_state.permutation(indices)

        if self.train_idx is None:
            self.val_idx = indices[:self.n_val]
            self.train_idx = indices[self.n_val: (self.n_val + self.n_train)]
            #self.test_idx = indices[(self.n_val + self.n_train):]
        else:
            indices = np.setdiff1d(indices, self.train_idx)
            if self.val_idx is None:
                self.val_idx = indices[:n_val]
            indices = np.concatenate([self.train_idx, self.val_idx])
            #if not self.test_idx:
            #    self.test_idx = indices[len(self.val_idx) + len(self.train_idx):]

        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        labels = labels[indices]

        modalities_names = ["expression", "accessibility", "protein"]
        modalities_encoders = [self.module.z_encoder_expression, self.module.z_encoder_accessibility,
                               self.module.z_encoder_protein]
        # restrict_modalities
        if restrict_modalities is not False:
            if type(restrict_modalities) == str:
                restrict_modalities = [restrict_modalities]
            modalities_encoders = [modality_encoder for modality_name, modality_encoder in
                                   zip(modalities_names, modalities_encoders) if modality_name in restrict_modalities]
            modalities_names = [modality_name for modality_name in modalities_names if
                                modality_name in restrict_modalities]

        labels_registry = []
        mask_restrict_unpaired_registry = []
        covariate_attention_registry = defaultdict(dict)
        covariate_attention_registry_concat = defaultdict(dict)

        for z_encoder_name, z_encoder in zip(modalities_names, modalities_encoders):
            if hasattr(z_encoder.encoder, "gomodel"):
                covariate_attention_registry[z_encoder_name] = defaultdict(list)
                for attention_layer_index in z_encoder.encoder.gomodel.attention_layer.keys():
                    z_encoder.encoder.gomodel.attention_layer[attention_layer_index].enable_store_attention()

        ###

        for tensors in scdl:
            inputs = self.module._get_inference_input(tensors)
            x = inputs["x"]
            y = inputs["y"]
            if self.module.n_input_genes == 0:
                x_rna = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
            else:
                x_rna = x[:, : self.module.n_input_genes]
            if self.module.n_input_regions == 0:
                x_chr = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
            else:
                x_chr = x[:, self.module.n_input_genes: (self.module.n_input_genes + self.module.n_input_regions)]

            if z_encoder_name == "expression":
                mask = x_rna.sum(dim=1) > 0
            elif z_encoder_name == "accessibility":
                mask = x_chr.sum(dim=1) > 0
            elif z_encoder_name == "protein":
                mask = y.sum(dim=1) > 0

            mask_restrict_unpaired_registry.append(mask.to("cpu").numpy().squeeze())

            ###

            _ = self.module.inference(**inputs)
            labels_registry.append(tensors["labels"].to("cpu").numpy().squeeze())

            for z_encoder_name, z_encoder in zip(modalities_names, modalities_encoders):
                if hasattr(z_encoder.encoder, "gomodel"):
                    for attention_layer_index in z_encoder.encoder.gomodel.attention_layer.keys():
                        covariate_attention_registry[z_encoder_name][attention_layer_index].append(z_encoder.encoder.gomodel.attention_layer[attention_layer_index].get_attention())

        for z_encoder_name, z_encoder in zip(modalities_names, modalities_encoders):
            if hasattr(z_encoder.encoder, "gomodel"):
                for attention_layer_index in z_encoder.encoder.gomodel.attention_layer.keys():
                    covariate_attention_registry_concat[z_encoder_name][attention_layer_index] = np.concatenate(covariate_attention_registry[z_encoder_name][attention_layer_index], axis=0)

        ###

        if labels_column is None:
            labels = np.concatenate(labels_registry)
            labels_column = "labels_registry"
        if len(np.unique(labels)) == 1:
            raise ValueError("Only 1-type label.")
        if restrict_unpaired_activations_mask:
            mask = np.concatenate(mask_restrict_unpaired_registry)
            if restrict_unpaired_activations_mask_flip:
                mask = ~mask
        else:
            mask = np.full(labels.shape, True, dtype=bool)

        labels_testing = set(labels)

        if labels_selection is not False:
            if type(labels_selection) == set:
                pass
            elif type(labels_selection) == list or type(labels_selection) == np.ndarray:
                labels_selection = set(labels_selection)
            else:
                labels_selection = set([labels_selection])
            labels_testing = labels_testing & labels_selection
        elif labels_selection_groups is not False:
            labels_testing = [sorted(labels_testing & set(label_selection_group)) for label_selection_group in
                              labels_selection_groups]

        ###

        def calculate_attention_statistics(modalities_names, modalities_encoders, mask):

            covariate_attention_registry_mean = defaultdict(dict)
            covariate_attention_registry_group_mean = defaultdict(dict)
            covariate_attention_registry_std = defaultdict(dict)
            covariate_attention_registry_group_std = defaultdict(dict)
            for z_encoder_name, z_encoder in zip(modalities_names, modalities_encoders):
                if hasattr(z_encoder.encoder, "gomodel"):
                    index_gene = 0
                    for go_term_ensg in list(z_encoder.encoder.goobj.keys()) + list(z_encoder.genemodel.geneobj.keys()):
                        if go_term_ensg.startswith("ENSG"):
                            out_slice = slice(0 + self.module.standard_gene_size * index_gene, self.module.standard_gene_size + self.module.standard_gene_size * index_gene)
                            depth = -1
                        elif go_term_ensg.startswith("GO:") and z_encoder.encoder.goobj[go_term_ensg].depth != self.module.n_layers_encoder:
                            out_slice = z_encoder.encoder.goobj[go_term_ensg].out_slice
                            depth = z_encoder.encoder.goobj[go_term_ensg].depth
                        else:
                            out_slice = None
                            depth = None
                            index_gene += 1

                        if out_slice is not None:
                            covariate_attention_registry_mean[z_encoder_name][go_term_ensg] = np.mean(np.mean(
                                covariate_attention_registry_concat[z_encoder_name][depth][mask, out_slice], axis=0),
                                axis=0)
                            covariate_attention_registry_std[z_encoder_name][go_term_ensg] = np.mean(np.std(
                                covariate_attention_registry_concat[z_encoder_name][depth][mask, out_slice], axis=0),
                                axis=0)

                            covariate_attention_registry_group_mean_sub = []
                            covariate_attention_registry_group_std_sub = []
                            for i in range(len(covariate_lengths[:-1])):
                                start_idx = covariate_lengths[i]
                                end_idx = covariate_lengths[i + 1]
                                group_mean = np.mean(
                                    covariate_attention_registry_mean[z_encoder_name][go_term_ensg][start_idx:end_idx])
                                group_std = np.std(
                                    covariate_attention_registry_mean[z_encoder_name][go_term_ensg][start_idx:end_idx])
                                covariate_attention_registry_group_mean_sub.append(group_mean)
                                covariate_attention_registry_group_std_sub.append(group_std)
                            covariate_attention_registry_group_mean[z_encoder_name][go_term_ensg] = np.array(
                                covariate_attention_registry_group_mean_sub)
                            covariate_attention_registry_group_std[z_encoder_name][go_term_ensg] = np.array(
                                covariate_attention_registry_group_std_sub)

            return covariate_attention_registry_mean, covariate_attention_registry_group_mean, covariate_attention_registry_std, covariate_attention_registry_group_std

        covariate_attention_registry_mean, covariate_attention_registry_group_mean, covariate_attention_registry_std, covariate_attention_registry_group_std = calculate_attention_statistics(modalities_names, modalities_encoders, mask)

        covariate_attention_registry_mean_phenotypes = {}
        covariate_attention_registry_group_mean_phenotypes = {}
        covariate_attention_registry_std_phenotypes = {}
        covariate_attention_registry_group_std_phenotypes = {}

        for phenotype in labels_testing:

            mask_phenotype = mask.copy()

            if type(phenotype) == str or type(phenotype) == int or type(phenotype) == float:
                mask_phenotype = mask_phenotype & np.array(labels == phenotype)
            else:
                mask_phenotype = mask_phenotype & np.isin(labels, phenotype)

            if mask_phenotype.sum() == 0:
                continue
            ###

            if type(phenotype) == str or type(phenotype) == int or type(phenotype) == float:
                phenotype_str = str(phenotype).replace(" ", "_").replace("/", "_")
            else:
                phenotype_str = f"{'_'.join(phenotype)}"

            covariate_attention_registry_mean_phenotypes[phenotype_str], covariate_attention_registry_group_mean_phenotypes[
                phenotype_str], covariate_attention_registry_std_phenotypes[phenotype_str], \
            covariate_attention_registry_group_std_phenotypes[phenotype_str] = calculate_attention_statistics(
                modalities_names, modalities_encoders, mask_phenotype)

        if save_results:
            np.save(os.path.join(results_dir, f'modality_categorical_covariate_keys.npy'), np.array(modality_categorical_covariate_keys))
            np.save(os.path.join(results_dir, f'continuous_covariate_keys.npy'), np.array(continuous_covariate_keys))
            np.save(os.path.join(results_dir, f'covariate_lengths.npy'), np.array(covariate_lengths))
            for covariate_attention_registry_statistic, covariate_attention_registry_statistic_name in zip(
                [covariate_attention_registry_mean, covariate_attention_registry_group_mean, covariate_attention_registry_std, covariate_attention_registry_group_std, covariate_attention_registry_mean_phenotypes, covariate_attention_registry_group_mean_phenotypes, covariate_attention_registry_std_phenotypes, covariate_attention_registry_group_std_phenotypes],
                ["covariate_attention_registry_mean", "covariate_attention_registry_group_mean", "covariate_attention_registry_std", "covariate_attention_registry_group_std", "covariate_attention_registry_mean_phenotypes", "covariate_attention_registry_group_mean_phenotypes", "covariate_attention_registry_std_phenotypes", "covariate_attention_registry_group_std_phenotypes"],
            ):
                with open(os.path.join(results_dir, f'{covariate_attention_registry_statistic_name}.pkl'), 'wb') as f:
                    pickle.dump(covariate_attention_registry_statistic, f)

        return covariate_attention_registry_mean, covariate_attention_registry_group_mean, covariate_attention_registry_std, covariate_attention_registry_group_std, covariate_attention_registry_mean_phenotypes, covariate_attention_registry_group_mean_phenotypes, covariate_attention_registry_std_phenotypes, covariate_attention_registry_group_std_phenotypes

    @torch.inference_mode()
    def get_library_size_factors(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] = None,
        batch_size: int = 128,
    ) -> dict[str, np.ndarray]:
        """Return library size factors.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        batch_size
            Minibatch size for data loading into model. Defaults to `networkvi.settings.batch_size`.

        Returns
        -------
        Library size factor for expression and accessibility
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        lib_exp = []
        lib_acc = []
        for tensors in scdl:
            outputs = self.module.inference(**self.module._get_inference_input(tensors))
            lib_exp.append(outputs["libsize_expr"].cpu())
            lib_acc.append(outputs["libsize_acc"].cpu())

        return {
            "expression": torch.cat(lib_exp).numpy().squeeze(),
            "accessibility": torch.cat(lib_acc).numpy().squeeze(),
        }

    @torch.inference_mode()
    def get_region_factors(self) -> np.ndarray:
        """Return region-specific factors."""
        if self.n_regions == 0:
            return np.zeros(1)
        else:
            if self.module.region_factors is None:
                raise RuntimeError("region factors were not included in this model")
            return torch.sigmoid(self.module.region_factors).cpu().numpy()

    @torch.inference_mode()
    def get_latent_representation(
        self,
        adata: AnnData | None = None,
        modality: Literal["joint", "expression", "accessibility", "protein"] = "joint",
        indices: Sequence[int] | None = None,
        give_mean: bool = True,
        batch_size: int | None = None,
    ) -> np.ndarray:
        r"""Return the latent representation for each cell.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        modality
            Return modality specific or joint latent representation.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        give_mean
            Give mean of distribution or sample from it.
        batch_size
            Minibatch size for data loading into model. Defaults to `networkvi.settings.batch_size`.

        Returns
        -------
        latent_representation : np.ndarray
            Low-dimensional representation for each cell
        """
        if not self.is_trained_:
            raise RuntimeError("Please train the model first.")
        self._check_adata_modality_weights(adata)
        keys = {"z": "z", "qz_m": "qz_m", "qz_v": "qz_v"}
        if self.fully_paired and modality != "joint":
            raise RuntimeError("A fully paired model only has a joint latent representation.")
        if not self.fully_paired and modality != "joint":
            if modality == "expression":
                keys = {"z": "z_expr", "qz_m": "qzm_expr", "qz_v": "qzv_expr"}
            elif modality == "accessibility":
                keys = {"z": "z_acc", "qz_m": "qzm_acc", "qz_v": "qzv_acc"}
            elif modality == "protein":
                keys = {"z": "z_pro", "qz_m": "qzm_pro", "qz_v": "qzv_pro"}
            else:
                raise RuntimeError(
                    "modality must be 'joint', 'expression', 'accessibility', or 'protein'."
                )

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs)
            qz_m = outputs[keys["qz_m"]]
            qz_v = outputs[keys["qz_v"]]
            z = outputs[keys["z"]]

            if give_mean:
                # does each model need to have this latent distribution param?
                if self.module.latent_distribution == "ln":
                    samples = Normal(qz_m, qz_v.sqrt()).sample([1])
                    z = torch.nn.functional.softmax(samples, dim=-1)
                    z = z.mean(dim=0)
                else:
                    z = qz_m

            latent += [z.cpu()]
        return torch.cat(latent).numpy()

    @torch.inference_mode()
    def get_accessibility_estimates(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] = None,
        n_samples_overall: int | None = None,
        region_list: Sequence[str] | None = None,
        transform_batch: str | int | None = None,
        use_z_mean: bool = True,
        threshold: float | None = None,
        normalize_cells: bool = False,
        normalize_regions: bool = False,
        batch_size: int = 128,
        return_numpy: bool = False,
    ) -> np.ndarray | csr_matrix | pd.DataFrame:
        """Impute the full accessibility matrix.

        Returns a matrix of accessibility probabilities for each cell and genomic region in the
        input (for return matrix A, A[i,j] is the probability that region j is accessible in cell
        i).

        Parameters
        ----------
        adata
            AnnData object that has been registered with networkvi. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of samples to return in total
        region_list
            Regions to use. if `None`, all regions are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used
            - int, then batch transform_batch is used
        use_z_mean
            If True (default), use the distribution mean. Otherwise, sample from the distribution.
        threshold
            If provided, values below the threshold are replaced with 0 and a sparse matrix
            is returned instead. This is recommended for very large matrices. Must be between 0 and
            1.
        normalize_cells
            Whether to reintroduce library size factors to scale the normalized probabilities.
            This makes the estimates closer to the input, but removes the library size correction.
            False by default.
        normalize_regions
            Whether to reintroduce region factors to scale the normalized probabilities. This makes
            the estimates closer to the input, but removes the region-level bias correction. False
            by default.
        batch_size
            Minibatch size for data loading into model
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        post = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if region_list is None:
            region_mask = slice(None)
        else:
            region_mask = [region in region_list for region in adata.var_names[self.n_genes :]]

        if threshold is not None and (threshold < 0 or threshold > 1):
            raise ValueError("the provided threshold must be between 0 and 1")

        imputed = []
        for tensors in post:
            get_generative_input_kwargs = {"transform_batch": transform_batch[0]}
            generative_kwargs = {"use_z_mean": use_z_mean}
            inference_outputs, generative_outputs = self.module.forward(
                tensors=tensors,
                get_generative_input_kwargs=get_generative_input_kwargs,
                generative_kwargs=generative_kwargs,
                compute_loss=False,
            )
            p = generative_outputs["p"].cpu()

            if normalize_cells:
                p *= inference_outputs["libsize_acc"].cpu()
            if normalize_regions:
                p *= torch.sigmoid(self.module.region_factors).cpu()
            if threshold:
                p[p < threshold] = 0
                p = csr_matrix(p.numpy())
            if region_mask is not None:
                p = p[:, region_mask]
            imputed.append(p)

        if threshold:  # imputed is a list of csr_matrix objects
            imputed = vstack(imputed, format="csr")
        else:  # imputed is a list of tensors
            imputed = torch.cat(imputed).numpy()

        if return_numpy:
            return imputed
        elif threshold:
            return pd.DataFrame.sparse.from_spmatrix(
                imputed,
                index=adata.obs_names[indices],
                columns=adata.var_names[self.n_genes :][region_mask],
            )
        else:
            return pd.DataFrame(
                imputed,
                index=adata.obs_names[indices],
                columns=adata.var_names[self.n_genes :][region_mask],
            )

    @torch.inference_mode()
    def get_normalized_expression(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        n_samples_overall: int | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        gene_list: Sequence[str] | None = None,
        use_z_mean: bool = True,
        n_samples: int = 1,
        batch_size: int | None = None,
        return_mean: bool = True,
        return_numpy: bool = False,
    ) -> np.ndarray | pd.DataFrame:
        r"""Returns the normalized (decoded) gene expression.

        This is denoted as :math:`\rho_n` in the scVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If `None`, defaults to the
            AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        n_samples_overall
            Number of observations to sample from ``indices`` if ``indices`` is provided.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            - None, then real observed batch is used.
            - int, then batch transform_batch is used.
        gene_list
            Return frequencies of expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        use_z_mean
            If True, use the mean of the latent distribution, otherwise sample from it
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `networkvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a numpy array instead of a pandas DataFrame.

        Returns
        -------
        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is
        :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        adata_manager = self.get_anndata_manager(adata, required=True)
        if indices is None:
            indices = np.arange(adata.n_obs)
        if n_samples_overall is not None:
            indices = np.random.choice(indices, n_samples_overall)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        transform_batch = _get_batch_code_from_category(adata_manager, transform_batch)

        if gene_list is None:
            gene_mask = slice(None)
        else:
            all_genes = adata.var_names[: self.n_genes]
            gene_mask = [gene in gene_list for gene in all_genes]

        exprs = []
        for tensors in scdl:
            per_batch_exprs = []
            for batch in transform_batch:
                if batch is not None:
                    batch_indices = tensors[REGISTRY_KEYS.BATCH_KEY]
                    tensors[REGISTRY_KEYS.BATCH_KEY] = torch.ones_like(batch_indices) * batch
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs={"n_samples": n_samples},
                    generative_kwargs={"use_z_mean": use_z_mean},
                    compute_loss=False,
                )
                output = generative_outputs["px_scale"]
                output = output[..., gene_mask]
                output = output.cpu().numpy()
                per_batch_exprs.append(output)
            per_batch_exprs = np.stack(
                per_batch_exprs
            )  # shape is (len(transform_batch) x batch_size x n_var)
            exprs += [per_batch_exprs.mean(0)]

        if n_samples > 1:
            # The -2 axis correspond to cells.
            exprs = np.concatenate(exprs, axis=-2)
        else:
            exprs = np.concatenate(exprs, axis=0)
        if n_samples > 1 and return_mean:
            exprs = exprs.mean(0)

        if return_numpy:
            return exprs
        else:
            return pd.DataFrame(
                exprs,
                columns=adata.var_names[: self.n_genes][gene_mask],
                index=adata.obs_names[indices],
            )

    @de_dsp.dedent
    def differential_accessibility(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.05,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        two_sided: bool = True,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for differential accessibility analysis.

        Implements ``'vanilla'`` DE :cite:p:`Lopez18` and ``'change'`` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        two_sided
            Whether to perform a two-sided test, or a one-sided test.
        **kwargs
            Keyword args for :meth:`networkvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential accessibility DataFrame with the following columns:
        prob_da
            the probability of the region being differentially accessible
        is_da_fdr
            whether the region passes a multiple hypothesis correction procedure with the
            target_fdr threshold
        bayes_factor
            Bayes Factor indicating the level of significance of the analysis
        effect_size
            the effect size, computed as (accessibility in population 2) - (accessibility in
            population 1)
        emp_effect
            the empirical effect, based on observed detection rates instead of the estimated
            accessibility scores from the PeakVI model
        est_prob1
            the estimated probability of accessibility in population 1
        est_prob2
            the estimated probability of accessibility in population 2
        emp_prob1
            the empirical (observed) probability of accessibility in population 1
        emp_prob2
            the empirical (observed) probability of accessibility in population 2

        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)
        col_names = adata.var_names[self.n_genes :]
        model_fn = partial(
            self.get_accessibility_estimates, use_z_mean=False, batch_size=batch_size
        )

        # TODO check if change_fn in kwargs and raise error if so
        def change_fn(a, b):
            return a - b

        if two_sided:

            def m1_domain_fn(samples):
                return np.abs(samples) >= delta

        else:

            def m1_domain_fn(samples):
                return samples >= delta

        all_stats_fn = partial(
            scatac_raw_counts_properties,
            var_idx=np.arange(adata.shape[1])[self.n_genes :],
        )

        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            change_fn=change_fn,
            m1_domain_fn=m1_domain_fn,
            silent=silent,
            **kwargs,
        )

        return result

    @de_dsp.dedent
    def differential_expression(
        self,
        adata: AnnData | None = None,
        groupby: str | None = None,
        group1: Iterable[str] | None = None,
        group2: str | None = None,
        idx1: Sequence[int] | Sequence[bool] | None = None,
        idx2: Sequence[int] | Sequence[bool] | None = None,
        mode: Literal["vanilla", "change"] = "change",
        delta: float = 0.25,
        batch_size: int | None = None,
        all_stats: bool = True,
        batch_correction: bool = False,
        batchid1: Iterable[str] | None = None,
        batchid2: Iterable[str] | None = None,
        fdr_target: float = 0.05,
        silent: bool = False,
        **kwargs,
    ) -> pd.DataFrame:
        r"""A unified method for differential expression analysis.

        Implements `"vanilla"` DE :cite:p:`Lopez18` and `"change"` mode DE :cite:p:`Boyeau19`.

        Parameters
        ----------
        %(de_adata)s
        %(de_groupby)s
        %(de_group1)s
        %(de_group2)s
        %(de_idx1)s
        %(de_idx2)s
        %(de_mode)s
        %(de_delta)s
        %(de_batch_size)s
        %(de_all_stats)s
        %(de_batch_correction)s
        %(de_batchid1)s
        %(de_batchid2)s
        %(de_fdr_target)s
        %(de_silent)s
        **kwargs
            Keyword args for :meth:`networkvi.model.base.DifferentialComputation.get_bayes_factors`

        Returns
        -------
        Differential expression DataFrame.
        """
        self._check_adata_modality_weights(adata)
        adata = self._validate_anndata(adata)

        col_names = adata.var_names[: self.n_genes]
        model_fn = partial(
            self.get_normalized_expression,
            batch_size=batch_size,
        )
        all_stats_fn = partial(
            scrna_raw_counts_properties,
            var_idx=np.arange(adata.shape[1])[: self.n_genes],
        )
        result = _de_core(
            adata_manager=self.get_anndata_manager(adata, required=True),
            model_fn=model_fn,
            representation_fn=None,
            groupby=groupby,
            group1=group1,
            group2=group2,
            idx1=idx1,
            idx2=idx2,
            all_stats=all_stats,
            all_stats_fn=all_stats_fn,
            col_names=col_names,
            mode=mode,
            batchid1=batchid1,
            batchid2=batchid2,
            delta=delta,
            batch_correction=batch_correction,
            fdr=fdr_target,
            silent=silent,
            **kwargs,
        )

        return result

    @torch.no_grad()
    def get_protein_foreground_probability(
        self,
        adata: AnnData | None = None,
        indices: Sequence[int] | None = None,
        transform_batch: Sequence[Number | str] | None = None,
        protein_list: Sequence[str] | None = None,
        n_samples: int = 1,
        batch_size: int | None = None,
        use_z_mean: bool = True,
        return_mean: bool = True,
        return_numpy: bool | None = None,
    ):
        r"""Returns the foreground probability for proteins.

        This is denoted as :math:`(1 - \pi_{nt})` in the totalVI paper.

        Parameters
        ----------
        adata
            AnnData object with equivalent structure to initial AnnData. If ``None``, defaults to
            the AnnData object used to initialize the model.
        indices
            Indices of cells in adata to use. If `None`, all cells are used.
        transform_batch
            Batch to condition on.
            If transform_batch is:

            * ``None`` - real observed batch is used
            * ``int`` - batch transform_batch is used
            * ``List[int]`` - average over batches in list
        protein_list
            Return protein expression for a subset of genes.
            This can save memory when working with large datasets and few genes are
            of interest.
        n_samples
            Number of posterior samples to use for estimation.
        batch_size
            Minibatch size for data loading into model. Defaults to `networkvi.settings.batch_size`.
        return_mean
            Whether to return the mean of the samples.
        return_numpy
            Return a :class:`~numpy.ndarray` instead of a :class:`~pandas.DataFrame`. DataFrame
            includes gene names as columns. If either ``n_samples=1`` or ``return_mean=True``,
            defaults to ``False``. Otherwise, it defaults to `True`.

        Returns
        -------
        - **foreground_probability** - probability foreground for each protein

        If `n_samples` > 1 and `return_mean` is False, then the shape is `(samples, cells, genes)`.
        Otherwise, shape is `(cells, genes)`. In this case, return type is
        :class:`~pandas.DataFrame` unless `return_numpy` is True.
        """
        adata = self._validate_anndata(adata)
        post = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)

        if protein_list is None:
            protein_mask = slice(None)
        else:
            all_proteins = self.scvi_setup_dict_["protein_names"]
            protein_mask = [True if p in protein_list else False for p in all_proteins]

        if n_samples > 1 and return_mean is False:
            if return_numpy is False:
                warnings.warn(
                    "`return_numpy` must be `True` if `n_samples > 1` and `return_mean` is "
                    "`False`, returning an `np.ndarray`.",
                    UserWarning,
                    stacklevel=settings.warnings_stacklevel,
                )
            return_numpy = True
        if indices is None:
            indices = np.arange(adata.n_obs)

        py_mixings = []
        if not isinstance(transform_batch, IterableClass):
            transform_batch = [transform_batch]

        transform_batch = _get_batch_code_from_category(self.adata_manager, transform_batch)
        for tensors in post:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
            py_mixing = torch.zeros_like(y[..., protein_mask])
            if n_samples > 1:
                py_mixing = torch.stack(n_samples * [py_mixing])
            for _ in transform_batch:
                # generative_kwargs = dict(transform_batch=b)
                generative_kwargs = {"use_z_mean": use_z_mean}
                inference_kwargs = {"n_samples": n_samples}
                _, generative_outputs = self.module.forward(
                    tensors=tensors,
                    inference_kwargs=inference_kwargs,
                    generative_kwargs=generative_kwargs,
                    compute_loss=False,
                )
                py_mixing += torch.sigmoid(generative_outputs["py_"]["mixing"])[
                    ..., protein_mask
                ].cpu()
            py_mixing /= len(transform_batch)
            py_mixings += [py_mixing]
        if n_samples > 1:
            # concatenate along batch dimension -> result shape = (samples, cells, features)
            py_mixings = torch.cat(py_mixings, dim=1)
            # (cells, features, samples)
            py_mixings = py_mixings.permute(1, 2, 0)
        else:
            py_mixings = torch.cat(py_mixings, dim=0)

        if return_mean is True and n_samples > 1:
            py_mixings = torch.mean(py_mixings, dim=-1)

        py_mixings = py_mixings.cpu().numpy()

        if return_numpy is True:
            return 1 - py_mixings
        else:
            pro_names = self.protein_state_registry.column_names
            foreground_prob = pd.DataFrame(
                1 - py_mixings,
                columns=pro_names[protein_mask],
                index=adata.obs_names[indices],
            )
            return foreground_prob

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        patient_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        protein_expression_obsm_key: str | None = None,
        protein_names_uns_key: str | None = None,
        **kwargs,
    ):
        """%(summary)s.

        Parameters
        ----------
        %(param_adata)s
        %(param_layer)s
        %(param_batch_key)s
        %(param_size_factor_key)s
        %(param_cat_cov_keys)s
        %(param_cont_cov_keys)s
        protein_expression_obsm_key
            key in `adata.obsm` for protein expression data.
        protein_names_uns_key
            key in `adata.uns` for protein names. If None, will use the column names of
            `adata.obsm[protein_expression_obsm_key]` if it is a DataFrame, else will assign
            sequential names to proteins.
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        adata.obs["_indices"] = np.arange(adata.n_obs)
        batch_field = CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key)
        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            batch_field,
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.PATIENT_KEY, patient_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            NumericalObsField(REGISTRY_KEYS.INDICES_KEY, "_indices"),
        ]
        if protein_expression_obsm_key is not None:
            anndata_fields.append(
                ProteinObsmField(
                    REGISTRY_KEYS.PROTEIN_EXP_KEY,
                    protein_expression_obsm_key,
                    use_batch_mask=True,
                    batch_field=batch_field,
                    colnames_uns_key=protein_names_uns_key,
                    is_count_data=True,
                )
            )

        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    def _check_adata_modality_weights(self, adata):
        """Checks if adata is None and weights are per cell.

        :param adata: anndata object
        :return:
        """
        if (adata is not None) and (self.module.modality_weights == "cell"):
            raise RuntimeError("Held out data not permitted when using per cell weights")
