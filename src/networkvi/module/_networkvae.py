from collections.abc import Iterable
from typing import Literal, Optional, Union

import math
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.distributions import Normal, Poisson
from torch.distributions import kl_divergence as kld
from torch.nn import functional as F
import os
import sys

from networkvi import REGISTRY_KEYS
from networkvi.distributions import (
    NegativeBinomial,
    NegativeBinomialMixture,
    ZeroInflatedNegativeBinomial
)
from networkvi.module._peakvae import Decoder as DecoderPeakVI
from networkvi.module.base import BaseModuleClass, LossOutput, auto_move_data
from networkvi.nn import DecoderSCVI, Encoder, FCLayers

from ._utils import masked_softmax

class GatingNetwork(nn.Module):
    """
    Gating network for the Mixture of Experts (MoE) model.

    The GatingNetwork is responsible for calculating normalized gating weights
    for a set of experts in the Mixture of Experts framework. The gating weights
    are computed based on the latent means, variances, and a binary mask that
    indicates the active experts.

    Parameters
    ----------
    latent_dim : int
        The dimensionality of the latent space for each expert.
    num_experts : int
        The total number of experts in the Mixture of Experts model.

    Examples
    --------
    >>> gating_net = GatingNetwork(latent_dim=16, num_experts=4)
    >>> means = torch.randn(8, 4, 16)  # (batch_size=8, num_experts=4, latent_dim=16)
    >>> logvars = torch.randn(8, 4, 16)
    >>> masks = torch.randint(0, 2, (8, 4))  # Binary mask (active/inactive experts)
    >>> gating_weights = gating_net(means, logvars, masks)
    >>> print(gating_weights.shape)
    torch.Size([8, 4])
    """

    def __init__(self, latent_dim, num_experts):
        super(GatingNetwork, self).__init__()
        input_dim = 2 * num_experts * latent_dim + num_experts  # Means + logvars + mask
        self.fc = nn.Linear(input_dim, num_experts)

    def forward(self, means, logvars, masks):
        """
        Args:
            means (torch.Tensor): (batch_size, num_experts, latent_dim)
            logvars (torch.Tensor): (batch_size, num_experts, latent_dim)
            mask (torch.Tensor): (batch_size, num_experts), binary mask

        Returns:
            gating_weights (torch.Tensor): (batch_size, num_experts), normalized weights
        """
        means = torch.stack(means, dim=1)
        logvars = torch.stack(logvars, dim=1)
        mask = torch.stack(masks, dim=1)

        batch_size, num_experts, latent_dim = means.size()
        means_flat = means.view(batch_size, -1)  # (batch_size, num_experts * latent_dim)
        logvars_flat = logvars.view(batch_size, -1)  # (batch_size, num_experts * latent_dim)
        mask_flat = mask.view(batch_size, -1)  # (batch_size, num_experts)
        gating_input = torch.cat([means_flat, logvars_flat, mask_flat], dim=-1)  # (batch_size, input_dim)

        logits = self.fc(gating_input)  # (batch_size, num_experts)
        gating_weights = F.softmax(logits, dim=-1)  # Normalize weights across experts
        gating_weights = gating_weights * mask
        gating_weights = gating_weights / (gating_weights.sum(dim=-1, keepdim=True) + 1e-8)

        return gating_weights

class RBF(nn.Module):
    """
    From: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
    """
    def __init__(self, n_kernels=5, mul_factor=2.0, bandwidth=None):
        super().__init__()
        self.bandwidth_multipliers = mul_factor ** (torch.arange(n_kernels) - n_kernels // 2)
        self.bandwidth = bandwidth

    def get_bandwidth(self, L2_distances):
        if self.bandwidth is None:
            n_samples = L2_distances.shape[0]
            return L2_distances.data.sum() / (n_samples ** 2 - n_samples)

        return self.bandwidth

    def forward(self, X):
        L2_distances = torch.cdist(X, X) ** 2
        return torch.exp(
            -L2_distances[None, ...] / (self.get_bandwidth(L2_distances) * self.bandwidth_multipliers.to(L2_distances.device))[:, None, None]
        ).sum(dim=0)


class MMDLoss(nn.Module):
    """
    From: https://github.com/yiftachbeer/mmd_loss_pytorch/blob/master/mmd_loss.py
    """
    def __init__(self, kernel=RBF()):
        super().__init__()
        self.kernel = kernel

    def forward(self, X, Y):
        K = self.kernel(torch.vstack([X, Y]))

        X_size = X.shape[0]
        XX = K[:X_size, :X_size].mean()
        XY = K[:X_size, X_size:].mean()
        YY = K[X_size:, X_size:].mean()
        return XX - 2 * XY + YY

class LibrarySizeEncoder(torch.nn.Module):
    """Library size encoder. Adapted from MultiVI :cite:p:`Ashuach2023`."""

    def __init__(
        self,
        n_input: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
        gene_layer_type: Literal["none", "standard", "interaction"] = "interaction",
        gene_layer_interaction_source: Literal["pp", "tf", "tad"] = "tad",
        layers_type: Literal["linear", "go"] = "go",
        **kwargs,
    ):
        super().__init__()

        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            inject_covariates=deep_inject_covariates,
            **kwargs,
        )
        self.output = torch.nn.Sequential(torch.nn.Linear(n_hidden, 1), torch.nn.LeakyReLU())

    def forward(self, x: torch.Tensor, *cat_list: int, cont_input: torch.Tensor = None):
        """Forward pass."""
        return self.output(self.px_decoder(x, *cat_list, cont_input=cont_input))


class DecoderADT(torch.nn.Module):
    """Decoder for just surface proteins (ADT). Adapted from TOTALVI :cite:p:`Gayoso2021`."""

    def __init__(
        self,
        n_input: int,
        n_output_proteins: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        deep_inject_covariates: bool = False,
        layers_type: Literal["linear"] = "linear",
        sparsities: list = [0.9, 0.9],
        dynamic: bool = True,
        dynamic_update_rate: Optional[int] = None,
        dynamic_end_update_rate: Optional[str] = None,
        activation_fn: nn.Module = nn.ReLU,
    ):
        super().__init__()
        self.n_output_proteins = n_output_proteins

        linear_args = {
            "n_layers": 1,
            "use_activation": False,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0,
        }

        self.py_fore_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation_fn=activation_fn,
        )

        self.py_fore_scale_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            activation_fn=activation_fn,
        )

        self.py_background_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            activation_fn=activation_fn,
            **linear_args,
        )

        # dropout (mixture component for proteins, ZI probability for genes)
        self.sigmoid_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation_fn=activation_fn,
        )

        # background mean parameters second decoder
        self.py_back_mean_log_alpha = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            activation_fn=activation_fn,
            **linear_args,
        )

        self.py_back_mean_log_beta = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            activation_fn=activation_fn,
            **linear_args,
        )

        # background mean first decoder
        self.py_back_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            activation_fn=activation_fn,
        )

    def forward(self, z: torch.Tensor, *cat_list: int, cont_input: torch.Tensor = None):
        """Forward pass."""
        # z is the latent repr
        py_ = {}

        py_back = self.py_back_decoder(z, *cat_list, cont_input=cont_input)
        py_back_cat_z = torch.cat([py_back, z], dim=-1)

        py_["back_alpha"] = self.py_back_mean_log_alpha(py_back_cat_z, *cat_list, cont_input=cont_input)
        py_["back_beta"] = torch.exp(self.py_back_mean_log_beta(py_back_cat_z, *cat_list, cont_input=cont_input))
        log_pro_back_mean = Normal(py_["back_alpha"], py_["back_beta"]).rsample()
        py_["rate_back"] = torch.exp(log_pro_back_mean)

        py_fore = self.py_fore_decoder(z, *cat_list, cont_input=cont_input)
        py_fore_cat_z = torch.cat([py_fore, z], dim=-1)
        py_["fore_scale"] = self.py_fore_scale_decoder(py_fore_cat_z, *cat_list, cont_input=cont_input) + 1 + 1e-8
        py_["rate_fore"] = py_["rate_back"] * py_["fore_scale"]

        p_mixing = self.sigmoid_decoder(z, *cat_list, cont_input=cont_input)
        p_mixing_cat_z = torch.cat([p_mixing, z], dim=-1)
        py_["mixing"] = self.py_background_decoder(p_mixing_cat_z, *cat_list, cont_input=cont_input)

        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        py_["scale"] = torch.nn.functional.normalize(
            (1 - protein_mixing) * py_["rate_fore"], p=1, dim=-1
        )

        return py_, log_pro_back_mean


class NETWORKVAE(BaseModuleClass):
    """Integration of multi-moda data employing domain knowledge-driven neural networks :cite:p:`Arnoldt2024`.

    NETWORKVAE performs paired and mosaic integration of multiomic datasets using sparse encoders. NetworkVI has been built on top of MULTIVI :cite:p:`Ashuach2023`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~networkvi.model.networkvi.setup_anndata`.
    n_input_regions
        The number of accessibility features (genomic regions).
    ensembl_ids_regions
        ENSEMBL-IDs of accessibility features (genomic regions).
    n_input_genes
        The number of gene expression features (genes).
    ensembl_ids_genes
        ENSEMBL-IDs of gene expression features (genes).
    n_input_proteins
        The number of epitope features (proteins).
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
    """

    def __init__(
        self,
        n_input_regions: int = 0,
        ensembl_ids_regions: np.ndarray | None = None,
        n_input_genes: int = 0,
        ensembl_ids_genes: np.ndarray | None = None,
        n_input_proteins: int = 0,
        ensembl_ids_proteins: np.ndarray | None = None,
        n_patient_covariates: int = 0,
        modality_weights: Literal["equal", "cell", "universal", "moe"] = "equal",
        modality_penalty: Literal["Jeffreys", "MMD", "None"] = "Jeffreys",
        n_batch: int = 0,
        n_obs: int = 0,
        n_labels: int = 0,
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
        gene_dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        n_hidden: int = None,
        n_latent: int = None,
        n_layers_encoder: int = 2,
        layers_encoder_type: Literal["linear", "go"] = "go",
        n_layers_decoder: int = 2,
        layers_decoder_type: Literal["linear", "go"] = "go",
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
        library_size_layers_type: Literal["linear", "go"] = "go",
        gene_layer_interaction_source: Optional[str] = None,
        standard_gene_size: int = 4,
        standard_go_size: int = 6,
        obo_file: Optional[str] = None,
        map_ensembl_go: Optional[Union[list, np.ndarray]] = None,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: float = 0.1,
        region_factors: bool = True,
        use_batch_norm: Literal["encoder", "decoder", "none", "both"] = "none",
        use_layer_norm: Literal["encoder", "decoder", "none", "both"] = "both",
        latent_distribution: Literal["normal", "ln"] = "normal",
        deeply_inject_covariates: bool = False,
        decoder_deeply_inject_covariates: bool = False,
        first_layer_inject_covariates: bool = False,
        last_layer_inject_covariates: bool = False,
        encode_covariates: bool = False,
        use_size_factor_key: bool = False,
        protein_background_prior_mean: Optional[np.ndarray] = None,
        protein_background_prior_scale: Optional[np.ndarray] = None,
        protein_dispersion: str = "protein",
        activation_fn: nn.Module = nn.ReLU,
        use_mean_mixing: bool = False,
        use_product_of_experts: bool = False,
        use_mixture_of_experts: bool = True,
        **kwargs,
    ):
        super().__init__()

        # INIT PARAMS
        self.n_input_regions = n_input_regions
        self.ensembl_ids_regions = ensembl_ids_regions
        self.n_input_genes = n_input_genes
        self.ensembl_ids_genes = ensembl_ids_genes
        self.n_input_proteins = n_input_proteins
        self.ensembl_ids_proteins = ensembl_ids_proteins
        self.n_patient_covariates = n_patient_covariates
        if n_hidden is None:
            if n_input_regions == 0:
                self.n_hidden = np.min([128, int(np.sqrt(self.n_input_genes))])
            else:
                self.n_hidden = int(np.sqrt(self.n_input_regions))
        else:
            self.n_hidden = n_hidden
        self.n_batch = n_batch

        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution

        self.n_latent = int(np.sqrt(self.n_hidden)) if n_latent is None else n_latent
        self.n_layers_encoder = n_layers_encoder
        self.layers_encoder_type = layers_encoder_type
        self.n_layers_decoder = n_layers_decoder
        self.layers_decoder_type = layers_decoder_type
        self.expression_gene_layer_type = expression_gene_layer_type
        self.accessibility_gene_layer_type = accessibility_gene_layer_type
        self.protein_gene_layer_type = protein_gene_layer_type
        self.library_size_layers_type = library_size_layers_type
        #self.expression_library_size_gene_layer_type = expression_library_size_gene_layer_type
        #self.accessibility_library_size_gene_layer_type = accessibility_library_size_gene_layer_type
        #self.protein_library_size_gene_layer_type = protein_library_size_gene_layer_type
        self.gene_layer_interaction_source = gene_layer_interaction_source
        self.standard_gene_size = standard_gene_size
        self.standard_go_size = standard_go_size
        self.obo_file = obo_file
        self.map_ensembl_go = map_ensembl_go
        self.n_cats_per_cov = n_cats_per_cov
        self.n_continuous_cov = n_continuous_cov
        self.dropout_rate = dropout_rate

        self.use_batch_norm_encoder = use_batch_norm in ("encoder", "both")
        self.use_batch_norm_decoder = use_batch_norm in ("decoder", "both")
        self.use_layer_norm_encoder = use_layer_norm in ("encoder", "both")
        self.use_layer_norm_decoder = use_layer_norm in ("decoder", "both")
        self.encode_covariates = encode_covariates
        self.deeply_inject_covariates = deeply_inject_covariates
        self.decoder_deeply_inject_covariates = decoder_deeply_inject_covariates
        self.first_layer_inject_covariates = first_layer_inject_covariates
        self.last_layer_inject_covariates = last_layer_inject_covariates
        self.use_size_factor_key = use_size_factor_key

        cat_list = [n_batch] + (list(n_cats_per_cov) if n_cats_per_cov is not None else []) + ([n_continuous_cov] if n_continuous_cov is not None and n_continuous_cov != 0 else [])
        cat_list_encoder = [n_batch] + (list(n_cats_per_cov) if n_cats_per_cov is not None else []) + ([n_continuous_cov] if n_continuous_cov is not None and n_continuous_cov != 0 else [])
        encoder_cat_list_library = cat_list if encode_covariates else None
        encoder_cat_list = cat_list_encoder if encode_covariates else None

        # expression
        # expression dispersion parameters
        self.gene_likelihood = gene_likelihood
        self.gene_dispersion = gene_dispersion
        if self.gene_dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes))
        elif self.gene_dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_batch))
        elif self.gene_dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input_genes, n_labels))
        elif self.gene_dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )
        self.activation_fn = activation_fn
        self.sparsities = sparsities
        self.dynamic = dynamic
        self.dynamic_update_rate = dynamic_update_rate
        self.dynamic_end_update_rate = dynamic_end_update_rate
        self.gene_interaction_layer_dynamic = gene_interaction_layer_dynamic
        self.gene_interaction_layer_pruning_frac = gene_interaction_layer_pruning_frac
        self.gene_interaction_layer_dynamic_update_rate = gene_interaction_layer_dynamic_update_rate
        self.gene_interaction_layer_dynamic_end_update_rate = gene_interaction_layer_dynamic_end_update_rate
        self.gene_interaction_layer_dynamic_save_path = gene_interaction_layer_dynamic_save_path
        self.keep_activations = keep_activations

        self.use_mean_mixing = use_mean_mixing
        self.use_product_of_experts = use_product_of_experts
        self.use_mixture_of_experts = use_mixture_of_experts

        # expression encoder
        if self.n_input_genes == 0:
            input_exp = 1
        else:
            input_exp = self.n_input_genes
        #n_input_encoder_exp = input_exp + n_continuous_cov * encode_covariates
        n_input_encoder_exp = input_exp
        if self.ensembl_ids_genes is not None:
            input_ensembl_ids_genes = self.ensembl_ids_genes
            input_ensembl_ids_genes = np.where(pd.isna(input_ensembl_ids_genes), "ENSG00000000000", input_ensembl_ids_genes)
        else:
            input_ensembl_ids_genes = None
        self.z_encoder_expression = Encoder(
            n_input=n_input_encoder_exp,
            n_layers=self.n_layers_encoder,
            ensembl_ids=input_ensembl_ids_genes,
            n_output=self.n_latent,
            n_cat_list=encoder_cat_list,
            n_hidden=self.n_hidden,
            dropout_rate=self.dropout_rate,
            distribution=self.latent_distribution,
            inject_covariates=deeply_inject_covariates,
            first_layer_inject_covariates=first_layer_inject_covariates,
            last_layer_inject_covariates=last_layer_inject_covariates,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            activation_fn=self.activation_fn,
            var_eps=0,
            return_dist=False,
            layers_type=self.layers_encoder_type,
            gene_layer_type=self.expression_gene_layer_type,
            gene_layer_interaction_source=self.gene_layer_interaction_source,
            standard_gene_size=self.standard_gene_size,
            standard_go_size=self.standard_go_size,
            obo_file=self.obo_file,
            map_ensembl_go=self.map_ensembl_go,
            sparsities=self.sparsities,
            dynamic=self.dynamic,
            dynamic_update_rate=self.dynamic_update_rate,
            dynamic_end_update_rate=self.dynamic_end_update_rate,
            gene_interaction_layer_dynamic=self.gene_interaction_layer_dynamic,
            gene_interaction_layer_pruning_frac=self.gene_interaction_layer_pruning_frac,
            gene_interaction_layer_dynamic_update_rate=self.gene_interaction_layer_dynamic_update_rate,
            gene_interaction_layer_dynamic_end_update_rate=self.gene_interaction_layer_dynamic_end_update_rate,
            gene_interaction_layer_dynamic_save_path=os.path.join(self.gene_interaction_layer_dynamic_save_path,"expression") if self.gene_interaction_layer_dynamic_save_path else None,
            keep_activations=self.keep_activations,
        )

        # expression library size encoder
        self.l_encoder_expression = LibrarySizeEncoder(
            n_input_encoder_exp,
            #n_input_encoder_exp + n_continuous_cov * encode_covariates,
            n_cat_list=encoder_cat_list_library,
            n_layers=self.n_layers_encoder,
            n_hidden=self.n_hidden,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            deep_inject_covariates=self.decoder_deeply_inject_covariates,
            layers_type=self.library_size_layers_type,
            #gene_layer_type=self.expression_library_size_gene_layer_type,
            #gene_layer_interaction_source=self.gene_layer_interaction_source,
            sparsities=self.sparsities,
            dynamic=self.dynamic,
            dynamic_update_rate=self.dynamic_update_rate,
            dynamic_end_update_rate=self.dynamic_end_update_rate,
            activation_fn=self.activation_fn,
        )

        # expression decoder
        #n_input_decoder = self.n_latent + self.n_continuous_cov
        n_input_decoder = self.n_latent
        self.z_decoder_expression = DecoderSCVI(
            n_input_decoder,
            n_input_genes,
            n_cat_list=cat_list,
            n_layers=n_layers_decoder,
            n_hidden=self.n_hidden,
            inject_covariates=self.decoder_deeply_inject_covariates,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            layers_type=self.layers_decoder_type,
            sparsities=self.sparsities,
            dynamic=self.dynamic,
            dynamic_update_rate=self.dynamic_update_rate,
            dynamic_end_update_rate=self.dynamic_end_update_rate,
            activation_fn=self.activation_fn,
        )

        # accessibility
        # accessibility encoder
        if self.n_input_regions == 0:
            input_acc = 1
        else:
            input_acc = self.n_input_regions
        if self.ensembl_ids_regions is not None:
            input_ensembl_ids_regions = self.ensembl_ids_regions
            input_ensembl_ids_regions = np.where(pd.isna(input_ensembl_ids_regions), "ENSG00000000000", input_ensembl_ids_regions)
        else:
            input_ensembl_ids_regions = None
        #n_input_encoder_acc = input_acc + n_continuous_cov * encode_covariates
        n_input_encoder_acc = input_acc #+ n_continuous_cov * encode_covariates
        self.z_encoder_accessibility = Encoder(
            n_input=n_input_encoder_acc,
            n_layers=self.n_layers_encoder,
            ensembl_ids=input_ensembl_ids_regions,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
            distribution=self.latent_distribution,
            inject_covariates=deeply_inject_covariates,
            first_layer_inject_covariates=first_layer_inject_covariates,
            last_layer_inject_covariates=last_layer_inject_covariates,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            return_dist=False,
            layers_type=self.layers_encoder_type,
            gene_layer_type=self.accessibility_gene_layer_type,
            gene_layer_interaction_source=self.gene_layer_interaction_source,
            standard_gene_size=self.standard_gene_size,
            standard_go_size=self.standard_go_size,
            obo_file=self.obo_file,
            map_ensembl_go=self.map_ensembl_go,
            sparsities=self.sparsities,
            dynamic=self.dynamic,
            dynamic_update_rate=self.dynamic_update_rate,
            dynamic_end_update_rate=self.dynamic_end_update_rate,
            gene_interaction_layer_dynamic=self.gene_interaction_layer_dynamic,
            gene_interaction_layer_pruning_frac=self.gene_interaction_layer_pruning_frac,
            gene_interaction_layer_dynamic_update_rate=self.gene_interaction_layer_dynamic_update_rate,
            gene_interaction_layer_dynamic_end_update_rate=self.gene_interaction_layer_dynamic_end_update_rate,
            gene_interaction_layer_dynamic_save_path=os.path.join(self.gene_interaction_layer_dynamic_save_path,"accessibility") if self.gene_interaction_layer_dynamic_save_path else None,
            keep_activations=self.keep_activations,
        )

        # accessibility region-specific factors
        self.region_factors = None
        if region_factors:
            self.region_factors = torch.nn.Parameter(torch.zeros(self.n_input_regions))

        # accessibility decoder
        self.z_decoder_accessibility = DecoderPeakVI(
            n_input=self.n_latent, #+ self.n_continuous_cov
            n_output=n_input_regions,
            n_hidden=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.decoder_deeply_inject_covariates,
            layers_type=self.layers_decoder_type,
            gene_layer_type="none",
            sparsities=self.sparsities,
            dynamic=self.dynamic,
            dynamic_update_rate=self.dynamic_update_rate,
            dynamic_end_update_rate=self.dynamic_end_update_rate,
            activation_fn=self.activation_fn,
        )

        # accessibility library size encoder
        self.l_encoder_accessibility = DecoderPeakVI(
            n_input=n_input_encoder_acc,
            #n_input=n_input_encoder_acc + n_continuous_cov * encode_covariates,
            n_output=1,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list_library,
            n_layers=self.n_layers_encoder,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            deep_inject_covariates=self.decoder_deeply_inject_covariates,
            layers_type=self.library_size_layers_type,
            #gene_layer_type=self.accessibility_library_size_gene_layer_type,
            #gene_layer_interaction_source=self.gene_layer_interaction_source,
            sparsities=self.sparsities,
            dynamic=self.dynamic,
            dynamic_update_rate=self.dynamic_update_rate,
            dynamic_end_update_rate=self.dynamic_end_update_rate,
            activation_fn=self.activation_fn,
        )

        # protein
        # protein encoder
        self.protein_dispersion = protein_dispersion
        if protein_background_prior_mean is None:
            if n_batch > 0:
                self.background_pro_alpha = torch.nn.Parameter(
                    torch.randn(n_input_proteins, n_batch)
                )
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins, n_batch), -10, 1)
                )
            else:
                self.background_pro_alpha = torch.nn.Parameter(torch.randn(n_input_proteins))
                self.background_pro_log_beta = torch.nn.Parameter(
                    torch.clamp(torch.randn(n_input_proteins), -10, 1)
                )
        else:
            if protein_background_prior_mean.shape[1] == 1 and n_batch != 1:
                init_mean = protein_background_prior_mean.ravel()
                init_scale = protein_background_prior_scale.ravel()
            else:
                init_mean = protein_background_prior_mean
                init_scale = protein_background_prior_scale
            self.background_pro_alpha = torch.nn.Parameter(
                torch.from_numpy(init_mean.astype(np.float32))
            )
            self.background_pro_log_beta = torch.nn.Parameter(
                torch.log(torch.from_numpy(init_scale.astype(np.float32)))
            )

        # protein encoder
        if self.n_input_proteins == 0:
            input_pro = 1
        else:
            input_pro = self.n_input_proteins
        #n_input_encoder_pro = input_pro + n_continuous_cov * encode_covariates
        n_input_encoder_pro = input_pro
        if self.ensembl_ids_proteins is not None:
            input_ensembl_ids_proteins = self.ensembl_ids_proteins
            input_ensembl_ids_proteins = np.where(pd.isna(input_ensembl_ids_proteins), "ENSG00000000000", input_ensembl_ids_proteins)
        else:
            input_ensembl_ids_proteins = None
        self.z_encoder_protein = Encoder(
            n_input=n_input_encoder_pro,
            n_layers=self.n_layers_encoder,
            ensembl_ids=input_ensembl_ids_proteins,
            n_output=self.n_latent,
            n_hidden=self.n_hidden,
            n_cat_list=encoder_cat_list,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
            distribution=self.latent_distribution,
            inject_covariates=deeply_inject_covariates,
            first_layer_inject_covariates=first_layer_inject_covariates,
            last_layer_inject_covariates=last_layer_inject_covariates,
            var_eps=0,
            use_batch_norm=self.use_batch_norm_encoder,
            use_layer_norm=self.use_layer_norm_encoder,
            return_dist=False,
            layers_type=self.layers_encoder_type,
            gene_layer_type=self.protein_gene_layer_type,
            gene_layer_interaction_source=self.gene_layer_interaction_source,
            standard_gene_size=self.standard_gene_size,
            standard_go_size=self.standard_go_size,
            obo_file=self.obo_file,
            map_ensembl_go=self.map_ensembl_go,
            sparsities=self.sparsities,
            dynamic=self.dynamic,
            dynamic_update_rate=self.dynamic_update_rate,
            dynamic_end_update_rate=self.dynamic_end_update_rate,
            gene_interaction_layer_dynamic=self.gene_interaction_layer_dynamic,
            gene_interaction_layer_pruning_frac=self.gene_interaction_layer_pruning_frac,
            gene_interaction_layer_dynamic_update_rate=self.gene_interaction_layer_dynamic_update_rate,
            gene_interaction_layer_dynamic_end_update_rate=self.gene_interaction_layer_dynamic_end_update_rate,
            gene_interaction_layer_dynamic_save_path=os.path.join(self.gene_interaction_layer_dynamic_save_path,"protein") if self.gene_interaction_layer_dynamic_save_path else None,
            keep_activations=self.keep_activations,
        )

        # protein decoder
        self.z_decoder_pro = DecoderADT(
            n_input=n_input_decoder,
            n_output_proteins=n_input_proteins,
            n_hidden=self.n_hidden,
            n_cat_list=cat_list,
            n_layers=self.n_layers_decoder,
            use_batch_norm=self.use_batch_norm_decoder,
            use_layer_norm=self.use_layer_norm_decoder,
            deep_inject_covariates=self.deeply_inject_covariates,
            layers_type=self.layers_decoder_type,
            sparsities=self.sparsities,
            dynamic=self.dynamic,
            dynamic_update_rate=self.dynamic_update_rate,
            dynamic_end_update_rate=self.dynamic_end_update_rate,
            activation_fn=self.activation_fn,
        )

        # protein dispersion parameters
        if self.protein_dispersion == "protein":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins))
        elif self.protein_dispersion == "protein-batch":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins, n_batch))
        elif self.protein_dispersion == "protein-label":
            self.py_r = torch.nn.Parameter(2 * torch.rand(self.n_input_proteins, n_labels))
        else:  # protein-cell
            pass

        # modality alignment
        self.n_obs = n_obs
        self.modality_weights = modality_weights
        self.modality_penalty = modality_penalty
        self.n_modalities = int(n_input_genes > 0) + int(n_input_regions > 0)
        max_n_modalities = 4
        if modality_weights == "equal":
            mod_weights = torch.ones(max_n_modalities)
            self.register_buffer("mod_weights", mod_weights)
        elif modality_weights == "universal":
            self.mod_weights = torch.nn.Parameter(torch.ones(max_n_modalities))
        elif modality_weights == "moe":
            self.gating_network = GatingNetwork(latent_dim=self.n_latent, num_experts=3)
        else:  # cell-specific weights
            self.mod_weights = torch.nn.Parameter(torch.ones(n_obs, max_n_modalities))

    def _get_inference_input(self, tensors):
        """Get input tensors for the inference model."""
        x = tensors[REGISTRY_KEYS.X_KEY]
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        patient_index = tensors[REGISTRY_KEYS.PATIENT_KEY]
        cell_idx = tensors.get(REGISTRY_KEYS.INDICES_KEY).long().ravel()
        cont_covs = tensors.get(REGISTRY_KEYS.CONT_COVS_KEY)
        cat_covs = tensors.get(REGISTRY_KEYS.CAT_COVS_KEY)
        label = tensors[REGISTRY_KEYS.LABELS_KEY]
        input_dict = {
            "x": x,
            "y": y,
            "batch_index": batch_index,
            "patient_index": patient_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "label": label,
            "cell_idx": cell_idx,
        }
        return input_dict

    @auto_move_data
    def mix_modalities_poe(self, mus, vars, masks): #, weights, weight_transform: callable = None, mode=None):
        """Compute the PoE of the Xs while masking unmeasured modality values.

        Parameters
        ----------
        Xs
            Sequence of Xs to mix, each should be (N x D)
        masks
            Sequence of masks corresponding to the Xs, indicating whether the values
            should be included in the mix or not (N)
        """

        mus = torch.stack(mus, dim=1)
        vars = torch.stack(vars, dim=1)
        masks = torch.stack(masks, dim=1).float().unsqueeze(-1)
        mus_joint = torch.sum(mus * masks / vars, dim=1)
        vars_joint = torch.ones_like(mus_joint)
        vars_joint += torch.sum(masks / vars, dim=1)
        vars_joint = 1.0 / vars_joint
        mus_joint *= vars_joint

        return mus_joint, vars_joint

    @auto_move_data
    def mix_modalities_moe(self, mus, vars, masks, weights): #, weights, weight_transform: callable = None, mode=None):
        """Compute the PoE of the Xs while masking unmeasured modality values.

        Parameters
        ----------
        Xs
            Sequence of Xs to mix, each should be (N x D)
        masks
            Sequence of masks corresponding to the Xs, indicating whether the values
            should be included in the mix or not (N)
        """

        mus = torch.stack(mus, dim=1)
        vars = torch.stack(vars, dim=1)
        masks = torch.stack(masks, dim=1).float().unsqueeze(-1)
        if weights is None:
            weights = masks / torch.sum(masks, dim=1, keepdim=True)
        else:
            weights = weights.unsqueeze(axis=-1)
        mus_mixture = torch.sum(weights * mus, dim=1)
        vars_mixture = torch.sum(weights**2 * vars, dim=1)

        return mus_mixture, vars_mixture


    @auto_move_data
    def mix_modalities(self, Xs, masks, weights, weight_transform: callable = None, mode=None):
        """Compute the weighted mean of the Xs while masking unmeasured modality values.

        Parameters
        ----------
        Xs
            Sequence of Xs to mix, each should be (N x D)
        masks
            Sequence of masks corresponding to the Xs, indicating whether the values
            should be included in the mix or not (N)
        weights
            Weights for each modality (either K or N x K)
        weight_transform
            Transformation to apply to the weights before using them
        """

        # (batch_size x latent) -> (batch_size x modalities x latent)
        Xs = torch.stack(Xs, dim=1)
        # (batch_size) -> (batch_size x modalities)
        masks = torch.stack(masks, dim=1).float()
        weights = masked_softmax(weights, masks, dim=-1)

        # (batch_size x modalities) -> (batch_size x modalities x latent)
        weights = weights.unsqueeze(-1)
        if weight_transform is not None:
            weights = weight_transform(weights)

        # sum over modalities, so output is (batch_size x latent)
        return (weights * Xs).sum(1)

    @auto_move_data
    def inference(
        self,
        x,
        y,
        batch_index,
        patient_index,
        cont_covs,
        cat_covs,
        label,
        cell_idx,
        n_samples=1,
    ) -> dict[str, torch.Tensor]:
        """Run the inference model."""
        # Get Data and Additional Covs
        if self.n_input_genes == 0:
            x_rna = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_rna = x[:, : self.n_input_genes]
        if self.n_input_regions == 0:
            x_chr = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            x_chr = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_chr.sum(dim=1) > 0
        mask_pro = y.sum(dim=1) > 0

        encoder_input_expression = x_rna
        encoder_input_accessibility = x_chr
        encoder_input_protein = y
        """
        if cont_covs is not None and self.encode_covariates:
            encoder_input_expression = torch.cat((x_rna, cont_covs), dim=-1)
            encoder_input_accessibility = torch.cat((x_chr, cont_covs), dim=-1)
            encoder_input_protein = torch.cat((y, cont_covs), dim=-1)
        else:
            encoder_input_expression = x_rna
            encoder_input_accessibility = x_chr
            encoder_input_protein = y
        """

        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        if cont_covs is not None and self.encode_covariates:
            continuous_input = cont_covs
        else:
            continuous_input = ()

        # Z Encoders
        qzm_acc, qzv_acc, z_acc = self.z_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input, cont_input=continuous_input
        )
        qzm_expr, qzv_expr, z_expr = self.z_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input, cont_input=continuous_input
        )
        qzm_pro, qzv_pro, z_pro = self.z_encoder_protein(
            encoder_input_protein, batch_index, *categorical_input, cont_input=continuous_input
        )

        # L encoders
        libsize_expr = self.l_encoder_expression(
            encoder_input_expression, batch_index, *categorical_input, cont_input=continuous_input
        ) #encoder_input_expression
        libsize_acc = self.l_encoder_accessibility(
            encoder_input_accessibility, batch_index, *categorical_input, cont_input=continuous_input
        ) #encoder_input_accessibility


        if self.modality_weights == "cell":
            weights = self.mod_weights[cell_idx, :]
        elif self.modality_weights == "moe":
            weights = self.gating_network((qzm_expr, qzm_acc, qzm_pro), (qzv_expr, qzv_acc, qzv_pro), (mask_expr, mask_acc, mask_pro))
        else:
            weights = self.mod_weights.unsqueeze(0).expand(len(cell_idx), -1)

        if self.use_product_of_experts:
            qz_m, qz_v = self.mix_modalities_poe((qzm_expr, qzm_acc, qzm_pro), (qzv_expr, qzv_acc, qzv_pro),  (mask_expr, mask_acc, mask_pro))

            untran_z = Normal(qz_m, qz_v.sqrt()).rsample()

        elif self.use_mixture_of_experts:
            qz_m, qz_v = self.mix_modalities_moe((qzm_expr, qzm_acc, qzm_pro), (qzv_expr, qzv_acc, qzv_pro),  (mask_expr, mask_acc, mask_pro), weights)

            untran_z = Normal(qz_m, qz_v.sqrt()).rsample()
        elif self.use_mean_mixing:

            qz_m = self.mix_modalities(
                (qzm_expr, qzm_acc, qzm_pro), (mask_expr, mask_acc, mask_pro), weights, mode="mean"
            )
            qz_v = self.mix_modalities(
                (qzv_expr, qzv_acc, qzv_pro),
                (mask_expr, mask_acc, mask_pro),
                weights,
                torch.sqrt,
                mode="variance"
            )

            # sample
            if n_samples > 1:

                def unsqz(zt, n_s):
                    return zt.unsqueeze(0).expand((n_s, zt.size(0), zt.size(1)))

                untran_za = Normal(qzm_acc, qzv_acc.sqrt()).sample((n_samples,))
                z_acc = self.z_encoder_accessibility.z_transformation(untran_za)
                untran_ze = Normal(qzm_expr, qzv_expr.sqrt()).sample((n_samples,))
                z_expr = self.z_encoder_expression.z_transformation(untran_ze)
                untran_zp = Normal(qzm_pro, qzv_pro.sqrt()).sample((n_samples,))
                z_pro = self.z_encoder_protein.z_transformation(untran_zp)

                libsize_expr = unsqz(libsize_expr, n_samples)
                libsize_acc = unsqz(libsize_acc, n_samples)

            # sample from the mixed representation
            untran_z = Normal(qz_m, qz_v.sqrt()).rsample()
        else:
            raise ValueError("")

        z = self.z_encoder_accessibility.z_transformation(untran_z)

        outputs = {
            "untran_z": untran_z,
            "z": z,
            "qz_m": qz_m if 'qz_m' in locals() else None,
            "qz_v": qz_v if 'qz_v' in locals() else None,
            "z_expr": z_expr,
            "qzm_expr": qzm_expr,
            "qzv_expr": qzv_expr,
            "z_acc": z_acc,
            "qzm_acc": qzm_acc,
            "qzv_acc": qzv_acc,
            "z_pro": z_pro,
            "qzm_pro": qzm_pro,
            "qzv_pro": qzv_pro,
            "libsize_expr": libsize_expr,
            "libsize_acc": libsize_acc,
        }
        return outputs

    def _get_generative_input(self, tensors, inference_outputs, transform_batch=None):
        """Get the input for the generative model."""
        z = inference_outputs["z"]
        qz_m = inference_outputs["qz_m"]
        libsize_expr = inference_outputs["libsize_expr"]

        size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY
        size_factor = (
            torch.log(tensors[size_factor_key]) if size_factor_key in tensors.keys() else None
        )

        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]
        patient_index = tensors[REGISTRY_KEYS.PATIENT_KEY]
        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        label = tensors[REGISTRY_KEYS.LABELS_KEY]

        input_dict = {
            "z": z,
            "qz_m": qz_m,
            "batch_index": batch_index,
            "patient_index": patient_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
            "libsize_expr": libsize_expr,
            "size_factor": size_factor,
            "label": label,
        }
        return input_dict

    @auto_move_data
    def generative(
        self,
        z,
        qz_m,
        batch_index,
        patient_index,
        cont_covs=None,
        cat_covs=None,
        libsize_expr=None,
        size_factor=None,
        use_z_mean=False,
        label: torch.Tensor = None,
    ):
        """Runs the generative model."""
        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        latent = z if not use_z_mean else qz_m
        """
        if cont_covs is None:
            decoder_input = latent
        elif latent.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [latent, cont_covs.unsqueeze(0).expand(latent.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([latent, cont_covs], dim=-1)
        """
        decoder_input = latent
        if cont_covs is not None:
            continuous_input = cont_covs
        else:
            continuous_input = ()

        # Accessibility Decoder
        p = self.z_decoder_accessibility(decoder_input, batch_index, *categorical_input, cont_input=continuous_input)

        # Expression Decoder
        if not self.use_size_factor_key:
            size_factor = libsize_expr
        px_scale, _, px_rate, px_dropout = self.z_decoder_expression(
            self.gene_dispersion,
            decoder_input,
            size_factor,
            batch_index,
            *categorical_input,
            #label,
            cont_input=continuous_input
        )
        # Expression Dispersion
        if self.gene_dispersion == "gene-label":
            px_r = F.linear(
                F.one_hot(label.squeeze(-1), self.n_labels).float(), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.gene_dispersion == "gene-batch":
            px_r = F.linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.px_r)
        elif self.gene_dispersion == "gene":
            px_r = self.px_r
        px_r = torch.exp(px_r)

        # Protein Decoder
        py_, log_pro_back_mean = self.z_decoder_pro(decoder_input, batch_index, *categorical_input, cont_input=continuous_input)
        # Protein Dispersion
        if self.protein_dispersion == "protein-label":
            # py_r gets transposed - last dimension is n_proteins
            py_r = F.linear(F.one_hot(label.squeeze(-1), self.n_labels).float(), self.py_r)
        elif self.protein_dispersion == "protein-batch":
            py_r = F.linear(F.one_hot(batch_index.squeeze(-1), self.n_batch).float(), self.py_r)
        elif self.protein_dispersion == "protein":
            py_r = self.py_r
        py_r = torch.exp(py_r)
        py_["r"] = py_r

        return {
            "p": p,
            "patient_index": patient_index,
            "px_scale": px_scale,
            "px_r": torch.exp(self.px_r),
            "px_rate": px_rate,
            "px_dropout": px_dropout,
            "py_": py_,
            "log_pro_back_mean": log_pro_back_mean,
        }

    def loss(self, tensors, inference_outputs, generative_outputs, kl_weight: float = 1.0):
        """Computes the loss function for the model."""
        # Get the data
        x = tensors[REGISTRY_KEYS.X_KEY]

        x_rna = x[:, : self.n_input_genes]
        x_chr = x[:, self.n_input_genes : (self.n_input_genes + self.n_input_regions)]
        if self.n_input_proteins == 0:
            y = torch.zeros(x.shape[0], 1, device=x.device, requires_grad=False)
        else:
            y = tensors[REGISTRY_KEYS.PROTEIN_EXP_KEY]

        mask_expr = x_rna.sum(dim=1) > 0
        mask_acc = x_chr.sum(dim=1) > 0
        mask_pro = y.sum(dim=1) > 0

        if mask_acc.sum().gt(0):
            # Compute Accessibility loss
            p = generative_outputs["p"]
            libsize_acc = inference_outputs["libsize_acc"]
            rl_accessibility = self.get_reconstruction_loss_accessibility(x_chr, p, libsize_acc)
        else:
            rl_accessibility = torch.zeros(x.shape[0], device=x.device, requires_grad=False)

        # Compute Expression loss
        px_rate = generative_outputs["px_rate"]
        px_r = generative_outputs["px_r"]
        px_dropout = generative_outputs["px_dropout"]
        x_expression = x[:, : self.n_input_genes]
        rl_expression = self.get_reconstruction_loss_expression(
            x_expression, px_rate, px_r, px_dropout
        )

        # Compute Protein loss - No ability to mask minibatch (Param:None)
        if mask_pro.sum().gt(0):
            py_ = generative_outputs["py_"]
            rl_protein = get_reconstruction_loss_protein(y, py_, None)
        else:
            rl_protein = torch.zeros(x.shape[0], device=x.device, requires_grad=False)

        # calling without weights makes this act like a masked sum
        recon_loss_expression = rl_expression * mask_expr
        recon_loss_accessibility = rl_accessibility * mask_acc
        recon_loss_protein = rl_protein * mask_pro
        recon_loss = recon_loss_expression + recon_loss_accessibility + recon_loss_protein

        # Compute KLD between Z and N(0,I)
        if inference_outputs["qz_m"] is None and inference_outputs["qz_v"] is None: #TILLMANN
            kl_div_z = kld(
                inference_outputs["untran_z"],
                Normal(0, 1),
            ).sum(dim=1)
        else:
            qz_m = inference_outputs["qz_m"]
            qz_v = inference_outputs["qz_v"]
            kl_div_z = kld(
                Normal(qz_m, torch.sqrt(qz_v)),
                Normal(0, 1),
            ).sum(dim=1)

        # Compute KLD between distributions for paired data
        kl_div_paired = self._compute_mod_penalty(
            (inference_outputs["qzm_expr"], inference_outputs["qzv_expr"]),
            (inference_outputs["qzm_acc"], inference_outputs["qzv_acc"]),
            (inference_outputs["qzm_pro"], inference_outputs["qzv_pro"]),
            mask_expr,
            mask_acc,
            mask_pro,
            generative_outputs["patient_index"],
        )

        # KL WARMUP
        kl_local_for_warmup = kl_div_z
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_div_paired

        # TOTAL LOSS
        loss = torch.mean(recon_loss + weighted_kl_local)

        recon_losses = {
            "reconstruction_loss_expression": recon_loss_expression,
            "reconstruction_loss_accessibility": recon_loss_accessibility,
            "reconstruction_loss_protein": recon_loss_protein,
        }
        kl_local = {
            "kl_divergence_z": kl_div_z,
            "kl_divergence_paired": kl_div_paired,
        }
        return LossOutput(loss=loss, reconstruction_loss=recon_losses, kl_local=kl_local)

    def get_reconstruction_loss_expression(self, x, px_rate, px_r, px_dropout):
        """Computes the reconstruction loss for the expression data."""
        rl = 0.0
        if self.gene_likelihood == "zinb":
            rl = (
                -ZeroInflatedNegativeBinomial(mu=px_rate, theta=px_r, zi_logits=px_dropout)
                .log_prob(x)
                .sum(dim=-1)
            )
        elif self.gene_likelihood == "nb":
            rl = -NegativeBinomial(mu=px_rate, theta=px_r).log_prob(x).sum(dim=-1)
        elif self.gene_likelihood == "poisson":
            rl = -Poisson(px_rate).log_prob(x).sum(dim=-1)
        return rl

    def get_reconstruction_loss_accessibility(self, x, p, d):
        """Computes the reconstruction loss for the accessibility data."""
        reg_factor = torch.sigmoid(self.region_factors) if self.region_factors is not None else 1
        return torch.nn.BCELoss(reduction="none")(p * d * reg_factor, (x > 0).float()).sum(dim=-1)

    def _compute_mod_penalty(self, mod_params1, mod_params2, mod_params3, mask1, mask2, mask3, patient_index):
        """Computes Similarity Penalty across modalities given selection (None, Jeffreys, MMD).

        Parameters
        ----------
        mod_params1/2/3
            Posterior parameters for for modality 1/2/3
        mask1/2/3
            mask for modality 1/2/3
        """
        mask12 = torch.logical_and(mask1, mask2)
        mask13 = torch.logical_and(mask1, mask3)
        mask23 = torch.logical_and(mask3, mask2)

        if self.modality_penalty == "None":
            pair_penalty = torch.tensor([0.0]*mask1.shape[0], device=mask1.device, requires_grad=True)
        elif self.modality_penalty == "Jeffreys":
            pair_penalty = torch.zeros(mask1.shape[0], device=mask1.device, requires_grad=True)
            if mask12.sum().gt(0):
                penalty12 = sym_kld(
                    mod_params1[0],
                    mod_params1[1].sqrt(),
                    mod_params2[0],
                    mod_params2[1].sqrt(),
                )
                penalty12 = torch.where(mask12, penalty12.T, torch.zeros_like(penalty12).T).sum(
                    dim=0
                )
                pair_penalty = pair_penalty + penalty12
            if mask13.sum().gt(0):
                penalty13 = sym_kld(
                    mod_params1[0],
                    mod_params1[1].sqrt(),
                    mod_params3[0],
                    mod_params3[1].sqrt(),
                )
                penalty13 = torch.where(mask13, penalty13.T, torch.zeros_like(penalty13).T).sum(
                    dim=0
                )
                pair_penalty = pair_penalty + penalty13
            if mask23.sum().gt(0):
                penalty23 = sym_kld(
                    mod_params2[0],
                    mod_params2[1].sqrt(),
                    mod_params3[0],
                    mod_params3[1].sqrt(),
                )
                penalty23 = torch.where(mask23, penalty23.T, torch.zeros_like(penalty23).T).sum(
                    dim=0
                )
                pair_penalty = pair_penalty + penalty23

        elif self.modality_penalty == "MMD":
            pair_penalty = torch.zeros(mask1.shape[0], device=mask1.device, requires_grad=True)
            if mask12.sum().gt(0):
                penalty12 = torch.linalg.norm(mod_params1[0] - mod_params2[0], dim=1)
                penalty12 = torch.where(mask12, penalty12.T, torch.zeros_like(penalty12).T).sum(
                    dim=0
                )
                pair_penalty = pair_penalty + penalty12
            if mask13.sum().gt(0):
                penalty13 = torch.linalg.norm(mod_params1[0] - mod_params3[0], dim=1)
                penalty13 = torch.where(mask13, penalty13.T, torch.zeros_like(penalty13).T).sum(
                    dim=0
                )
                pair_penalty = pair_penalty + penalty13
            if mask23.sum().gt(0):
                penalty23 = torch.linalg.norm(mod_params2[0] - mod_params3[0], dim=1)
                penalty23 = torch.where(mask23, penalty23.T, torch.zeros_like(penalty23).T).sum(
                    dim=0
                )
                pair_penalty = pair_penalty + penalty23

        elif self.modality_penalty == "realMMD":
            mmd_loss = MMDLoss()

            pair_penalty = torch.zeros(mask1.shape[0], device=mask1.device, requires_grad=True)

            if mask12.sum().gt(0):
                penalty12 = mmd_loss(mod_params1[0][mask12], mod_params2[0][mask12])
                pair_penalty = pair_penalty + penalty12

            if mask13.sum().gt(0):
                penalty13 = mmd_loss(mod_params1[0][mask13], mod_params3[0][mask13])
                pair_penalty = pair_penalty + penalty13

            if mask23.sum().gt(0):
                penalty23 = mmd_loss(mod_params2[0][mask23], mod_params3[0][mask23])
                pair_penalty = pair_penalty + penalty23

            pair_penalty *= 40
        elif self.modality_penalty == "RF":
            pair_penalty = torch.zeros(mask1.shape[0], device=mask1.device, requires_grad=True)
            alpha_rf = 0.1
            if sum(bool(mask.item()) for mask in [mask12.sum().gt(0), mask13.sum().gt(0), mask14.sum().gt(0), mask23.sum().gt(0), mask24.sum().gt(0), mask34.sum().gt(0) ]) > 1:
                raise ValueError("RF only available for biomodal datasets.")
            if alpha_rf > 0:
                if mask12.sum().gt(0) and self.flow_top_to_bottom is not None:
                    penalty12 = self.flow_top_to_bottom(mod_params1[0][mask12], mod_params2[0][mask12]) + self.flow_bottom_to_top(mod_params2[0][mask12], mod_params1[0][mask12])
                    pair_penalty = pair_penalty + penalty12
                if mask13.sum().gt(0) and self.flow_top_to_bottom is not None:
                    penalty13 = self.flow_top_to_bottom(mod_params1[0][mask13], mod_params3[0][mask13]) + self.flow_bottom_to_top(mod_params3[0][mask13], mod_params1[0][mask13])
                    pair_penalty = pair_penalty + penalty13
                if mask23.sum().gt(0) and self.flow_top_to_bottom is not None:
                    penalty23 = self.flow_top_to_bottom(mod_params2[0][mask23], mod_params3[0][mask23]) + self.flow_bottom_to_top(mod_params3[0][mask23], mod_params2[0][mask23])
                    pair_penalty = pair_penalty + penalty23
        else:
            raise ValueError("modality penalty not supported")

        return pair_penalty

@auto_move_data
def sym_kld(qzm1, qzv1, qzm2, qzv2):
    """Symmetric KL divergence between two Gaussians."""
    rv1 = Normal(qzm1, qzv1.sqrt())
    rv2 = Normal(qzm2, qzv2.sqrt())

    return kld(rv1, rv2) + kld(rv2, rv1)


@auto_move_data
def get_reconstruction_loss_protein(y, py_, pro_batch_mask_minibatch=None):
    """Get the reconstruction loss for protein data."""
    py_conditional = NegativeBinomialMixture(
        mu1=py_["rate_back"],
        mu2=py_["rate_fore"],
        theta1=py_["r"],
        mixture_logits=py_["mixing"],
    )

    reconst_loss_protein_full = -py_conditional.log_prob(y)

    if pro_batch_mask_minibatch is not None:
        temp_pro_loss_full = pro_batch_mask_minibatch.bool() * reconst_loss_protein_full
        rl_protein = temp_pro_loss_full.sum(dim=-1)
    else:
        rl_protein = reconst_loss_protein_full.sum(dim=-1)

    return rl_protein
