import collections
from collections.abc import Iterable
from typing import Callable, Literal, Optional

import torch
from torch import nn
from torch.distributions import Normal
from torch.nn import ModuleList


def _identity(x):
    return x


import logging


logger = logging.getLogger(__name__)

import ast
import warnings
from collections import defaultdict
from typing import Sequence, Union, Tuple, Set, Dict, List
import numpy as np
import pandas as pd
import math
import torch
import torch.nn as nn
from tabulate import tabulate
from collections import Counter
import os

import sys
import pickle


import networkvi.nn._go_utils as utils
import networkvi.nn._go_utils as goo
from networkvi.nn._go_utils import GOTermExtended, lstrip_multiline
from goatools.obo_parser import GODag, GOTerm
from networkvi.nn._gene_models import GeneModelClassic
from networkvi.nn._go_graph import GoGraph
import logging
from addict import Dict as AttributeDict


class GeneLayers(nn.Module):
    """A helper class to build GeneLayers for a neural network.

    Parameters
    ----------
    ensembl_ids
        ENSEMBL-IDs of features.
    standard_gene_size
        Standard size of gene nodes in Gene Layers.
    input_dropout
        Dropout rate to apply to each of the hidden layers
    gene_layer_interaction_source
        Gene layer interaction source. One of the following
        * ``'pp'`` - Protein-Protein
        * ``'tf'`` - Transcription Factor
        * ``'tad'`` - Topologically Associated Domains
    n_hidden
        The number of nodes per hidden layer
    activation_fn
        Which activation function to use
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    """

    def __init__(
        self,
        ensembl_ids,
        standard_gene_size: int = 4,
        input_dropout: float = 0.1,
        gene_layer_interaction_source: Optional[str] = None,
        n_hidden: int = 128,
        activation_fn: nn.Module = nn.ReLU,
        n_cat_list=None,
        *args,
        **kwargs,
    ):

        super().__init__()

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            #self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
            self.n_cat_list = [n_cat if n_cat > 1 or (n_cat_index + 1 == len(n_cat_list) and len(n_cat_list) > 1) else 0 for n_cat_index, n_cat in enumerate(n_cat_list)]
        else:
            self.n_cat_list = []

        if activation_fn == nn.ReLU:
            activation_fn = "relu"
        elif activation_fn == nn.LeakyReLU:
            activation_fn = "lrelu"
        elif activation_fn == nn.Hardswish:
            activation_fn = "swish"
        else:
            raise NotImplementedError("")

        genemodel = {'gene_block_depth': 1,
                      'gene_scaling_layer': False,
                      'gene_intermediate_scaling': 1,
                      'gene_intermediate_mode': 'out_out',
                      'gene_embedding_linear_layer_enabled': True,
                      'standard_gene_size': standard_gene_size,
                      'embedder_standard_intermediate_size': 3,
                      'variant_embedding': False}
                      #'ppi': None,
                      #'tfi': None}

        if gene_layer_interaction_source is not None:
            ppi = {
                "ppi_file": gene_layer_interaction_source,
                "model": "LinearInteraction",
                "residual": True,
                "nlayers": 1,
            }
        else:
            ppi = None

        layer = {'input_dropout': input_dropout,
                  'track_running_stats': True,
                  'activation': activation_fn,
                  'bn_momentum': 0.1,
                  'residual': True,
                  'batch_norm': False,
                  'dropout': 0.0}

        self.ensembl_ids = ensembl_ids

        self.keep_activations = "False"

        self.geneobj = self.get_geneobj(
            ensembl_ids=ensembl_ids,
            standard_gene_size=genemodel["standard_gene_size"],
            #n_cat_list=n_cat_list,
        )

        logger.info("Prepared gene object.")

        self.genemodel = GeneModelClassic(
            self.geneobj,
            layer=layer,
            ppi=ppi,
            tfi=None,
            **genemodel,
        )

        if ppi is not None and "pruning_save_path" in ppi.keys() and ppi["pruning_save_path"] is not None:
            with open(os.path.join(ppi["pruning_save_path"], "geneobj.pickle"), "wb") as f:
                pickle.dump(self.genemodel.geneobj, f)

        #list(list(self.genemodel.interaction_layer.modules())[0][0].modules())[2].block_structure

    @staticmethod
    def get_geneobj(ensembl_ids, standard_gene_size): #n_cat_list

        """Iterate through datasets and create entries for each gene"""
        geneobj = {}

        di = 0
        for ci, eid in enumerate(ensembl_ids):
            # Insert a new snp into the geneobject if it does not exist yet
            feature = {
                "index": ci,  # calc_offset(), # self.dataset, (di, ci)),
                "index_separted": (di, ci),
            }
            geneobj.setdefault(
                eid,
                {"inputs": [], "layersize": standard_gene_size},
            )["inputs"].append(feature)

        """
        if n_cat_list is not None:
            for index_n_cat_list in range(len(n_cat_list)):
                covariate = {
                    "index": len(ensembl_ids)+index_n_cat_list,
                    "index_separted": (di, len(ensembl_ids)+index_n_cat_list),
                }
                for eid in geneobj.keys():
                    geneobj[eid]["inputs"].append(covariate)
        """

        return geneobj

    #def forward(
    #    self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], *args
    #) -> torch.Tensor:
    def forward(self, x: torch.Tensor): #, *cat_list: int, cont_input: torch.Tensor = None

        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list) + 1 if cont_input is not None else 0:
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]

        out = self.genemodel(torch.cat((x, *one_hot_cat_list), dim=-1))
        """
        out = self.genemodel(x)

        return out

class GOLayers(nn.Module):
    """A helper class to build GOLayers for a neural network.

    Parameters
    ----------
    geneobj
        geneobj generated using GeneLayers.
    genemodel_out_blocks
        out blocks sparse matrix of object generated using GeneLayers.
    ensembl_ids
        ENSEMBL-IDs of features.
    obo_file
        Path .obo file of GO.
    map_ensembl_go
        List of .gaf files with mappings of Ensembl IDs to GO.
    standard_go_size
        Standard size of GO nodes in GO Layers.
    input_dropout
        Dropout rate to apply to each of the hidden layers
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    activation_fn
        Which activation function to use
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    inject_covariates
        Whether to deeply inject covariates into all layers of the endecoder. If False,
        covariates will only be included in the input layer.
    first_layer_inject_covariates
        Whether to deeply inject covariates into all layers of the decoder. If False,
        covariates will only be included in the input layer.
    last_layer_inject_covariates
        Whether to inject covariates into all layers of the decoder. If False,
        covariates will only be included in the input layer.
    """

    def __init__(
        self,
        geneobj,
        genemodel_out_blocks,
        ensembl_ids,
        obo_file: str,
        map_ensembl_go: list | np.ndarray,
        standard_go_size: int = 6,
        input_dropout: float = 0.1,
        n_layers: int = 5,
        n_hidden: int = 128,
        activation_fn: nn.Module = nn.ReLU,
        dynamic_go_size: bool = False,
        register_rancon: bool = False,
        remove_rancon: bool = False,
        n_cat_list=None,
        inject_covariates: bool = True,
        first_layer_inject_covariates: bool = True,
        last_layer_inject_covariates: bool = True,
        *args,
        **kwargs,
    ):

        super().__init__()

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            #self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
            self.n_cat_list = [n_cat if n_cat > 1 or (n_cat_index + 1 == len(n_cat_list) and len(n_cat_list) > 1) else 0 for n_cat_index, n_cat in enumerate(n_cat_list)]
        else:
            self.n_cat_list = []
        self.inject_covariates = inject_covariates
        self.first_layer_inject_covariates = first_layer_inject_covariates
        self.last_layer_inject_covariates = last_layer_inject_covariates

        if activation_fn == nn.ReLU:
            activation_fn_str = "relu"
        elif activation_fn == nn.LeakyReLU:
            activation_fn_str = "lrelu"
        elif activation_fn == nn.Hardswish:
            activation_fn_str = "swish"
        else:
            raise NotImplementedError("")

        hierarchy = {'relations': {'is_a': True,
                                    'part_of': False,
                                    'regulates': False,
                                    'regulates_neg': False,
                                    'regulates_pos': False},
                      'min_genes_disrupted': 0,
                      'mode': 'depth',
                      'genetic_json': "",
                      'obo_file': obo_file,
                      'map_ensembl_go': map_ensembl_go,
                      # goa_human_databases_uniprot_reactome_ensembl_intact_gocentral_ensembl_gene_mapping.gaf
                      # goa_human_npinter_nc_ensembl_gene_mapping.gaf
                      # goa_human_npinter_tar_v2_ensembl_gene_mapping.gaf
                      'gene_transcript': 'gene',
                      'max_level': {'depth': n_layers},
                      'ontology': 'biological_process',
                      'regulates_init': False,
                      'regulates_masking': False,
                      'dynamic_go_size': 'disabled',
                      'dynamic_go': None,
                      'filter_namespace': True}

        n_latent = n_hidden

        if not inject_covariates and not first_layer_inject_covariates:
            attention = None
            """"{'enabled': False,
                          'mode': None,
                          'levels': None,
                          'inter_dim': None,
                          'values_from': None,
                          'dropout': None,
                          'aggregate': None,
                          'residual': None,
                          'layer_norm': None,
                          'reduce': None,
                          'head': None,
                          'multihead': None}
            """
        else:
            attention = {'enabled': True,
                          'mode': "series",
                          'at_levels': "all" if inject_covariates else [-1],
                          'inter_dim': 1,
                          'values_from': "covariates",
                          'dropout': 0.0,
                          'aggregate': False,
                          'combine': False,
                          'residual': True,
                          'layer_norm': True,
                          'reduce': False,
                          #'head': "AttentionDecoder",
                          'multihead': 1}
        if last_layer_inject_covariates:
            self.fclayer = FCLayers(
                n_in=n_latent,
                n_out=n_latent,
                n_cat_list=n_cat_list,
                n_layers=1,
                n_hidden=n_latent,
                dropout_rate=input_dropout,
                use_batch_norm=False,
                use_layer_norm=True,
                use_activation=True,
                activation_fn=activation_fn,
            )
        else:
            self.act = activation_fn()
            #self.batch_norm = nn.BatchNorm1d(n_latent, momentum=0.01, eps=0.001)
            self.layer_norm = nn.LayerNorm(n_latent, elementwise_affine=False)

        gomodel = {'go_unknown_go_size': 12,
                    'go_unknown_hierarchy_depth': 4,
                    'standard_go_block_depth': 1,
                    'standard_go_size': standard_go_size,
                    'go_intermediate_scaling': 1.0,
                    'go_intermediate_mode': 'out_out',
                    }

        layer = {'input_dropout': input_dropout,
                  'track_running_stats': True,
                  'activation': activation_fn_str,
                  'bn_momentum': 0.1,
                  'residual': True,
                  'batch_norm': False,
                  'dropout': 0.0}

        self.ensembl_ids = ensembl_ids
        self.goobj = None
        self.goobj_unfiltered = None
        self.neurons_available = None
        self.is_layersize_set = False
        self.hierarchy = AttributeDict(hierarchy)
        self.keep_activations = "False"

        if geneobj is None:
            geneobj = GeneLayers.get_geneobj(
                ensembl_ids=ensembl_ids,
                standard_gene_size=1,
                #n_cat_list=n_cat_list,
            )
            for gene_index, gene in enumerate(geneobj):
                geneobj[gene]["block_index"] = [gene_index]

        self.initialize_go_obj(
            n_latent=n_latent,
            go_unknown_hierarchy_depth=gomodel["go_unknown_hierarchy_depth"],
            go_unknown_go_size=gomodel["go_unknown_go_size"],
            standard_go_size=gomodel["standard_go_size"],
            dynamic_go_cfg=self.hierarchy["dynamic_go"],
            **hierarchy,
        )

        self.gomodel = GoGraph(
            self.goobj,
            geneobj,
            genemodel_out_blocks,
            n_covariates=sum(self.n_cat_list) if n_cat_list is not None else 0,
            register_rancon=register_rancon,
            remove_rancon=remove_rancon,
            attention=attention,
            hierarchy=hierarchy,
            layer=layer,
            n_latent=n_latent,
            **gomodel,
        )

    def get_parents(self, term):
        parents = set()
        if term == None:
            return {}
        direct_parents = term.parents
        for parent in direct_parents:
            parentparents = self.get_parents(parent)
            parents |= parentparents
        parents |= direct_parents
        return parents

    def get_sizes_by_level(self, name="depth") -> Dict[int, List[int]]:
        """Returns the OGM's layersizes ordered per depth/ogmdepth/height as dict mapping level -> list of layersizes"""
        assert name in ["depth", "ogm_depth", "height"]
        if not self.is_layersize_set:
            return None
        sizes = defaultdict(list)
        goobj = self.goobj
        for goid in goobj:
            go = goobj[goid]
            level = getattr(go, name)
            if hasattr(go, "layersize"):
                sizes[level].append(go.layersize)
            else:
                warnings.warn(f"GOTerm {goid} has no attribute `layersize`!")
                sizes[level].append(-1)
        return sizes

    def get_total_size(self) -> int:
        """Returns the OGM's total number of neurons."""
        goobj = self.goobj
        total_size = 0
        for goid in goobj:
            go = goobj[goid]
            if hasattr(go, "layersize"):
                total_size += go.layersize
            else:
                warnings.warn(f"GOTerm {goid} has no attribute `layersize`!")
        return max(1, total_size)

    def get_neurons_available(
        self, dynamic_go_cfg, standard_go_size: int, n_latent: int
    ) -> Dict[int, int]:
        """Returns the OGM's number of neurons available for distribution per height level."""
        # if self.neurons_available is not None:
        #     return self.neurons_available

        height_histogram = utils.get_height_dist(self.goobj)
        max_height = max(height_histogram.keys())
        heights = list(range(max_height + 1))
        ngos = [
            height_histogram[h] if h in height_histogram else 0
            for h in range(max_height + 1)
        ]

        input_size = standard_go_size * ngos[0]
        scaling_strategy = dynamic_go_cfg.scaling_strategy
        neurons_available = defaultdict(int)
        for h in heights:
            DEFAULT = standard_go_size * ngos[h]
            if h == 0:
                neurons_available[h] = DEFAULT
            elif h == max_height:
                neurons_available[h] = n_latent
            # Increase the available number of neurons per height level and neuron by constant scaling factor
            elif scaling_strategy.type == "constant_scaling":
                neurons_available[h] = int(
                    (dynamic_go_cfg.scaling_strategy.factor**h) * DEFAULT
                )
            # Keep the available number of neurons per height level constant
            elif scaling_strategy.type == "constant_total":
                neurons_available[h] = standard_go_size * ngos[0]
            elif scaling_strategy.type == "differential_ngos":
                scaling = n_latent / input_size
                # The scaling base that needs to be chose s.t. input_size * base^max_level = n_latent
                level_cnt = len(heights) - 1
                base = 10 ** (np.log10(scaling) / level_cnt)
                # logging.info(f'For scaling strategy {scaling_strategy.type}, calculated a base of {base:.6f}.')
                neurons_available[h] = int(input_size * base**h)
            elif scaling_strategy.type == "differential_go_size":
                scaling = n_latent / standard_go_size
                level_cnt = len(heights) - 1
                base = 10 ** (np.log10(scaling) / level_cnt)
                neurons_available[h] = int((base**h) * DEFAULT)
            else:
                neurons_available[h] = DEFAULT
        return neurons_available

    def get_goobj_info(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        goobj = self.goobj
        # Get statistics on the graph structure:
        if self.is_layersize_set:
            total_size = self.get_total_size()
        total_cnt = len(goobj)

        logger.info(f"\nGO object with {total_cnt} GOTerms.")
        relations_used = ", ".join(
            [k for k, v in self.hierarchy["relations"].items() if v]
        )

        def get_stats_table(go_sizes, histogram):
            max_key = max(histogram.keys())
            depth_table = [
                ["Level", *range(max_key + 1)],
                [
                    "#GOTerms",
                    *[
                        histogram[d] if d in histogram else 0
                        for d in range(max_key + 1)
                    ],
                ],
            ]
            if go_sizes is not None:
                if [
                    np.min(go_sizes[d]) if d in histogram else 0
                    for d in range(max_key + 1)
                ][0] != -1:
                    depth_table += [
                        [
                            "min Size",
                            *[
                                np.min(go_sizes[d]) if d in histogram else 0
                                for d in range(max_key + 1)
                            ],
                        ],
                        [
                            "med Size",
                            *[
                                np.median(go_sizes[d]) if d in histogram else 0
                                for d in range(max_key + 1)
                            ],
                        ],
                        [
                            "max Size",
                            *[
                                np.max(go_sizes[d]) if d in histogram else 0
                                for d in range(max_key + 1)
                            ],
                        ],
                        [
                            "sum Size",
                            *[
                                np.sum(go_sizes[d]) if d in histogram else 0
                                for d in range(max_key + 1)
                            ],
                        ],
                        [
                            "%   Size",
                            *[
                                int(1000 * np.sum(go_sizes[d]) / total_size) / 10
                                if d in histogram
                                else 0
                                for d in range(max_key + 1)
                            ],
                        ],
                    ]
            return depth_table

        depth_histogram = goo.get_depth_dist(self.goobj)
        sizes_by_depth = self.get_sizes_by_level(name="depth")
        logger.info(f"\nDepth distribution (using is_a only):")
        depth_table = get_stats_table(sizes_by_depth, depth_histogram)
        logger.info("\n" + tabulate(depth_table, tablefmt="github"))

        ogm_depth_histogram = goo.get_ogm_depth_dist(self.goobj)
        sizes_by_ogm_depth = self.get_sizes_by_level(name="ogm_depth")
        logger.info(f"\nDepth distribution (using {relations_used}):")
        ogm_depth_table = get_stats_table(sizes_by_ogm_depth, ogm_depth_histogram)
        logger.info("\n" + tabulate(ogm_depth_table, tablefmt="github"))

        height_histogram = goo.get_height_dist(self.goobj)
        sizes_by_height = self.get_sizes_by_level(name="height")
        logger.info(f"\nHeight distribution (using {relations_used}):")
        height_table = get_stats_table(sizes_by_height, height_histogram)
        logger.info("\n" + tabulate(height_table, tablefmt="github"))

        logger.info("\n")
        depth_df = pd.DataFrame({line[0]: line[1:] for line in depth_table})
        height_df = pd.DataFrame({line[0]: line[1:] for line in height_table})
        return depth_df, height_df

    def get_overflow_info(
        self, dynamic_go_cfg, standard_go_size: int, n_latent: int
    ) -> pd.DataFrame:
        goobj = self.goobj
        max_height = -1
        for goid in goobj:
            go = goobj[goid]
            max_height = max(max_height, go.height)
        heights = range(max_height)
        neurons_available = self.get_neurons_available(
            dynamic_go_cfg, standard_go_size, n_latent
        )
        scaling_strategy = dynamic_go_cfg.scaling_strategy
        distribution_strategy = dynamic_go_cfg.distribution_strategy

        neurons_used = defaultdict(int)
        for goid in goobj:
            go = goobj[goid]
            neurons_used[go.height] += go.layersize
        overflows = [neurons_used[h] - neurons_available[h] for h in heights]
        overflow_table = [
            ["Height", *heights],
            ["#Neurons scheduled", *[neurons_available[h] for h in heights]],
            ["#Neurons used", *[neurons_used[h] for h in heights]],
            ["#Overflow", *overflows],
        ]
        logger.info(
            f"\nOverflows using scaling strategy {scaling_strategy.type}, distributions strategy {distribution_strategy.type}:"
        )
        logger.info("\n" + tabulate(overflow_table, tablefmt="github"))
        overflow_df = pd.DataFrame({col[0]: col[1:] for col in overflow_table})
        return overflow_df

    def get_ds_terms_and_parents(
        self, unknown: GOTerm, goobj: GODag, gos: Sequence[GOTermExtended]
    ) -> Tuple[Set[GOTermExtended], Set[GOTermExtended]]:
        ds_terms = set()
        # def calc_offset(separate_idx):  # ds, separate_idx):
        #     offset = 0
        #     ds_idx, idx = separate_idx
        #     return offset + idx # this is lecay code that allows to combine multiple genetic datasets, which we don't do anymore
        #
        #     # Returns index of separate_idx indexer transformed to a concatenated dataset
        #     offset = 0
        #     ds_idx, idx = separate_idx
        #
        #     # For each Dataset with a lower index than ds_idx add the number of columns of that dataset
        #     for _ds_idx in range(ds_idx):
        #         offset += ds.ds.datasets[_ds_idx].get_width()
        #
        #     return offset + idx
        for goterms in gos:
            filtered_col = {term for term in goterms if term in goobj}
            # If the term has no parents in the Graph, then add it to the Unknown Go node
            if len(filtered_col) == 0:
                filtered_col.add(unknown.id)
            ds_terms = ds_terms | filtered_col

        # Collect ds_terms and ds_terms parents into one set
        ds_parents = set().union(
            *[self.get_parents(goobj[term]) for term in ds_terms]
        )  # Collect a set of sets of parents for each node
        ds_parents = {
            x.id for x in ds_parents
        }  # Unify the set of lists into one set of parents
        return ds_terms, ds_parents

    def set_ensemblids(self, unknown, goobj, gos, ensemblids):
        """Sets the ensemblid attribute for every GOTerm of the GO object."""
        assert len(gos) == len(ensemblids)
        for goid in goobj:
            goobj[goid].ensemblids = set()
        for ci, (goterms, ensembleid) in enumerate(zip(gos, ensemblids)):
            # Create a queue which contains all go terms for this ensemble id
            filtered_col = {term for term in goterms if term in goobj}
            # If the term has no parents in the Graph, then add it to the Unknown Go node
            if len(filtered_col) == 0:
                filtered_col.add(unknown.id)
            for t in filtered_col:
                goobj[t].ensemblids.add(ensembleid)

    def set_layersize_dynamic(
        self,
        dynamic_go_cfg,
        goobj: GODag,
        relations,
        n_latent: int,
        standard_go_size: int,
    ):
        distribution_strategy = dynamic_go_cfg.distribution_strategy
        ngenes_cumsum = goo.get_ngenes_cumsum(goobj)
        nchildren_cumsum = goo.get_nchildren_cumsum(goobj, relations)
        self.neurons_available = self.get_neurons_available(
            dynamic_go_cfg, standard_go_size, n_latent
        )
        ngos = goo.get_height_dist(goobj)

        def _get_layersize_go(go: GOTermExtended) -> int:
            # Take n_latent dimension for the ontology root node
            if go.depth == 0:
                return n_latent
            # Take the standard_go_size for leaf nodes
            elif go.height == 0:
                return standard_go_size
            # Assign go size to inner nodes according to distribution strategy
            else:
                if distribution_strategy.type == "equal":
                    share = 1 / ngos[go.height]
                elif distribution_strategy.type == "genes_below":
                    share = len(go.genes_disrupted) / ngenes_cumsum[go.height]
                elif distribution_strategy.type == "gos_below":
                    share = len(go.children) / nchildren_cumsum[go.height]
                else:
                    return standard_go_size
                layersize = max(
                    dynamic_go_cfg.min_go_size,
                    int(share * self.neurons_available[go.height]),
                )
            return layersize

        for goid in goobj:
            go = goobj[goid]
            go.layersize = _get_layersize_go(go)

    def set_layersize(
        self,
        goobj,
        relations,
        standard_gene_size,
        n_latent,
        dynamic_go_size,
        go_unknown_go_size,
        standard_go_size,
    ):
        """Sets the layersize attribute for every GOTerm of the GO object. NOTE: DEPRECATED."""

        def _get_go_block_size(
            name, layer, num_genes_below, num_children, num_neurons_below
        ):
            if (
                layer == 0 or layer == -1
            ):  # 0 is the root node, -1 is the top node above unknown and the three root nodes (MF, BP, CC)
                return n_latent

            if "UNKNOWN" in name:
                return go_unknown_go_size
            else:
                return standard_go_size

        for go in self.order:
            if not go in goobj:  # we filtered this one out before
                continue
            children = utils.get_all_children(goobj[go], **relations)
            neurons_below = 0
            for c in children:
                try:
                    neurons_below += c.layersize
                except AttributeError:
                    pass
            if neurons_below == 0:
                neurons_below = len(goobj[go].genes_disrupted) * standard_gene_size

            goobj[go].layersize = _get_go_block_size(
                go,
                goobj[go].level,
                len(goobj[go].genes_disrupted),
                len(children),
                neurons_below,
            )
            goobj[go].neurons_below = neurons_below
            if n_latent == -1:
                if goobj[go].level == 0:
                    n_latent = goobj[
                        go
                    ].layersize
                    logger.debug(
                        f"Changed adaptive n_latent -1 to {n_latent} dynamically."
                    )
            # logger.debug(goobj[go].level, goobj[go].genes_disrupted, len(goobj[go].children), '-->', goobj[go].layersize)

    def set_topnode_and_unknown(
        self, goobj, ontology, go_unknown_hierarchy_depth
    ) -> Tuple[GOTermExtended, GOTermExtended]:
        """Sets and returns the top (root) node and UNKNOWN node of the ontology."""
        # Prepare topnode
        roots = [go_name for go_name in goobj if len(goobj[go_name].parents) == 0]
        if len(roots) == 1:
            root = roots[0]
            topnode = goobj[root]
        else:
            t = GOTermExtended()
            t.parents = set([])
            t.level = -1
            t.depth = -1
            t.ogm_depth = -1
            tid = "TOPNODE"
            t.id = tid
            t.item_id = tid
            t.namespace = ontology
            for root in roots:
                t.children.add(goobj[root])
                goobj[root].parents.add(t)
            topnode = t
            goobj[tid] = t

        # Prepare UNKNOWN node
        UNKNOWN_ID = "GO:UNKNOWN"
        for go_unknown_hierarchy in list(range(1, go_unknown_hierarchy_depth + 1)):
            if go_unknown_hierarchy <= self.hierarchy.max_level.depth:
                unknown = GOTermExtended()
                unknown.level = go_unknown_hierarchy
                unknown.depth = go_unknown_hierarchy
                unknown.ogm_depth = go_unknown_hierarchy
                unknown.id = UNKNOWN_ID + str(go_unknown_hierarchy)
                unknown.item_id = UNKNOWN_ID + str(go_unknown_hierarchy)
                unknown.namespace = ontology
                goobj[UNKNOWN_ID + str(go_unknown_hierarchy)] = unknown

        for go_unknown_hierarchy in list(range(1, go_unknown_hierarchy_depth + 1)):
            if go_unknown_hierarchy <= self.hierarchy.max_level.depth:
                if go_unknown_hierarchy == 1:
                    goobj[UNKNOWN_ID + str(go_unknown_hierarchy)].parents = set(
                        [topnode]
                    )
                    if go_unknown_hierarchy_depth > 1 and self.hierarchy.max_level.depth > 1:
                        goobj[UNKNOWN_ID + str(go_unknown_hierarchy)].children.add(
                            goobj[UNKNOWN_ID + str(go_unknown_hierarchy + 1)]
                        )
                    topnode.children.add(unknown)
                else:
                    goobj[UNKNOWN_ID + str(go_unknown_hierarchy)].parents.add(
                        goobj[UNKNOWN_ID + str(go_unknown_hierarchy - 1)]
                    )
                    if (
                        go_unknown_hierarchy != go_unknown_hierarchy_depth
                        and go_unknown_hierarchy < self.hierarchy.max_level.depth
                    ):
                        goobj[UNKNOWN_ID + str(go_unknown_hierarchy)].children.add(
                            goobj[UNKNOWN_ID + str(go_unknown_hierarchy + 1)]
                        )

        return topnode, unknown

    def patch_bup_ontology(self, goobj: GODag, relations):
        """Patch ontology built from the bottom up (i.e. with filtering for max_level height)."""
        if "height" not in self.hierarchy.max_level:
            logger.info("Nothing to patch since `model.max_level.height` is not set.")
        else:
            order = sorted(list(goobj.keys()), key=lambda x: goobj[x].height)
            greatest_height = goobj[order[-1]].height

            root_gos = []
            for gokey in self.order[::-1]:
                go = goobj[gokey]
                if go.height == greatest_height:
                    root_gos.append(go)
                else:
                    break

            # Make all GOTerms of highest non-root layer children of root:
            for in_goid in order[::-1]:
                in_go = goobj[in_goid]
                if in_go.height == self.hierarchy.max_level.height:
                    for root in root_gos:
                        root.children.add(in_go)
                        in_go.parents.add(root)
                elif in_go.height < self.hierarchy.max_level.height:
                    break

            # If a node has a deleted parent that is connected to a root node,
            # add a connection from that node to the root node.
            for root in root_gos:
                all_children = utils.get_all_children(root, **relations)
                for c_go in all_children:
                    if in_go.height > self.hierarchy.max_level.height:
                        all_children_children = utils.get_all_children(
                            c_go, **relations
                        )
                        for cc_go in c_go in all_children_children:
                            if (
                                cc_go.height < self.hierarchy.max_level.height
                                and cc_go.id in goobj
                            ):
                                root.children.add(cc_go)

    def get_ordered_gos(
        self,
        ensemblids,
        genetic_json,
        map_ensembl_go: str,
        ontology: str,
        obo_file: str,
        gene_transcript: str,
    ):
        if map_ensembl_go is None:
            logger.warn("No ensembl to GO mapping provided.")
            ordered_gos = None
        else:

            ordered_gos = goo.map2gos_v2(
                ensemblids_input=ensemblids,
                genetic_json=genetic_json,
                map_ensembl_go=map_ensembl_go,
                ontology=ontology,
                obo_file=obo_file,
                gene_transcript=gene_transcript,
                logger=logger,
            )

        return ordered_gos

    def initialize_go_obj(
        self,
        mode,
        obo_file,
        max_level,
        relations,
        ontology,
        map_ensembl_go,
        genetic_json,
        gene_transcript,
        go_unknown_go_size,
        go_unknown_hierarchy_depth,
        standard_go_size,
        min_genes_disrupted,
        n_latent,
        # regulates_init,
        # regulates_masking,
        dynamic_go_size,
        dynamic_go_cfg,
        filter_namespace=None,
        *args,
        **kwargs,
    ):

        goobj = GODag(obo_file, optional_attrs=["relationship"])
        goobj_copy = GODag(obo_file, optional_attrs=["relationship"])
        utils.filter_goobj(
            goobj, lambda goid: goobj[goid].id != goid, bubble_up_ensemblids=False
        )
        utils.filter_goobj(
            goobj_copy,
            lambda goid: goobj_copy[goid].id != goid,
            bubble_up_ensemblids=False,
        )
        if filter_namespace:
            namespaces = ontology.split(",")
            utils.filter_goobj(
                goobj,
                lambda goid: goobj[goid].namespace not in namespaces,
                bubble_up_ensemblids=False,
            )
        self.goobj = goobj
        self.goobj_unfiltered = goobj_copy
        topnode, unknown = self.set_topnode_and_unknown(
            goobj, ontology, go_unknown_hierarchy_depth
        )
        utils.extend_godag(self.goobj)
        utils.set_heights(self.goobj, **relations)
        utils.set_ogm_depths(self.goobj, **relations)
        logger.info("Before filtering:")
        self.get_goobj_info()
        # Collect all GO Terms in the dataset and mark those that are also part of our GOObj
        self.ordered_gos = self.get_ordered_gos(
            self.ensembl_ids,
            genetic_json,
            map_ensembl_go,
            ontology,
            obo_file,
            gene_transcript,
        )

        self.set_ensemblids(
            unknown,
            self.goobj,
            self.ordered_gos,
            self.ensembl_ids,
        )
        ds_terms, ds_parents = self.get_ds_terms_and_parents(
            unknown, self.goobj, self.ordered_gos
        )
        self.ds_terms_and_parents = (
            ds_parents | ds_terms
        )  # Unify the parents and the children
        utils.filter_goobj(
            self.goobj,
            lambda goid: goid not in self.ds_terms_and_parents,
            remove_obsolete_refs=True,
        )
        logger.info(
            str(len(self.goobj.keys()))
            + " Go Concepts and their parents associated with genes"
        )
        logger.info(str(len(ds_parents)) + " unique GO terms found in the dataset")
        if len(ds_parents) < 10:
            ds_parents_annotated = {
                k: self.goobj_unfiltered[k].name for k in ds_parents
            }
            logger.info(f"Specifically these ones: {ds_parents_annotated}")

        self.order = sorted(
            list(self.goobj.keys()), key=lambda x: -self.goobj[x].ogm_depth
        )

        def _filter_levels(goid):
            go = self.goobj[goid]

            if "depth" in max_level:
                if go.depth > max_level["depth"]:
                    return True
            if "ogm_depth" in max_level:
                if go.ogm_depth > max_level["ogm_depth"]:
                    return True
            if "height" in max_level:
                if go.height > max_level["height"] and not (go.depth == 0):
                    return True
            return False

        utils.filter_goobj(self.goobj, _filter_levels)
        utils.set_genes_disrupted(self.goobj)
        utils.filter_goobj(
            self.goobj,
            lambda goid: len(self.goobj[goid].genes_disrupted) < min_genes_disrupted,
        )
        if "height" in max_level:
            self.order = sorted(
                list(self.goobj.keys()), key=lambda x: self.goobj[x].height
            )
            self.patch_bup_ontology(self.goobj, relations)
            utils.prune_obsolete_refs(self.goobj)
        if mode == "height":
            self.set_layersize_dynamic(
                dynamic_go_cfg,
                self.goobj,
                self.hierarchy["relations"],
                n_latent,
                standard_go_size,
            )
            self.get_overflow_info(dynamic_go_cfg, standard_go_size, n_latent)
        else:
            standard_gene_size = 6
            self.set_layersize(
                self.goobj,
                relations,
                standard_gene_size,
                n_latent,
                dynamic_go_size,
                go_unknown_go_size,
                standard_go_size,
            )
        self.is_layersize_set = True
        logger.info(f"After filtering:")
        self.get_goobj_info()
        filter_info = f"""
        {len(goobj.keys())} Go Concepts are affected by at least {min_genes_disrupted-1} genes,
        structured by hierarchy mode {mode} and below max_levels:
        """
        logger.info(lstrip_multiline(filter_info))
        logger.info("\n ".join([f"\t{key}: {val}" for key, val in max_level.items()]))
        if unknown.id in goobj.keys():
            logger.info(f"{len(goobj[unknown.id].ensemblids)} features map to unknown GO Term.")

    def get_n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    #def forward(
    #    self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], *args
    #) -> torch.Tensor:
    def forward(self, x: torch.Tensor, *cat_list: int, cont_input: torch.Tensor = None):

        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list) + 1 if cont_input is not None else 0:
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]

        if self.inject_covariates or self.first_layer_inject_covariates:
            if cont_input is not None and len(cont_input) != 0:
                covariates = torch.concat(one_hot_cat_list + [cont_input], axis=1).float()
            else:
                covariates = torch.concat(one_hot_cat_list, axis=1).float()
            #out = self.gomodel((x, one_hot_cat_list[0].float()))
            #x = torch.cat([x, cont_input], dim=1)
            #out = self.gomodel((x, torch.concat(one_hot_cat_list, axis=1).float()))
            out = self.gomodel((x, covariates))
        else:
            out = self.gomodel(x)
        #out = self.gomodel(x)
        if self.last_layer_inject_covariates:
            out = self.fclayer(out, *cat_list, cont_input=cont_input)
        else:
            out = self.act(self.layer_norm(out))
        #torch.cat((out, *one_hot_cat_list), dim=-1)

        return out


class FCLayers(nn.Module):
    """A helper class to build fully-connected layers for a neural network.

    Parameters
    ----------
    n_in
        The dimensionality of the input
    n_out
        The dimensionality of the output
    n_cat_list
        A list containing, for each category of interest,
        the number of categories. Each category will be
        included using a one-hot encoding.
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    use_batch_norm
        Whether to have `BatchNorm` layers or not
    use_layer_norm
        Whether to have `LayerNorm` layers or not
    use_activation
        Whether to have layer activation or not
    bias
        Whether to learn bias in linear layers or not
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    activation_fn
        Which activation function to use
    """

    def __init__(
        self,
        n_in: int,
        n_out: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        use_activation: bool = True,
        bias: bool = True,
        inject_covariates: bool = True,
        activation_fn: nn.Module = nn.ReLU,
        keep_activations: bool = False,
        **kwargs
    ):
        super().__init__()
        self.inject_covariates = inject_covariates
        layers_dim = [n_in] + (n_layers - 1) * [n_hidden] + [n_out]

        if n_cat_list is not None:
            # n_cat = 1 will be ignored
            #self.n_cat_list = [n_cat if n_cat > 1 else 0 for n_cat in n_cat_list]
            self.n_cat_list = [n_cat if n_cat > 1 or (n_cat_index+1 == len(n_cat_list) and len(n_cat_list) > 1) else 0 for n_cat_index, n_cat in enumerate(n_cat_list)]
        else:
            self.n_cat_list = []

        cat_dim = sum(self.n_cat_list)
        self.fc_layers = nn.Sequential(
            collections.OrderedDict(
                [
                    (
                        f"Layer {i}",
                        nn.Sequential(
                            nn.Linear(
                                n_in + cat_dim * self.inject_into_layer(i),
                                n_out,
                                bias=bias,
                            ),
                            # non-default params come from defaults in original Tensorflow
                            # implementation
                            nn.BatchNorm1d(n_out, momentum=0.01, eps=0.001)
                            if use_batch_norm
                            else None,
                            nn.LayerNorm(n_out, elementwise_affine=False)
                            if use_layer_norm
                            else None,
                            activation_fn() if use_activation else None,
                            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else None,
                        ),
                    )
                    for i, (n_in, n_out) in enumerate(zip(layers_dim[:-1], layers_dim[1:]))
                ]
            )
        )

        self.keep_activations = keep_activations
        self.accumulated_activations = defaultdict(list)

    def inject_into_layer(self, layer_num) -> bool:
        """Helper to determine if covariates should be injected."""
        user_cond = layer_num == 0 or (layer_num > 0 and self.inject_covariates)
        return user_cond

    def set_online_update_hooks(self, hook_first_layer=True):
        """Set online update hooks."""
        self.hooks = []

        def _hook_fn_weight(grad):
            categorical_dims = sum(self.n_cat_list)
            new_grad = torch.zeros_like(grad)
            if categorical_dims > 0:
                new_grad[:, -categorical_dims:] = grad[:, -categorical_dims:]
            return new_grad

        def _hook_fn_zero_out(grad):
            return grad * 0

        for i, layers in enumerate(self.fc_layers):
            for layer in layers:
                if i == 0 and not hook_first_layer:
                    continue
                if isinstance(layer, nn.Linear):
                    if self.inject_into_layer(i):
                        w = layer.weight.register_hook(_hook_fn_weight)
                    else:
                        w = layer.weight.register_hook(_hook_fn_zero_out)
                    self.hooks.append(w)
                    b = layer.bias.register_hook(_hook_fn_zero_out)
                    self.hooks.append(b)

    def reset_accumulated_activations(self):
        self.accumulated_activations = defaultdict(list)

    def merge_activations(self):
        return {
            i: np.concatenate(self.accumulated_activations[i], axis=0) #torch.cat
            for i in self.accumulated_activations
        }

    def forward(self, x: torch.Tensor, *cat_list: int, cont_input: torch.Tensor = None):
        """Forward computation on ``x``.

        Parameters
        ----------
        x
            tensor of values with shape ``(n_in,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        :class:`torch.Tensor`
            tensor of shape ``(n_out,)``
        """
        one_hot_cat_list = []  # for generality in this list many indices useless.

        if len(self.n_cat_list) > len(cat_list) + 1 if cont_input is not None else 0:
            raise ValueError("nb. categorical args provided doesn't match init. params.")
        for n_cat, cat in zip(self.n_cat_list, cat_list):
            if n_cat and cat is None:
                raise ValueError("cat not provided while n_cat != 0 in init. params.")
            if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
                if cat.size(1) != n_cat:
                    if cat.squeeze(-1).max() + 1 > n_cat:
                        one_hot_cat = torch.zeros(cat.squeeze(-1).shape[0], n_cat, dtype=torch.int64).to(cat.device)
                    else:
                        one_hot_cat = nn.functional.one_hot(cat.squeeze(-1), n_cat)
                else:
                    one_hot_cat = cat  # cat has already been one_hot encoded
                one_hot_cat_list += [one_hot_cat]
        for i, layers in enumerate(self.fc_layers):
            for i_layer, layer in enumerate(layers):
                if layer is not None:
                    if isinstance(layer, nn.BatchNorm1d):
                        if x.dim() == 3:
                            x = torch.cat([(layer(slice_x)).unsqueeze(0) for slice_x in x], dim=0)
                        else:
                            x = layer(x)
                    else:
                        if isinstance(layer, nn.Linear) and self.inject_into_layer(i):
                            if x.dim() == 3:
                                one_hot_cat_list_layer = [
                                    o.unsqueeze(0).expand((x.size(0), o.size(0), o.size(1)))
                                    for o in one_hot_cat_list
                                ]
                            else:
                                one_hot_cat_list_layer = one_hot_cat_list
                            x = torch.cat((x, *one_hot_cat_list_layer), dim=-1)
                            if cont_input is not None and len(cont_input) != 0:
                                x = torch.cat([x, cont_input], dim=1)
                        x = layer(x)
                        if self.keep_activations and i_layer == len(layers)-1:
                           #self.activations[i] = np.concatenate([self.activations[i], x.cpu().detach().numpy()], axis=0)
                           self.accumulated_activations[i].append(x.cpu().detach().numpy())
        return x

# Encoder
class Encoder(nn.Module):
    """Encode data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of z
    var_eps
        Minimum value for the variance;
        used for numerical stability
    var_activation
        Callable used to ensure positivity of the variance.
        Defaults to :meth:`torch.exp`.
    return_dist
        Return directly the distribution of z instead of its parameters.
    gene_layer_type
        Type of gene layer. One of the following
        * ``'none'`` - No gene layer
        * ``'standard'`` - Standard Gene Layer
        * ``'interaction'`` - Interaction Gene Layer
    gene_layer_interaction_source
        Gene layer interaction source. One of the following
        * ``'pp'`` - Protein-Protein
        * ``'tf'`` - Transcription Factor
        * ``'tad'`` - Topologically Associated Domains
    layers_type
        Type of layer. One of the following
        * ``'linear'`` - Linear Layers
        * ``'go'`` - GO Layers
    **kwargs
        Keyword args for :class:`~networkvi.nn.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        ensembl_ids: np.ndarray | None = None,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        dropout_rate: float = 0.1,
        distribution: str = "normal",
        var_eps: float = 1e-4,
        var_activation: Optional[Callable] = None,
        return_dist: bool = False,
        gene_layer_type: Literal["none", "standard", "interaction"] = "interaction",
        gene_layer_interaction_source: Optional[str] = None,
        layers_type: Literal["linear", "go"] = "go",
        standard_gene_size: int = 4,
        standard_go_size: int = 6,
        obo_file: Optional[str] = None,
        map_ensembl_go: Optional[Union[list, np.ndarray]] = None,
        keep_activations: bool = False,
        **kwargs,
    ):
        super().__init__()

        self.distribution = distribution
        self.var_eps = var_eps

        if gene_layer_type == "standard" and ensembl_ids is not None and len(ensembl_ids) > 1:

            n_input = len(np.unique(ensembl_ids)) * standard_gene_size

            self.genemodel = GeneLayers(
                ensembl_ids=ensembl_ids,
                standard_gene_size=standard_gene_size,
                interaction_file=None,
                input_dropout=dropout_rate,
                n_cat_list=n_cat_list,
                **kwargs,
            )

        elif gene_layer_type == "interaction" and ensembl_ids is not None and len(ensembl_ids) > 1:

            n_input = len(np.unique(ensembl_ids)) * standard_gene_size

            self.genemodel = GeneLayers(
                ensembl_ids=ensembl_ids,
                standard_gene_size=standard_gene_size,
                gene_layer_interaction_source=gene_layer_interaction_source,
                input_dropout=dropout_rate,
                n_cat_list=n_cat_list,
                **kwargs,
            )

        if layers_type == "go" and ensembl_ids is not None and len(ensembl_ids) > 1:

            self.encoder = GOLayers(
                geneobj=self.genemodel.geneobj if hasattr(self, "genemodel") else None,
                genemodel_out_blocks=self.genemodel.genemodel.out_blocks if hasattr(self, "genemodel") else [1] * n_input,
                ensembl_ids=ensembl_ids,
                register_rancon=True,
                remove_rancon=True,
                obo_file=obo_file,
                map_ensembl_go=map_ensembl_go,
                standard_go_size=standard_go_size,
                n_hidden=n_hidden,
                n_layers=n_layers,
                input_dropout=dropout_rate,
                n_cat_list=n_cat_list,
                **kwargs,
            )

        else:
            self.encoder = FCLayers(
                n_in=n_input,
                n_out=n_hidden,
                n_cat_list=n_cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                dropout_rate=dropout_rate,
                keep_activations=keep_activations,
                **kwargs,
            )

        self.mean_encoder = nn.Linear(n_hidden, n_output)
        self.var_encoder = nn.Linear(n_hidden, n_output)
        self.return_dist = return_dist

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity
        self.var_activation = torch.exp if var_activation is None else var_activation

    def forward(self, x: torch.Tensor, *cat_list: int, cont_input: torch.Tensor = None):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. multivariate normal
            \\( \\sim Ne(q_m, \\mathbf{I}q_v) \\)

        Parameters
        ----------
        x
            tensor with shape (n_input,)
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        if hasattr(self, "genemodel"):
            x = self.genemodel(x) #, *cat_list, cont_input

        # Parameters for latent distribution
        q = self.encoder(x, *cat_list, cont_input=cont_input)
        q_m = self.mean_encoder(q)
        q_v = self.var_activation(self.var_encoder(q)) + self.var_eps
        dist = Normal(q_m, q_v.sqrt())
        latent = self.z_transformation(dist.rsample())
        if self.return_dist:
            return dist, latent
        return q_m, q_v, latent


# Decoder
class DecoderSCVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions into ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    inject_covariates
        Whether to inject covariates in each layer, or just the first (default).
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    layers_type
        Type of layer. One of the following
        * ``'linear'`` - Linear Layers
    **kwargs
        Keyword args for :class:`~networkvi.nn.FCLayers`.
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        inject_covariates: bool = True,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
        layers_type: Literal["linear"] = "linear",
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
            inject_covariates=inject_covariates,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            **kwargs,
        )

        # mean gamma
        if scale_activation == "softmax":
            px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            px_scale_activation = nn.Softplus()
        self.px_scale_decoder = nn.Sequential(
            nn.Linear(n_hidden, n_output),
            px_scale_activation,
        )

        # dispersion: here we only deal with gene-cell dispersion case
        self.px_r_decoder = nn.Linear(n_hidden, n_output)

        # dropout
        self.px_dropout_decoder = nn.Linear(n_hidden, n_output)

    def forward(
        self,
        dispersion: str,
        z: torch.Tensor,
        library: torch.Tensor,
        *cat_list: int,
        cont_input: torch.Tensor = None,
    ):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns parameters for the ZINB distribution of expression
         #. If ``dispersion != 'gene-cell'`` then value for that param will be ``None``

        Parameters
        ----------
        dispersion
            One of the following

            * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
            * ``'gene-batch'`` - dispersion can differ between different batches
            * ``'gene-label'`` - dispersion can differ between different labels
            * ``'gene-cell'`` - dispersion can differ for every gene in every cell
        z :
            tensor with shape ``(n_input,)``
        library_size
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        4-tuple of :py:class:`torch.Tensor`
            parameters for the ZINB distribution of expression

        """
        # The decoder returns values for the parameters of the ZINB distribution
        px = self.px_decoder(z, *cat_list, cont_input=cont_input)
        px_scale = self.px_scale_decoder(px)
        px_dropout = self.px_dropout_decoder(px)
        # Clamp to high value: exp(12) ~ 160000 to avoid nans (computational stability)
        px_rate = torch.exp(library) * px_scale  # torch.clamp( , max=12)
        px_r = self.px_r_decoder(px) if dispersion == "gene-cell" else None
        return px_scale, px_r, px_rate, px_dropout


class LinearDecoderSCVI(nn.Module):
    """Linear decoder for networkvi."""

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        use_batch_norm: bool = False,
        use_layer_norm: bool = False,
        bias: bool = False,
        **kwargs,
    ):
        super().__init__()

        # mean gamma
        self.factor_regressor = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            **kwargs,
        )

        # dropout
        self.px_dropout_decoder = FCLayers(
            n_in=n_input,
            n_out=n_output,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=False,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
            bias=bias,
            dropout_rate=0,
            **kwargs,
        )

    def forward(self, dispersion: str, z: torch.Tensor, library: torch.Tensor, *cat_list: int):
        """Forward pass."""
        # The decoder returns values for the parameters of the ZINB distribution
        raw_px_scale = self.factor_regressor(z, *cat_list)
        px_scale = torch.softmax(raw_px_scale, dim=-1)
        px_dropout = self.px_dropout_decoder(z, *cat_list)
        px_rate = torch.exp(library) * px_scale
        px_r = None

        return px_scale, px_r, px_rate, px_dropout


# Decoder
class Decoder(nn.Module):
    """Decodes data from latent space to data space.

    ``n_input`` dimensions to ``n_output``
    dimensions using a fully-connected neural network of ``n_hidden`` layers.
    Output is the mean and variance of a multivariate Gaussian

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output
        The dimensionality of the output (data space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    kwargs
        Keyword args for :class:`~networkvi.module._base.FCLayers`
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=0,
            **kwargs,
        )

        self.mean_decoder = nn.Linear(n_hidden, n_output)
        self.var_decoder = nn.Linear(n_hidden, n_output)

    def forward(self, x: torch.Tensor, *cat_list: int):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns tensors for the mean and variance of a multivariate distribution

        Parameters
        ----------
        x
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        2-tuple of :py:class:`torch.Tensor`
            Mean and variance tensors of shape ``(n_output,)``

        """
        # Parameters for latent distribution
        p = self.decoder(x, *cat_list)
        p_m = self.mean_decoder(p)
        p_v = torch.exp(self.var_decoder(p))
        return p_m, p_v

class DecoderTOTALVI(nn.Module):
    """Decodes data from latent space of ``n_input`` dimensions ``n_output`` dimensions.

    Uses a linear decoder.

    Parameters
    ----------
    n_input
        The dimensionality of the input (latent space)
    n_output_genes
        The dimensionality of the output (gene space)
    n_output_proteins
        The dimensionality of the output (protein space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm in layers
    scale_activation
        Activation layer to use for px_scale_decoder
    """

    def __init__(
        self,
        n_input: int,
        n_output_genes: int,
        n_output_proteins: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 1,
        n_hidden: int = 256,
        dropout_rate: float = 0,
        use_batch_norm: float = True,
        use_layer_norm: float = False,
        scale_activation: Literal["softmax", "softplus"] = "softmax",
    ):
        super().__init__()
        self.n_output_genes = n_output_genes
        self.n_output_proteins = n_output_proteins

        linear_args = {
            "n_layers": 1,
            "use_activation": False,
            "use_batch_norm": False,
            "use_layer_norm": False,
            "dropout_rate": 0,
        }

        self.px_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        # mean gamma
        self.px_scale_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_genes,
            n_cat_list=n_cat_list,
            **linear_args,
        )
        if scale_activation == "softmax":
            self.px_scale_activation = nn.Softmax(dim=-1)
        elif scale_activation == "softplus":
            self.px_scale_activation = nn.Softplus()

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
        )
        # background mean parameters second decoder
        self.py_back_mean_log_alpha = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )
        self.py_back_mean_log_beta = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        # foreground increment decoder step 1
        self.py_fore_decoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        # foreground increment decoder step 2
        self.py_fore_scale_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            n_layers=1,
            use_activation=True,
            use_batch_norm=False,
            use_layer_norm=False,
            dropout_rate=0,
            activation_fn=nn.ReLU,
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
        )
        self.px_dropout_decoder_gene = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_genes,
            n_cat_list=n_cat_list,
            **linear_args,
        )

        self.py_background_decoder = FCLayers(
            n_in=n_hidden + n_input,
            n_out=n_output_proteins,
            n_cat_list=n_cat_list,
            **linear_args,
        )

    def forward(self, z: torch.Tensor, library_gene: torch.Tensor, *cat_list: int):
        """The forward computation for a single sample.

         #. Decodes the data from the latent space using the decoder network
         #. Returns local parameters for the ZINB distribution for genes
         #. Returns local parameters for the Mixture NB distribution for proteins

         We use the dictionary `px_` to contain the parameters of the ZINB/NB for genes.
         The rate refers to the mean of the NB, dropout refers to Bernoulli mixing parameters.
         `scale` refers to the quanity upon which differential expression is performed. For genes,
         this can be viewed as the mean of the underlying gamma distribution.

         We use the dictionary `py_` to contain the parameters of the Mixture NB distribution for
         proteins. `rate_fore` refers to foreground mean, while `rate_back` refers to background
         mean. `scale` refers to foreground mean adjusted for background probability and scaled to
         reside in simplex. `back_alpha` and `back_beta` are the posterior parameters for
         `rate_back`. `fore_scale` is the scaling factor that enforces `rate_fore` > `rate_back`.

        Parameters
        ----------
        z
            tensor with shape ``(n_input,)``
        library_gene
            library size
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        3-tuple (first 2-tuple :py:class:`dict`, last :py:class:`torch.Tensor`)
            parameters for the ZINB distribution of expression

        """
        px_ = {}
        py_ = {}

        px = self.px_decoder(z, *cat_list)
        px_cat_z = torch.cat([px, z], dim=-1)
        unnorm_px_scale = self.px_scale_decoder(px_cat_z, *cat_list)
        px_["scale"] = self.px_scale_activation(unnorm_px_scale)
        px_["rate"] = library_gene * px_["scale"]

        py_back = self.py_back_decoder(z, *cat_list)
        py_back_cat_z = torch.cat([py_back, z], dim=-1)

        py_["back_alpha"] = self.py_back_mean_log_alpha(py_back_cat_z, *cat_list)
        py_["back_beta"] = torch.exp(self.py_back_mean_log_beta(py_back_cat_z, *cat_list))
        log_pro_back_mean = Normal(py_["back_alpha"], py_["back_beta"]).rsample()
        py_["rate_back"] = torch.exp(log_pro_back_mean)

        py_fore = self.py_fore_decoder(z, *cat_list)
        py_fore_cat_z = torch.cat([py_fore, z], dim=-1)
        py_["fore_scale"] = self.py_fore_scale_decoder(py_fore_cat_z, *cat_list) + 1 + 1e-8
        py_["rate_fore"] = py_["rate_back"] * py_["fore_scale"]

        p_mixing = self.sigmoid_decoder(z, *cat_list)
        p_mixing_cat_z = torch.cat([p_mixing, z], dim=-1)
        px_["dropout"] = self.px_dropout_decoder_gene(p_mixing_cat_z, *cat_list)
        py_["mixing"] = self.py_background_decoder(p_mixing_cat_z, *cat_list)

        protein_mixing = 1 / (1 + torch.exp(-py_["mixing"]))
        py_["scale"] = torch.nn.functional.normalize(
            (1 - protein_mixing) * py_["rate_fore"], p=1, dim=-1
        )

        return (px_, py_, log_pro_back_mean)


# Encoder
class SparseEncoderTOTALVI(nn.Module):
    """Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        ensembl_ids: np.ndarray | None = None,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
        gene_layer_type: Literal["none", "standard", "interaction"] = "interaction",
        gene_layer_interaction_source: Optional[str] = None,
        layers_type: Literal["linear", "go"] = "go",
        standard_gene_size: int = 4,
        standard_go_size: int = 6,
        obo_file: Optional[str] = None,
        map_ensembl_go: Optional[Union[list, np.ndarray]] = None,
        activation_fn: nn.Module = nn.ReLU,
        **kwargs
    ):
        super().__init__()


        if gene_layer_type == "standard" and ensembl_ids is not None:
            n_input = len(np.unique(ensembl_ids)) * standard_gene_size
            self.genemodel = GeneLayers(
                ensembl_ids=ensembl_ids,
                standard_gene_size=standard_gene_size,
                interaction_file=None,
                input_dropout=dropout_rate,
                activation_fn=activation_fn,
                **kwargs,
            )

        elif gene_layer_type == "interaction" and ensembl_ids is not None:
            n_input = len(np.unique(ensembl_ids)) * standard_gene_size
            self.genemodel = GeneLayers(
                ensembl_ids=ensembl_ids,
                standard_gene_size=standard_gene_size,
                gene_layer_interaction_source=gene_layer_interaction_source,
                input_dropout=dropout_rate,
                activation_fn=activation_fn,
                **kwargs,
            )

        if layers_type == "go" and ensembl_ids is not None:
            self.encoder = GOLayers(
                geneobj=self.genemodel.geneobj if hasattr(self, "genemodel") else None,
                genemodel_out_blocks=self.genemodel.genemodel.out_blocks if hasattr(self, "genemodel") else [1] * n_input,
                ensembl_ids=ensembl_ids,
                register_rancon=True,
                remove_rancon=True,
                obo_file=obo_file,
                map_ensembl_go=map_ensembl_go,
                standard_go_size=standard_go_size,
                n_hidden=n_hidden,
                n_layers=n_layers,
                input_dropout=dropout_rate,
                activation_fn=activation_fn,
                **kwargs,
            )

        else:
            self.encoder = FCLayers(
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

        self.z_mean_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_encoder = nn.Linear(n_hidden, n_output)

        self.l_gene_encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity

        self.l_transformation = torch.exp

    def reparameterize_transformation(self, mu, var):
        """Reparameterization trick to sample from a normal distribution."""
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary ``latent`` contains the samples of the latent variables, while
        ``untran_latent`` contains the untransformed versions of these latent variables. For
        example, the library size is log normally distributed, so ``untran_latent["l"]`` gives the
        normal sample that was later exponentiated to become ``latent["l"]``. The logistic normal
        distribution is equivalent to applying softmax to a normal sample.

        Parameters
        ----------
        data
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        6-tuple. First 4 of :py:class:`torch.Tensor`, next 2 are `dict` of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        if hasattr(self, "genemodel"):
            data = self.genemodel(data)

        # Parameters for latent distribution
        q = self.encoder(data, *cat_list)
        qz_m = self.z_mean_encoder(q)
        qz_v = torch.exp(self.z_var_encoder(q)) + 1e-4
        q_z = Normal(qz_m, qz_v.sqrt())
        untran_z = q_z.rsample()
        z = self.z_transformation(untran_z)

        ql_gene = self.l_gene_encoder(data, *cat_list)
        ql_m = self.l_gene_mean_encoder(ql_gene)
        ql_v = torch.exp(self.l_gene_var_encoder(ql_gene)) + 1e-4
        q_l = Normal(ql_m, ql_v.sqrt())
        log_library_gene = q_l.rsample()
        log_library_gene = torch.clamp(log_library_gene, max=15)
        library_gene = self.l_transformation(log_library_gene)

        latent = {}
        untran_latent = {}
        latent["z"] = z
        latent["l"] = library_gene
        untran_latent["z"] = untran_z
        untran_latent["l"] = log_library_gene

        return q_z, q_l, latent, untran_latent

# Encoder
class EncoderTOTALVI(nn.Module):
    """Encodes data of ``n_input`` dimensions into a latent space of ``n_output`` dimensions.

    Uses a fully-connected neural network of ``n_hidden`` layers.

    Parameters
    ----------
    n_input
        The dimensionality of the input (data space)
    n_output
        The dimensionality of the output (latent space)
    n_cat_list
        A list containing the number of categories
        for each category of interest. Each category will be
        included using a one-hot encoding
    n_layers
        The number of fully-connected hidden layers
    n_hidden
        The number of nodes per hidden layer
    dropout_rate
        Dropout rate to apply to each of the hidden layers
    distribution
        Distribution of the latent space, one of

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal
    use_batch_norm
        Whether to use batch norm in layers
    use_layer_norm
        Whether to use layer norm
    """

    def __init__(
        self,
        n_input: int,
        n_output: int,
        n_cat_list: Iterable[int] = None,
        n_layers: int = 2,
        n_hidden: int = 256,
        dropout_rate: float = 0.1,
        distribution: str = "ln",
        use_batch_norm: bool = True,
        use_layer_norm: bool = False,
    ):
        super().__init__()

        self.encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.z_mean_encoder = nn.Linear(n_hidden, n_output)
        self.z_var_encoder = nn.Linear(n_hidden, n_output)

        self.l_gene_encoder = FCLayers(
            n_in=n_input,
            n_out=n_hidden,
            n_cat_list=n_cat_list,
            n_layers=1,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )
        self.l_gene_mean_encoder = nn.Linear(n_hidden, 1)
        self.l_gene_var_encoder = nn.Linear(n_hidden, 1)

        self.distribution = distribution

        if distribution == "ln":
            self.z_transformation = nn.Softmax(dim=-1)
        else:
            self.z_transformation = _identity

        self.l_transformation = torch.exp

    def reparameterize_transformation(self, mu, var):
        """Reparameterization trick to sample from a normal distribution."""
        untran_z = Normal(mu, var.sqrt()).rsample()
        z = self.z_transformation(untran_z)
        return z, untran_z

    def forward(self, data: torch.Tensor, *cat_list: int):
        r"""The forward computation for a single sample.

         #. Encodes the data into latent space using the encoder network
         #. Generates a mean \\( q_m \\) and variance \\( q_v \\)
         #. Samples a new value from an i.i.d. latent distribution

        The dictionary ``latent`` contains the samples of the latent variables, while
        ``untran_latent`` contains the untransformed versions of these latent variables. For
        example, the library size is log normally distributed, so ``untran_latent["l"]`` gives the
        normal sample that was later exponentiated to become ``latent["l"]``. The logistic normal
        distribution is equivalent to applying softmax to a normal sample.

        Parameters
        ----------
        data
            tensor with shape ``(n_input,)``
        cat_list
            list of category membership(s) for this sample

        Returns
        -------
        6-tuple. First 4 of :py:class:`torch.Tensor`, next 2 are `dict` of :py:class:`torch.Tensor`
            tensors of shape ``(n_latent,)`` for mean and var, and sample

        """
        # Parameters for latent distribution
        q = self.encoder(data, *cat_list)
        qz_m = self.z_mean_encoder(q)
        qz_v = torch.exp(self.z_var_encoder(q)) + 1e-4
        q_z = Normal(qz_m, qz_v.sqrt())
        untran_z = q_z.rsample()
        z = self.z_transformation(untran_z)

        ql_gene = self.l_gene_encoder(data, *cat_list)
        ql_m = self.l_gene_mean_encoder(ql_gene)
        ql_v = torch.exp(self.l_gene_var_encoder(ql_gene)) + 1e-4
        q_l = Normal(ql_m, ql_v.sqrt())
        log_library_gene = q_l.rsample()
        log_library_gene = torch.clamp(log_library_gene, max=15)
        library_gene = self.l_transformation(log_library_gene)

        latent = {}
        untran_latent = {}
        latent["z"] = z
        latent["l"] = library_gene
        untran_latent["z"] = untran_z
        untran_latent["l"] = log_library_gene

        return q_z, q_l, latent, untran_latent
