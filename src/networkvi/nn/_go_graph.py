import logging

logger = logging.getLogger(__name__)

from collections import OrderedDict, defaultdict
from typing import Tuple, Union
from goatools.obo_parser import GODag
import torch
from torch.nn import Sequential
import numpy as np
from collections import Counter
import logging

import networkvi.nn._go_utils as utils
from networkvi.nn._torch_utils import get_activation
from networkvi.nn._go_embedding_layer import GOEmbeddingLayer
from networkvi.nn._block_sparse_linear import BlockSparseLinear
from networkvi.nn._go_cov_attention import GOCovAttention

class GoGraph(torch.nn.Module):
    """
    GoGraph modelling the Gene Ontology.
    """

    def __init__(
        self,
        goobj: GODag,
        geneobj,
        genemodel_out_blocks,
        attention,
        hierarchy,
        layer,
        standard_go_block_depth,
        go_intermediate_scaling,
        go_intermediate_mode,
        n_covariates: int = 0,
        register_rancon: bool = False,
        remove_rancon: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes the GoGraph model.

        Args:
            goobj (GODag): go object.
            geneobj: gene obj
            genemodel_out_blocks: Defines the structure of the gene model output blocks.
            attention: Attention mechanism configuration.
            hierarchy: Hierarchical structure defining the GO relationships.
            layer: Layer configuration parameters.
            standard_go_block_depth: Standard depth of GO blocks.
            go_intermediate_scaling: Scaling factor for intermediate layers.
            go_intermediate_mode: Defines the mode of intermediate layer operation.
            n_covariates (int, optional): Number of covariates. Defaults to 0.
            register_rancon (bool, optional): Whether to register rancon. Defaults to False.
            remove_rancon (bool, optional): Whether to remove rancon. Defaults to False.
        """
        super().__init__()
        self.goobj = goobj
        # self.goobj_unfiltered = goobj_unfiltered
        self.geneobj = geneobj
        self.genemodel_out_blocks = genemodel_out_blocks
        self.attention = attention
        self.layer = layer

        # has to be set, can be changed with set_keep_activations('accumulate') method
        self.keep_activations = "False"

        self.n_covariates = n_covariates if attention is not None else 0
        self.use_series_attention = (
            attention is not None
            and attention["enabled"]
            and attention["mode"] == "series"
        )
        if self.use_series_attention and attention["at_levels"] == "all":
            attention["at_levels"] = [-1, *range(hierarchy["max_level"]["depth"])]
        if self.use_series_attention and attention["at_levels"] == "all_but_genes":
            attention["at_levels"] = [*range(hierarchy["max_level"]["depth"])]
        self.order = None
        self.block_structure = None
        self.block_structure_pos = None
        self.block_structure_neg = None
        self.block_sizes_per_layer = None
        self.block_depth_per_layer = None

        self.register_rancon = register_rancon
        self.remove_rancon = remove_rancon

        self.block_layer = None
        self.attention_layer = None

        self.block_layer_activations = {}  # Used to store outputs of each block layer
        self.attention_layer_activations = (
            {}
        )  # Used to store outputs of each attention layer

        # if hierarchy.regulates_init and not (ogmdef.model.use_regulates_pos or ogmdef.model.use_regulates_neg):
        #     logger.warn(f'Will ignore regulates_init, since use_regulates_pos and use_regulates_neg are both set to False', UserWarning)

        self.construct_graph(
            hierarchy,
            layer,
            attention,
            standard_go_block_depth,
            go_intermediate_scaling,
            go_intermediate_mode,
        )

    def construct_graph(
        self,
        hierarchy,
        layer,
        attention,
        standard_go_block_depth,
        go_intermediate_scaling,
        go_intermediate_mode,
    ):
        """
        Constructs the GO graph structure and initializes layers.
        """
        self.block_structure = (
            OrderedDict()
        )  # This will contain the block structure that connects the layers embeddings
        self.block_structure_pos = OrderedDict()
        self.block_structure_neg = OrderedDict()
        if self.use_series_attention:
            self.attention_layer = OrderedDict()
        self.block_layer = (
            OrderedDict()
        )  # This will contain the BlockSparseLinear layers
        # each block in a layer can be seen as an edge in the go graph
        self.block_sizes_per_layer = {}  # This will contain embedding sizes per layer
        # each embedding is one go term and can be seen as a node
        self.block_depth_per_layer = {}

        self.generate_block_structure(hierarchy, standard_go_block_depth)
        self.generate_layers(
            hierarchy, layer, attention, go_intermediate_scaling, go_intermediate_mode
        )

        self.reset_accumulated_activations()
        if len(np.unique([term.split(":")[0] for term in self.order])) > 1:
            logger.debug(
                f"Prepared total {len(self.order)} GO-like nodes ({Counter([term.split(':')[0] for term in self.order])}) with {self.get_n_parameters()} parameters.\n"
            )
        else:
            logger.debug(
                f"Prepared {len(self.order)} go nodes with {self.get_n_parameters()} parameters.\n"
            )


    def reset_accumulated_activations(self):
        self.accumulated_activations = {k: [] for k in self.block_layer}

    def merge_activations(self):
        return {
            k: torch.cat(self.accumulated_activations[k], axis=0)
            for k in self.accumulated_activations
        }

    def generate_block_structure(self, hierarchy, standard_go_block_depth):
        """
        Generates the block structure for the GO graph.
        """
        connection_cnt = defaultdict(int)

        if hierarchy["mode"] in ["depth", "ogm_depth"]:
            self.order = sorted(
                list(self.goobj.keys()), key=lambda x: -self.goobj[x].ogm_depth
            )  # Topological ordering
            rev = -1
            mode = "ogm_depth"
            smallest_idx = self.goobj[self.order[0]].ogm_depth

        elif hierarchy["mode"] == "height":
            self.order = sorted(
                list(self.goobj.keys()), key=lambda x: self.goobj[x].height
            )
            rev = 1
            mode = "height"
            smallest_idx = 0
            # greatest_height = self.goobj[self.order[-1]].height

        layer_idx = -1  # Initialize to minimal value
        current_block_index = 0  # Index of block in current layer
        current_out_slice_pos = 0

        for goid in self.order:
            # figure everything out for the layer
            go = self.goobj[goid]

            if (
                not getattr(go, mode) == layer_idx
            ):  # New depth encountered, needs to initialize variable for that new layer
                layer_idx = getattr(go, mode)
                current_block_index = 0
                current_out_slice_pos = 0
                # Initialize an empty block structure array for each possible incoming layer (i.e. all layers before current)
                #   -1 represents the genelayer
                self.block_structure[layer_idx] = {
                    **{l: [] for l in range(smallest_idx, getattr(go, mode), rev)},
                    **{-1: []},
                }
                self.block_structure_pos[layer_idx] = {
                    **{l: [] for l in range(smallest_idx, getattr(go, mode), rev)},
                    **{-1: []},
                }
                self.block_structure_neg[layer_idx] = {
                    **{l: [] for l in range(smallest_idx, getattr(go, mode), rev)},
                    **{-1: []},
                }

                self.block_sizes_per_layer[layer_idx] = []
                self.block_depth_per_layer[layer_idx] = standard_go_block_depth

            go.block_index = current_block_index
            go.out_slice = slice(
                current_out_slice_pos, current_out_slice_pos + go.layersize
            )
            self.block_sizes_per_layer[layer_idx].append(go.layersize)

            # Add connections from genes to layer embeddings (each embedding is one go term)
            in_genes = [self.geneobj[ensembl] for ensembl in go.ensemblids]
            for in_gene in in_genes:
                for in_gene_block_index in in_gene["block_index"]:
                    self.block_structure[layer_idx][-1].append(
                        (in_gene_block_index, current_block_index)
                    )

            all_children = utils.get_all_children(
                go, **hierarchy["relations"], as_dict=True, ret_empty=True
            )
            for rel in all_children:
                rel_children = all_children[rel]
                connection_cnt[rel] += len(rel_children)

                if hierarchy["regulates_init"] and hierarchy["regulates_masking"]:
                    raise ValueError(
                        f"regulates_init and regulates_masking must not both be True!"
                    )
                should_mask_or_init = (
                    hierarchy["regulates_init"] or hierarchy["regulates_masking"]
                )

                if rel == "positively_regulates" and should_mask_or_init:
                    for in_go in rel_children:
                        self.block_structure_pos[layer_idx][
                            getattr(in_go, mode)
                        ].append((in_go.block_index, current_block_index))
                elif rel == "negatively_regulates" and should_mask_or_init:
                    for in_go in rel_children:
                        self.block_structure_neg[layer_idx][
                            getattr(in_go, mode)
                        ].append((in_go.block_index, current_block_index))
                else:
                    for in_go in rel_children:
                        self.block_structure[layer_idx][getattr(in_go, mode)].append(
                            (in_go.block_index, current_block_index)
                        )

            current_block_index = current_block_index + 1
            current_out_slice_pos = current_out_slice_pos + go.layersize
        for rel, cnt in connection_cnt.items():
            logger.info(f"Used {cnt} connections of the {rel} relationship.")

    def generate_layers(
        self, hierarchy, layer, attention, go_intermediate_scaling, go_intermediate_mode
    ):
        """
        Creates the GO model layers.
        """
        # Create the actual BlockSparseLinear Layers according to above specification
        self.block_layer = OrderedDict()
        self.rancon_layer = []

        if self.use_series_attention and -1 in attention["at_levels"]:
            print(f"Using attention in GO model at level -1.")
            out_blocks = self.genemodel_out_blocks
            gene_dim = len(out_blocks) if attention["reduce"] else sum(out_blocks)
            cov_dim = self.n_covariates
            go_cov_attention = GOCovAttention(
                gene_dim,
                cov_dim,
                inter_dim=attention["inter_dim"],
                values_from=attention["values_from"],
                dropout=attention["dropout"],
                aggregate=attention["aggregate"],
                residual=attention["residual"],
                layer_norm=attention["layer_norm"],
            )
            if attention["reduce"]:
                layers = OrderedDict()
                diagonal_block_structure = [(i, i) for i in range(0, len(out_blocks))]
                reduced_blocks = [1 for block in out_blocks]
                layers["scale_down"] = BlockSparseLinear(
                    out_blocks, reduced_blocks, diagonal_block_structure
                )
                layers["go_cov_attention"] = go_cov_attention
                layers["scale_up"] = BlockSparseLinear(
                    reduced_blocks, out_blocks, diagonal_block_structure
                )
                self.attention_layer[-1] = Sequential(*layers)
            else:
                self.attention_layer[-1] = go_cov_attention

            self.add_module("AttentionLayer -1", self.attention_layer[-1])

        for o_layer in self.block_structure:
            self.block_layer[o_layer] = {}
            out_blocks = self.block_sizes_per_layer[o_layer]

            if self.use_series_attention and o_layer in attention["at_levels"]:
                print(f"Using attention in GO model at level {o_layer}.")
                go_dim = len(out_blocks) if attention["reduce"] else sum(out_blocks)
                cov_dim = self.n_covariates
                go_cov_attention = GOCovAttention(
                    go_dim,
                    cov_dim,
                    inter_dim=attention["inter_dim"],
                    values_from=attention["values_from"],
                    dropout=attention["dropout"],
                    aggregate=attention["aggregate"],
                    residual=attention["residual"],
                    layer_norm=attention["layer_norm"],
                )
                if attention["reduce"]:
                    layers = OrderedDict()
                    diagonal_block_structure = [
                        (i, i) for i in range(0, len(out_blocks))
                    ]
                    reduced_blocks = [1 for block in out_blocks]
                    layers["scale_down"] = BlockSparseLinear(
                        out_blocks, reduced_blocks, diagonal_block_structure
                    )
                    layers["go_cov_attention"] = go_cov_attention
                    layers["scale_up"] = BlockSparseLinear(
                        reduced_blocks, out_blocks, diagonal_block_structure
                    )
                    self.attention_layer[o_layer] = Sequential(*layers)
                else:
                    self.attention_layer[o_layer] = go_cov_attention

                self.add_module(
                    f"AttentionLayer {o_layer}", self.attention_layer[o_layer]
                )

            for i_layer in self.block_structure[o_layer]:

                if len(self.block_structure[o_layer][i_layer]) == 0:
                    self.block_layer[o_layer][i_layer] = None
                    continue

                if i_layer == -1:
                    in_blocks = self.genemodel_out_blocks
                else:
                    in_blocks = self.block_sizes_per_layer[i_layer]

                self.block_layer[o_layer][i_layer] = GOEmbeddingLayer(
                    self.block_depth_per_layer[o_layer],
                    in_blocks,
                    out_blocks,
                    self.block_structure[o_layer][i_layer],
                    block_structure_pos=self.block_structure_pos[o_layer][i_layer],
                    block_structure_neg=self.block_structure_neg[o_layer][i_layer],
                    intermediate_scaling=go_intermediate_scaling,
                    intermediate_mode=go_intermediate_mode,
                    batch_norm=layer["batch_norm"],
                    dropout=layer["dropout"],
                    track_running_stats=layer["track_running_stats"],
                    activation_function=get_activation(layer["activation"]),
                    momentum=layer["bn_momentum"],
                    residual=layer["residual"],
                    pos_neg_init=hierarchy["regulates_init"],
                    masking=hierarchy["regulates_masking"],
                )
                layer_name = "BlockLayer " + str(o_layer) + " " + str(i_layer)
                self.add_module(layer_name, self.block_layer[o_layer][i_layer])
                if self.register_rancon:
                    self.rancon_layer.append(self.block_layer[o_layer][i_layer])

    def get_n_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def forward(
        self, inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]], *args
    ):
        """
        Forward pass through the GO model.
        """
        if self.use_series_attention:
            gene_inputs, cov_inputs = inputs
            if -1 in self.attention["at_levels"]:
                gene_inputs = self.attention_layer[-1](inputs) #(gene_inputs)
        else:
            gene_inputs = inputs

        for o_layer in self.block_layer:
            t_block_layer_activations = []
            for i_layer in self.block_layer[o_layer]:
                if self.block_layer[o_layer][i_layer] is None:
                    continue
                if i_layer == -1:
                    t_block_layer_activations.append(
                        self.block_layer[o_layer][i_layer](gene_inputs)
                    )
                else:
                    t_block_layer_activations.append(
                        self.block_layer[o_layer][i_layer](
                            self.block_layer_activations[i_layer]
                        )
                    )

            outputs = torch.sum(torch.stack(t_block_layer_activations, dim=-1), dim=-1)
            if self.use_series_attention and o_layer in self.attention_layer:
                go_cov_inputs = (outputs, cov_inputs)
                outputs = self.attention_layer[o_layer](go_cov_inputs)

            self.block_layer_activations[o_layer] = outputs

        last_layer = list(self.block_layer.keys())[-1]
        final = self.block_layer_activations[last_layer]
        if self.keep_activations == "keep":  # What about a callback?
            pass  # or put these to cpu, and detach them?
        elif self.keep_activations == "accumulate":
            for level in self.block_layer_activations:
                self.accumulated_activations[level].append(
                    self.block_layer_activations[level].detach().cpu()
                )
                #self.accumulated_activations[level].append(
                #    self.block_layer_activations[level].detach().cpu()
                #)
        elif self.keep_activations == "accumulate_balanced":
            for level in self.block_layer_activations:
                self.accumulated_activations[level].append(
                    self.block_layer_activations[level][self.batch_keep_idx, :]
                    .detach()
                    .cpu()
                )
        # elif self.ogmdef.eval.keep_activations == 'low_mem_accumulate':
        #     for level in self.block_layer_activations:
        #         if isinstance(self.accumulated_activations[level], list):
        #             self.accumulated_activations[level] = self.block_layer_activations[level].detach().cpu()
        #         else:
        #             # This seems to be really heavy on the cpu, write to file instead?
        #             self.accumulated_activations[level] = torch.cat(
        #                 [self.accumulated_activations[level],
        #                  self.block_layer_activations[level].detach().cpu()], axis=0)
        else:
            self.reset_accumulated_activations()
        # else:
        #   del self.block_layer_activations   # just testing, if this is the memory problem (if so, detach these to be able to work with them for activation monitoring etc.)

        return final
