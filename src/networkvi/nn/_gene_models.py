import logging

logger = logging.getLogger(__name__)
import torch
import torch.nn as nn
from networkvi.nn._block_sparse_linear import BlockSparseLinear
from torch.nn.parameter import Parameter
import pandas as pd
from addict import Dict as AttributeDict

from networkvi.nn._torch_utils import get_activation
from networkvi.nn._gene_gene_interactions import LinearPPI

class GeneModelClassic(torch.nn.Module):
    """
    Layer integrates features mapped on genes and enables gene-gene interaction modeling.
    """

    def __init__(
        self,
        geneobj,
        gene_block_depth,
        gene_intermediate_mode,
        gene_scaling_layer,
        gene_intermediate_scaling,
        variant_embedding,
        layer,
        ppi=None,
        *args,
        **kwargs,
    ):
        """
        Initializes the GeneModelClassic.

        Args:
            geneobj (dict): gene object
            gene_block_depth (int): The number of blocks (layers) in the gene model.
            gene_intermediate_mode (str): Mode for intermediate gene layer transformations. Options: "in_in", "out_out", "in_out".
            gene_scaling_layer (bool): Whether to apply a scaling layer to the gene blocks.
            gene_intermediate_scaling (float or list): The scaling factor for intermediate layers or a list of scalings for each layer.
            variant_embedding (bool): Whether to apply embedding to variants.
            layer (dict): Dictionary of additional layer configurations (e.g., dropout rate, activation function).
            ppi (dict, optional): Configuration for gene-gene interactions (PPI). Default is None.
            *args: Additional positional arguments for flexibility.
            **kwargs: Additional keyword arguments for flexibility.
        """

        super().__init__()

        self.geneobj = geneobj
        self.variant_embedding = variant_embedding
        self.gene_scaling_layer = gene_scaling_layer

        # has to be set, can be changed with set_keep_activations('accumulate') method
        self.keep_activations = "False"
        self.construct_graph(
            gene_block_depth=gene_block_depth,
            gene_intermediate_mode=gene_intermediate_mode,
            gene_intermediate_scaling=gene_intermediate_scaling,
            gene_scaling_layer=gene_scaling_layer,
            ppi=ppi,
            variant_embedding=variant_embedding,
            **layer,
        )

    def construct_graph(
        self,
        activation=None,
        batch_norm=None,
        bn_momentum=None,
        dropout=0.0,
        gene_block_depth=None,
        gene_intermediate_mode=None,
        gene_intermediate_scaling=None,
        gene_scaling_layer=None,
        input_dropout=0.0,
        ppi=None,
        residual=None,
        track_running_stats=None,
        variant_embedding=None,
        *args,
        **kwargs,
    ):
        """
        Constructs the network graph by setting up layers and block structures.

        Args:
            activation (str, optional): The activation function for layers. Default is None.
            batch_norm (bool, optional): Whether to apply batch normalization. Default is None.
            dropout (float, optional): The dropout rate to apply between layers. Default is 0.0.
            gene_block_depth (int): Number of layers in the gene model.
            gene_intermediate_mode (str): Mode for intermediate layer transformations (options: "in_in", "out_out", "in_out").
            gene_intermediate_scaling (float or list): Scaling factor for intermediate layers or list for each layer.
            gene_scaling_layer (bool): Whether to apply scaling layers to the blocks.
            input_dropout (float): Dropout rate applied to the input layer. Default is 0.0.
            ppi (dict, optional): Configuration for gene-gene interactions. Default is None.
            residual (bool, optional): Whether to apply residual connections between layers. Default is None.
            track_running_stats (bool, optional): Whether to track running stats for batch normalization. Default is None.
            variant_embedding (bool): Whether to apply variant embedding. Default is None.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.
        """

        self.in_blocks = []
        self.out_blocks = []
        self.block_structure = []

        self.in_blocks_one_hot = []
        self.out_blocks_one_hot = []
        self.block_structure_one_hot = []

        n_snps = 0

        for i, ensembl in enumerate(self.geneobj):
            gene = self.geneobj[ensembl]
            gene["block_index"] = [
                i
            ]  # Is a result of no 'without gene layer, only GO model' benchmark, since groups of SNPs are directly mapped to GO layers. these groups are represented as lists in the block_index list, while with the gene layer only on index can be found in the block_index list
            self.out_blocks.append(gene["layersize"])
            for idx in gene[
                "inputs"
            ]:  # also works if there are no inputs directly from the dataset
                self.block_structure.append((idx["index"], i))
                n_snps += 1

        self.in_blocks = [1 for x in range(0, n_snps)]
        self.block_structure_diagonal = [(x, x) for x in range(0, n_snps)]
        # self.in_blocks_one_hot = [4 for x in range(0, n_snps)]
        # self.out_blocks_one_hot = [1 for x in range(0, n_snps)]
        # self.block_structure_one_hot = [(1, 1) for x in range(0, n_snps)]

        # self.block_layer_one_hot = BlockSparseLinear(self.in_blocks_one_hot, self.out_blocks_one_hot,
        #                                     self.block_structure_one_hot, batch_norm=False
        #                                     , dropout=False
        #                                     , activation_function=lambda x: x)

        if (
            gene_scaling_layer == True
        ):  # most likely not contributing anything (anymore)
            self.scale_layer = BlockSparseLinear(
                self.in_blocks,
                self.in_blocks,
                self.block_structure_diagonal,
                residual=False,
                batch_norm=False,
                dropout=input_dropout,
                activation_function=torch.nn.Identity(),
            )

        dropout = dropout if gene_scaling_layer else input_dropout
        num_layers = gene_block_depth

        intermediate_scaling = gene_intermediate_scaling
        if intermediate_scaling != 1.0 and num_layers < 2:
            raise ValueError(
                f"If intermediate scaling is enabled, the number of layers muste be >= 2, but you specified {num_layers}!"
            )

        if (
            isinstance(intermediate_scaling, list)
            and len(intermediate_scaling) != num_layers - 1
        ):
            raise ValueError(
                f"If intermediate_scaling is provided as list, the length must match num_layers - 1, but {len(intermediate_scaling)} != {num_layers - 1}!"
            )

        if not isinstance(intermediate_scaling, list):
            if num_layers == 1:
                intermediate_scaling = [intermediate_scaling]
            elif num_layers > 1:
                intermediate_scaling = [intermediate_scaling] * (num_layers - 1)

        intermediate_mode = gene_intermediate_mode
        block_layers = []
        in_blocks = self.in_blocks
        out_blocks = self.out_blocks
        block_structure = self.block_structure

        if num_layers == 1:
            block = BlockSparseLinear(
                in_blocks,
                out_blocks,
                block_structure,
                residual=residual,
                batch_norm=batch_norm,
                momentum=bn_momentum,
                dropout=dropout,
                activation_function=get_activation(activation),
            )
            block_layers.append(block)

        elif num_layers >= 2:

            if intermediate_mode == "in_in":
                diagonal_in_block_structure = [(i, i) for i in range(0, len(in_blocks))]
            elif intermediate_mode == "out_out":
                diagonal_out_block_structure = [
                    (i, i) for i in range(0, len(out_blocks))
                ]

            for layer in range(num_layers):
                if intermediate_mode == "in_in":
                    if layer == 0:
                        scaling = intermediate_scaling[layer]
                        in_blocks_scaled = [int(i * scaling) for i in in_blocks]

                        block = BlockSparseLinear(
                            in_blocks,
                            in_blocks_scaled,
                            diagonal_in_block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )

                    elif layer == num_layers - 1:
                        block = BlockSparseLinear(
                            in_blocks_scaled,
                            out_blocks,
                            block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )
                    else:
                        scaling = intermediate_scaling[layer]
                        in_blocks_scaled_new = [int(i * scaling) for i in in_blocks]

                        block = BlockSparseLinear(
                            in_blocks_scaled,
                            in_blocks_scaled_new,
                            diagonal_in_block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )
                        in_blocks_scaled = in_blocks_scaled_new

                elif intermediate_mode == "out_out":
                    if layer == 0:
                        scaling = intermediate_scaling[layer]
                        out_blocks_scaled = [int(i * scaling) for i in out_blocks]

                        block = BlockSparseLinear(
                            in_blocks,
                            out_blocks_scaled,
                            block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )

                    elif layer == num_layers - 1:
                        block = BlockSparseLinear(
                            out_blocks_scaled,
                            out_blocks,
                            diagonal_out_block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )

                    else:
                        scaling = intermediate_scaling[layer]
                        out_blocks_scaled_new = [int(i * scaling) for i in out_blocks]

                        block = BlockSparseLinear(
                            out_blocks_scaled,
                            out_blocks_scaled_new,
                            diagonal_out_block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )
                        out_blocks_scaled = out_blocks_scaled_new

                elif intermediate_mode == "in_out":
                    if layer == 0:
                        scaling = intermediate_scaling[layer]
                        ham_blocks_scaled = [int(scaling) for i, j in block_structure]
                        in_ham_block_structure = [
                            (i, k) for k, (i, j) in enumerate(block_structure)
                        ]

                        block = BlockSparseLinear(
                            in_blocks,
                            ham_blocks_scaled,
                            in_ham_block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )

                    elif layer == num_layers - 1:
                        ham_out_block_structure = [
                            (k, j) for k, (i, j) in enumerate(block_structure)
                        ]
                        block = BlockSparseLinear(
                            ham_blocks_scaled,
                            out_blocks,
                            ham_out_block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )

                    else:
                        scaling = intermediate_scaling[layer]
                        ham_blocks_scaled_new = [
                            int(scaling) for i, j in block_structure
                        ]
                        ham_ham_block_structure = [
                            (k, k) for k, (i, j) in enumerate(block_structure)
                        ]

                        block = BlockSparseLinear(
                            ham_blocks_scaled,
                            ham_blocks_scaled_new,
                            ham_ham_block_structure,
                            residual=residual,
                            batch_norm=batch_norm,
                            momentum=bn_momentum,
                            dropout=dropout,
                            activation_function=get_activation(activation),
                        )
                        ham_blocks_scaled = ham_blocks_scaled_new

                block_layers.append(block)

        self.block_layers = nn.Sequential(*block_layers)

        # Prepare Interaction layers
        if ppi is not None:
            interaction_layer_list = []

            # Prepare PPI layers
            if ppi is not None:
                ppi = AttributeDict(ppi)
                if ppi.model == "LinearInteraction":
                    for i in range(ppi.nlayers):
                        ppi["threshold"] = 0
                        ppi_layer = LinearPPI(
                            self.geneobj,
                            activation_function=get_activation(activation),
                            **ppi,
                        )
                        interaction_layer_list.append(ppi_layer)

                else:
                    I_MODELS = ["LinearInteraction"]
                    raise ValueError(
                        f'You set ppi_model={ppi.model}, but is has to be one of {", ".join(I_MODELS)}!'
                    )
            self.interaction_layer = nn.Sequential(*interaction_layer_list)
        else:
            self.interaction_layer = None

        logger.debug(
            f"Prepared {len(self.geneobj)} gene nodes with {self.get_n_parameters()} parameters.\n"
        )

        if variant_embedding:
            self.variant_embedding = nn.Embedding(4, 1)

    def forward(self, inputs, *args, **kwargs):
        """
        Forward pass of the model.

        Args:
            inputs (torch.Tensor): Input tensor for the forward pass.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: Output of the model after applying all layers and interactions.
        """

        device = inputs.device
        # inputs = to_one_hot(inputs, n_dims=4).flatten(start_dim=1).to(device)
        # out = self.block_layer_one_hot(inputs)
        out = inputs

        if self.variant_embedding:
            out = self.variant_embedding(out.long()).squeeze(-1)

        if self.gene_scaling_layer:
            out = self.scale_layer(out)

        try:
            out = self.block_layers(out)
        except AssertionError as e:
            raise Exception(
                "Most likely, the dataset does not fit the model specification, i.e. the number of variants is incorrect."
            ).with_traceback(e.__traceback__)

        if self.interaction_layer:
            out = self.interaction_layer(out)

        if self.keep_activations == "keep":  # What about a callback?
            pass  # or put these to cpu, and detach them?
        elif self.keep_activations == "accumulate":
            self.accumulated_activations.append(out.detach().cpu())
        elif self.keep_activations == "accumulate_balanced":
            self.accumulated_activations.append(
                out[self.batch_keep_idx, :].detach().cpu()
            )

        return out

    def get_n_parameters(self):
        """
        Get the number of parameters in the model.

        Returns:
            int: The total number of parameters in the model.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def reset_accumulated_activations(self):
        """
        Reset the list of accumulated activations.
        """
        self.accumulated_activations = []

    def merge_activations(self):
        """
        Merge all accumulated activations into a single tensor.

        Returns:
            torch.Tensor: The concatenated activations.
        """
        return torch.cat(self.accumulated_activations, axis=0)
