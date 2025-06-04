import torch
from typing import Optional, Union, List, Tuple
from torch.nn.modules.module import Module
from networkvi.nn._block_sparse_linear import BlockSparseLinear


class GOEmbeddingLayer(Module):
    """
    A PyTorch module for embedding layers with block sparse linear transformations.
    Supports multiple layers, various intermediate scaling modes, and residual connections.
    """

    def __init__(
        self,
        num_layers: int,
        in_blocks: List[int],
        out_blocks: List[int],
        block_structure: List[Tuple[int, int]],
        block_structure_pos: Optional[List[Tuple[int, int]]] = [],
        block_structure_neg: Optional[List[Tuple[int, int]]] = [],
        intermediate_scaling: Optional[Union[int, List[int]]] = 1.0,
        intermediate_mode: Optional[str] = "out_out",
        randomize: Optional[float] = 0.0,
        residual: Optional[bool] = False,
        **kwargs,
    ):
        super(type(self), self).__init__()

        self.num_layers = num_layers
        self.embedding_blocks = torch.nn.ModuleList()
        self.intermediate_scaling = intermediate_scaling
        self.intermediate_mode = intermediate_mode
        self.residual = residual

        if (
            isinstance(intermediate_scaling, list)
            and len(intermediate_scaling) != num_layers - 1
        ):
            raise ValueError(
                f"If intermediate_scaling is provided as list, the length must match num_layers - 1, but {len(intermediate_scaling)} != {num_layers - 1}!"
            )

        if intermediate_scaling != 1.0 and num_layers < 2:
            raise ValueError(
                f"If intermediate scaling is enabled, the number of layers muste be >= 2, but you specified {num_layers}!"
            )

        if not isinstance(intermediate_scaling, list):
            if num_layers == 1:
                intermediate_scaling = [intermediate_scaling]
            elif num_layers > 1:
                intermediate_scaling = [intermediate_scaling] * (num_layers - 1)

        if self.intermediate_mode == "in_in":
            diagonal_in_block_structure = [(i, i) for i in range(0, len(in_blocks))]
        elif self.intermediate_mode == "out_out":
            diagonal_out_block_structure = [(i, i) for i in range(0, len(out_blocks))]

        if num_layers == 1:
            block = BlockSparseLinear(
                in_blocks,
                out_blocks,
                block_structure,
                block_structure_pos=block_structure_pos,
                block_structure_neg=block_structure_neg,
                **kwargs,
            )
            self.embedding_blocks.append(block)

        elif num_layers >= 2:
            for layer in range(self.num_layers):
                if self.intermediate_mode == "in_in":
                    if layer == 0:
                        scaling = intermediate_scaling[layer]
                        in_blocks_scaled = [int(i * scaling) for i in in_blocks]

                        block = BlockSparseLinear(
                            in_blocks,
                            in_blocks_scaled,
                            diagonal_in_block_structure,
                            **kwargs,
                        )

                    elif layer == num_layers - 1:
                        block = BlockSparseLinear(
                            in_blocks_scaled,
                            out_blocks,
                            block_structure,
                            block_structure_pos=block_structure_pos,
                            block_structure_neg=block_structure_neg,
                            **kwargs,
                        )

                    else:
                        scaling = intermediate_scaling[layer]
                        in_blocks_scaled_new = [int(i * scaling) for i in in_blocks]

                        block = BlockSparseLinear(
                            in_blocks_scaled,
                            in_blocks_scaled_new,
                            diagonal_in_block_structure,
                            **kwargs,
                        )
                        in_blocks_scaled = in_blocks_scaled_new

                elif self.intermediate_mode == "out_out":
                    if layer == 0:
                        scaling = intermediate_scaling[layer]
                        out_blocks_scaled = [int(i * scaling) for i in out_blocks]

                        block = BlockSparseLinear(
                            in_blocks,
                            out_blocks_scaled,
                            block_structure,
                            block_structure_pos=block_structure_pos,
                            block_structure_neg=block_structure_neg,
                            **kwargs,
                        )

                    elif layer == num_layers - 1:
                        block = BlockSparseLinear(
                            out_blocks_scaled,
                            out_blocks,
                            diagonal_out_block_structure,
                            **kwargs,
                        )

                    else:
                        scaling = intermediate_scaling[layer]
                        out_blocks_scaled_new = [int(i * scaling) for i in out_blocks]

                        block = BlockSparseLinear(
                            out_blocks_scaled,
                            out_blocks_scaled_new,
                            diagonal_out_block_structure,
                            **kwargs,
                        )
                        out_blocks_scaled = out_blocks_scaled_new

                elif self.intermediate_mode == "in_out":
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
                            **kwargs,
                        )

                    elif layer == num_layers - 1:
                        ham_out_block_structure = [
                            (k, j) for k, (i, j) in enumerate(block_structure)
                        ]
                        # To have an edge have a positive/negative contribution, it suffices to to force that over the last edges.
                        ham_out_block_structure_pos = [
                            (k, j) for k, (i, j) in enumerate(block_structure_pos)
                        ]
                        ham_out_block_structure_neg = [
                            (k, j) for k, (i, j) in enumerate(block_structure_neg)
                        ]

                        block = BlockSparseLinear(
                            ham_blocks_scaled,
                            out_blocks,
                            ham_out_block_structure,
                            block_structure_pos=ham_out_block_structure_pos,
                            block_structure_neg=ham_out_block_structure_neg,
                            **kwargs,
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
                            **kwargs,
                        )
                        ham_blocks_scaled = ham_blocks_scaled_new

                else:
                    INTERMEDIATE_MODES = ["in_in", "in_out", "out_out"]
                    raise ValueError(
                        f'Expected go_intermediate_mode to be one of {", ".join(INTERMEDIATE_MODES)}, but got {self.intermediate_mode}!'
                    )

                self.embedding_blocks.append(block)

        if residual:
            kwargs.pop("activation_function", None)
            self.residual_block = BlockSparseLinear(
                in_blocks,
                out_blocks,
                block_structure,
                activation_function=torch.nn.Identity(),
                **kwargs,
            )

    def forward(self, input):
        out = input
        for layer in range(self.num_layers):
            out = self.embedding_blocks[layer](out)
        if self.residual:
            out = out + self.residual_block(input)
        return out

    def resample_connections(self):
        for block in self.embedding_blocks:
            block.resample_random()
        if self.residual:
            self.residual_block.resample_random()
