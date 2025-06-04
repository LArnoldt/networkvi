from abc import ABC, abstractmethod
import math
import torch
import torch.nn as nn
import numpy as np
from operator import itemgetter
try:
    import torch_sparse
except ImportError:
    print("Install torch-sparse and torch-scatter via: pip install -U torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-[VERSION]+cu[VERSION].html")
from collections import defaultdict
from typing import Optional, Callable, List, Tuple
from torch.nn import init
from torch.nn.modules.module import Module
from torch.nn.parameter import Parameter
import os


def kaiming_uniform_(tensor, a=0, fan=1, mode="fan_in", nonlinearity="leaky_relu"):
    r"""Fills the input `Tensor` with values according to the method
    described in `Delving deep into rectifiers: Surpassing human-level
    performance on ImageNet classification` - He, K. et al. (2015), using a
    uniform distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{U}(-\text{bound}, \text{bound})` where

    .. math::
        \text{bound} = \text{gain} \times \sqrt{\frac{3}{\text{fan\_mode}}}

    Also known as He initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        a: the negative slope of the rectifier used after this layer (only
            used with ``'leaky_relu'``)
        mode: either ``'fan_in'`` (default) or ``'fan_out'``. Choosing ``'fan_in'``
            preserves the magnitude of the variance of the weights in the
            forward pass. Choosing ``'fan_out'`` preserves the magnitudes in the
            backwards pass.
        nonlinearity: the non-linear function (`nn.functional` name),
            recommended to use only with ``'relu'`` or ``'leaky_relu'`` (default).

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    """
    gain = init.calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)


class BlockSparseLinear(Module):
    __constants__ = ["in_features", "out_features"]

    def __init__(
        self,
        in_blocks: List[int],
        out_blocks: List[int],
        block_structure: List[Tuple[int, int]],
        block_structure_pos: List[Tuple[int, int]] = [],
        block_structure_neg: List[Tuple[int, int]] = [],
        dropout: Optional[int] = 0,
        momentum: Optional[int] = 0.1,
        activation_function: Optional[Callable] = lambda x: x,
        bias: Optional[bool] = True,
        batch_norm: Optional[bool] = False,
        debug: Optional[bool] = False,
        track_running_stats: Optional[bool] = False,
        residual: Optional[bool] = False,
        masking: Optional[bool] = False,
        pos_neg_init: Optional[bool] = False,
        **kwargs,
    ):
        """Represents a weighted connection between GO terms of one layer and GO terms of another layer.

        Parameters:
        -----------
        in_blocks: list
            A list containing the size for each input block, i.e. the number of neurons to be used for each incoming GOTerm.
        out_blocks: list
            A list containing the size for each output block, i.e. the number of neurons to be used for each outgoing GOTerm.
        block_structure: List[Tuple[int, int]]
            A dictionary containing tuples (i, j), reprensenting a connection from GOTerm i to GOTerm j.
            The indices i, j should address the matching block sizes in_blocks and out_blocks, respectively.
        block_structure_pos: List[Tuple[int, int]], optional
            A dictionary containing tuples (i, j), reprensenting a `positively_regulates` connection from GOTerm i to GOTerm j.
            Should be used with `masking` or `pos_neg_init` set to True.
        block_structure_neg: List[Tuple[int, int]], optional
            A dictionary containing tuples (i, j), reprensenting a `negatively_regulates` connection from GOTerm i to GOTerm j.
            Should be used with `masking` or `pos_neg_init` set to True.
        dropout: float, optional
            If greater than 0, specifies the dropout to apply on the output of the BlockSparseLinear layer.
            Value range: 0 <= dropout <= 1
        momentum: float, optional
            Sets the momentum for the BatchNormalization layer if `batch_norm` is set to True.
            Value range: 0 <= dropout <= 1
        activation_function: Callable, optional
            Activation function to be applied after BlockSparseLinear layer.
        bias: bool, optional
            Specifies if bias should be applied after weight layer.
        batch_norm: bool, optional
            Specifies if batch normalization should be used.
        debug:
            Initializes this module in debug mode.
        track_running_stats:
            Sets `track_running_stats` of BachNorm1d layer if used.
        residual:
            If set, uses residual connection, i.e. the layer output becomes f'(x) = f(x) + x
        masking:
            If set, will constrain weights for connections provided with `block_structure_pos` and `block_structure_neg`:
            -`block_structure_pos`: Weights are always positive.
            -`block_structure_neg`: Weights are always negative.
        pos_neg_init:
            If set, will initialize weights for connections provided with `block_structure_pos` and `block_structure_neg`:
            -`block_structure_pos`: Weights are initialized positive.
            -`block_structure_neg`: Weights are initialized negative.
        """

        super(type(self), self).__init__()

        in_features, out_features = sum(in_blocks), sum(out_blocks)

        assert isinstance(in_blocks, list), "Argument of wrong type!"
        assert isinstance(out_blocks, list), "Argument of wrong type!"

        self.in_features = in_features
        self.out_features = out_features
        self.out_blocks = out_blocks
        self.in_blocks = in_blocks
        self.in_blocks_starts = np.cumsum([0] + self.in_blocks)
        self.out_blocks_starts = np.cumsum([0] + self.out_blocks)
        self.block_structure = block_structure
        self.block_structure_pos = block_structure_pos
        self.block_structure_neg = block_structure_neg
        self.debug = debug
        self.residual = residual
        self.masking = masking

        # Indices are initialized using register_buffer()
        # self.indices = None
        # self.indices_pos = None
        # self.indices_neg = None
        self.values = None
        self.values_pos = None
        self.values_neg = None

        self.values_residual = None
        self.values_residual_pos = None
        self.values_residual_neg = None

        self.pos_neg_init = pos_neg_init
        self.activation_function = activation_function

        if bias:
            self.bias = Parameter(torch.Tensor(out_features, 1))
        else:
            self.register_parameter("bias", None)

        self.dropout = dropout
        self.drop_layer = (
            torch.nn.Dropout(p=dropout, inplace=False) if dropout > 0 else None
        )
        self.batch_norm_layer = (
            torch.nn.BatchNorm1d(
                sum(out_blocks),
                track_running_stats=track_running_stats,
                momentum=momentum,
                affine=True,
            )
            if batch_norm
            else None
        )

        self.reset_parameters()

        print(f"Generated block structure with len {len(self.block_structure)} ({len(self.block_structure)/(len(self.in_blocks)*len(self.out_blocks))} blocks).")

    def _get_index(self, block):
        line_index, col_index = (
            self.in_blocks_starts[block[0]],
            self.out_blocks_starts[block[1]],
        )
        # Size of block
        block_lines, block_cols = self.in_blocks[block[0]], self.out_blocks[block[1]]

        # Generate index grid shape: (columns, lines, 2), last dimension for either line or column index
        indices = np.indices((block_cols, block_lines))
        indices = indices.reshape(2, -1)
        indices[0] += col_index
        indices[1] += line_index
        indices = indices.T
        indices = torch.LongTensor(indices)
        return indices

    def _get_index_values(self, block, fan_out, fan_in):
        line_index, col_index = (
            self.in_blocks_starts[block[0]],
            self.out_blocks_starts[block[1]],
        )
        # Size of block
        block_lines, block_cols = self.in_blocks[block[0]], self.out_blocks[block[1]]

        # Generate index grid shape: (columns, lines, 2), last dimension for either line or column index
        indices = np.indices((block_cols, block_lines))
        indices = indices.reshape(2, -1)
        indices[0] += col_index
        indices[1] += line_index
        indices = indices.T
        indices = torch.LongTensor(indices)

        # Initialize weights for a linear layer of block size
        if self.debug:
            block_weights = torch.ones([block_cols, block_lines])
        else:
            block_weights = torch.Tensor(block_cols, block_lines)
            kaiming_uniform_(
                block_weights,
                a=math.sqrt(5),
                fan=(fan_out[block[0]] + fan_in[block[1]]),
            )

        return block_weights, indices

    def reset_parameters(self):
        i_list, i_list_pos, i_list_neg = [], [], []
        v_list, v_list_pos, v_list_neg = [], [], []
        r_list, r_list_pos, r_list_neg = [], [], []
        # Generate a tensor in which we save all indices and values that are added to the sparse matrix
        # Will be initialized to a size that must be larger than the number of elements added
        # max_params = max(self.in_blocks) * max(self.out_blocks) * len(self.block_structure)
        # i, v = torch.zeros([max_params, 2], dtype=torch.long), torch.zeros([max_params])
        out_block_weights = (
            {}
        )  # Collects the weights that account for each output block for bias computation
        # idx = 0 # Points to index in list of elements to add to sparse matrix

        # Initialize fan_in and fan_out:

        self.fan_in = defaultdict(int)
        self.fan_out = defaultdict(int)
        for block in self.block_structure:
            self.fan_in[block[1]] = self.fan_in[block[1]] + self.in_blocks[block[0]]
            self.fan_out[block[0]] = self.fan_out[block[0]] + self.out_blocks[block[1]]

        for block in self.block_structure_pos:
            self.fan_in[block[1]] = self.fan_in[block[1]] + self.in_blocks[block[0]]
            self.fan_out[block[0]] = self.fan_out[block[0]] + self.out_blocks[block[1]]

        for block in self.block_structure_neg:
            self.fan_in[block[1]] = self.fan_in[block[1]] + self.in_blocks[block[0]]
            self.fan_out[block[0]] = self.fan_out[block[0]] + self.out_blocks[block[1]]

        for block in self.block_structure:
            # Line and column index of block start in sparse matrix (index of upper, lefthand corner)

            # Insert the flattened indices transformed to the sparse matrix indixes
            # i[idx:idx+block_cols*block_lines] = indices
            # indices = np.indices((block_cols, block_lines))
            # i[idx:idx+block_cols*block_lines, 0] = torch.LongTensor(indices[0].flatten() + col_index)
            # i[idx:idx+block_cols*block_lines, 1] = torch.LongTensor(indices[1].flatten() + line_index)
            # Insert the flattened values from the initialized block matrix for each index
            # v[idx:idx+block_cols*block_lines] = block_weights.flatten()

            block_weights, indices = self._get_index_values(block, self.fan_out, self.fan_in)

            i_list.append(indices)
            v_list.append(block_weights.flatten())
            r_list.append(
                block_weights.flatten()
            )  # torch.eye(block_weights.shape[0], block_weights.shape[1]).flatten())

            # Append the block matrix values to a list for each this output block for bias bounds
            out_block_weights.setdefault(block[1], []).append(block_weights)

            # Increase index into list of elements to add according to the number of elements added
            # idx = idx+block_cols*block_lines

        if self.pos_neg_init and self.block_structure_pos:
            for block in self.block_structure_pos:
                block_weights, indices = self._get_index_values(block, self.fan_out, self.fan_in)
                block_weights = torch.abs(block_weights)

                i_list.append(indices)
                v_list.append(block_weights.flatten())
                r_list.append(block_weights.flatten())

        if self.pos_neg_init and self.block_structure_neg:
            for block in self.block_structure_neg:
                block_weights, indices = self._get_index_values(block, self.fan_out, self.fan_in)
                block_weights = -torch.abs(block_weights)

                i_list.append(indices)
                v_list.append(block_weights.flatten())
                r_list.append(block_weights.flatten())

        if self.masking and self.block_structure_pos:
            for block in self.block_structure_pos:
                block_weights, indices = self._get_index_values(block, self.fan_out, self.fan_in)
                i_list_pos.append(indices)
                v_list_pos.append(block_weights.flatten())
                r_list_pos.append(block_weights.flatten())

        if self.masking and self.block_structure_neg:
            for block in self.block_structure_neg:
                block_weights, indices = self._get_index_values(block, self.fan_out, self.fan_in)
                i_list_neg.append(indices)
                v_list_neg.append(block_weights.flatten())
                r_list_neg.append(block_weights.flatten())

        if self.bias is not None:
            for block in range(0, len(self.out_blocks)):
                # if (self.debug):
                #    self.bias[self.out_blocks_starts[block]:self.out_blocks_starts[block] + self.out_blocks[block],
                #    0] = 0.0
                #    continue
                if not block in out_block_weights:
                    bound = 0.5
                else:
                    bound = 1 / math.sqrt(self.fan_in[block])
                init.uniform_(
                    self.bias[
                        self.out_blocks_starts[block] : self.out_blocks_starts[block]
                        + self.out_blocks[block],
                        0,
                    ],
                    -bound,
                    bound,
                )

        # Remove the superflous entries (we had to initialize more entries than are needed)
        # i = i[0:idx].T
        # v = v[0:idx]
        def _prepare_ivr(i_list, v_list, r_list):
            if len(i_list) == 0:
                i = torch.tensor([], dtype=torch.int64).reshape([2, 0])
                v = torch.tensor([])
                r = torch.tensor([])
            else:
                i, v, r = torch.cat(i_list).T, torch.cat(v_list), torch.cat(r_list)
            i, v = torch_sparse.coalesce(i, v, self.out_features, self.in_features)
            i.requires_grad = False
            return i, v, r

        i, v, r = _prepare_ivr(i_list, v_list, r_list)
        self.register_buffer("indices", i)
        self.values = Parameter(v)

        if self.residual:
            i, r = torch_sparse.coalesce(i, r, self.out_features, self.in_features)
            self.values_residual = Parameter(r, requires_grad=True)

        if self.masking:
            if bool(i_list_pos) and bool(v_list_pos):
                i_pos, v_pos, r_pos = _prepare_ivr(i_list_pos, v_list_pos, r_list_pos)
                self.register_buffer("indices_pos", i_pos)
                self.values_pos = Parameter(v_pos, requires_grad=True)

            if bool(i_list_neg) and bool(v_list_neg):
                i_neg, v_neg, r_neg = _prepare_ivr(i_list_neg, v_list_neg, r_list_neg)
                self.register_buffer("indices_neg", False)
                self.values_neg = Parameter(v_neg, requires_grad=True)

            if self.residual:
                i_pos, r_pos = torch_sparse.coalesce(
                    i_pos, r_pos, self.out_features, self.in_features
                )
                i_neg, r_neg = torch_sparse.coalesce(
                    i_neg, r_neg, self.out_features, self.in_features
                )

                self.values_residual_pos = Parameter(r_pos, requires_grad=True)
                self.values_residual_neg = Parameter(r_neg, requires_grad=True)

    def reset_parameters_(self):
        # Generate a tensor in which we save all indices and values that are added to the sparse matrix
        # Will be initialized to a size that must be larger than the number of elements added
        # max_params = max(self.in_blocks) * max(self.out_blocks) * len(self.block_structure)
        # i, v = torch.zeros([max_params, 2], dtype=torch.long), torch.zeros([max_params])
        out_block_weights = (
            {}
        )  # Collects the weights that account for each output block for bias computation
        # idx = 0 # Points to index in list of elements to add to sparse matrix
        self.fan_in = defaultdict(int)
        self.fan_out = defaultdict(int)
        count_total = 0
        for block in self.block_structure:
            self.fan_in[block[1]] = self.fan_in[block[1]] + self.in_blocks[block[0]]
            self.fan_out[block[0]] = self.fan_out[block[0]] + self.out_blocks[block[1]]

    def _generate_new_indices(self, block_structure, in_blocks, out_blocks, input):

        block_structure = set(block_structure)
        len_block_structure = len(block_structure)
        #row_indices = torch.randint(0, len(in_blocks), (self.k,))
        #col_indices = torch.randint(0, len(out_blocks), (self.k,))
        while len(block_structure) < self.k+len_block_structure:
            row_indices = torch.randint(0, len(in_blocks), (self.k,))
            col_indices = torch.randint(0, len(out_blocks), (self.k,))
            sampled_combinations = set(zip(row_indices.cpu().numpy(), col_indices.cpu().numpy()))
            block_structure.update(sampled_combinations)
        block_structure = list(block_structure)[:self.k+len_block_structure]
        #block_structure = block_structure + [(row_index, col_index) for row_index, col_index in zip(row_indices.cpu().numpy(), col_indices.cpu().numpy())]
        #(lm_indices.unsqueeze(1) * self.n_group + torch.arange(self.n_group).to(lm_indices.device)).view(-1)
        #row_indices = (row_indices.unsqueeze(1) * in_blocks[0] + torch.arange(in_blocks[0]).unsqueeze(0)).view(-1)
        #col_indices = (col_indices.unsqueeze(1) * out_blocks[0] + torch.arange(out_blocks[0]).unsqueeze(0)).view(-1)
        row_indices = torch.repeat_interleave((row_indices*in_blocks[0]).repeat_interleave(in_blocks[0]) + torch.arange(out_blocks[0]).repeat(len(row_indices)), out_blocks[0])
        col_indices = torch.repeat_interleave((col_indices*in_blocks[0]).repeat_interleave(in_blocks[0]) + torch.arange(out_blocks[0]).repeat(len(col_indices)), out_blocks[0])

        fan_in = defaultdict(int)
        fan_out = defaultdict(int)
        for block in block_structure:
            fan_in[block[1]] = fan_in[block[1]] + in_blocks[block[0]]
            fan_out[block[0]] = fan_out[block[0]] + out_blocks[block[1]]

        return block_structure, fan_in, fan_out, torch.stack((row_indices, col_indices), dim=0)

    def forward(self, input):

        values = self.values
        out = torch_sparse.spmm(
            self.indices, values, self.out_features, self.in_features, input.T
        )

        if self.masking:
            if isinstance(self.indices_pos, torch.Tensor) and isinstance(
                self.values_pos, torch.Tensor
            ):
                values_pos = torch.abs(self.values_pos)
                out = out + torch_sparse.spmm(
                    self.indices_pos,
                    values_pos,
                    self.out_features,
                    self.in_features,
                    input.T,
                )

            if isinstance(self.indices_neg, torch.Tensor) and isinstance(
                self.values_neg, torch.Tensor
            ):
                values_neg = -torch.abs(self.values_neg)
                out = out + torch_sparse.spmm(
                    self.indices_neg,
                    values_neg,
                    self.out_features,
                    self.in_features,
                    input.T,
                )

        if self.bias is not None:
            out = out + self.bias
        out = self.activation_function(out)
        if self.residual:
            out_residual = torch_sparse.spmm(
                self.indices,
                self.values_residual,
                self.out_features,
                self.in_features,
                input.T,
            )
            if self.masking:
                if bool(self.indices_pos) and bool(self.values_residual_pos):
                    out_residual = out_residual + torch_sparse.spmm(
                        self.indices_pos,
                        self.values_residual_pos,
                        self.out_features,
                        self.in_features,
                        input.T,
                    )
                if bool(self.indices_neg) and bool(self.values_residual_neg):
                    out_residual = out_residual + torch_sparse.spmm(
                        self.indices_neg,
                        self.values_residual_neg,
                        self.out_features,
                        self.in_features,
                        input.T,
                    )

            out = out + out_residual
        out = out.T
        if self.dropout > 0.0:
            out = self.drop_layer(out)
        if self.batch_norm_layer:
            out = self.batch_norm_layer(out)
        return out

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )
