from typing import Optional, Tuple
import warnings
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GOCovAttention(nn.Module):
    def __init__(
        self,
        go_dim: int,
        cov_dim: int,
        inter_dim: Optional[int] = 1,
        values_from: Optional[str] = "covariates",
        multihead: Optional[int] = 1,
        dropout: Optional[float] = 0.0,
        aggregate: Optional[bool] = False,
        residual: Optional[bool] = True,
        layer_norm: Optional[bool] = True,
        store_attention: Optional[bool] = False,
    ):
        """GO-Term Covariate Attention

        Parameters:
        -----------
        go_dim: int
            The dimension of the GO input vector passed to the attention mechanism.
            This should correspond to the total number of neurons in the preceding GO level.
        cov_dim: int
            The dimension of the covariates input vector passed to the attention mechanism.
        inter_dim: int, optional
            The intermediate dimension used in query matrix Wq [N x go_dim x inter_dim] and key matrix Wk [N x cov_dim x inter_dim]
            before computing the attention matrix. This dimension will be summed over when computing the attention matrix.
        multihead: int, optional
            Number of attention heads to use. Defaults to 1.
        values_from: str, optional
            The input vector to be used for computing the value matrix. There are two available options:
            - 'covariates': The covariate input vector (default behaviour)
            - 'gos': The go input vector
            In most cases, using 'gos' will lead to a higher number of parameters.
        dropout: float, optional
            If larger than 0, will apply a dropout layer after the attention mechanism.
        aggregate: bool, optional
            If set to True, will aggregate the attention matrices over the batch dimension into a single attention matrix of dimensions [go_dim x cov-dim]
            By default, will calculate an attention matrix for each sample of minibatch, i.e. the attention A will have dimensions [N x go_dim x cov-dim].
        residual: bool, optional
            If set to True, will use a residual connection.
        layer_norm: bool, optional
            If set to True, will apply Layer Normalization on the outputs.
        store_attention: bool, optional
            If set to True, will store the attention A for later retrieval with the get_attention() method.
            The behaviour can also be controlled after initialization using the enable_store_attention() and disable_store_attention() methods.
        """
        super(GOCovAttention, self).__init__()
        VALUES_FROM = ["covariates", "gos"]
        if not values_from in VALUES_FROM:
            raise ValueError(
                f'Parameters `keys_from` must be one of {", ".join(VALUES_FROM)}, but you specified {values_from}!'
            )
        elif values_from == "covariates":
            self.use_cov = True
        else:
            self.use_cov = False
        self.go_dim = go_dim
        self.cov_dim = cov_dim
        self.inter_dim = inter_dim
        self.multihead = multihead
        self.residual = residual
        self.layer_norm = nn.LayerNorm(go_dim) if layer_norm else None

        self.should_aggregate = aggregate
        self.should_store_attention = store_attention

        self.Wq = []
        self.Wk = []
        self.Wv = []
        for i in range(multihead):
            Wq = torch.rand(go_dim, inter_dim)
            Wq = Parameter(Wq)
            self.register_parameter(f"Wq{i}", Wq)
            self.Wq.append(Wq)

            Wk = torch.rand(cov_dim, inter_dim)
            Wk = Parameter(Wk)
            self.register_parameter(f"Wk{i}", Wk)
            self.Wk.append(Wk)

            # Alternative: input_go [batch_size, cov_dim] x Wv [cov_dim, cov_dim] -> V [batch_size, cov_dim]
            if self.use_cov:
                Wv = torch.rand(cov_dim, cov_dim)
            else:
                Wv = torch.rand(go_dim, cov_dim)
            Wv = Parameter(Wv)
            self.register_parameter(f"Wv{i}", Wv)
            self.Wv.append(Wv)

        if multihead > 1:
            Wo = torch.rand(multihead * go_dim, go_dim)
            self.Wo = Parameter(Wo)

        self.softmax = nn.Softmax(dim=1 if aggregate else 2)
        self.attention = None
        self.dropout = dropout
        self.dropout_layer = (
            nn.Dropout(p=dropout, inplace=True) if dropout > 0 else None
        )

    def enable_store_attention(self):
        self.should_store_attention = True

    def disable_store_attention(self):
        self.should_store_attention = False

    def get_attention(
        self, input: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
    ) -> torch.Tensor:
        if input:
            if self.should_store_attention:
                warnings.warn(
                    f"You tried to compute an attention map from an input, but storing attention maps is disabled. Use `enable_store_attention()` to store attention maps!",
                    UserWarning,
                )
            self.forward(input)
            attention = self.attention
        else:
            attention = self.attention
        return attention

    def forward(self, input: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        go_input, cov_input = input
        assert go_input.shape[0] == cov_input.shape[0]
        B = go_input.shape[0]
        output = torch.empty([B, self.go_dim * self.multihead])
        if self.should_aggregate:
            offset = 0
            for Wq, Wk, Wv in zip(self.Wq, self.Wk, self.Wv):
                # input_go [batch_size, go_dim] x Wq [go_dim, inter_dim] -> Q [go_dim, inter_dim]
                Q = 1 / B * torch.einsum("bj,jk->jk", go_input, Wq)
                # input_cov [batch_size, cov_dim] x Wk [cov_dim, inter_dim] -> K [cov_dim, inter_dim]
                K = 1 / B * torch.einsum("bj,jk->jk", cov_input, Wk)
                if self.use_cov:
                    # input_go [batch_size, cov_dim] x Wv [cov_dim, cov_dim] -> V [batch_size, cov_dim]
                    V = torch.einsum("bj,jk->bk", cov_input, Wv)
                else:
                    # input_go [batch_size, go_dim] x Wv [go_dim, cov_dim] -> V [batch_size, cov_dim]
                    V = torch.einsum("bj,jk->bk", go_input, Wv)
                # Q [batch_size, go_dim, inter_dim] x K [batch_size, cov_dim, inter_dim] x V [batch_size, cov_dim] -> A [go_dim, cov_dim] x V [batch_size, cov_dim] -> output [batch_size, go_dim]
                attention = self.softmax(
                    1 / np.sqrt(self.go_dim) * torch.einsum("jl,kl->jk", Q, K)
                )
                if self.should_store_attention:
                    self.attention = attention.clone().cpu().detach()

                out = torch.einsum("jk,bk->bj", attention, V)
                if self.multihead > 1:
                    output[:, offset : offset + self.go_dim] = out
                    offset += self.go_dim
                else:
                    output = out

            if self.multihead > 1:
                output = torch.einsum("ij,jk->ik", output, self.Wo)

        else:
            offset = 0
            for Wq, Wk, Wv in zip(self.Wq, self.Wk, self.Wv):
                # input_go [batch_size, go_dim] x Wq [go_dim, inter_dim] -> Q [batch_size, go_dim, inter_dim]
                Q = torch.einsum("bj,jk->bjk", go_input, Wq)
                # input_cov [batch_size, cov_dim] x Wk [cov_dim, inter_dim] -> K [batch_size, cov_dim, inter_dim]
                K = torch.einsum("bj,jk->bjk", cov_input, Wk)
                if self.use_cov:
                    # input_go [batch_size, cov_dim] x Wv [cov_dim, cov_dim] -> V [batch_size, cov_dim]
                    V = torch.einsum("bj,jk->bk", cov_input, Wv)
                else:
                    # input_go [batch_size, go_dim] x Wv [go_dim, cov_dim] -> V [batch_size, cov_dim]
                    V = torch.einsum("bj,jk->bk", go_input, Wv)
                # Q [batch_size, go_dim, inter_dim] x K [batch_size, cov_dim, inter_dim] x V [batch_size, cov_dim] -> A [batch_size, go_dim, cov_dim] x V [batch_size, cov_dim] -> output [batch_size, go_dim]
                attention = self.softmax(
                    1 / np.sqrt(self.go_dim) * torch.einsum("bjl,bkl->bjk", Q, K)
                )
                if self.should_store_attention:
                    self.attention = attention.clone().cpu().detach()

                out = torch.einsum("bjk,bk->bj", attention, V)
                if self.multihead > 1:
                    output[:, offset : offset + self.go_dim] = out
                    offset += self.go_dim
                else:
                    output = out

            if self.multihead > 1:
                output = torch.einsum("ij,jk->ik", output, self.Wo)

        if self.residual:
            output = output + go_input
        if self.dropout_layer:
            output = self.dropout_layer(output)
        if self.layer_norm:
            output = self.layer_norm(output)

        return output
