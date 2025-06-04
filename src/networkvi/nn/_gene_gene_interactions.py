from typing import Callable, Optional
import pandas as pd
import torch.nn as nn

from networkvi.nn._block_sparse_linear import BlockSparseLinear

class LinearPPI(nn.Module):
    def __init__(
        self,
        geneobj,
        ppi_file: str,
        threshold: int = 899,
        score: str = "combined_score",
        activation_function: Optional[Callable] = lambda x: x,
        residual: bool = True,
        reflexive: bool = True,
        expand_gene_nodes: bool = False,
        **kwargs
    ):
        """A layer representing protein-protein interactions PPI through a learnable BlockSparseLinear layer.
        Each weight of the sparse layer represents a PPI (or the respective gene-tuple) above the score threshold provided.

        Parameters:
        -----------
        geneobj:
            A geneobj annoated with 'layersize' and 'block_index' for all gene entries.
        ppi_file: str
            Path to a prepared space-separated .csv file of the PPIs containing at least three columns:

            <gene1> | <gene2> | <score>

            The gene columns should be called 'gene1' and 'gene2' and use a gene id also used in geneobj.
            The score column should be called 'combined_score' by default,
            but an arbitrary column name can be selected using the 'score' key.
        threshold: int
            All ppis with scores below this threshold will be filtered.
        score: str
            The name of the score ot use.
        reflexive: bool
            If True, will include connections from a gene to itself (regardless of the ppi scores of its respective proteins).
            NOTE: If G1 codes codes for P1 and P2 and there is an interaction {P1, P2},
            then this will lead to a connection (G1, G1) in this layer.

        """
        super().__init__()
        self.geneobj = geneobj
        self.blocks = []
        self.block_structure = []
        self.ppi_score = score
        self.reflexive = reflexive
        self.residual = residual
        self.input_dim = 0

        for i, ensg in enumerate(geneobj):
            gene = geneobj[ensg]
            self.blocks.append(gene["layersize"])
        self.input_dim = sum(self.blocks)

        ppi_df = pd.read_csv(ppi_file)
        # Keep only interactions scoring above threshold
        ppi_df = ppi_df[ppi_df[self.ppi_score] >= threshold]
        cols_without_prot = [*(set(ppi_df.columns) - {"protein1", "protein2"})]
        # A gene may map to several proteins. This leads to duplicate tuples('gene1', 'gene2').
        # Take the maxmimum interaction score of the respective ppis.
        ggi_df = ppi_df[cols_without_prot]
        ggi_df = ggi_df.groupby(["gene1", "gene2"]).max().reset_index()
        # Filter out genes that are not in geneobj

        ggi_df = ggi_df[
            ggi_df["gene1"].isin([*geneobj.keys()])
            & ggi_df["gene2"].isin([*geneobj.keys()])
        ]
        # Map genes to block indizes
        ggi_df["block1"] = ggi_df["gene1"].map(lambda x: geneobj[x]["block_index"][0])
        ggi_df["block2"] = ggi_df["gene2"].map(lambda x: geneobj[x]["block_index"][0])
        bl = ggi_df[["block1", "block2"]].drop_duplicates().to_dict("list")
        self.block_structure = [*zip(bl["block1"], bl["block2"])]

        if reflexive:
            diagonal_block_structure = [
                (i, i)
                for i in range(len(geneobj))
                if (i, i) not in self.block_structure
            ]
            self.block_structure.extend(diagonal_block_structure)

        self.block_sparse = BlockSparseLinear(
            self.blocks,
            self.blocks,
            self.block_structure,
            activation_function=activation_function,
        )

    def forward(self, input):
        output = self.block_sparse(input)
        if self.residual:
            output = output + input
        return output
