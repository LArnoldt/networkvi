import logging

logger = logging.getLogger(__name__)

import sys
import os
import json
import torch
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Mapping, Any
import numpy as np
import pandas as pd

sys.path.append(os.getcwd())
import pathlib
import re

import warnings
import pickle
from pytorch_lightning import Trainer
from tqdm import tqdm



def create_html(tree_data, results_dir, evaluate_selector, phenotype):
    filename = "resources/genopheno/ogm_overview.html"
    f = open(filename, "r", encoding="utf-8")
    template = f.read()
    f.close()

    filename = "resources/genopheno/ogm_overview.css"
    f = open(filename, "r", encoding="utf-8")
    style = f.read()
    f.close()

    template = template.replace("{$style$}", style)
    template = template.replace("{$dag_data$}", tree_data)

    save_path = os.path.join(results_dir, f"{evaluate_selector}_{phenotype}_graph.html")
    f = open(save_path, "w", encoding="utf-8")
    f.write(template)
    f.close()

    return save_path


def goobj_to_graphml(
    goobj,
    output_directory,
    enrichment_names=[],
    interpret_covariates=None,
    label_only=True,
):
    with open(output_directory, "w") as graphml:
        graphml.write('<?xml version="1.0" encoding="UTF - 8" standalone="no"?>\n')
        graphml.write('<graphml xmlns="http://graphml.graphdrawing.org/xmlns">\n')

        graphml.write(
            '\t<key attr.name="_parents" attr.type="string" for="node" id="_parents"/>\n'
        )
        graphml.write(
            '\t<key attr.name="activation_diff" attr.type="double" for="node" id="activation_diff"/>\n'
        )
        graphml.write(
            '\t<key attr.name="alt_ids" attr.type="string" for="node" id="alt_ids"/>\n'
        )
        graphml.write(
            '\t<key attr.name="block_index" attr.type="int" for="node" id="block_index"/>\n'
        )
        graphml.write(
            '\t<key attr.name="depth" attr.type="int" for="node" id="depth"/>\n'
        )
        graphml.write('\t<key attr.name="id" attr.type="string" for="node" id="id"/>\n')
        graphml.write(
            '\t<key attr.name="is_obsolete" attr.type="string" for="node" id="is_obsolete"/>\n'
        )
        graphml.write(
            '\t<key attr.name="item_id" attr.type="string" for="node" id="item_id"/>\n'
        )
        graphml.write(
            '\t<key attr.name="layersize" attr.type="string" for="node" id="layersize"/>\n'
        )
        graphml.write(
            '\t<key attr.name="level" attr.type="int" for="node" id="level"/>\n'
        )
        graphml.write(
            '\t<key attr.name="name" attr.type="string" for="node" id="name"/>\n'
        )
        graphml.write(
            '\t<key attr.name="namespace" attr.type="string" for="node" id="namespace"/>\n'
        )
        graphml.write(
            '\t<key attr.name="ogm_depth" attr.type="int" for="node" id="ogm_depth"/>\n'
        )
        graphml.write(
            '\t<key attr.name="parents" attr.type="string" for="node" id="parents"/>\n'
        )
        graphml.write(
            '\t<key attr.name="predictors_label_roc" attr.type="double" for="node" id="predictors_label_roc"/>\n'
        )
        graphml.write(
            '\t<key attr.name="predictors_label_spearman_correlation" attr.type="double" for="node" id="predictors_label_spearman_correlation"/>\n'
        )
        graphml.write(
            '\t<key attr.name="predictors_label_spearman_p" attr.type="double" for="node" id="predictors_label_spearman_p"/>\n'
        )
        graphml.write(
            '\t<key attr.name="predictors_label_valid_roc" attr.type="double" for="node" id="predictors_label_valid_roc"/>\n'
        )
        graphml.write(
            '\t<key attr.name="predictors_label_valid_spearman_correlation" attr.type="double" for="node" id="predictors_label_valid_spearman_correlation"/>\n'
        )
        graphml.write(
            '\t<key attr.name="predictors_label_valid_spearman_p" attr.type="double" for="node" id="predictors_label_valid_spearman_p"/>\n'
        )
        if interpret_covariates and not label_only:
            for cov in interpret_covariates:

                graphml.write(
                    f'\t<key attr.name="predictors_{cov}_spearman_correlation" attr.type="double" for="node" id="predictors_{cov}_spearman_correlation"/>\n'
                )
                graphml.write(
                    f'\t<key attr.name="predictors_{cov}_spearman_p" attr.type="double" for="node" id="predictors_{cov}_spearman_p"/>\n'
                )
                graphml.write(
                    f'\t<key attr.name="predictors_{cov}_valid_spearman_correlation" attr.type="double" for="node" id="predictors_{cov}_valid_spearman_correlation"/>\n'
                )
                graphml.write(
                    f'\t<key attr.name="predictors_{cov}_valid_spearman_p" attr.type="double" for="node" id="predictors_{cov}_valid_spearman_p"/>\n'
                )
        for enrichment_name in enrichment_names:
            graphml.write(
                f'\t<key attr.name="{enrichment_name}" attr.type="double" for="node" id="{enrichment_name}"/>\n'
            )
        graphml.write(
            '\t<key attr.name="relationship" attr.type="string" for="node" id="relationship"/>\n'
        )
        graphml.write(
            '\t<key attr.name="width" attr.type="int" for="node" id="width"/>\n'
        )
        graphml.write(
            '\t<key attr.name="SUID" attr.type="string" for="edge" id="width"/>\n'
        )

        graphml.write(
            '\t<graph edgedefault="directed" id="test_evaluated_graph.graphml">\n'
        )

        for go_term in goobj.keys():
            if go_term.startswith("GO:"):
                graphml.write(f'\t<node id="{go_term}">\n')
                graphml.write(
                    f'\t\t<data key="_parents">{",".join(list(goobj[go_term]._parents))}</data>\n'
                )
                graphml.write(
                    f'\t\t<data key="activation_diff">{goobj[go_term].activation_diff}</data>\n'
                )

                graphml.write(
                    f'\t\t<data key="alt_ids">{",".join(list(goobj[go_term].alt_ids))}</data>\n'
                )
                graphml.write(
                    f'\t\t<data key="block_index">{goobj[go_term].block_index if isinstance(goobj[go_term].block_index, int) else goobj[go_term].block_index[0]}</data>\n'
                )

                graphml.write(f'\t\t<data key="depth">{goobj[go_term].depth}</data>\n')

                graphml.write(f'\t\t<data key="id">{goobj[go_term].id}</data>\n')
                graphml.write(
                    f'\t\t<data key="is_obsolete">{goobj[go_term].is_obsolete}</data>\n'
                )
                graphml.write(
                    f'\t\t<data key="item_id">{goobj[go_term].item_id}</data>\n'
                )
                graphml.write(
                    f'\t\t<data key="layersize">{goobj[go_term].layersize}</data>\n'
                )
                graphml.write(f'\t\t<data key="level">{goobj[go_term].level}</data>\n')
                graphml.write(f'\t\t<data key="name">{goobj[go_term].name}</data>\n')
                graphml.write(
                    f'\t\t<data key="namespace">{goobj[go_term].namespace}</data>\n'
                )
                graphml.write(
                    f'\t\t<data key="ogm_depth">{goobj[go_term].ogm_depth}</data>\n'
                )

                try:
                    graphml.write(
                        f'\t\t<data key="parents">{",".join([pr.item_id for pr in goobj[go_term].parents])}</data>\n'
                    )
                except:
                    graphml.write(
                        f'\t\t<data key="parents">{",".join([pr for pr in goobj[go_term].parents])}</data>\n'
                    )

                if len(vars(goobj[go_term].predictors["label"]).keys()) > 0:
                    graphml.write(
                        f'\t\t<data key="predictors_label_roc">{goobj[go_term].predictors["label"].roc}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_spearman_correlation">{goobj[go_term].predictors["label"].spearman_correlation}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_spearman_p">{goobj[go_term].predictors["label"].spearman_p}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_valid_roc">{goobj[go_term].predictors["label"].valid_roc}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_valid_spearman_correlation">{goobj[go_term].predictors["label"].valid_spearman_correlation}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_valid_spearman_p">{goobj[go_term].predictors["label"].spearman_p}</data>\n'
                    )
                else:
                    graphml.write(
                        f'\t\t<data key="predictors_label_roc">{"NA"}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_spearman_correlation">{"NA"}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_spearman_p">{"NA"}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_valid_roc">{"NA"}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_valid_spearman_correlation">{"NA"}</data>\n'
                    )
                    graphml.write(
                        f'\t\t<data key="predictors_label_valid_spearman_p">{"NA"}</data>\n'
                    )
                if interpret_covariates and not label_only:
                    for cov in interpret_covariates:

                        graphml.write(
                            f'\t\t<data key="predictors_{cov}_spearman_correlation">{goobj[go_term].predictors[cov].spearman_correlation}</data>\n'
                        )
                        graphml.write(
                            f'\t\t<data key="predictors_{cov}_spearman_p">{goobj[go_term].predictors[cov].spearman_p}</data>\n'
                        )
                        graphml.write(
                            f'\t\t<data key="predictors_{cov}_valid_spearman_correlation">{goobj[go_term].predictors[cov].valid_spearman_correlation}</data>\n'
                        )
                        graphml.write(
                            f'\t\t<data key="predictors_{cov}_valid_spearman_p">{goobj[go_term].predictors[cov].spearman_p}</data>\n'
                        )
                for enrichment_name in enrichment_names:
                    graphml.write(
                        f'\t\t<data key="{enrichment_name}">{goobj[go_term].predictors[enrichment_name]}</data>\n'
                    )

                graphml.write(f'\t\t<data key="width">{goobj[go_term].width}</data>\n')
                graphml.write("\t</node>\n")

        edge_counter = 0
        for go_term in goobj.keys():
            if go_term.startswith("GO:"):
                for parent in goobj[go_term]._parents:
                    graphml.write(f'\t<edge source="{go_term}" target="{parent}">\n')
                    graphml.write(f'\t\t<data key="SUID">{edge_counter}</data>\n')
                    graphml.write("\t</edge>\n")
                    edge_counter += 1

        graphml.write("</graph>\n")
        graphml.write("</graphml>\n")


def make_dataframe_from_goobj(goobj, goorder):
    # Create json representation of graph
    data = {"root": "", "nodes": {}, "links": [], "labels": []}
    labels_set = set()
    for go in goorder:
        term = goobj[go]
        if len(term.parents) == 0:
            data["root"] = term.item_id
        data_elem = {
            "id": term.item_id,
            "name": term.name,
            "children": [
                goobj[ch.item_id].item_id for ch in term.children
            ],  # CHANGE: ch -> ch.item_id
            "parents": [
                goobj[pr.item_id].item_id for pr in term.parents
            ],  # CHANGE: pr -> pr.item_id
            "depth": term.depth,
        }
        predictors = {}
        for predictor in goobj[go].predictors:
            labels_set.add(predictor)
            try:
                predictors[predictor] = str(
                    term.predictors[predictor].spearman_correlation
                )
            except AttributeError:
                predictors[predictor] = str(term.predictors[predictor])
        data_elem["predictors"] = predictors
        data_elem["label_roc"] = term.predictors["label"].roc
        data["nodes"][go] = data_elem
        data["links"] += [
            [go, goobj[ch.item_id].item_id] for ch in term.children
        ]  # CHANGE: ch -> ch.item_id
    data["labels"] = list(labels_set)

    return data


def make_json_str(goobj, goorder):
    data = make_dataframe_from_goobj(goobj, goorder)
    json_str = json.dumps(data)
    logger.info(f"Created json string")
    return json_str


