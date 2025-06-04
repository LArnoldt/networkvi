import logging

logger = logging.getLogger(__name__)

from sklearn.linear_model import (
    LogisticRegression,
    LogisticRegressionCV,
    Ridge,
)

# from sklearn.neural_network import MLPClassifier
# from sklearn.model_selection import GridSearchCV
# from catboost import CatBoostClassifier

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from pqdm.threads import pqdm

from argparse import Namespace
import numpy as np
from scipy import stats
import pandas as pd
import ast
from collections import defaultdict
import os
import matplotlib.pyplot as plt
from goatools.obo_parser import GODag
from goatools.anno.gaf_reader import GafReader
import pickle
import torch.nn as nn
import sys
import torch
import fcntl

def goobj_to_csv(goobj, interpret_covariates=None, label_only=True):
    columns = [
        "_parents",
        "activation_diff",
        "activations_test",
        "activations_valid",
        "alt_ids",
        "block_index",
        "case_activations",
        "case_means",
        #"children",
        "ctrl_activations",
        "ctrl_means",
        "depth",
        "ensemblids",
        "genes_disrupted",
        #"get_all_child_edges",
        #"get_all_children",
        #"get_all_lower",
        #"get_all_parent_edges",
        #"get_all_parents",
        #"get_all_upper",
        #"get_goterms_lower",
        # 'get_goterms_lower_rels',
        #"get_goterms_upper",
        # 'get_goterms_upper_rels',
        # 'has_child',
        # 'has_parent',
        "id",
        "is_obsolete",
        "item_id",
        "layersize",
        "level",
        "name",
        "namespace",
        "ogm_depth",
        "out_slice",
        #"parents",
        # 'predictors_label_clf',
        "predictors_label_preds_test",
        "predictors_label_preds_valid",
        "predictors_label_test_roc",
        "predictors_label_test_spearman_correlation",
        "predictors_label_test_spearman_p",
        "predictors_label_valid_roc",
        "predictors_label_valid_spearman_correlation",
        "predictors_label_valid_spearman_p",
        "predictors_label_valid_rlipp",
        "predictors_label_valid_roc_improvement",
        "predictors_label_valid_spearman_improvement",
        #"relationship",
        #"relationship_rev",
        "width",
    ]

    if interpret_covariates and not label_only:
        for cov in interpret_covariates:
            columns.append(f"predictors_{cov}_clf")
            columns.append(f"predictors_{cov}_preds_test")
            columns.append(f"predictors_{cov}_preds_valid")
            columns.append(f"predictors_{cov}_test_roc")
            columns.append(f"predictors_{cov}_test_spearman_correlation")
            columns.append(f"predictors_{cov}_test_spearman_p")
            columns.append(f"predictors_{cov}_valid_roc")
            columns.append(f"predictors_{cov}_valid_spearman_correlation")
            columns.append(f"predictors_{cov}_valid_spearman_p")

    goobj_csv = pd.DataFrame(columns=columns, index=goobj.keys())

    for go_term in goobj.keys():
        try:
            if go_term.startswith("GO:"):
                goobj_csv.loc[go_term, "_parents"] = goobj[go_term]._parents
            elif go_term.startswith("ENSG"):
                goobj_csv.loc[go_term, "_parents"] = np.nan
            goobj_csv.loc[go_term, "activation_diff"] = goobj[go_term].activation_diff
            goobj_csv.loc[go_term, "activations_test"] = goobj[go_term].activations_test
            goobj_csv.loc[go_term, "activations_valid"] = goobj[go_term].activations_valid
            if go_term.startswith("GO:"):
                goobj_csv.loc[go_term, "alt_ids"] = goobj[go_term].alt_ids
                goobj_csv.loc[go_term, "block_index"] = (
                    goobj[go_term].block_index
                    if isinstance(goobj[go_term].block_index, int)
                    else goobj[go_term].block_index[0]
                )
            elif go_term.startswith("ENSG"):
                goobj_csv.loc[go_term, "alt_ids"] = np.nan
                goobj_csv.loc[go_term, "block_index"] = np.nan
            goobj_csv.loc[go_term, "case_activations"] = goobj[go_term].case_activations
            goobj_csv.loc[go_term, "case_means"] = goobj[go_term].case_means
            #goobj_csv.loc[go_term, "children"] = goobj[go_term].children
            goobj_csv.loc[go_term, "ctrl_activations"] = goobj[go_term].ctrl_activations
            goobj_csv.loc[go_term, "ctrl_means"] = goobj[go_term].ctrl_means
            goobj_csv.loc[go_term, "depth"] = goobj[go_term].depth
            if go_term.startswith("GO:"):
                goobj_csv.loc[go_term, "ensemblids"] = goobj[go_term].ensemblids
                goobj_csv.loc[go_term, "genes_disrupted"] = goobj[go_term].genes_disrupted
                #goobj_csv.loc[go_term, "get_all_child_edges"] = goobj[
                #    go_term
                #].get_all_child_edges
                #goobj_csv.loc[go_term, "get_all_children"] = goobj[go_term].get_all_children
                #goobj_csv.loc[go_term, "get_all_lower"] = goobj[go_term].get_all_lower
                #goobj_csv.loc[go_term, "get_all_parent_edges"] = goobj[
                #    go_term
                #].get_all_parent_edges
                #goobj_csv.loc[go_term, "get_all_parents"] = goobj[go_term].get_all_parents
                #goobj_csv.loc[go_term, "get_all_upper"] = goobj[go_term].get_all_upper
                #goobj_csv.loc[go_term, "get_goterms_lower"] = goobj[
                #    go_term
                #].get_goterms_lower
                # goobj_csv.loc[go_term, 'get_goterms_lower_rels'] = goobj[go_term].get_goterms_lower_rels
                #goobj_csv.loc[go_term, "get_goterms_upper"] = goobj[
                #    go_term
                #].get_goterms_upper
                # goobj_csv.loc[go_term, 'get_goterms_upper_rels'] = goobj[go_term].get_goterms_upper_rels
                # goobj_csv.loc[go_term, 'has_child'] = goobj[go_term].has_child
                # goobj_csv.loc[go_term, 'has_parent'] = goobj[go_term].has_parent
            elif go_term.startswith("ENSG"):
                goobj_csv.loc[go_term, "ensemblids"] = np.nan
                goobj_csv.loc[go_term, "genes_disrupted"] = np.nan
                #goobj_csv.loc[go_term, "get_all_child_edges"] = np.nan
                #goobj_csv.loc[go_term, "get_all_children"] = np.nan
                #goobj_csv.loc[go_term, "get_all_lower"] = np.nan
                #goobj_csv.loc[go_term, "get_all_parent_edges"] = np.nan
                #goobj_csv.loc[go_term, "get_all_parents"] = np.nan
                #goobj_csv.loc[go_term, "get_all_upper"] = np.nan
                #goobj_csv.loc[go_term, "get_goterms_lower"] = np.nan
                # goobj_csv.loc[go_term, 'get_goterms_lower_rels'] = np.nan
                #goobj_csv.loc[go_term, "get_goterms_upper"] = np.nan
                # goobj_csv.loc[go_term, 'get_goterms_upper_rels'] = np.nan
                # goobj_csv.loc[go_term, 'has_child'] = np.nan
                # goobj_csv.loc[go_term, 'has_parent'] = np.nan
            goobj_csv.loc[go_term, "id"] = goobj[go_term].id
            if go_term.startswith("GO:"):
                goobj_csv.loc[go_term, "is_obsolete"] = goobj[go_term].is_obsolete
                goobj_csv.loc[go_term, "item_id"] = goobj[go_term].item_id
                goobj_csv.loc[go_term, "layersize"] = goobj[go_term].layersize
                goobj_csv.loc[go_term, "level"] = goobj[go_term].level
                goobj_csv.loc[go_term, "name"] = goobj[go_term].name
                goobj_csv.loc[go_term, "namespace"] = goobj[go_term].namespace
                goobj_csv.loc[go_term, "ogm_depth"] = goobj[go_term].ogm_depth
                # goobj_csv.loc[go_term, 'out_slice'] = goobj[go_term].out_slice
                #goobj_csv.loc[go_term, "parents"] = goobj[go_term].parents
            elif go_term.startswith("ENSG"):
                goobj_csv.loc[go_term, "is_obsolete"] = np.nan
                goobj_csv.loc[go_term, "item_id"] = np.nan
                goobj_csv.loc[go_term, "layersize"] = np.nan
                goobj_csv.loc[go_term, "level"] = np.nan
                goobj_csv.loc[go_term, "name"] = np.nan
                goobj_csv.loc[go_term, "namespace"] = np.nan
                goobj_csv.loc[go_term, "ogm_depth"] = np.nan
                # goobj_csv.loc[go_term, 'out_slice'] = np.nan
                #goobj_csv.loc[go_term, "parents"] = np.nan
            # goobj_csv.loc[go_term, 'predictors_label_clf'] = goobj[go_term].predictors['label'].clf
            if len(vars(goobj[go_term].predictors["label"]).keys()) > 0:
                goobj_csv.loc[go_term, "predictors_label_preds_test"] = (
                    goobj[go_term].predictors["label"].preds_test
                )
                goobj_csv.loc[go_term, "predictors_label_preds_valid"] = (
                    goobj[go_term].predictors["label"].preds_valid
                )
                goobj_csv.loc[go_term, "predictors_label_test_roc"] = (
                    goobj[go_term].predictors["label"].roc
                )
                goobj_csv.loc[go_term, "predictors_label_test_spearman_correlation"] = (
                    goobj[go_term].predictors["label"].spearman_correlation
                )
                goobj_csv.loc[go_term, "predictors_label_test_spearman_p"] = (
                    goobj[go_term].predictors["label"].spearman_p
                )
                goobj_csv.loc[go_term, "predictors_label_valid_roc"] = (
                    goobj[go_term].predictors["label"].valid_roc
                )
                goobj_csv.loc[go_term, "predictors_label_valid_spearman_correlation"] = (
                    goobj[go_term].predictors["label"].valid_spearman_correlation
                )
                goobj_csv.loc[go_term, "predictors_label_valid_spearman_p"] = (
                    goobj[go_term].predictors["label"].spearman_p
                )
                goobj_csv.loc[go_term, "predictors_label_valid_rlipp"] = (
                    goobj[go_term].predictors["label"].valid_rlipp
                )
                goobj_csv.loc[go_term, "predictors_label_valid_roc_improvement"] = (
                    goobj[go_term].predictors["label"].valid_roc_improvement
                )
                goobj_csv.loc[go_term, "predictors_label_valid_spearman_improvement"] = (
                    goobj[go_term].predictors["label"].valid_spearman_improvement
                )
            else:
                goobj_csv.loc[go_term, "predictors_label_preds_test"] = np.nan
                goobj_csv.loc[go_term, "predictors_label_preds_valid"] = np.nan
                goobj_csv.loc[go_term, "predictors_label_test_roc"] = np.nan
                goobj_csv.loc[
                    go_term, "predictors_label_test_spearman_correlation"
                ] = np.nan
                goobj_csv.loc[go_term, "predictors_label_test_spearman_p"] = np.nan
                goobj_csv.loc[go_term, "predictors_label_valid_roc"] = np.nan
                goobj_csv.loc[
                    go_term, "predictors_label_valid_spearman_correlation"
                ] = np.nan
                goobj_csv.loc[go_term, "predictors_label_valid_spearman_p"] = np.nan
            if interpret_covariates and not label_only:
                for cov in interpret_covariates:
                    goobj_csv.loc[go_term, f"predictors_{cov}_preds_test"] = (
                        goobj[go_term].predictors[cov].preds_test
                    )
                    goobj_csv.loc[go_term, f"predictors_{cov}_preds_valid"] = (
                        goobj[go_term].predictors[cov].preds_valid
                    )
                    goobj_csv.loc[
                        go_term, f"predictors_{cov}_test_spearman_correlation"
                    ] = (goobj[go_term].predictors[cov].spearman_correlation)
                    goobj_csv.loc[go_term, f"predictors_{cov}_test_spearman_p"] = (
                        goobj[go_term].predictors[cov].spearman_p
                    )
                    goobj_csv.loc[
                        go_term, f"predictors_{cov}_valid_spearman_correlation"
                    ] = (goobj[go_term].predictors[cov].valid_spearman_correlation)
                    goobj_csv.loc[go_term, f"predictors_{cov}_valid_spearman_p"] = (
                        goobj[go_term].predictors[cov].spearman_p
                    )
            if go_term.startswith("GO:"):
                goobj_csv.loc[go_term, "width"] = goobj[go_term].width
                #goobj_csv.loc[go_term, "relationship"] = goobj[go_term].relationship
                #goobj_csv.loc[go_term, "relationship_rev"] = goobj[go_term].relationship_rev
            elif go_term.startswith("ENSG"):
                goobj_csv.loc[go_term, "width"] = np.nan
                #goobj_csv.loc[go_term, "relationship"] = np.nan
                #goobj_csv.loc[go_term, "relationship_rev"] = np.nan
        except AttributeError:
            continue

    return goobj_csv


def calculate_node(
    goobj,
    go_term,
    case_idces_valid,
    case_idces_test,
    calculate_roc=False,
    n_max=20000000,
    cv=3,
    Cs=4,
    max_iter=20,
    label_only=True,
    load_save_fit=False,
    save_fit=False,
    overwrite_save_fit=False,
    path_save_fit=None
):
    term_obj = goobj[go_term]

    term_obj.predictors = {}

    def fit_and_predict(
        go_term,
        covariate,
        valid_act,
        test_act,
        case_idces_valid,
        case_idces_test,
        binary,
        prefix,
        load_save_fit,
        save_fit,
        overwrite_save_fit,
        path_save_fit
    ):
        not_nan_selector = ~np.isnan(case_idces_valid[covariate])
        if not_nan_selector.size == 0 and len(case_idces_valid[covariate]) > 0:
            logger.info("Covariate label doesn't exist.")

        if binary:
            if load_save_fit:
                with open(os.path.join(path_save_fit, "go_term_clf", f"{prefix}{go_term}_{covariate}_{binary}.pickle"), 'rb') as file:
                    clf = pickle.load(file)

                term_obj.predictors[covariate].preds_valid = None

            else:
                if cv > 1:
                    clf = LogisticRegressionCV(
                        solver="liblinear",
                        max_iter=max_iter,
                        cv=cv,
                        class_weight="balanced",
                        Cs=Cs,
                    )
                else:
                    clf = LogisticRegression(
                        solver="liblinear", max_iter=max_iter, class_weight="balanced"
                    )
                # grid = {
                #        'depth': [2, 4],  # , 12, 15 Makes it slow
                #        'l2_leaf_reg': [0.5, 1, 3]
                #        }
                # classes = np.unique(case_idces_valid[covariate][not_nan_selector][0:n_max])
                # weights = compute_class_weight(class_weight='balanced', classes=classes, y=case_idces_valid[covariate][not_nan_selector][0:n_max])

                # clf = CatBoostClassifier(iterations=100, devices='5', task_type="GPU", verbose=0
                #                         , od_type = "Iter", od_wait = 10, class_weights=weights)
                # clf.randomized_search(grid,valid_act[not_nan_selector, :][0:n_max]
                #                      , case_idces_valid[covariate][not_nan_selector][0:n_max],n_iter=4)

                # clf = Perceptron(penalty='elasticnet')
                # clf = GridSearchCV(clf, {'alpha': [0.0001, 0.001, 0.01]}, cv=2, verbose=0, scoring="roc_auc", refit=True)

                # clf = MLPClassifier(max_iter=100, hidden_layer_sizes=20, early_stopping=True, activation='logistic')
                # clf = GridSearchCV(clf, {'alpha': [0.0001, 0.001, 0.01]}, cv=2, verbose=0, scoring="roc_auc", refit=True)

                # clf = LogisticRegression(solver='saga', max_iter=max_iter, class_weight='balanced', n_jobs=1)
                clf.fit(
                    valid_act[not_nan_selector, :][0:n_max],
                    case_idces_valid[covariate][not_nan_selector][0:n_max],
                )

                term_obj.predictors[covariate].preds_valid = clf.predict_proba(valid_act)[
                                                             :, 1
                                                             ]

                if save_fit:
                    os.makedirs(os.path.join(path_save_fit, "go_term_clf"), exist_ok=True)
                    if not os.path.isfile(os.path.join(path_save_fit, "go_term_clf", f"{prefix}{go_term}_{covariate}_{binary}.pickle")) or overwrite_save_fit:
                        with open(os.path.join(path_save_fit, "go_term_clf", f"{prefix}{go_term}_{covariate}_{binary}.pickle"), 'wb') as file:
                            pickle.dump(clf, file)

            term_obj.predictors[covariate].preds_test = clf.predict_proba(test_act)[
                :, 1
            ]

        else:
            if load_save_fit:
                with open(os.path.join(path_save_fit, "go_term_clf", f"{prefix}{go_term}_{covariate}_{binary}.pickle"), 'rb') as file:
                    clf = pickle.load(file)
                term_obj.predictors[covariate].preds_valid = None
            else:
                clf = Ridge(max_iter=max_iter).fit(
                    valid_act[not_nan_selector[:, 0], :][0:n_max],
                    case_idces_valid[covariate][not_nan_selector][0:n_max],
                )
                term_obj.predictors[covariate].preds_valid = clf.predict(valid_act)
            if save_fit:
                if not os.path.isfile(os.path.join(path_save_fit, "go_term_clf", f"{prefix}{go_term}_{covariate}_{binary}.pickle")) or overwrite_save_fit:
                    with open(os.path.join(path_save_fit, "go_term_clf", f"{prefix}{go_term}_{covariate}_{binary}.pickle"), 'wb') as file:
                        pickle.dump(clf, file)

            term_obj.predictors[covariate].preds_test = clf.predict(test_act)


        term_obj.predictors[covariate].clf = clf

        def evaluate_predictions(preds, case_idces, attr_prefix=""):
            not_nan = ~np.isnan(case_idces[covariate])

            if binary:
                if preds is not None:
                    roc = roc_auc_score(
                        y_true=case_idces[covariate][not_nan], y_score=preds[not_nan]
                    )
                    setattr(term_obj.predictors[covariate], attr_prefix + "roc", roc)
                else:
                    setattr(term_obj.predictors[covariate], attr_prefix + "roc", None)

            if preds is not None:
                corr = stats.spearmanr(case_idces[covariate][not_nan], preds[not_nan])
                setattr(
                    term_obj.predictors[covariate],
                    attr_prefix + "spearman_correlation",
                    corr[0],
                )
                setattr(term_obj.predictors[covariate], attr_prefix + "spearman_p", corr[1])
            else:
                setattr(
                    term_obj.predictors[covariate],
                    attr_prefix + "spearman_correlation",
                    None,
                )
                setattr(term_obj.predictors[covariate], attr_prefix + "spearman_p", None)

        evaluate_predictions(
            term_obj.predictors[covariate].preds_test, case_idces_test, "" + prefix
        )
        evaluate_predictions(
            term_obj.predictors[covariate].preds_valid,
            case_idces_valid,
            "valid_" + prefix,
        )

        return (
            term_obj.predictors[covariate].preds_test,
            term_obj.predictors[covariate].preds_valid,
        )
        # return  term_obj.predictors[covariate].preds_valid

    for covariate in case_idces_valid:
        if not (covariate == "label") and label_only:
            continue
        binary = case_idces_valid[covariate].dtype == "bool"

        term_obj.predictors[covariate] = Namespace()

        try:
            # fit and evaluate logistic regression for the terms activation
            valid_act = term_obj.activations_valid
            test_act = term_obj.activations_test
            preds_valid = fit_and_predict(
                go_term,
                covariate,
                valid_act,
                test_act,
                case_idces_valid,
                case_idces_test,
                binary,
                "",
                load_save_fit,
                save_fit,
                overwrite_save_fit,
                path_save_fit
            )
        except (ValueError, AttributeError) as e:
            logger.warning(f"Encountered an issue for covariate {covariate}: {e}")
            continue

        try:
            children = [
                goobj[c] for c in term_obj.children
            ]
            if len(children) > 0 and children[0].activations_valid is not None:
                valid_act = np.concatenate(
                    [c.activations_valid for c in children][0:n_max], axis=1
                )
            else:
                valid_act = None
            test_act = np.concatenate(
                [c.activations_test for c in children][0:n_max], axis=1
            )
            fit_and_predict(
                go_term,
                covariate,
                valid_act,
                test_act,
                case_idces_valid,
                case_idces_test,
                binary,
                "child_",
                load_save_fit,
                save_fit,
                overwrite_save_fit,
                path_save_fit
            )
        except (KeyError, AttributeError, ValueError):
            if binary:
                term_obj.predictors[
                    covariate
                ].child_roc = 0.5
                term_obj.predictors[covariate].valid_child_roc = 0.5
            term_obj.predictors[covariate].child_spearman_correlation = 0.0
            term_obj.predictors[covariate].valid_child_spearman_correlation = 0.0

        if binary:
            term_obj.predictors[covariate].roc_improvement = (
                term_obj.predictors[covariate].roc
                - term_obj.predictors[covariate].child_roc
            )

            if term_obj.predictors[covariate].valid_roc is not None:
                term_obj.predictors[covariate].valid_roc_improvement = (
                    term_obj.predictors[covariate].valid_roc
                    - term_obj.predictors[covariate].valid_child_roc
                )
            else:
                term_obj.predictors[covariate].valid_roc_improvement = None

            term_obj.predictors[covariate].rlipp = (
                term_obj.predictors[covariate].roc_improvement
                / term_obj.predictors[covariate].child_roc
            )

            if term_obj.predictors[covariate].valid_roc is not None:
                term_obj.predictors[covariate].valid_rlipp = (
                    term_obj.predictors[covariate].valid_roc_improvement
                    / term_obj.predictors[covariate].valid_child_roc
                )
            else:
                term_obj.predictors[covariate].valid_rlipp = None

        term_obj.predictors[covariate].spearman_improvement = (
            term_obj.predictors[covariate].spearman_correlation
            - term_obj.predictors[covariate].child_spearman_correlation
        )
        if term_obj.predictors[covariate].valid_roc is not None:
            term_obj.predictors[covariate].valid_spearman_improvement = (
                term_obj.predictors[covariate].valid_spearman_correlation
                - term_obj.predictors[covariate].valid_child_spearman_correlation
            )
        else:
            term_obj.predictors[covariate].valid_spearman_improvement = None


def compare_acts(
    goorder,
    goobj,
    case_idces_valid,
    case_idces_test,
    calculate_roc=False,
    n_max=20000000,
    cv=3,
    Cs=4,
    max_iter=100,
    label_only=True,
    load_save_fit=False,
    save_fit=False,
    overwrite_save_fit=False,
    path_save_fit=None
):
    logger.info("Starting fitting and evaluating GO Terms")

    def f(go_term):
        calculate_node(
            goobj,
            go_term,
            case_idces_valid,
            case_idces_test,
            calculate_roc=calculate_roc,
            n_max=n_max,
            cv=cv,
            Cs=Cs,
            max_iter=max_iter,
            label_only=label_only,
            load_save_fit=load_save_fit,
            save_fit=save_fit,
            overwrite_save_fit=overwrite_save_fit,
            path_save_fit=path_save_fit
        )
        return None

    # print('5 jobs.')
    # pqdm(goorder, f, n_jobs=5)
    for go_term in goorder:
        f(go_term)

    return goobj


def set_acts(
    goorder,
    goobj,
    acts_valid,
    acts_test,
    case_idces_valid,
    case_idces_test,
    control_idces_test=None,
):
    if isinstance(case_idces_test, pd.DataFrame):
        case_idces_test = case_idces_test["label"]
    if control_idces_test is None:
        control_idces_test = ~case_idces_test
    logger.info("Setting activations")

    def set_act(go_term):
        term_obj = goobj[go_term]

        depth = term_obj.depth

        term_obj.activations_valid = acts_valid[depth][:, term_obj.out_slice]
        # term_obj.mean_activations_valid = term_obj.activations_valid.mean(axis=0)

        term_obj.activations_test = acts_test[depth][:, term_obj.out_slice]

        # term_obj.mean_activations_test = term_obj.activations_test.mean(axis=0)
        term_obj.width = term_obj.activations_test.shape[1]

        term_obj.case_activations = term_obj.activations_test[case_idces_test, :]
        term_obj.ctrl_activations = term_obj.activations_test[~control_idces_test, :]

        term_obj.case_means = np.mean(term_obj.case_activations, axis=0)
        term_obj.ctrl_means = np.mean(term_obj.ctrl_activations, axis=0)
        term_obj.activation_diff = np.linalg.norm(
            term_obj.case_means - term_obj.ctrl_means, axis=0
        ) / np.sqrt(term_obj.width)

    # set_act(goorder[0]) # for testing purposes only
    pqdm(goorder, set_act, n_jobs=5, disable=True)

    return


class GeneTerm:
    def __init__(self, _id, name, out_slice, depth, children, snp_data):
        self.id = _id
        self.name = name
        self.out_slice = out_slice
        self.depth = depth
        self.children = children
        self.parents = []
        self.snp_data = snp_data


def get_case_indices(results, cov_labels, index_phenotype, interpret_covariates):

    all_covariates = np.concatenate(
        [
            results["logistic_train"]["covariates"][k]
            for k in results["logistic_train"]["covariates"]
            if len(results["logistic_train"]["covariates"][k].shape) != 1
        ],
        axis=1,
    )
    case_idces_valid = {
        "label": results["logistic_train"]["labels"]["categorical"][
            :, index_phenotype
        ].astype(bool)
    }
    for cov in interpret_covariates:
        case_idces_valid[cov] = all_covariates[:, cov_labels == cov]

    all_covariates = np.concatenate(
        [
            results["logistic_eval"]["covariates"][k]
            for k in results["logistic_eval"]["covariates"]
            if len(results["logistic_eval"]["covariates"][k].shape) != 1
        ],
        axis=1,
    )
    case_idces_test = {
        "label": np.array(
            results["logistic_eval"]["labels"]["categorical"][:, index_phenotype]
        ).astype(bool)
    }
    for cov in interpret_covariates:
        case_idces_test[cov] = all_covariates[:, cov_labels == cov]

    return case_idces_valid, case_idces_test


def calculate_interpretations(
    goobj,
    goorder,
    activations_train,
    activations_eval,
    case_idces_train,
    case_idces_eval,
    geneobj=None,
    geneobj_genetic_effects=None,
    geneobj_genetic_effects_input=None,
    geneobj_snps_input=None,
    interpret_covariates=None,
    load_go_terms=None,
    load_genes=None,
    load_gene_groups=None,
    expand_gene_nodes=False,
    label_only=True,
    load_save_fit=False,
    save_fit=False,
    overwrite_save_fit=False,
    path_save_fit=None
):
    if load_go_terms == False:
        goorder = []
        goobj = {}
    elif load_go_terms != "all" and load_go_terms != True:
        if type(load_go_terms) == list:
            go_terms_selection = load_go_terms
        else:
            go_terms_selection = np.load(load_go_terms, allow_pickle=True)
        for go_term in goorder:
            if go_term not in go_terms_selection:
                del goobj[go_term]
                goorder.remove(go_term)
    if load_genes != False and load_genes != "False":
        # geneobj_genetic_effects=None,
        # geneobj_snps_input=None,
        # geneobj_genetic_effects_input=None,
        # Combine gene objects
        if load_genes == True or load_genes == "all":
            """
            for gene in geneobj:
                s = geneobj[gene]["layersize"]
                p = geneobj[gene]["block_index"] * s
                goobj[gene] = GeneTerm(
                    gene, gene, slice(p, p + s), "genes", [], geneobj[gene]["inputs"]
                )
            goorder = list(geneobj.keys()) + goorder
            """
            # activations_on_cpu["genes"], activations_on_cpu["snps_to_genetic_effects"], activations_on_cpu["snps_to_genes"], activations_on_cpu["genetic_effects_to_genes"]
            for geneobj_selected, depth_name in zip(
                [
                    geneobj,
                    geneobj_genetic_effects,
                    geneobj_snps_input,
                    geneobj_genetic_effects_input,
                ],
                [
                    "genes",
                    "snps_to_genetic_effects",
                    "snps_to_genes",
                    "genetic_effects_to_genes",
                ],
            ):
                if geneobj_selected:
                    print("Adding genes to goobj.")
                    for gene in geneobj_selected:
                        s = geneobj_selected[gene]["layersize"]
                        p = geneobj_selected[gene]["block_index"][0] * s
                        if depth_name == "genes":
                            goobj[gene] = GeneTerm(
                                gene,
                                gene,
                                slice(p, p + s),
                                depth_name,
                                [],
                                geneobj_selected[gene]["inputs"],
                            )
                        else:
                            goobj[gene + "_" + depth_name] = GeneTerm(
                                gene + "_" + depth_name,
                                gene + "_" + depth_name,
                                slice(p, p + s),
                                depth_name,
                                [],
                                geneobj_selected[gene]["inputs"],
                            )
                    if depth_name == "genes":
                        goorder = list(geneobj_selected.keys()) + goorder
                    else:
                        goorder = [
                            gene + "_" + depth_name for gene in geneobj_selected.keys()
                        ] + goorder
        else:
            if not geneobj:
                raise NotImplementedError(
                    "Analysis with selected genes only implemented for geneobj, not for variant layer."
                )
            if type(load_genes) == list:
                genes_selection = load_genes
            else:
                genes_selection = list(np.load(load_genes, allow_pickle=True))
            genes_selection_filtered = []
            for gene in geneobj:
                if expand_gene_nodes:
                    try:
                        genes_select = ast.literal_eval(gene)
                        if any(
                            gene_select in genes_selection
                            for gene_select in genes_select
                        ):
                            genes_selection_filtered.append(gene)
                    except:
                        if gene in genes_selection:
                            genes_selection_filtered.append(gene)
                else:
                    if gene in genes_selection:
                        genes_selection_filtered.append(gene)
            for gene in genes_selection_filtered:
                s = geneobj[gene]["layersize"]
                p = geneobj[gene]["block_index"] * s
                goobj[gene] = GeneTerm(
                    gene, gene, slice(p, p + s), "genes", [], geneobj[gene]["inputs"]
                )
            goorder = genes_selection_filtered + goorder
    if load_gene_groups:
        if not geneobj:
            raise NotImplementedError(
                "Analysis with gene groups only implemented for geneobj, not for variant layer."
            )
        geneobj_keys = np.array(list(geneobj.keys()))
        geneobj_keys_to_activation = defaultdict(set)
        for index, gene in enumerate(geneobj):
            if expand_gene_nodes:
                try:
                    genes_select = ast.literal_eval(gene)
                    for gene_select in genes_select:
                        index_gene = np.where(geneobj_keys == gene_select)[0]
                        if len(index_gene) == 1:
                            geneobj_keys_to_activation[gene_select].add(index)
                except:
                    geneobj_keys_to_activation[gene].add(index)
            else:
                geneobj_keys_to_activation[gene].add(index)
        gene_group_names = []
        if type(load_gene_groups) == list:
            gene_groups_selection = load_gene_groups
        else:
            gene_groups_selection = list(np.load(load_gene_groups, allow_pickle=True))
        for index_gene_group, gene_group in enumerate(gene_groups_selection):
            gene_group_name = []
            gene_group_indices = set()
            gene_group_inputs = []
            for gene in gene_group:
                if gene in geneobj_keys_to_activation:
                    gene_group_name.append(gene)
                    gene_group_indices = (
                        gene_group_indices | geneobj_keys_to_activation[gene]
                    )
                    gene_group_inputs += geneobj[gene]["inputs"]
            if len(gene_group_name) > 1:
                gene_group_name_str = "_".join(gene_group_name)
                gene_group_names.append(gene_group_name_str)
                goobj[gene_group_name_str] = GeneTerm(
                    gene_group_name_str,
                    gene_group_name_str,
                    list(gene_group_indices),
                    "genes",
                    gene_group_name,
                    gene_group_inputs,
                )
        goorder = gene_group_names + goorder

    #print("Set acts")
    set_acts(
        goorder,
        goobj,
        activations_train,
        activations_eval,
        case_idces_train,
        case_idces_eval,
    )

    #print("Compare acts")
    goobj = compare_acts(
        goorder,
        goobj,
        case_idces_train,
        case_idces_eval,
        calculate_roc=True,
        n_max=500000,
        cv=3,
        Cs=3,
        max_iter=20,
        label_only=label_only,
        load_save_fit=load_save_fit,
        save_fit=save_fit,
        overwrite_save_fit=overwrite_save_fit,
        path_save_fit=path_save_fit
    )

    return goobj

