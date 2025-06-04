import warnings
from addict import Dict as AttrDict
from collections import defaultdict
from typing import Callable, Optional, List, Dict, Union
from goatools.obo_parser import GODag, GOTerm
import json
import os
import pickle as pkl
from collections import Counter, defaultdict
from typing import Sequence

import numpy as np
import pandas as pd
from goatools.anno.gaf_reader import GafReader
from goatools.obo_parser import GODag
from prettytable import PrettyTable

ensembl2go = None

def map2gos_v2(
    ensemblids_input,
    genetic_json,
    map_ensembl_go,
    ontology,
    obo_file,
    gene_transcript,
    logger,
    remove_redundant_parents=False,
    gene_type_dict=None,
    transcript_type_dict=None,
):

    global ensembl2go
    if ontology is None:
        ontology = ""
    else:
        ontology = "_" + ontology.split("/")[-1].split(".")[0]

    if obo_file is None:
        obo_file_str = ""
    else:
        obo_file_str = "_" + obo_file.split("/")[-1].split(".")[0]

    if map_ensembl_go is None:
        map_ensembl_go_str = ""
    else:
        if isinstance(map_ensembl_go, str):
            map_ensembl_go_str = "_" + map_ensembl_go.split("/")[-1].split(".")[0]
        elif isinstance(map_ensembl_go, Sequence):
            map_ensembl_go_str = ""
            for map in map_ensembl_go:
                map_ensembl_go_str += "_" + map.split("/")[-1].split(".")[0]

    if ensembl2go is None:
        print(
            genetic_json
            + ontology
            + obo_file_str
            + map_ensembl_go_str
            + ".v2_processed.pkl"
        )
        if os.path.isfile(
            genetic_json
            + ontology
            + obo_file_str
            + map_ensembl_go_str
            + ".v2_processed.pkl"
        ):
            with open(
                genetic_json
                + ontology
                + obo_file_str
                + map_ensembl_go_str
                + ".v2_processed.pkl",
                "rb",
            ) as ifile:
                ensembl2go = pkl.load(ifile)
        else:
            ensembl2go_list = []

            for map_ensembl in map_ensembl_go:
                objanno = GafReader(map_ensembl)
                ensembl2go_list.append(objanno.get_id2gos_nss())

            ensembl2go = defaultdict(set)

            for d in ensembl2go_list:
                for key, value in d.items():
                    ensembl2go[key] = ensembl2go[key] | value

            if remove_redundant_parents:
                obodag = GODag(obo_file)

                total_go_term_anno = 0
                total_go_term_anno_after_parent_removal = 0
                total_number_gene_parents_removal = 0
                total_number_genes = len(ensembl2go.keys())
                for eid in ensembl2go.keys():
                    parents_go_terms = set()
                    for go_term in ensembl2go[eid]:
                        try:
                            parents_go_terms = (
                                parents_go_terms | obodag[go_term].get_all_parents()
                            )
                        except KeyError:
                            continue
                    total_go_term_anno_eid = len(ensembl2go[eid])
                    ensembl2go[eid] = ensembl2go[eid] - parents_go_terms
                    total_go_term_anno_after_parent_removal_eid = len(ensembl2go[eid])
                    if (
                        total_go_term_anno_eid
                        != total_go_term_anno_after_parent_removal_eid
                    ):
                        total_number_gene_parents_removal += 1
                    total_go_term_anno += total_go_term_anno_eid
                    total_go_term_anno_after_parent_removal += (
                        total_go_term_anno_after_parent_removal_eid
                    )

                logger.info(
                    f"Of {total_number_genes} genes with a total number of {total_go_term_anno} GO annotations in {total_number_gene_parents_removal} genes {total_go_term_anno - total_go_term_anno_after_parent_removal} GO annotations have been removed."
                )

            with open(
                genetic_json
                + ontology
                + obo_file_str
                + map_ensembl_go_str
                + ".v2_processed.pkl",
                "wb",
            ) as f:
                pkl.dump(ensembl2go, f)


    go_list = []

    for set_of_ensemblids in ensemblids_input:
        if (
            len(set_of_ensemblids) > 0 and set_of_ensemblids[0] == "E"
        ):  # backwards compatibilty, single ensembl id
            gos = ensembl2go.get(set_of_ensemblids, set())

        else:
            gos = set()
            for ensemblid in set_of_ensemblids[1:-1].split(", "):
                gos = gos.union(ensembl2go.get(ensemblid.strip("'"), set()))

        go_list.append(gos)

    return np.array(go_list)

def lstrip_multiline(x):
    lines = x.split("\n")
    lines_stripped = [l.lstrip() for l in lines]
    return "\n".join(lines_stripped)

def filter_goobj(
    goobj,
    condition: Callable[[str], bool],
    remove_obsolete_refs=True,
    bubble_up_ensemblids=True,
):
    """Removes GO terms from the GO object inplace if they satisfy the condition provided.

    Parameters:
    -----------
    goobj: GODag
    condition: Callable[[str], bool]
      A function that takes a GO id and returns True if the respective GOTerm should be removed from the GO object.
    remove_obsolute_refs: bool, optional
      If set, removes connections to children that are not contained in the GO object.
    bubble_up_eids: bool, optional
      If set and GOTerm has 'ensemblid' property, bubbles up ensemblids to parent nodes.
      E.g. consider the following set of GOTerms in the goobj:

      p1: {children: {c1, c2}, ensemblids: {eid3}}
      p2: {children: {c2, c3}, ensemblids: {}}
      c1: {children: {}, ensemblids: {eid1, eid2}}
      c2: {children: {}, ensemblids: {eid2, eid3}}
      c3: {children: {}, ensemblids: {eid4}}

      Upon deletion of c2 from the goobj, the goobj will be updated to:

      p1: {children: {c1}, ensemblids: {eid2, eid3}}
      p2: {children: {c2, c3}, ensemblids: {eid2, eid3}}
      c1: {children: {}, ensemblids: {eid1, eid2}}
      c3: {children: {}, ensemblids: {eid4}}

    """
    del_list = []
    order = sorted(list(goobj.keys()), key=lambda x: -goobj[x].depth)
    for goid in order:
        if condition(goid):
            goterm = goobj[goid]
            del_list.append(goid)
            for p in goterm.parents:
                if hasattr(p, "ensemblids") and bubble_up_ensemblids:
                    p.ensemblids = p.ensemblids | goterm.ensemblids
    # Delete all marked terms from the goobj
    for goid in del_list:
        del goobj[goid]
    # Remove references pointing to GOTerms no longer in the goobj
    if remove_obsolete_refs:
        prune_obsolete_refs(goobj)


def prune_obsolete_refs(goobj):
    order = sorted(list(goobj.keys()), key=lambda x: -goobj[x].depth)
    for goid in order:
        if goid in goobj:
            goterm = goobj[goid]
            for c in list(goterm.children):
                if not c.id in goobj:
                    goterm.children.remove(c)
            for p in list(goterm.parents):
                if not p.id in goobj:
                    goterm.parents.remove(p)
            if hasattr(goterm, "relationship_rev"):
                relations_rev = goterm.relationship_rev
                for rel in relations_rev:
                    for c in list(relations_rev[rel]):
                        if not c.id in goobj:
                            relations_rev[rel].remove(c)
            if hasattr(goterm, "relationship"):
                relations = goterm.relationship
                for rel in relations:
                    for c in list(relations[rel]):
                        if not c.id in goobj:
                            relations[rel].remove(c)


def set_ogm_depths(
    goobj: GODag,
    is_a: Optional[bool] = True,
    part_of: Optional[bool] = True,
    regulates: Optional[bool] = True,
    regulates_pos: Optional[bool] = True,
    regulates_neg: Optional[bool] = True,
):
    """Sets the depths (maximum depth from root) of all nodes in the GODag, accounting for all relationships (unless specified otherwise).
    The GOTerms will be annotated with a new attribute `depth`.

    Parameters:
    -----------
    goobj: GODag
      The graph for which to annotate the depths.

    """
    rel_flags = AttrDict(
        {
            "is_a": is_a,
            "part_of": part_of,
            "regulates": regulates,
            "regulates_pos": regulates_pos,
            "regulates_neg": regulates_neg,
        }
    )

    def _init_depth(rec, **rel_flags):
        if not hasattr(rec, "ogm_depth"):
            all_parents = get_all_parents(rec, **rel_flags)
            if all_parents:
                depth = 0
                for rec_ in all_parents:
                    depth = max(_init_depth(rec_, **rel_flags), depth)
                rec.ogm_depth = depth + 1
            else:
                rec.ogm_depth = 0
        return rec.ogm_depth

    for rec in goobj.values():
        if not hasattr(rec, "ogm_depth"):
            _init_depth(rec, **rel_flags)


def set_heights(
    goobj: GODag,
    is_a: Optional[bool] = True,
    part_of: Optional[bool] = True,
    regulates: Optional[bool] = True,
    regulates_pos: Optional[bool] = True,
    regulates_neg: Optional[bool] = True,
):
    """Calculates and sets the height of all GOTerms in the GODag graph.
    The GOTerms will be annotated with a new attribute `height`.

    Parameters:
    -----------
    go: GOTerm
      The GOTerm from which to extract all children.
    is_a: bool, optional
      Specifies if 'is_a' relationships should be used, if available. This is the default relationship used in the Gene Ontology.
    part_of: bool, optional
      Specifies if 'part_of' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'part_of' relationship, this means 'GO2 is part of GO1'.
    regulates: bool, optional
      Specifies if 'regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'regulates' relationship, this means 'GO2 necessarily regulates GO1.'
    regulates_pos: bool, optional
      Specifies if 'positively_regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'positively_regulates' relationship, this means 'GO2 necessarily positively regulates GO1.'
    regulates_neg: bool, optional
      Specifies if 'negatively_regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'negatively_regulates' relationship, this means 'GO2 necessarily negativley regulates GO1.'

    Returns:
    --------
    A list of GOTerm children.

    """
    rel_flags = AttrDict(
        {
            "is_a": is_a,
            "part_of": part_of,
            "regulates": regulates,
            "regulates_pos": regulates_pos,
            "regulates_neg": regulates_neg,
        }
    )

    def _init_height(rec, **rel_flags):
        if not hasattr(rec, "height"):
            all_children = get_all_children(rec, **rel_flags)
            if all_children:
                height = 0
                for rec_ in all_children:
                    height = max(_init_height(rec_, **rel_flags), height)
                rec.height = height + 1
            else:
                rec.height = 0
        return rec.height

    for rec in goobj.values():
        if not hasattr(rec, "height"):
            _init_height(rec, **rel_flags)


def extend_godag(goobj: GODag):
    """Replaces all references to GOTerm objects in the GODag provided with GOTermExtended instances inplace."""
    # Replace all GOTerms in goobj
    for goid in goobj:
        goobj[goid] = GOTermExtended.from_goterm(goobj[goid])
    # Replace references in goobj with new ones
    for goid in goobj:
        gox = goobj[goid]
        if gox.children:
            gox.children = {goobj[c.id] for c in gox.children}
        if gox.parents:
            gox.parents = {goobj[p.id] for p in gox.parents}
        REL_KEYS = [
            "part_of",
            "regulates",
            "positively_regulates",
            "negatively_regulates",
        ]
        for rel in REL_KEYS:
            if hasattr(gox, "relationship") and (rel in gox.relationship):
                gox.relationship[rel] = {goobj[c.id] for c in gox.relationship[rel]}
            if hasattr(gox, "relationship_rev") and (rel in gox.relationship_rev):
                gox.relationship_rev[rel] = {
                    goobj[p.id] for p in gox.relationship_rev[rel]
                }


def set_genes_disrupted(
    goobj: GODag,
    is_a: Optional[bool] = True,
    part_of: Optional[bool] = True,
    regulates: Optional[bool] = True,
    regulates_pos: Optional[bool] = True,
    regulates_neg: Optional[bool] = True,
):
    """Sets the genes_disrupted attribute for every GOTerm in the GO object which counts the number of genes below said GOTerm."""
    rel_flags = AttrDict(
        {
            "is_a": is_a,
            "part_of": part_of,
            "regulates": regulates,
            "regulates_pos": regulates_pos,
            "regulates_neg": regulates_neg,
        }
    )
    order = sorted(list(goobj.keys()), key=lambda x: -goobj[x].ogm_depth)
    for gokey in order:
        goobj[gokey].genes_disrupted = set()
    for gokey in order:
        go = goobj[gokey]
        go.genes_disrupted = go.genes_disrupted | go.ensemblids
        all_children = get_all_children(go, **rel_flags)
        for c in all_children:
            go.genes_disrupted = go.genes_disrupted | c.genes_disrupted


def get_leafs(goobj):
    return list(
        filter(
            lambda x: len(map(lambda y: y.id, goobj[x].children) & goobj.keys()) == 0,
            goobj,
        )
    )


def get_parents(gos):
    if isinstance(gos, dict):
        gos = set(gos.values())
    results = gos
    if sum([len(go.parents) for go in gos]) == 0:
        return results
    for go in results:
        results = results | get_parents(go.parents)
    return results


def get_height_dist(goobj: GODag) -> Dict[int, int]:
    height_histogram = defaultdict(int)
    for goid in goobj:
        go = goobj[goid]
        if hasattr(go, "height"):
            height_histogram[go.height] += 1
        else:
            warnings.warn(f"GOTerm {goid} has no attribute `height`!")
    return height_histogram


def get_depth_dist(goobj: GODag) -> Dict[int, int]:
    depth_histogram = defaultdict(int)
    for goid in goobj:
        go = goobj[goid]
        if hasattr(go, "depth"):
            depth_histogram[go.depth] += 1
        else:
            warnings.warn(f"GOTerm {goid} has no attribute `depth`!")
    return depth_histogram


def get_ogm_depth_dist(goobj: GODag) -> Dict[int, int]:
    depth_histogram = defaultdict(int)
    for goid in goobj:
        go = goobj[goid]
        if hasattr(go, "ogm_depth"):
            depth_histogram[go.ogm_depth] += 1
        else:
            warnings.warn(f"GOTerm {goid} has no attribute `ogm_depth`!")
    return depth_histogram


def get_ngenes_cumsum(goobj: GODag) -> Dict[int, int]:
    """Calculates the cumulative number of genes below each height level and returns them as dict mapping height -> ngenes."""
    order = sorted(list(goobj.keys()), key=lambda x: goobj[x].height)
    ngenes_cumsum = defaultdict(int)
    for goid in order:
        go = goobj[goid]
        ngenes_cumsum[go.height] += len(go.genes_disrupted)
    return ngenes_cumsum


def get_nchildren_cumsum(
    goobj: GODag,
    is_a: Optional[bool] = True,
    part_of: Optional[bool] = False,
    regulates: Optional[bool] = False,
    regulates_pos: Optional[bool] = False,
    regulates_neg: Optional[bool] = False,
) -> Dict[int, int]:
    """Calculates the cumulative number of children below each height level and returns them as dict mapping height -> nchildren."""
    rel_flags = AttrDict(
        {
            "is_a": is_a,
            "part_of": part_of,
            "regulates": regulates,
            "regulates_pos": regulates_pos,
            "regulates_neg": regulates_neg,
        }
    )
    order = sorted(list(goobj.keys()), key=lambda x: goobj[x].height)
    nchildren_cumsum = defaultdict(int)
    for goid in order:
        go = goobj[goid]
        all_children = get_all_children(go, **rel_flags)
        nchildren_cumsum[go.height] += len(all_children)
    return nchildren_cumsum


def get_all_children(
    go: GOTerm,
    is_a: Optional[bool] = True,
    part_of: Optional[bool] = True,
    regulates: Optional[bool] = True,
    regulates_pos: Optional[bool] = True,
    regulates_neg: Optional[bool] = True,
    as_dict: Optional[bool] = False,
    ret_empty: Optional[bool] = False,
) -> Union[List[GOTerm], Dict[str, GOTerm]]:
    """Given a GO Term, returns children over all relationships, unless specified otherwise.

    Parameters:
    -----------
    go: GOTerm
      The GOTerm from which to extract all children.
    is_a: bool, optional
      Specifies if 'is_a' relationships should be used, if available. This is the default relationship used in the Gene Ontology.
    part_of: bool, optional
      Specifies if 'part_of' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'part_of' relationship, this means 'GO2 is part of GO1'.
    regulates: bool, optional
      Specifies if 'regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'regulates' relationship, this means 'GO2 necessarily regulates GO1.'
    regulates_pos: bool, optional
      Specifies if 'positively_regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'positively_regulates' relationship, this means 'GO2 necessarily positively regulates GO1.'
    regulates_neg: bool, optional
      Specifies if 'negatively_regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'negatively_regulates' relationship, this means 'GO2 necessarily negativley regulates GO1.'
    as_dict: bool, optional
      If True, returns a dict with GOTerms assigned to their respective relation key.
      Otherweise, returns a list.
    ret_empty: bool, optional
      If True, returns an empty list for relation keys for which the GOTerm has no children (or all are filtered).
      NOTE: This parameter only has an effect if as_dict is set to True.

    Returns:
    --------
    A list of GOTerm children.

    """
    if not isinstance(go, GOTerm):
        raise TypeError(f"Expeected GOTerm, but got object of type {type(go)}!")

    all_children = {} if as_dict else []

    if is_a:
        if as_dict:
            all_children["is_a"] = list(go.children)
        else:
            all_children += list(go.children)

    if hasattr(go, "relationship_rev"):
        arg_key_pairs = [
            (part_of, "part_of"),
            (regulates, "regulates"),
            (regulates_pos, "positively_regulates"),
            (regulates_neg, "negatively_regulates"),
        ]

        for arg, key in arg_key_pairs:
            if arg and key in go.relationship_rev:
                if as_dict:
                    all_children[key] = list(go.relationship_rev[key])
                else:
                    all_children += list(go.relationship_rev[key])
            elif as_dict and ret_empty:
                all_children[key] = []

    return all_children


def get_all_parents(
    go: GOTerm,
    is_a: Optional[bool] = True,
    part_of: Optional[bool] = True,
    regulates: Optional[bool] = True,
    regulates_pos: Optional[bool] = True,
    regulates_neg: Optional[bool] = True,
    as_dict: Optional[bool] = False,
    ret_empty: Optional[bool] = False,
) -> Union[List[GOTerm], Dict[str, GOTerm]]:
    """Given a GO Term, returns parents over all relationships, unless specified otherwise.

    Parameters:
    -----------
    go: GOTerm
      The GOTerm from which to extract all children.
    is_a: bool, optional
      Specifies if 'is_a' relationships should be used, if available. This is the default relationship used in the Gene Ontology.
    part_of: bool, optional
      Specifies if 'part_of' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'part_of' relationship, this means 'GO2 is part of GO1'.
    regulates: bool, optional
      Specifies if 'regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'regulates' relationship, this means 'GO2 necessarily regulates GO1.'
    regulates_pos: bool, optional
      Specifies if 'positively_regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'positively_regulates' relationship, this means 'GO2 necessarily positively regulates GO1.'
    regulates_neg: bool, optional
      Specifies if 'negatively_regulates' relationships should be used, if available.
      NOTE: If GO1 has a child GO2 over the 'negatively_regulates' relationship, this means 'GO2 necessarily negativley regulates GO1.'
    as_dict: bool, optional
      If True, returns a dict with GOTerms assigned to their respective relation key.
      Otherwise, returns a list.
    ret_empty: bool, optional
      If True, returns an empty list for relation keys for which the GOTerm has no children (or all are filtered).
      NOTE: This parameter only has an effect if as_dict is set to True.

    Returns:
    -----------
    A list of GOTerm parents.

    """
    if not isinstance(go, GOTerm):
        raise TypeError(f"Expeected GOTerm, but got object of type {type(go)}!")

    all_parents = {} if as_dict else []

    if is_a:
        if as_dict:
            all_parents["is_a"] = list(go.parents)
        else:
            all_parents += list(go.parents)

    if hasattr(go, "relationship"):
        arg_key_pairs = [
            (part_of, "part_of"),
            (regulates, "regulates"),
            (regulates_pos, "positively_regulates"),
            (regulates_neg, "negatively_regulates"),
        ]

        for arg, key in arg_key_pairs:
            if arg and key in go.relationship:
                if as_dict:
                    all_parents[key] = list(go.relationship[key])
                else:
                    all_parents += list(go.relationship[key])
            elif as_dict and ret_empty:
                all_parents[key] = []

    return all_parents


class GOTermExtended(GOTerm):
    def __init__(self):
        super(type(self), self).__init__()

    def __repr__(self):
        """Print GO ID and all attributes in GOTerm class."""
        ret = ["GOTermExtended('{ID}'):".format(ID=self.item_id)]
        for key, val in self.__dict__.items():
            if isinstance(val, int) or isinstance(val, str):
                ret.append("{K}:{V}".format(K=key, V=val))
            elif val is not None:
                if key == "children" or key == "parents":
                    ret.append(f"{key}: ")
                    for go in val:
                        ret.append(f"GOTermExtended('{go.item_id}'')")
                else:
                    representation = repr(val)
                    ret.append(f"{key}: {representation}")
            else:
                ret.append("{K}: None".format(K=key))
        return "\n  ".join(ret)

    @classmethod
    def from_goterm(cls, goterm: GOTerm):
        """Copies all attributes of GOTerm object and create ExtendedGOTerm object"""
        goterm_ext = GOTermExtended()
        for key, val in goterm.__dict__.items():
            setattr(goterm_ext, key, val)
        return goterm_ext
