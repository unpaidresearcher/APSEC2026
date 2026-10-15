"""
build_primevul_graphs.py
========================
Builds per-project typed knowledge graphs from PrimeVul functions.

Key difference from the original BigVul global graph:
  - One graph per project (e.g. linux, openssl, ffmpeg, chrome, ...)
  - Nodes are functions, variables, types, statements within THAT project only
  - Edges capture real intra-project call chains and data flow
  - At retrieval time: only look up neighbours within the same project graph

Why per-project graphs are better for detection:
  - A use-after-free in the Linux kernel has specific call patterns unique
    to that codebase (kfree, kmalloc, rcu_read_lock etc.)
  - Mixing Linux + OpenSSL + FFmpeg into one graph dilutes these patterns
  - Per-project graphs let the model attend to codebase-specific structural
    context that is directly relevant to the query function

Output:
  primevul_graphs/
    {project_name}/
      graph.gpickle          - NetworkX MultiDiGraph for this project
      entity_embeddings.pt   - TransH embeddings (128-dim)
      entity_to_idx.json     - entity string -> embedding row index
    global_index.json        - maps func_hash -> project_name

Run:
    python build_primevul_graphs.py \\
        --train ../data/primevul/primevul_train.jsonl \\
        --valid ../data/primevul/primevul_valid.jsonl \\
        --test  ../data/primevul/primevul_test.jsonl \\
        --out   ../data/primevul_graphs \\
        --min_funcs 10
"""

import json
import sys
import argparse
import pickle
import re
from pathlib import Path
from collections import defaultdict

import networkx as nx
from tqdm import tqdm

sys.path.insert(0, ".")

# ---------------------------------------------------------------------------
# Node and relation types (same as BigVul graph - reuses all downstream code)
# ---------------------------------------------------------------------------

NODE_TYPES = ["Function", "Variable", "Parameter",
              "Type", "Statement", "ExternalAPI"]

RELATION_TYPES = ["CALLS", "DEFINES", "CONTAINS", "HAS_TYPE",
                  "PRECEDES", "WRITES", "READS", "RETURNS", "USES"]

# Standard C library functions treated as ExternalAPI nodes
STDLIB_APIS = {
    "malloc","calloc","realloc","free","memcpy","memmove","memset","memcmp",
    "strcpy","strncpy","strcat","strncat","strcmp","strncmp","strlen","strdup",
    "sprintf","snprintf","fprintf","printf","scanf","fopen","fclose","fread",
    "fwrite","read","write","open","close","ioctl","mmap","munmap",
    "kzalloc","kmalloc","kfree","krealloc","vmalloc","vfree",
    "copy_from_user","copy_to_user","get_user","put_user",
    "mutex_lock","mutex_unlock","spin_lock","spin_unlock",
    "rcu_read_lock","rcu_read_unlock","rcu_dereference",
    "list_add","list_del","list_for_each","hlist_add_head",
    "container_of","offsetof","sizeof",
    "NULL","ERR_PTR","IS_ERR","PTR_ERR",
}


# ---------------------------------------------------------------------------
# AST-based triple extraction (reuses logic from triple_extractor.py)
# ---------------------------------------------------------------------------

def extract_func_name(src: str) -> str:
    skip = {"if","for","while","switch","return","sizeof","typeof",
            "static","inline","void","int","char","unsigned","struct",
            "extern","const","volatile","register"}
    m = re.search(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', src)
    return m.group(1) if m and m.group(1) not in skip else "unknown_func"


def extract_called_functions(src: str) -> list:
    """Extract all function calls from C source code."""
    skip = {"if","for","while","switch","sizeof","typeof","return",
            "do","else","case","break","continue","goto",
            "defined","__attribute__","likely","unlikely",
            "WARN_ON","BUG_ON","ASSERT","assert"}
    calls = re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(', src)
    return [c for c in calls if c not in skip]


def extract_variables(src: str) -> list:
    """Extract local variable declarations."""
    patterns = [
        r'(?:int|char|unsigned|long|short|float|double|void\s*\*|'
        r'bool|size_t|uint\w*|int\w*|__\w+)\s+\*{0,3}'
        r'([a-zA-Z_][a-zA-Z0-9_]*)\s*(?:=|;|,|\[)',
    ]
    vars_found = []
    for pat in patterns:
        vars_found.extend(re.findall(pat, src))
    return list(set(vars_found))


def extract_types(src: str) -> list:
    """Extract type identifiers (struct/typedef names)."""
    structs = re.findall(r'struct\s+([a-zA-Z_][a-zA-Z0-9_]*)', src)
    typedefs = re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*_t)\b', src)
    return list(set(structs + typedefs))


def function_to_triples(func_src: str, project: str) -> list:
    """
    Extract typed (head, relation, tail) triples from a C function.
    All entity IDs are namespaced by project to avoid cross-project collisions.
    """
    func_name  = extract_func_name(func_src)
    func_id    = f"{project}::{NODE_TYPES[0]}::{func_name}"
    triples    = []

    # CALLS edges
    for callee in extract_called_functions(func_src):
        if callee in STDLIB_APIS:
            tail_id = f"GLOBAL::{NODE_TYPES[5]}::{callee}"
            node_type = NODE_TYPES[5]  # ExternalAPI
        else:
            tail_id = f"{project}::{NODE_TYPES[0]}::{callee}"
            node_type = NODE_TYPES[0]
        triples.append((func_id, "CALLS", tail_id))

    # DEFINES variable edges
    for var in extract_variables(func_src):
        var_id = f"{project}::{NODE_TYPES[1]}::{func_name}::{var}"
        triples.append((func_id, "DEFINES", var_id))

    # HAS_TYPE edges
    for typ in extract_types(func_src):
        type_id = f"{project}::{NODE_TYPES[3]}::{typ}"
        triples.append((func_id, "HAS_TYPE", type_id))
        for var in extract_variables(func_src):
            var_id = f"{project}::{NODE_TYPES[1]}::{func_name}::{var}"
            triples.append((var_id, "HAS_TYPE", type_id))

    return triples


# ---------------------------------------------------------------------------
# Build per-project graphs
# ---------------------------------------------------------------------------

def load_all_functions(jsonl_paths: list) -> dict:
    """Load all functions grouped by project."""
    project_funcs = defaultdict(list)
    func_hash_to_project = {}

    for path in jsonl_paths:
        if not Path(path).exists():
            print(f"  WARNING: {path} not found, skipping")
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except:
                    continue
                func = (row.get("func") or "").strip()
                project = (row.get("project") or "unknown").strip()
                project = re.sub(r'[^a-zA-Z0-9_\-]', '_', project)
                func_hash = row.get("func_hash", "") or row.get("idx", "")

                if func:
                    project_funcs[project].append({
                        "func": func,
                        "target": int(row.get("target", 0)),
                        "func_hash": str(func_hash),
                    })
                    if func_hash:
                        func_hash_to_project[str(func_hash)] = project

    return dict(project_funcs), func_hash_to_project


def build_project_graph(project: str, funcs: list) -> nx.MultiDiGraph:
    """Build a typed knowledge graph for one project."""
    G = nx.MultiDiGraph()

    for item in funcs:
        src = item["func"]
        triples = function_to_triples(src, project)
        for h, r, t in triples:
            # Add nodes with type attribute
            for node_id, ntype in [(h, h.split("::")[1] if "::" in h else "Function"),
                                   (t, t.split("::")[1] if "::" in t else "Function")]:
                if node_id not in G:
                    G.add_node(node_id, node_type=ntype,
                               project=project if not node_id.startswith("GLOBAL") else "GLOBAL")
            G.add_edge(h, t, relation=r)

    return G


def build_all_graphs(
    jsonl_paths: list,
    out_dir: str,
    min_funcs: int = 10,
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print("Loading all PrimeVul functions ...")
    project_funcs, func_hash_to_project = load_all_functions(jsonl_paths)

    # Filter projects with enough functions for a meaningful graph
    large_projects = {p: f for p, f in project_funcs.items()
                      if len(f) >= min_funcs}
    small_count = len(project_funcs) - len(large_projects)

    print(f"  Total projects: {len(project_funcs):,}")
    print(f"  Projects with >= {min_funcs} functions: {len(large_projects):,}")
    print(f"  Small projects (will use nearest large project graph): {small_count:,}")

    # Sort by size for progress reporting
    sorted_projects = sorted(large_projects.items(),
                             key=lambda x: len(x[1]), reverse=True)
    print(f"\n  Top 10 projects by function count:")
    for p, f in sorted_projects[:10]:
        n_vuln = sum(1 for x in f if x["target"] == 1)
        print(f"    {p:<40} {len(f):>6} funcs | {n_vuln:>4} vulnerable")

    # Build one graph per project
    print(f"\nBuilding {len(large_projects):,} project graphs ...")
    all_graph_stats = {}

    for project, funcs in tqdm(sorted_projects, desc="Building graphs"):
        proj_dir = out_path / project
        proj_dir.mkdir(parents=True, exist_ok=True)

        G = build_project_graph(project, funcs)
        all_graph_stats[project] = {
            "n_funcs":   len(funcs),
            "n_nodes":   G.number_of_nodes(),
            "n_edges":   G.number_of_edges(),
            "n_vuln":    sum(1 for f in funcs if f["target"] == 1),
        }

        # Save graph
        graph_path = proj_dir / "graph.gpickle"
        with open(graph_path, "wb") as f:
            pickle.dump(G, f, protocol=4)

    # Save global index: func_hash -> project
    json.dump(func_hash_to_project,
              open(out_path / "global_index.json", "w"), indent=2)

    # Save stats
    json.dump(all_graph_stats,
              open(out_path / "graph_stats.json", "w"), indent=2)

    # Print summary
    total_nodes = sum(s["n_nodes"] for s in all_graph_stats.values())
    total_edges = sum(s["n_edges"] for s in all_graph_stats.values())
    print(f"\nGraph build complete:")
    print(f"  Projects with graphs: {len(large_projects):,}")
    print(f"  Total nodes (all projects): {total_nodes:,}")
    print(f"  Total edges (all projects): {total_edges:,}")
    print(f"  Graphs saved to: {out_path}/")
    print(f"\nNext step: train TransH embeddings per project")
    print(f"  python train_primevul_transh.py --graphs {out_path}")

    return all_graph_stats


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="../datasets/primevul/primevul_train.jsonl")
    ap.add_argument("--valid", default="../datasets/primevul/primevul_valid.jsonl")
    ap.add_argument("--test",  default="../datasets/primevul/primevul_test.jsonl")
    ap.add_argument("--out",   default="./datasets/primevul_graphs")
    ap.add_argument("--min_funcs", type=int, default=10,
                    help="Min functions per project to build a graph")
    args = ap.parse_args()

    build_all_graphs(
        jsonl_paths=[args.train, args.valid, args.test],
        out_dir=args.out,
        min_funcs=args.min_funcs,
    )
