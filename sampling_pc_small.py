import time
import pickle
import random
import numpy as np
from collections import defaultdict
from os.path import join, dirname
import networkx as nx
from paths_graph import CFPG, get_reachable_sets, PathsGraph, PathsTree, \
                        CombinedPathsGraph
from matplotlib import pyplot as plt

output_dir = join(dirname(__file__), 'output')

def run_pg_vs_nx(graph, source, target, depth):
    # PG sampling
    start = time.time()
    f_level, b_level = get_reachable_sets(graph, source, target, depth)
    pg_list = []
    for i in range(1, depth+1):
        pg = PathsGraph.from_graph(graph, source, target, i, f_level,
                                   b_level)
        pg_list.append(pg)
    combined_pg = CombinedPathsGraph(pg_list)
    cf_paths = combined_pg.sample_cf_paths(1000)
    end = time.time()
    print("Done sampling from PG")
    pg_elapsed = end - start

    # Networkx enumeration
    index = 0
    start = time.time()
    nx_paths = []
    for p in nx.all_simple_paths(graph, source, target, cutoff=depth):
        nx_paths.append(tuple(p))
        if index % 10000 == 0:
            print(index)
        index += 1
    paths_tree = PathsTree(nx_paths)
    nx_sampled_paths = paths_tree.sample(1000)
    end = time.time()
    nx_elapsed = end - start
    assert set(cf_paths) <= set(nx_paths)
    print("all_simple_paths done")
    print("Total paths (nx):", len(nx_paths))
    print("Unique sampled paths (pg):", len(set(cf_paths)))
    print("Unique sampled_paths (tree):", len(set(nx_sampled_paths)))
    print("NX time", nx_elapsed)
    print("PG time", pg_elapsed)
    return {'pg_paths': cf_paths, 'nx_paths': nx_sampled_paths,
            'pg_time': pg_elapsed, 'nx_time': nx_elapsed}


def run_timing_comparison(min_depth, max_depth, num_reps):
    depths = list(range(min_depth, max_depth + 1))
    results = np.empty((2, len(depths), num_reps))
    genes = graph.nodes()
    for depth_ix, depth in enumerate(depths):
        for rep_ix in range(NUM_REPS):
            source = random.choice(genes)
            target = random.choice(genes)
            print("depth", depth, "rep", rep_ix+1, "source", source, "target",
                  target)
            res_dict = run_pg_vs_nx(graph, source, target, depth)
            results[0, depth_ix, rep_ix] = res_dict['nx_time']
            results[1, depth_ix, rep_ix] = res_dict['pg_time']
    # Plot results
    plt.ion()
    plt.figure()
    nx_means = results.mean(axis=2)[0,:]
    cfpg_means = results.mean(axis=2)[1, :]
    nx_stds = results.std(axis=2)[0,:]
    cfpg_stds = results.std(axis=2)[1, :]
    plt.errorbar(depths, nx_means, yerr=nx_stds)
    plt.errorbar(depths, cfpg_means, yerr=cfpg_stds)
    plt.gca().set_yscale('log')


def get_node_distribution(paths):
    dist = defaultdict(lambda: 0)
    for path in paths:
        for node in path:
            dist[node] += 1
    pcts = {}
    for node, val in dist.items():
        pcts[node] = val / float(len(paths))
    return pcts


if __name__ == '__main__':
    random.seed(1)
    filename = join(output_dir, 'pc_digraph_small.pkl')
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    # Timing comparison
    MAX_DEPTH = 5
    NUM_REPS = 20
    run_timing_comparison(1, MAX_DEPTH, NUM_REPS)
