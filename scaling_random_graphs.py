import time
import itertools
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from paths_graph import PathsGraph, get_reachable_sets, CFPG, \
                        CombinedPathsGraph, PathsTree

seed = 1

def scaling_random_graphs(num_samples, min_size, max_size, edge_prob=0.5):
    data_shape =  (max_size - min_size + 1, num_samples)
    times_nx_paths = np.empty(data_shape)
    times_pg = np.empty(data_shape)
    times_cfpg = np.empty(data_shape)
    for i, num_nodes in enumerate(range(min_size, max_size+1)):
        print(num_nodes)
        for j in range(num_samples):
            print("Sample", j)
            # Generate a random graph
            rg = nx.erdos_renyi_graph(num_nodes, edge_prob, directed=True)
            source = 0
            target = num_nodes - 1
            # Time to compute all simple paths with path probabilities
            start = time.time()
            paths = list(nx.all_simple_paths(rg, source, target))
            pt = PathsTree(paths)
            path_probs = pt.path_probabilities()
            end = time.time()
            elapsed = end - start
            times_nx_paths[i, j] = elapsed
            # Time to compute paths_graphs
            pg_start = time.time()
            f_level, b_level = get_reachable_sets(rg, source, target, num_nodes)
            pg_list = []
            for length in range(1, num_nodes):
                pg = PathsGraph.from_graph(rg, source, target, length,
                                           f_level, b_level)
                pg_list.append(pg)
            combined_pg = CombinedPathsGraph(pg_list)
            total_paths = combined_pg.count_paths()
            print("Total paths (with cycles)", total_paths)
            #cf_paths = combined_pg.sample_cf_paths(100000)
            pg_elapsed = time.time() - pg_start
            times_pg[i, j] = pg_elapsed
            # Now compute the CFPG
            #cfpg_list = []
            #for pg in pg_list:
            #    cfpg = CFPG.from_pg(pg)
            #    cfpg_list.append(cfpg)
            #cfpg_elapsed = time.time() - pg_start
            #times_cfpg[i, j] = cfpg_elapsed
    return times_nx_paths, times_pg, times_cfpg


def prob_ascending_path(paths):
    count_ascending = 0
    if isinstance(paths, list):
        weighted_paths = list(itertools.product(paths, [1.]))
        norm_factor = float(len(paths))
    elif isinstance(paths, dict):
        weighted_paths = [(k, v) for k, v in paths.items()]
        norm_factor = 1.
    for path, weight in weighted_paths:
        last_node = path[0]
        ascending = True
        for node in path[1:]:
            if node < last_node:
                ascending = False
                break
            last_node = node
        if ascending:
            count_ascending += weight
    return count_ascending / norm_factor


# Generate a random graph
if __name__ == '__main__':
    num_nodes = 16
    rg = nx.erdos_renyi_graph(num_nodes, 0.5, directed=True)
    source = 0
    target = num_nodes - 1
    # Time to compute all simple paths with path probabilities
    start = time.time()
    print("Enumerating paths")
    paths = list(nx.all_simple_paths(rg, source, target))
    print("Done enumerating paths")
    pt = PathsTree(paths)
    path_probs = pt.path_probabilities()
    end = time.time()
    elapsed = end - start
    # Time to compute paths_graphs
    pg_start = time.time()
    f_level, b_level = get_reachable_sets(rg, source, target, num_nodes)
    pg_list = []
    for length in range(1, num_nodes):
        pg = PathsGraph.from_graph(rg, source, target, length,
                                   f_level, b_level)
        pg_list.append(pg)
    combined_pg = CombinedPathsGraph(pg_list)
    total_paths = combined_pg.count_paths()
    print("Total paths (with cycles)", total_paths)
    cf_paths = combined_pg.sample_cf_paths(10000)
    pg_elapsed = time.time() - pg_start
    print("Prob CF ascending (NX)", prob_ascending_path(path_probs))
    print("Prob CF ascending (PG)", prob_ascending_path(cf_paths))
    print("NX elapsed", elapsed)
    print("PG elapsed", pg_elapsed)

"""
if __name__ == '__main__':
    min_size = 8
    max_size = 9
    num_samples = 10
    lengths = range(min_size, max_size+1)
    nx_results, pg_results, cfpg_results = \
                    scaling_random_graphs(num_samples, min_size, max_size, 0.5)

    plt.ion()
    plt.figure()
    plt.errorbar(lengths, nx_results.mean(axis=1),
                 yerr=nx_results.std(axis=1, ddof=1))
    plt.errorbar(lengths, pg_results.mean(axis=1),
                 yerr=pg_results.std(axis=1, ddof=1))
    plt.errorbar(lengths, cfpg_results.mean(axis=1),
                 yerr=cfpg_results.std(axis=1, ddof=1))
    ax = plt.gca()
    ax.set_yscale('log')
"""
