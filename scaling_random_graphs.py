import time
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from paths_graph import PathsGraph, get_reachable_sets, CFPG

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
            rg = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=1, directed=True)
            source = 0
            target = num_nodes - 1
            # Time to compute all simple paths
            start = time.time()
            paths = list(nx.all_simple_paths(rg, source, target))
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
            pg_elapsed = time.time() - pg_start
            times_pg[i, j] = pg_elapsed
            # Now compute the CFPG
            cfpg_list = []
            for pg in pg_list:
                cfpg = CFPG.from_pg(pg)
                cfpg_list.append(cfpg)
            cfpg_elapsed = time.time() - pg_start
            times_cfpg[i, j] = cfpg_elapsed
    return times_nx_paths, times_pg, times_cfpg

if __name__ == '__main__':
    min_size = 3
    max_size = 13
    num_samples = 20
    lengths = range(min_size, max_size+1)
    nx_results, pg_results, cfpg_results = \
                    scaling_random_graphs(num_samples, min_size, max_size, 0.8)

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
