"""The purpose of this script is to generate random graphs of a given size
and test multiple path finding methods between a selected source and target.
The plot of how path finding time scales with the size of the graph is then
produced.
"""

import time
import itertools
import numpy as np
import networkx as nx
from matplotlib import pyplot as plt
from paths_graph import PathsGraph, get_reachable_sets, CFPG, \
                        CombinedPathsGraph, PathsTree, CombinedCFPG
from paths_graph.path_checker import HypothesisTester

seed = 1


def ordered_property(path):
    return sorted(path) == path


def exists_property(path, node):
    return node in path


def run_nx(rg, source, target):
    # Time to compute all simple paths with path probabilities
    start = time.time()
    paths = [tuple(p) for p in nx.all_simple_paths(rg, source, target)]
    #paths2 = [tuple(p) for p in nx.shortest_simple_paths(rg, source, target)]
    #assert(set(paths) == set(paths2))

    # Now build a path tree from the paths and calculate probabilities
    pt = PathsTree(paths)
    path_probs = pt.path_probabilities()
    pr = np.sum([w*exists_property(p, 5) for p, w in path_probs.items()])
    print(f'NX prob {pr}')
    # Save the time it took the calculate
    end = time.time()
    elapsed = end - start
    print(f'NX: {elapsed:.2f}s')
    return elapsed


def run_pg_cfpg(rg, source, target):
    # Time to compute paths_graphs and make combined graph
    pg_start = time.time()
    f_level, b_level = get_reachable_sets(rg, source, target, num_nodes)
    pg_list = []
    for length in range(1, num_nodes):
        pg = PathsGraph.from_graph(rg, source, target, length,
                                   f_level, b_level)
        pg_list.append(pg)
    combined_pg = CombinedPathsGraph(pg_list)

    ht = HypothesisTester(0.5, 0.1, 0.1, 0.05)
    tf = None
    tfs = []
    nsamples = 0
    batch = 10
    while tf is None:
        new_paths = combined_pg.sample_cf_paths(batch)
        if not new_paths:
            tf = 0
            break
        tfs += [exists_property(p, 5) for p in new_paths] 
        nsamples += batch
        tf = ht.test(tfs)
    print(f'PG: {tf} based on {nsamples} samples')

    # cf_paths = combined_pg.sample_cf_paths(10000)
    # print(prob_ascending_path(cf_paths))

    pg_elapsed = time.time() - pg_start
    print(f'PG: {pg_elapsed:.2f}s')

    # Now compute the CFPG
    cfpg_list = []
    for pg in pg_list:
        cfpg = CFPG.from_pg(pg)
        cfpg_list.append(cfpg)
    ccfpg = CombinedCFPG(cfpg_list)

    print('Sampling CFPG')
    ht = HypothesisTester(0.5, 0.1, 0.1, 0.05)
    tf = None
    tfs = []
    nsamples = 0
    batch = 10
    while tf is None:
        new_paths = ccfpg.sample_paths(batch)
        if not new_paths:
            tf = 0
            break
        tfs += [exists_property(p, 5) for p in new_paths] 
        nsamples += batch
        tf = ht.test(tfs)
    print(f'CFPG: {tf} based on {nsamples} samples')

    #cfpg_paths = ccfpg.sample_paths(10000)
    #print(prob_ascending_path(cfpg_paths))

    cfpg_elapsed = time.time() - pg_start
    print(f'CFPG: {cfpg_elapsed:.2f}s')
    return pg_elapsed, cfpg_elapsed


def scaling_random_graphs(num_samples, min_size, max_size, edge_prob=0.5):
    # Iterate over number of nodes in network
    for i, num_nodes in enumerate(range(min_size, max_size+1)):
        print(f'Number of nodes in network: {num_nodes}')

        # Iterate over num_samples random graphs of this size
        for j in range(num_samples):
            print(f'Sample {j}')

    return times_nx_paths, times_pg, times_cfpg


def to_directed(G):
    DG = nx.DiGraph()
    for u, v in G.edges():
        edge = (u, v) if np.random.rand() < 0.5 else (v, u)
        DG.add_edge(*edge)
    return DG


def run_all(rg, source, target, num_nodes):
    times = np.zeros(3)
    # Run NX
    nx_elapsed = run_nx(rg, source, target)
    times[0] = nx_elapsed
    # Run PG / CFPG
    pg_elapsed, cfpg_elapsed = run_pg_cfpg(rg, source, target)
    times[1] = pg_elapsed
    times[2] = cfpg_elapsed
    return times

if __name__ == '__main__':
    # Some basic parameters
    min_size = 6
    max_size = 15
    num_samples = 10

    lengths = range(min_size, max_size+1)
    graph_types = [
        lambda x: nx.erdos_renyi_graph(x, 0.5, directed=True),
        # lambda x: to_directed(nx.barabasi_albert_graph(x, m=int(x/2.0))),
        ]

    data_shape = (max_size - min_size + 1, len(graph_types), num_samples, 3)
    times = np.zeros(data_shape)

    for i, num_nodes in enumerate(range(min_size, max_size+1)):
        for j, graph_type in enumerate(graph_types):
            for k, sample in enumerate(range(num_samples)):
                print(f'{num_nodes},{j},{sample}')
                # Make graph
                rg = graph_type(num_nodes)
                source = 0
                target = num_nodes - 1
                # Get all times
                sample_times = run_all(rg, source, target, num_nodes)
                times[i, j, k, :] = sample_times

    # Plotting
    plt.ion()
    plt.figure()
    means = times.mean(axis=2).reshape(max_size - min_size + 1, 3)
    stds = times.std(axis=2).reshape(max_size - min_size + 1, 3)
    plt.errorbar(lengths, means[:,0], yerr=stds[:,0], label='NX')
    plt.errorbar(lengths, means[:,1], yerr=stds[:,1], label='PG')
    plt.errorbar(lengths, means[:,2], yerr=stds[:,2], label='CF')
    ax = plt.gca()
    ax.set_yscale('log')
    plt.legend()
