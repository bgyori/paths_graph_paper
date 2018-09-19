import time
import pickle
import random
import numpy as np
from collections import defaultdict
from os.path import join, dirname
import networkx as nx
from paths_graph import CFPG, get_reachable_sets
from matplotlib import pyplot as plt

output_dir = join(dirname(__file__), 'output')

if __name__ == '__main__':
    filename = join(output_dir, 'pc_digraph_small.pkl')
    with open(filename, 'rb') as f:
        graph = pickle.load(f)
    MAX_DEPTH = 7
    NUM_REPS = 1
    #depths = list(range(4, MAX_DEPTH+1))
    depths = [4]
    results = np.empty((2, len(depths), NUM_REPS))
    random.seed(1)
    genes = list(graph.nodes())
    for depth_ix, depth in enumerate(depths):
        print(depth)
        for rep_ix in range(NUM_REPS):
            print("---------")
            source = random.choice(genes)
            target = random.choice(genes)
            print("depth", depth, "rep", rep_ix+1, "source", source, "target", target)
            index = 0
            start = time.time()
            for p in nx.all_simple_paths(graph, source, target, cutoff=depth):
                if index % 10000 == 0:
                    print(index)
                index += 1
            end = time.time()
            nx_elapsed = end - start
            results[0, depth_ix, rep_ix] = nx_elapsed
            print("done")

            start = time.time()
            f_level, b_level = get_reachable_sets(graph, source, target, MAX_DEPTH)
            total_paths = 0
            for i in range(1, depth+1):
                print(i)
                cfpg = CFPG.from_graph(graph, source, target, i, f_level, b_level)
                path_count = cfpg.count_paths()
                print(path_count, "paths")
                total_paths += path_count
            print("total paths", total_paths)
            print("nx paths", index)
            end = time.time()
            pg_elapsed = end - start
            results[1, depth_ix, rep_ix] = pg_elapsed
            print("NX time", nx_elapsed)
            print("CFPG time", pg_elapsed)

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

