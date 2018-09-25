import time
import pickle
import random
import numpy as np
from collections import defaultdict, Counter
from os.path import join, dirname
import networkx as nx
from paths_graph import CFPG, get_reachable_sets, PathsGraph, PathsTree, \
                        CombinedPathsGraph
from matplotlib import pyplot as plt
from indra.util import plot_formatting as pf

output_dir = join(dirname(__file__), 'output')

def run_pg_vs_nx(graph, source, target, depth, num_samples):
    # PG sampling
    start = time.time()
    f_level, b_level = get_reachable_sets(graph, source, target, depth)
    pg_list = []
    for i in range(1, depth+1):
        pg = PathsGraph.from_graph(graph, source, target, i, f_level,
                                   b_level)
        pg_list.append(pg)
    combined_pg = CombinedPathsGraph(pg_list)
    print("Sampling from PG")
    cf_paths = []
    while len(cf_paths) < num_samples:
        print(f'{len(cf_paths)} / {num_samples}')
        cf_path_chunk = combined_pg.sample_paths(100)
    #cf_paths = []
    end = time.time()
    #print("Done sampling from PG")
    print("Done generating PGs")
    pg_elapsed = end - start

    # Networkx enumeration
    index = 0
    start = time.time()
    nx_paths = []
    nx_sampled_paths = []
    """
    for p in nx.all_simple_paths(graph, source, target, cutoff=depth):
        nx_paths.append(tuple(p))
        if index % 10000 == 0:
            print(index)
        index += 1
    #print("Making PathsTree")
    #paths_tree = PathsTree(nx_paths)
    #print("Sampling PathsTree")
    #nx_sampled_paths = paths_tree.sample(num_samples)
    end = time.time()
    nx_elapsed = end - start
    #assert set(cf_paths) <= set(nx_paths)
    print("all_simple_paths done")
    print("Total paths (nx):", len(nx_paths))
    print("Unique sampled paths (pg):", len(set(cf_paths)))
    #print("Unique sampled_paths (tree):", len(set(nx_sampled_paths)))
    print("NX time", nx_elapsed)
    print("PG time", pg_elapsed)

    nx_sampled_paths = []
    """
    nx_elapsed = 0
    return {'pg_list': pg_list,
            'pg_paths': cf_paths,
            'nx_paths': nx_paths,
            'nx_paths_sampled': nx_sampled_paths,
            'pg_time': pg_elapsed, 'nx_time': nx_elapsed}


def run_timing_comparison(min_depth, max_depth, num_reps, num_samples):
    depths = list(range(min_depth, max_depth + 1))
    results = np.empty((2, len(depths), num_reps))
    genes = graph.nodes()
    for depth_ix, depth in enumerate(depths):
        for rep_ix in range(NUM_REPS):
            source = random.choice(genes)
            target = random.choice(genes)
            print("depth", depth, "rep", rep_ix+1, "source", source, "target",
                  target)
            res_dict = run_pg_vs_nx(graph, source, target, depth, num_samples)
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


def get_node_distribution(paths, source, target, n_length=1):
    dist = defaultdict(lambda: 0)
    for path in paths:
        for ix in range(len(path) - n_length + 1):
            if n_length == 1 and \
               (source is not None and target is not None) and \
               (path[ix] == source or path[ix] == target):
                continue
            ngram = path[ix:ix + n_length]
            dist[ngram] += 1
    pcts = []
    for node, val in dist.items():
        pcts.append((node, val / float(len(paths))))
    return sorted(pcts, key=lambda x: x[1], reverse=True)


if __name__ == '__main__':
    random.seed(1)
    filename = join(output_dir, 'pc_digraph.pkl')
    with open(filename, 'rb') as f:
        graph = pickle.load(f)

    # Timing comparison
    MAX_DEPTH = 5
    #NUM_REPS = 20
    #run_timing_comparison(1, MAX_DEPTH, NUM_REPS, 10000)

    # Node distribution
    source = 'SRC'
    target = 'CHEK2'
    result = run_pg_vs_nx(graph, source, target, MAX_DEPTH, 10000)
    #print("Pickling")
    #with open('pc_egfr_mapk1_max%d.pkl' % MAX_DEPTH, 'wb') as f:
    # pickle.dump(result, f)

    #with open('egfr_mapk1_depth_10_result.pkl', 'rb') as f:
    #    result = pickle.load(f)

    path_counts = []
    for pg in result['pg_list']:
        path_counts.append(pg.count_paths())
    combined_pg = CombinedPathsGraph(result['pg_list'])
    total_paths = np.sum(path_counts)
    print(path_counts)
    print(total_paths)

    # Plot num paths vs length
    plt.show
    plt.figure(figsize=(5,2), dpi=150)
    ypos = list(range(1, MAX_DEPTH+1))
    plt.bar(ypos, path_counts, align='center')
    #plt.xticks(ypos, str_names[:num_genes], rotation='vertical')
    ax = plt.gca()
    plt.ylabel('Number of paths')
    plt.xlabel('Path length')
    ax.set_yscale('log')
    #plt.subplots_adjust(bottom=0.3)
    pf.format_axis(ax)
    """
    cfpg_list = []
    total_cf_paths = 0
    for i, pg in enumerate(result['pg_list']):
        print("Generating CFPG %d" % i)
        cfpg = CFPG.from_pg(pg)
        total_cf_paths += cfpg.count_paths()
    print("total paths (with cycles)", total_paths)
    print("total cycle-free paths", total_cf_paths)
    """

    # Length distribution
    pg_path_lengths = Counter([len(p)-1 for p in result['pg_paths']])
    #nx_path_lengths = Counter([len(p)-1 for p in result['nx_paths']])
    lengths = range(1, MAX_DEPTH+1)
    plt.figure(figsize=(5, 2), dpi=150)
    plt.bar(lengths, [pg_path_lengths.get(l, 0) for l in lengths])
    plt.xlabel('Path length')
    plt.ylabel('Number of paths')
    ax.set_yscale('log')
    #plt.subplots_adjust(bottom=0.3)
    pf.format_axis(ax)

    node_dist = get_node_distribution(result['pg_paths'], source, target, 1)
    names, freqs = zip(*node_dist)
    str_names = [', '.join(n) for n in names]
    num_genes = 30
    plt.ion()
    plt.figure(figsize=(5,2), dpi=150)
    ypos = np.array(range(num_genes)) * 1.0
    plt.bar(ypos, freqs[:num_genes], align='center')
    plt.xticks(ypos, str_names[:num_genes], rotation='vertical')
    ax = plt.gca()
    plt.ylabel('Frequency')
    plt.subplots_adjust(bottom=0.3)
    pf.format_axis(ax)

