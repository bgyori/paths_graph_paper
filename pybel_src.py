import pickle
import itertools
import numpy as np
import networkx as nx
from paths_graph import PathsGraph, get_reachable_sets, \
                        CombinedPathsGraph
from sampling_pc_small import get_node_distribution
from matplotlib import pyplot as plt
from indra.util import plot_formatting as pf

with open('large_corpus_pybel.pkl', 'rb') as f:
    pybel_graph = pickle.load(f)

src_nodes = [n for n in pybel_graph if n[0] == 'Protein' and
                                       n[2] == 'SRC']
#chek2_nodes = [n for n in pybel_graph
#               if n == ('Protein', 'HGNC', 'CHEK2',
#                       ('pmod', ('bel', 'Ph'), 'Thr', 68))]
chek2_node = ('Protein', 'HGNC', 'CHEK2', ('pmod', ('bel', 'Ph'), 'Thr', 68))

# Prepare a signed version of the Pybel Graph
pb_sign_edges = []
edge_count = 0
for u, v, data in pybel_graph.edges_iter(data=True):
    if data['relation'] == 'increases':
        sign = 0
    elif data['relation'] == 'decreases':
        sign = 1
    else:
        sign = 0
    edge_count += 1
    pb_sign_edges.append((u, v, {'sign': sign}))
print(edge_count)
pb_signed = nx.DiGraph()
pb_signed.add_edges_from(pb_sign_edges)

src_edges = list(itertools.product(['root'], src_nodes, [{'sign': 0}]))


graph = pb_signed
graph.add_edges_from(src_edges)
source = 'root'
target = chek2_node
depth = 6
num_samples = 1000

f_level, b_level = get_reachable_sets(graph, source, target, depth,
                                      signed=True)
pg_list = []

for i in range(1, depth+1):
    pg = PathsGraph.from_graph(graph, source, target, i, f_level,
                               b_level, signed=True, target_polarity=1)
    pg_list.append(pg)
combined_pg = CombinedPathsGraph(pg_list)
cf_paths = combined_pg.sample_cf_paths(num_samples)

"""
dist = get_node_distribution(cf_paths, None, None)
dist_filt = [(n[0][0][2], n[1]) for n in dist]
dist_filt = [n for n in dist_filt if n[0] not in ['o', 'SRC', 'CHEK2']]

node_dist = dist_filt
str_names, freqs = zip(*node_dist)
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
"""
