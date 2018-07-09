import sys
from os.path import join
import itertools
import networkx as nx
from paths_graph import PathsGraph, CombinedPathsGraph, CFPG, \
                        get_reachable_sets

"""
g_edges = [
    ('S', 'B'), ('S', 'E'),
    ('B', 'C'), ('B', 'S'), ('B', 'E'), ('B', 'T'),
    ('C', 'S'), ('C', 'T'), ('D', 'B'), ('D', 'S'), ('D', 'F'),
    ('E', 'F'), ('E', 'T'),
    ('F', 'E'), ('F', 'B'),
    ('T', 'C'), ('T', 'E'), ('T', 'D'),
    ]
g_edges = [
    ('A', 'B'), ('A', 'E'),
    ('B', 'C'), ('B', 'A'), ('B', 'E'),
    ('C', 'A'), ('D', 'B'), ('D', 'A'), ('D', 'F'),
    ('E', 'F'),
    ('F', 'E'), ('F', 'B'),
    ]
"""
g_edges = [
    ('S', 'A'), ('S', 'B'),
    ('A', 'C'), ('A', 'T'),
    ('B', 'C'), ('B', 'T'),
    ('C', 'A'), ('C', 'B'),
]

g = nx.DiGraph()
g.add_edges_from(g_edges)


def draw(g, filename):
    fixed_edges = []
    for u, v in g.edges():
        u_fix = str(u).replace("'", "").replace('"', '')
        v_fix = str(v).replace("'", "").replace('"', '')
        fixed_edges.append((u_fix, v_fix))
    g_fixed = nx.DiGraph()
    g_fixed.add_edges_from(fixed_edges)
    ag = nx.nx_agraph.to_agraph(g_fixed)
    ag.draw(filename, prog='dot')


def draw_reachset(g, level, direction, depth, output_dir):
    if direction not in ('forward', 'backward'):
        raise ValueError("direction must be 'forward' or 'backward'")
    edges = []
    for level_ix in range(1, depth+1):
        if direction == 'forward':
            prev_nodes = [(level_ix - 1, n) for n in level[level_ix - 1]]
            cur_nodes = [(level_ix, n) for n in level[level_ix]]
        elif direction == 'backward':
            prev_nodes = [(depth - (level_ix - 1), n)
                          for n in level[level_ix - 1]]
            cur_nodes = [(depth - level_ix, n) for n in level[level_ix]]
        for prev_node, cur_node in itertools.product(prev_nodes, cur_nodes):
            if direction == 'forward' and (cur_node[1] in g[prev_node[1]]):
                edges.append((prev_node, cur_node))
            elif direction == 'backward' and (prev_node[1] in g[cur_node[1]]):
                edges.append((prev_node, cur_node))
    graph = nx.DiGraph()
    graph.add_edges_from(edges)
    draw(graph, join(output_dir, '%s_graph.pdf' % direction))

if __name__ == '__main__':

    output_dir = sys.argv[1]

    # Draw G
    draw(g, join(output_dir, 'g.pdf'))

    depth = 4
    source = 'S'
    target = 'T'

    f_level, b_level = get_reachable_sets(g, source, target, depth)
    draw_reachset(g, f_level, 'forward', depth, output_dir)
    draw_reachset(g, b_level, 'backward', depth, output_dir)

    print("f_level", f_level)
    print("b_level", b_level)

    pg = PathsGraph.from_graph(g, source, target, depth)
    draw(pg.graph, join(output_dir, 'pg_%d.pdf' % depth))

    # Combined paths graph
    pg_list = []
    for i in range(1, 6+1):
        pg_list.append(PathsGraph.from_graph(g, source, target, i))
    cpg = CombinedPathsGraph(pg_list)
    draw(cpg.graph, join(output_dir, 'combined_pg.pdf'))

    # Cycle-free paths graph
    cfpg = CFPG.from_pg(pg)
    # Remove the frozensets for drawing
    cfpg_edges_fixed = []
    for u, v in cfpg.graph.edges():
        u_set = '{}' if u[2] == 0 else str(set(u[2]))
        v_set = '{}' if v[2] == 0 else str(set(v[2]))
        u_fixed = str((u[0], u[1], u_set))
        v_fixed = str((v[0], v[1], v_set))
        cfpg_edges_fixed.append((u_fixed, v_fixed))
    cfpg_fixed = nx.DiGraph()
    cfpg_fixed.add_edges_from(cfpg_edges_fixed)
    draw(cfpg_fixed, join(output_dir, 'cfpg_%d.pdf' % depth))

    # Non-uniform sampling
    # Graph for testing sampling uniformly vs. non-uniformly
    g_samp = nx.DiGraph()
    g_samp.add_edges_from([
        ('S', 'A1'), ('S', 'A2'),
        ('A1', 'B1'),
        ('A2', 'B2'), ('A2', 'B3'), ('A2', 'B4'), ('A2', 'B5'),
        ('B1', 'T'),
        ('B2', 'T'), ('B3', 'T'), ('B4', 'T'), ('B5', 'T')])
    draw(g_samp, join(output_dir, 'g_samp.pdf'))

