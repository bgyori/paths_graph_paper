import networkx as nx
from paths_graph import PathsGraph, CFPG, get_reachable_sets

g_edges = [
    ('A', 'B'), ('A', 'E'),
    ('B', 'C'), ('B', 'A'), ('B', 'E'), ('B', 'G'),
    ('C', 'A'), ('C', 'G'), ('D', 'B'), ('D', 'A'), ('D', 'F'),
    ('E', 'F'), ('E', 'G'),
    ('F', 'E'), ('F', 'B'),
    ('G', 'C'), ('G', 'E'), ('G', 'D'),
    ]
g = nx.DiGraph()
g.add_edges_from(g_edges)

def draw(g, filename):
    ag = nx.nx_agraph.to_agraph(g)
    ag.draw(filename, prog='dot')


# Draw G
draw(g, 'g.pdf')

f_level, b_level = get_reachable_sets(g, 'A', 'G', 4)


