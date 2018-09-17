import time
import libsbml
import networkx
from paths_graph import PathsGraph, CombinedPathsGraph, get_reachable_sets


fname = 'ReconMap-2.01-SBML3-Layout-Render.xml'


def get_combined_pg(rg, source, target, num_nodes):
    f_level, b_level = get_reachable_sets(rg, source, target, num_nodes)
    pg_list = []
    for length in range(1, num_nodes):
        pg = PathsGraph.from_graph(rg, source, target, length,
                                   f_level, b_level)
        pg_list.append(pg)
    combined_pg = CombinedPathsGraph(pg_list)
    return combined_pg


def make_networkx_graph(model):
    reactions = model.getListOfReactions()

    print('Building NetworkX graph')
    G = networkx.DiGraph()
    for reaction in reactions:
        for reactant in reaction.reactants:
            G.add_edge(reactant.species, reaction.id)
        for product in reaction.products:
            G.add_edge(reaction.id, product.species)
    return G


def prune_graph(G):
    compartments = ['c', 'm', 'l', 'e', 'r']
    blacklist = ['hco3', 'h2o', 'o2', 'h', 'nad', 'nadh', 'nadp', 'nadph',
                 'adp', 'atp', 'na1']
    print(f'{len(G.edges)} edges before pruning')
    for node in blacklist:
        for comp in compartments:
            try:
                G.remove_node(f'M_{node}_{comp}')
            except Exception:
                pass
    print(f'{len(G.edges)} edges after pruning')


def draw_paths(paths):
    vis_g = networkx.DiGraph()
    for path in paths:
        for s, t in zip(path[:-1], path[1:]):
            vis_g.add_edge(s, t)
    ag = networkx.nx_agraph.to_agraph(vis_g)
    # Add some visual styles to the graph
    ag.node_attr['shape'] = 'plaintext'
    ag.node_attr['rankdir'] = 'LR'
    ag.draw('vis_g.pdf', prog='dot')


def get_pg_paths(G, source, target, max_len, num_sample):
    ts = time.time()
    pg = get_combined_pg(G, source, target, max_len)
    paths = pg.sample_paths(num_sample)
    te = time.time()
    print(f'Got {num_sample} paths in {te-ts} seconds.')
    return paths


def get_nx_paths(G, source, target, max_len, num_sample):
    ts = time.time()
    path_gen = networkx.shortest_simple_paths(G, source, target)
    paths = []
    for idx, path in enumerate(path_gen):
        if idx >= num_sample:
            break
        if len(path) > max_len:
            break
        paths.append(tuple(path))
    te = time.time()
    print(f'Got {num_sample} paths in {te-ts} seconds.')
    return paths

if __name__ == '__main__':
    print(f'Loading SBML from {fname}')
    sbml_doc = libsbml.readSBMLFromFile(fname)
    model = sbml_doc.getModel()
    G = make_networkx_graph(model)
    prune_graph(G)
    pg_paths = get_pg_paths(G, 'M_glc_D_c', 'M_xylt_c', 30, 1000)
    nx_paths = get_nx_paths(G, 'M_glc_D_c', 'M_xylt_c', 30, 1000)
