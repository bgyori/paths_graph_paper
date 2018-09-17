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
    for node in blacklist:
        for comp in compartments:
            print(f'{len(G.edges)} edges before removing {node}_{comp}')
            try:
                G.remove_node(f'M_{node}_{comp}')
            except Exception:
                pass
            print(f'{len(G.edges)} edges after removing {node}_{comp}')


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


if __name__ == '__main__':
    print(f'Loading SBML from {fname}')
    sbml_doc = libsbml.readSBMLFromFile(fname)
    model = sbml_doc.getModel()
    G = make_networkx_graph(model)
    prune_graph(G)
    pg = get_combined_pg(G, 'M_glc_D_c', 'M_xylt_c', 25)
    path_set = set(pg.sample_paths(1000))
