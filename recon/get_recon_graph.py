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


def get_recon_model(fname)
    print(f'Loading SBML from {fname}')
    sbml_doc = libsbml.readSBMLFromFile(fname)
    model = sbml_doc.getModel()
    return model


def make_networkx_graph(model)
    reactions = model.getListOfReactions()

    print('Building NetworkX graph')
    G = networkx.DiGraph()
    for reaction in reactions:
        for reactant in reaction.reactants:
            G.add_edge(reactant.species, reaction.id)
        for product in reaction.products:
            G.add_edge(reaction.id, product.species)
    return G



if __name__ == '__main__':
    model = get_recon_model(fname)
    G = make_networkx_graph(model)
    pg = get_combined_pg(G, 'M_arachd_c', 'M_C06315_c', 5)
