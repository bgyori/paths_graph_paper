import sys
import csv
import pickle
from os.path import join, dirname
import networkx as nx
from indra.databases import hgnc_client

csv.field_size_limit(sys.maxsize)

output_dir = join(dirname(__file__), 'output')
data_dir = join(dirname(__file__), 'data')

def read_file(pc_filename):
    with open(pc_filename, 'rt') as f:
        pc_data_generator = csv.reader(f, delimiter='\t')
        next(pc_data_generator)
        for row in pc_data_generator:
            yield row

def load_pc_network(filter_genes=None, flatten=True):
    """Get Pathway Common gene network as a networkx MultiDiGraph.

    Parameters
    ----------
    filter_genes : list of str
        List of gene names to filter the network against. Only edges between
        nodes in this list will be added to the network.
    """
    # Column names in the TSV file
    col_names = ['PARTICIPANT_A', 'INTERACTION_TYPE', 'PARTICIPANT_B',
                 'INTERACTION_DATA_SOURCE', 'INTERACTION_PUBMED_ID',
                 'PATHWAY_NAMES', 'MEDIATOR_IDS']
    # Get the data, skipping the header line
    print("Processing Pathway Commons TSV file")
    # Load into a networkx MultiDiGraph
    if flatten:
        pc_graph = nx.DiGraph()
    else:
        pc_graph = nx.MultiDiGraph()
    for ix, row in enumerate(read_file(pc_filename)):
        # Handle possible missing rows
        if not row:
            continue
        if (ix+1) % 100000 == 0:
            print("Row %d" % (ix+1))
        subj = row[0]
        obj = row[2]
        if not hgnc_client.get_hgnc_id(subj) or not hgnc_client.get_hgnc_id(obj):
            continue
        # If desired, these lines can be uncommented to put more of the extended
        # SIF metadata into the network edges
        edge_data = dict(zip(col_names[3:-1], row[3:-1]))
        edge_data['relation'] = row[1]
        if not filter_genes or \
           (subj in filter_genes and obj in filter_genes):
            pc_graph.add_edge(subj, obj, attr_dict=edge_data)
    return pc_graph


if __name__ == '__main__':
    # Script to parse the PC data and cache the network
    pc_filename = join(data_dir, 'PathwayCommons9.All.hgnc.txt')
    prior_genes_file = join(data_dir, 'prior_genes.txt')
    with open(prior_genes_file, 'rt') as f:
        prior_genes = [line.strip() for line in f.readlines()]
    # The full network, flattened
    pc_graph = load_pc_network(flatten=True)
    pc_pickle = join(output_dir, 'pc_multidigraph.pkl')
    with open(pc_pickle, 'wb') as f:
        pickle.dump(pc_graph, f)
    # The filtered network, flattened
    pc_graph_filt = load_pc_network(flatten=True, filter_genes=prior_genes)
    pc_pickle_filt = join(output_dir, 'pc_multidigraph_prior.pkl')
    with open(pc_pickle_filt, 'wb') as f:
        pickle.dump(pc_graph_filt, f)

