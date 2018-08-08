from collections import defaultdict
import pickle
from indra.databases import hgnc_client
from indra.explanation.model_checker import ModelChecker
from indra.statements import *
from matplotlib import pyplot as plt
import numpy as np
from indra.util import plot_formatting as pf
from indra.assemblers import EnglishAssembler

from sampling_pc_small import get_node_distribution

def _stmt_from_rule(model, rule_name, stmts):
    """Return the INDRA Statement corresponding to a given rule by name."""
    stmt_uuid = None
    for ann in model.annotations:
        if ann.predicate == 'from_indra_statement':
            if ann.subject == rule_name:
                stmt_uuid = ann.object
                break
    if stmt_uuid:
        for stmt in stmts:
            if stmt.uuid == stmt_uuid:
                return stmt

def ag(gene_name):
    hgnc_id = hgnc_client.get_hgnc_id(gene_name)
    up_id = hgnc_client.get_uniprot_id(hgnc_id)
    db_refs = {'HGNC': hgnc_id, 'UP': up_id}
    return Agent(gene_name, db_refs=db_refs)

with open('data/korkut_pysb_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('data/korkut_model_pysb_before_pa.pkl', 'rb') as f:
    statements = pickle.load(f)

st1 = Phosphorylation(ag('KRAS'), ag('MAPK1'), 'T', '185')
st2 = Phosphorylation(ag('SRC'), ag('EIF4EBP1'), 'T', '37')
st3 = Phosphorylation(ag('SRC'), ag('CHEK2'), 'T', '68')
st4 = Dephosphorylation(ag('SRC'), ag('CHEK2'), 'T', '68')

mc = ModelChecker(model, [st1, st2, st3, st4], do_sampling=True)
mc.get_im()
mc.prune_influence_map()
pr = mc.check_statement(st4, max_paths=1000, max_path_length=10)


node_dist = get_node_distribution(pr.paths, None, None, 1)
dist_filt = [(n[0][0][0], n[1]) for n in node_dist]
dist_filt = [n for n in dist_filt if n[0] != 'CHEK2_T68_p_obs']

stmt_freq = [(_stmt_from_rule(model, r[0], statements), r[1])
             for r in dist_filt]
combined_freq = {}
for stmt, freq in stmt_freq:
    if stmt.uuid not in combined_freq:
        combined_freq[stmt.uuid] = (stmt, freq)
    else:
        _, old_freq = combined_freq[stmt.uuid]
        combined_freq[stmt.uuid] = (stmt, freq + old_freq)
top_stmts = list(combined_freq.values())
top_stmts.sort(key=lambda x: x[1], reverse=True)

desc = []
for s, freq in top_stmts:
    ea = EnglishAssembler([s])
    text = ea.make_model()
    desc.append((text, freq))
for t, f in desc[:30]:
    print('%s,%s' % (t, f))

"""
str_names, freqs = zip(*dist_filt)
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
