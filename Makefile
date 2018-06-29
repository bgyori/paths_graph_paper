OUTPUT := output
DATA := data
NET := networks
#DEPLOY := ~/Dropbox/DARPA\ projects/papers/INDRA\ paper\ 2/figure_panels/

all: preprocessing

preprocessing: \
        $(OUTPUT)/pc_multidigraph.pkl \
        $(OUTPUT)/pc_multidigraph_prior.pkl

$(BUILD)/pc_multidigraph.pkl: \
        $(OUTPUT)/PathwayCommons9.All.hgnc.txt \
        $(DATA)/prior_genes.txt
	python preprocess_pc.py



