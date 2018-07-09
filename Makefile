OUTPUT := output
DATA := data
NET := networks
#DEPLOY := ~/Dropbox/DARPA\ projects/papers/INDRA\ paper\ 2/figure_panels/

all: preprocessing toy_example

preprocessing: \
        $(OUTPUT)/pc_digraph.pkl \
        $(OUTPUT)/pc_digraph_small.pkl

$(OUTPUT)/pc_digraph.pkl: \
        $(DATA)/PathwayCommons9.All.hgnc.txt \
        $(DATA)/prior_genes.txt
	python preprocess_pc.py

toy_example: $(OUTPUT)/g.pdf

$(OUTPUT)/g.pdf: toy_example.py
	python toy_example.py $(OUTPUT)
