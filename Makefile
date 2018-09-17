OUTPUT := output
DATA := data
NET := networks
#DEPLOY := ~/Dropbox/DARPA\ projects/papers/INDRA\ paper\ 2/figure_panels/

all: preprocessing toy_example

clean_toy_example:
	rm -f $(OUTPUT)/toy_*.pdf

deploy:
	rsync -av $(OUTPUT)/*.pdf ../paths_graph_manuscript/figures/

preprocessing: \
        $(DATA)/PathwayCommons9.All.hgnc.txt \
        $(DATA)/prior_genes.txt
	python preprocess_pc.py

toy_example: $(OUTPUT)/toy_g.pdf

$(OUTPUT)/toy_g.pdf: toy_example.py
	python toy_example.py $(OUTPUT)


