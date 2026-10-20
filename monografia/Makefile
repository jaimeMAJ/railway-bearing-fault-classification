MAIN = main

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex referencias.bib
	pdflatex $(MAIN)
	bibtex $(MAIN)
	pdflatex $(MAIN)
	pdflatex $(MAIN)

latexmk:
	latexmk -pdf $(MAIN).tex

clean:
	rm -f *.aux *.log *.bbl *.blg *.toc *.lof *.lot *.out *.fls *.fdb_latexmk *.brf
	rm -f capitulos/*.aux pre_textuais/*.aux

distclean: clean
	rm -f $(MAIN).pdf
