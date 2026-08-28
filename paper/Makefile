# paper/Makefile -- build the IEEE paper.
#   make            -> main.pdf   (latexmk if available, else tectonic, else a clear error)
#   make clean      -> remove build artefacts (keeps main.pdf)
#   make distclean  -> also remove main.pdf
# The repo-level `make paper` target should simply call `$(MAKE) -C paper`.

TEX      := main
LATEXMK  := $(shell command -v latexmk 2>/dev/null)
TECTONIC := $(shell command -v tectonic 2>/dev/null || ls $(HOME)/.local/bin/tectonic 2>/dev/null)

.PHONY: all pdf clean distclean check

all: pdf

pdf: $(TEX).pdf

$(TEX).pdf: $(TEX).tex refs.bib $(wildcard figures/*)
ifneq ($(LATEXMK),)
	$(LATEXMK) -pdf -interaction=nonstopmode -halt-on-error $(TEX).tex
else ifneq ($(TECTONIC),)
	$(TECTONIC) --keep-logs $(TEX).tex
else
	@echo "No LaTeX toolchain found. Options:"
	@echo "  1) sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra latexmk"
	@echo "  2) user-space, no sudo: download the tectonic release binary into ~/.local/bin (see paper/README.md)"
	@echo "  3) upload the paper/ folder to Overleaf (IEEE Conference Template) and compile there"
	@exit 1
endif

# quick sanity checks used by the MSE1 DoD: no '??' citations, >= 15 bib entries
check: $(TEX).pdf
	@grep -c "^@" refs.bib | xargs -I{} echo "bib entries: {}"
	@if grep -q "Citation .* undefined" $(TEX).log; then echo "UNDEFINED CITATIONS"; exit 1; else echo "no undefined citations"; fi
	@pdfinfo $(TEX).pdf 2>/dev/null | grep Pages || true

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).fls $(TEX).fdb_latexmk $(TEX).synctex.gz $(TEX).toc

distclean: clean
	rm -f $(TEX).pdf
