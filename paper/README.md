# paper/ — IEEE conference paper

**Title (draft):** *Beyond Fake or Real: Evidence-Grounded, Explainable Five-Class News Verification with Temporal Context Checking*

| File | Purpose |
|---|---|
| `main.tex` | The paper (IEEEtran `conference` class). MSE1 state: abstract (placeholder numbers), §I Introduction, §II Related Work with Table I; §III–VIII are headed stubs with `%TODO(MSE2/ESE)` comments. |
| `refs.bib` | 33 references, IEEE style via `IEEEtran.bst`. Bibkeys are referenced from `docs/literature_table.md`; keep them stable. |
| `figures/` | Figures exported from `docs/figures/` (PDF preferred). Empty at MSE1 (`.gitkeep`). |
| `Makefile` | `make` → `main.pdf` using `latexmk` if installed, otherwise `tectonic`. `make check` prints the bib count and fails on undefined citations. |
| `main.pdf` | Built output (committed so examiners can open it without a TeX install). |

## Building

```bash
make -C paper            # or: cd paper && latexmk -pdf main.tex   /   tectonic main.tex
make -C paper check      # bib count, undefined-citation check, page count
```

### Toolchain notes (28 Aug 2026)

* The lab machine has `pdflatex`/`bibtex` but **no `latexmk`, no `IEEEtran.cls` and no Times (`ptm`) font metrics** (`texlive-fonts-recommended` / `texlive-latex-extra` are not installed), so `pdflatex main.tex` fails with `Font OT1/ptm/m/n/10=ptmr7t ... not loadable`. Installing those packages needs `sudo apt install texlive-latex-extra texlive-fonts-recommended texlive-bibtex-extra latexmk`.
* Without sudo, the working path is **Tectonic** (single static binary, downloads packages on demand, runs BibTeX itself). `uv tool install tectonic` does not exist on PyPI; the release binary was fetched from
  `https://github.com/tectonic-typesetting/tectonic/releases` (v0.17.0, `x86_64-unknown-linux-musl`) into `~/.local/bin/tectonic`. First run downloads ~50 MB of TeX packages and takes a few minutes; later builds take ~10 s. The `Makefile` finds it automatically.
* **Overleaf** works out of the box: upload the `paper/` folder (or `main.tex` + `refs.bib` + `figures/`), set compiler to pdfLaTeX. IEEEtran is preinstalled there.

## Writing rules (from the master doc §20)

* Write from notes in your own words — never paste sentences from papers or web pages (Turnitin target < 10 % excl. references).
* Every number carries a `\cite{}`. Anything unverified stays marked `%TODO-VERIFY` and is dropped before submission.
* Verdict names use `\cls{Real}` etc.; placeholders use `\todo{...}` (renders red so they cannot be missed).
* 6 pages target (up to 8 only if the venue allows). §I ≈ 1 page, §II ≈ 1–1.5 pages incl. Table I.
* Section owners: §I–II Nikhil (all review); §III Navishka; §IV Naveen + Prateek; §V–VI Naitik; §VII–VIII shared.

## Status log

* 2026-08-28 — v0.1: skeleton, abstract placeholder, §I and §II written, 33 verified references, builds with tectonic (4 pages incl. stubs, 0 undefined citations, 0 BibTeX warnings).
