"""main.tex -> main.md (GitHub-flavoured Markdown), kept in sync with the PDF.

Usage (from the repo root): make paper-md   (= python3 paper/tex2md.py)
Needs pandoc, tectonic (for main.aux/main.bbl: citation and table numbers exactly as in the PDF)
and pdftoppm (figures/pipeline.pdf -> figures/pipeline.png). Stdlib only.
"""
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
TECTONIC = shutil.which("tectonic") or str(Path.home() / ".local/bin/tectonic")


def pandoc(tex: str) -> str:
    return subprocess.run(["pandoc", "-f", "latex", "-t", "gfm", "--wrap=none"], input=tex,
                          capture_output=True, text=True, check=True).stdout.strip()


def braced(s: str, i: int) -> tuple[str, int]:
    """Content of the {...} group starting at s[i] == '{', and the index after it."""
    depth = 0
    for j in range(i, len(s)):
        depth += {"{": 1, "}": -1}.get(s[j], 0)
        if depth == 0:
            return s[i + 1:j], j + 1
    raise ValueError("unbalanced braces")


def main() -> None:
    tex = (HERE / "main.tex").read_text()
    with tempfile.TemporaryDirectory() as tmp:
        for f in ("main.tex", "refs.bib"):
            shutil.copy(HERE / f, tmp)
        shutil.copytree(HERE / "figures", Path(tmp) / "figures")
        subprocess.run([TECTONIC, "--keep-intermediates", "main.tex"], cwd=tmp, check=True, capture_output=True)
        aux, bbl = (Path(tmp) / "main.aux").read_text(), (Path(tmp) / "main.bbl").read_text()
    if shutil.which("pdftoppm"):
        subprocess.run(["pdftoppm", "-png", "-r", "150", "-singlefile", str(HERE / "figures/pipeline.pdf"),
                        str(HERE / "figures/pipeline")], check=True)

    cite = dict(re.findall(r"\\bibcite\{([^}]+)\}\{(\d+)\}", aux))
    label = dict(re.findall(r"\\newlabel\{([^}]+)\}\{\{([^}]+)\}", aux))

    # --- front matter: title, authors, abstract, keywords ---------------------------------------
    title = re.search(r"\\title\{(.+?)\}\n", tex).group(1)
    authors = []
    for name, aff in re.findall(r"\\IEEEauthorblockN\{(.+?)\}\s*\\IEEEauthorblockA\{(.+?)\}\s*(?:\\and|\})", tex, re.S):
        lines = [l.strip().replace("\\&", "&") for l in aff.split("\\\\")]
        authors.append(f"- **{name}** — " + ", ".join(lines[:-1]) + f" — <{lines[-1]}>")
    abstract = re.search(r"\\begin\{abstract\}(.+?)\\end\{abstract\}", tex, re.S).group(1)
    keywords = re.search(r"\\begin\{IEEEkeywords\}(.+?)\\end\{IEEEkeywords\}", tex, re.S).group(1)
    body = tex[tex.index("\\end{IEEEkeywords}") + len("\\end{IEEEkeywords}"):tex.index("\\bibliographystyle")]

    def prep(s: str) -> str:
        s = re.sub(r"~?\\cite\{([^}]+)\}", lambda m: " " + ", ".join(f"[{cite[k.strip()]}]" for k in m.group(1).split(",")), s)
        s = re.sub(r"\\ref\{([^}]+)\}", lambda m: label[m.group(1)], s)
        s = re.sub(r"\\section\*\{", r"\\section{UNNUMBERED ", s)
        s = re.sub(r"\\(?:cls|textsc)\{([^}]+)\}", r"\1", s).replace("\\checkmark", "✓")
        return s

    figs, n_fig = {}, 0

    def figure(m: re.Match) -> str:
        nonlocal n_fig
        n_fig += 1
        cap = pandoc(prep(braced(m.group(0), m.group(0).index("\\caption{") + 8)[0]))
        img = re.search(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", m.group(0)).group(1).replace(".pdf", ".png")
        figs[f"FIGURE{n_fig}X"] = f"![Fig. {n_fig}. {cap}](figures/{img})\n\n*Fig. {n_fig}. {cap}*"
        return f"\n\nFIGURE{n_fig}X\n\n"

    def table(m: re.Match) -> str:
        t = m.group(0)
        cap = braced(t, t.index("\\caption{") + 8)[0]
        num = label[re.search(r"\\label\{([^}]+)\}", t).group(1)]
        tabs = re.findall(r"\\begin\{tabular\}.+?\\end\{tabular\}", t, re.S)
        out = []
        for tab in tabs:
            tab = re.sub(r"\\multicolumn\{(\d)\}\{\w\}\{([^}]*)\}", lambda m: m.group(2) + " &" * (int(m.group(1)) - 1), tab)
            # p{..}/>{..} column specs make pandoc drop cell text: replace with plain l columns
            ncols = len(re.split(r"(?<!\\)&", tab.split("\\toprule")[1].split("\\\\")[0]))
            tab = re.sub(r"\\begin\{tabular\}\{[^\n]*\}\n", "\\\\begin{tabular}{" + "l" * ncols + "}\n", tab, count=1)
            tab = re.sub(r"\\cmidrule\S*", "", tab)
            out.append(tab)
        return f"\n\n\\textbf{{Table {num}.}} {cap}\n\n" + "\n\n".join(out) + "\n\n"

    body = re.sub(r"\\begin\{figure\}.+?\\end\{figure\}", figure, body, flags=re.S)
    body = re.sub(r"\\begin\{table\}.+?\\end\{table\}", table, body, flags=re.S)
    md = pandoc(prep(body)).replace("<!-- -->", "")
    for k, v in figs.items():
        md = md.replace(k, v)
    # pandoc drops IEEE section numbering; number sections I, II, ... and subsections A, B, ... like the PDF
    roman = iter(["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"])
    out, sub = [], 0
    for line in md.splitlines():
        if line.startswith("# UNNUMBERED "):
            line = "## " + line[len("# UNNUMBERED "):]
        elif line.startswith("# "):
            line, sub = f"## {next(roman)}. {line[2:]}", 0
        elif line.startswith("## "):
            sub += 1
            line = f"### {chr(64 + sub)}. {line[3:]}"
        out.append(line)
    md = "\n".join(out)

    # --- references from the BibTeX output, same order and numbering as the PDF -----------------
    items = re.findall(r"\\bibitem\{[^}]+\}\s*(.+?)(?=\n\s*\\bibitem|\n\s*\\end\{thebibliography\})", bbl, re.S)
    refs = [pandoc(re.sub(r"\\BIBentry\w+|\\newblock\s*|\\hskip[^\\]*\\relax", " ", it)).replace("\n", " ") for it in items]

    doc = [f"# {title}", "", *authors, "",
           f"**Abstract**—{pandoc(prep(abstract))}", "",
           f"**Index Terms**—{keywords.strip()}", "",
           "> Markdown version of `paper/main.tex`, generated by `make paper-md` (`paper/tex2md.py`). "
           "The PDF (`paper/main.pdf`) is the version of record; numbering of sections, tables, figures and "
           "references follows it.", "",
           md, "", "## References", "", *[f"{i}. {r}" for i, r in enumerate(refs, 1)], ""]
    (HERE / "main.md").write_text("\n".join(doc))
    print(f"wrote {HERE / 'main.md'} ({len(refs)} references)")


if __name__ == "__main__":
    os.chdir(HERE)
    main()
