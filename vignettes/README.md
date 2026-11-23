# Vignettes

Ten vignettes, ordered from basic to advanced, that together cover the package. Each
lives in its own directory `NN_name/NN_name.qmd` and is rendered to GitHub-flavoured
Markdown (`NN_name.md` plus the figures in `NN_name_files/`), self-contained HTML
(`NN_name.html`) and PDF (`NN_name.pdf`). All rendered files are kept next to the
`.qmd` and committed.

| | Vignette | Covers |
|---|---|---|
| 01 | [A first decision table](01_first_decision_table/01_first_decision_table.md) | `PayoffTable`, `individuals_payoff`, `FiniteBelief`, `PointBelief`, `DecisionProblem`, feasibility, expected utility, the baseline and gains over it |
| 02 | [Decision rules side by side](02_decision_rules/02_decision_rules.md) | expected utility, maximin, maximax, Hurwicz, Laplace, minimax regret, satisficing, CVaR and quantile, ties |
| 03 | [Regret](03_regret/03_regret.md) | regret tables, expected and maximum regret, the identity with EVPI, relative regret, menu-dependence of minimax regret |
| 04 | [Experiments and the value of sample information](04_experiments_and_evsi/04_experiments_and_evsi.md) | `Experiment`, `infection_payoff`, `bayes_invert`, `evsi`, zero-value information and `information_can_change_decision`, `enbs` |
| 05 | [Partial information](05_partial_information/05_partial_information.md) | `evppi` over partitions, beliefs from FiniteKernels states, `restrict_scenarios` |
| 06 | [Comparing experiments](06_comparing_experiments/06_comparing_experiments.md) | `garble`, `is_garbling_of`, `deficiency`, incomparable designs, experiments from kernels |
| 07 | [Uncertain beliefs](07_uncertain_beliefs/07_uncertain_beliefs.md) | `CredalSet`, `ScenarioSet`, Γ-maximin, α-maxmin, E-admissibility, maximality, interval dominance, the same linear programme by the default exact route, by HiGHS with exact verification and by CDDLib (`eadmissibility_model`, `optimizer=`), robust value of information, robustness metrics, `total_variation`, `decision_loss_bound` |
| 08 | [Several objectives](08_several_objectives/08_several_objectives.md) | `ParetoFront` over named payoff tables, arithmetic on tables, the weight sweep |
| 09 | [The decision report](09_decision_report/09_decision_report.md) | `DecisionReport` with named experiments, partitions and objectives, its Markdown and 80-column text forms, Tables.jl output |
| 10 | [Exact arithmetic and certificates](10_exact_arithmetic_and_certificates/10_exact_arithmetic_and_certificates.md) | rational problems, the three exporters, the Lean checker |

## Conventions of the examples

- **Utility is counted in individuals, never in money**: the number of animals of the
  focal species. For an infectious disease there are two counts, uninfected individuals
  (higher is better) and infected individuals (lower is better), combined as
  uninfected − w × infected and shown at w = 0, at w = 1 and as a sweep. Infected but
  living animals are not counted as population.
- **Every vignette has a no-intervention action**, sets it as the `baseline` of the
  `DecisionProblem`, and reports each action's gain over it.
- **All numbers are illustrative.** They are not data and not estimates for any real
  system, and each vignette says so.

## Rendering: the code runs once

```sh
julia scripts/render_vignettes.jl                # from the package root: all ten
julia scripts/render_vignettes.jl 04 07          # only some
julia scripts/render_vignettes.jl --no-execute   # HTML and PDF again from the .md on disk
julia scripts/sync_vignettes.jl                  # copy the Markdown into docs/src/tutorials
```

The script does the following for each vignette, in this order.

1. `quarto render NN_name.qmd --to gfm` executes the Julia code (native Julia engine)
   and writes `NN_name.md` and the PNG figures under `NN_name_files/`. **This is the
   only step that runs code.** The script then rewrites the `<img>` tags Quarto uses
   for figures as Markdown images, a change of markup only.
2. pandoc converts `NN_name.md` to `NN_name.html`: standalone, with the figures and the
   style sheet `_pandoc/vignette.css` embedded.
3. pandoc converts `NN_name.md` to `NN_name.pdf` with `lualatex` and the preamble
   `_pandoc/header.tex`.

HTML and PDF are made from the executed Markdown, so the three formats show the same
numbers and the same figures by construction. pandoc is the one bundled with Quarto
(`quarto pandoc`).

Do not run a bare `quarto render` for HTML or PDF, and do not add `html` or `pdf` to
`_quarto.yml` or to a vignette's front matter. Quarto executes a document once per
format, and its freeze cache is keyed by format, so it would not reuse the Markdown
run. `_quarto.yml` therefore declares the `gfm` format only.

Figures are PNG. The helper `png_figure` in `_common.jl` wraps a plot so that it is
stored as PNG and not SVG; a PNG goes into all three formats without a converter.

The first render on a machine may have to precompile the plotting extension of the
Quarto Julia worker, and Quarto would capture that message into the first cell. The
script detects this and renders that vignette once more; later runs execute once.

## Requirements

- Julia 1.12 or later, and this directory's environment (`Project.toml`: the package
  and FiniteKernels by path, Plots, Tables, and the solvers HiGHS and CDDLib that
  vignette 07 uses; the package itself depends on no solver). The script instantiates
  and precompiles it.
- Quarto 1.9 or later (Julia engine).
- For the PDF: `lualatex` with `koma-script`, `fvextra`, `float` and `unicode-math`.
  TinyTeX from `quarto install tinytex` is found even when it is not on the `PATH`.

## Fonts of the PDF

The sibling packages set text in STIX Two Text and STIX Two Math and code in JuliaMono,
loaded from a workspace directory `../fonts/JuliaMono/` beside the package. The script
uses exactly that when it is there. When it is not, it falls back, in order, to Menlo
and DejaVu Sans Mono if installed, and otherwise to the LaTeX default, and prints which
it used. STIX Two is used when `fc-list` finds it, otherwise the LaTeX default text
fonts. To get JuliaMono, unpack
[`JuliaMono-ttf.tar.gz`](https://github.com/cormullion/juliamono/releases/latest) into
`../fonts/JuliaMono/` relative to the package root.

The PDF does not set a document language, because pandoc's language support under
`lualatex` needs the `selnolig` package, which a minimal TinyTeX lacks. Hyphenation is
therefore the LaTeX default. The HTML is marked `en-GB`.

## Files

- `_quarto.yml`: the Quarto project (Julia engine, `gfm` only).
- `_common.jl`: plotting helpers and a small Markdown table for summaries assembled by
  hand, included by every vignette. It computes no result. Payoff tables, regret
  tables, rule results and reports display themselves as tables.
- `_pandoc/`: the style sheet of the HTML and the LaTeX preamble of the PDF.
- `Project.toml`, `Manifest.toml`: the environment, as in the sibling packages.
- `NN_name/_out/`: files a vignette writes while it runs (the certificates of
  vignette 10). Ignored by git.

Vignette 10 runs the Lean checker `proofs/.lake/build/bin/check_certificate` when it
has been built (`cd proofs && lake exe cache get && lake build check_certificate`) and
prints a notice in its place when it has not. Build it before rendering the copy that
is committed.
