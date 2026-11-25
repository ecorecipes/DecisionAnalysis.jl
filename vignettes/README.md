# Vignettes

Thirty-one vignettes. Twenty are tutorials on illustrative examples (1 to 15, ordered
from basic to advanced, and 21 and 25 to 28); the other eleven are case studies, in a
section of their own below, grouped by field. Each
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
| 11 | [Sample beliefs and outcome tables](11_sample_beliefs_and_outcome_tables/11_sample_beliefs_and_outcome_tables.md) | `SampleBelief`, the adapter verbs and `outcomes`, `OutcomeTable` from a table of runs, `crn_correlation`, `IndividualsUtility`, `InfectionUtility`, `payoff_table` with standard errors, `single_replicate`, rules on a sample, `policy_regions`, `weight_sweep`, `evpi` with a method, `variance_components`, the sample-based report |
| 12 | [EVPPI from a sample](12_evppi_methods/12_evppi_methods.md) | `evppi` with `Grouping`, `StrongOakley`, `Sadatsafavi`, `SplineRegression`, `GAMRegression`, `GaussianProcessRegression` and the choice of method by the number of focal parameters; `conditional_problem`, bootstrap standard errors, closed-form check on the normal model of the R `voi` vignette |
| 13 | [EVSI from a sample](13_evsi_methods/13_evsi_methods.md) | `evsi` with `SummaryStatistics`, `ImportanceSampling` (`importance_loglikelihoods`) and `GaussianApproximation`; the two parts of the importance-sampling standard error and its effective sample sizes; `enbs` in animals and `optimal_design`; closed-form check |
| 14 | [Bd epizootic response](14_bd_epizootic_response/14_bd_epizootic_response.md) | The cached Drawert et al. (2017) outcome table of `Chytridiomycosis.jl`: count and infection utilities, gains with standard errors, groups as scenarios, the weight sweep, EVPI, EVPPI, and the EVSI of a pre-intervention survey by importance sampling |
| 15 | [Risk attitude and several attributes](15_risk_attitude_and_attributes/15_risk_attitude_and_attributes.md) | `LinearUtility`, `ExponentialUtility`, `PowerUtility`, `certainty_equivalent`, `risk_premium`, a non-linear utility applied per run, `AdditiveUtility`, `MultiplicativeUtility` |
| 25 | [Survey first or act now, under a budget](25_survey_or_act/25_survey_or_act.md) | `survey_or_act` and `select_under_budget` on synthetic parcels, after Raymond et al. (2020): the closed form of the rule, never, always and mixed surveying under a budget, expected against largest cost, more than one species |
| 26 | [Act now, or wait and learn?](26_value_of_waiting/26_value_of_waiting.md) | `value_of_waiting` on an invented outbreak in a herd: the value of a test result to the late decision, the cost of the delay, the best length of wait, and why EVSI with the delay ignored overstates |
| 27 | [Learning while managing](27_adaptive_management/27_adaptive_management.md) | `adaptive_management` on an invented bird colony: no learning, passive and active adaptive management and perfect information over several years, the value of updating against the value of probing, the policy tree, and how both depend on the horizon and the prior |
| 28 | [Managing a state that changes](28_changing_state/28_changing_state.md) | `adaptive_management` with `transitions` on an invented reserve and weed: the fixed plan as a schedule, why the passive manager must plan ahead, an inspection as an action taken only to learn, and which state the signal is about |

## Case studies

Each case study replicates a published value-of-information analysis from the tables
printed in the paper and its supplement. They are grouped by field: ecology, which
includes diseases of animals, and human epidemiology. A case study's directory is named
`NN_ecology_*` or `NN_epidemiology_*`, and `docs/make.jl` files it by that name. Their numbers are the papers', not
illustrations; each result is shown beside the published one, and each vignette says
what was reproduced, what was not and why, and then adds something the paper did not
report. The same comparisons, with the provenance and licence of every input, are
`test/fixtures/replications/*.json` and `test/test_replications.jl`.

### Ecology (including diseases of animals)

| | Vignette | Paper and unit | Covers |
|---|---|---|---|
| 16 | [Adaptive culling for foot-and-mouth disease](16_ecology_shea_fmd/16_ecology_shea_fmd.md) | Shea et al. 2014, doi:10.1371/journal.pbio.1001970; cost of livestock lost, GBP million (the paper's valuation) | a misprint found in the paper's Table 2; `evpi`; `two_stage_value` for the adaptive policy and the share of EVPI it recovers; regret; a credal set over the kernel weights; a coarser partition for what monitoring must establish |
| 17 | [Learning by doing for whooping cranes](17_ecology_runge_cranes/17_ecology_runge_cranes.md) | Runge et al. 2011, doi:10.1016/j.biocon.2010.12.020; weighted score on 0-1, and fledging rate | `hypothesis_partition` (the paper's "EVPXI"); `gaussian_experiment` and the EVSI of trying a strategy; the trial net of the score forgone; `robust_evpi` |
| 19 | [Biological and operational uncertainty in weed control](19_ecology_li_thistle/19_ecology_li_thistle.md) | Li et al. 2021, doi:10.1111/1365-2664.13904; population growth rate λ | a payoff table from matrix models; `evppi` by model, by effectiveness and by agent; non-additivity of partial values; a replication that fails in part, with the reason |
| 20 | [The value of imperfect water-quality monitoring](20_ecology_luhede_monitoring/20_ecology_luhede_monitoring.md) | Luhede et al. 2024, doi:10.1016/j.ecolecon.2024.108128; EUR million per year (the paper's valuation) | `evsi` of single and repeated imperfect readings; `information_can_change_decision`; break-even cost and the best programme against its price; `optimal_design`; `robust_evsi` |
| 23 | [Recovery planning for a turtle](23_ecology_christensen_turtle/23_ecology_christensen_turtle.md) | Christensen et al. 2026, doi:10.1111/csp2.70421; decision-maker value on 0-1, and years | `evpi` under four value functions and three measures; `evppi` of threat magnitude and of extent |
| 24 | [Which species to monitor, and how](24_ecology_bal_monitoring/24_ecology_bal_monitoring.md) | Bal et al. 2018, doi:10.1111/2041-210X.13037; species declining, of three | a payoff table from a model of 64 hypotheses and a budget split; `Experiment` for surveillance and for targeted monitoring with detection errors; `evsi` against the budget |
| 29 | [When to stop managing or surveying a cryptic species](29_ecology_chades_tiger/29_ecology_chades_tiger.md) | Chadès et al. 2008, doi:10.1073/pnas.0805265105; USD per year (the paper's valuation) | `adaptive_management` with `transitions`, `signal_of=:next` and `method=:vectors`; `value_function` and the switching beliefs by years left; the long-run schedule against a 30-year horizon; the four ways of managing over 30 years |

### Human epidemiology

| | Vignette | Paper and unit | Covers |
|---|---|---|---|
| 18 | [ICU beds, and EVPXI against EVPPI](18_epidemiology_eze_icu/18_epidemiology_eze_icu.md) | Eze et al. 2023, doi:10.1186/s12913-023-09479-4; mismatch in ICU beds | EVPI against a weight in the loss; the value of testing a scenario, the collapsed "per-parameter EVPI" and the EVPPI of a parameter as three different numbers; non-additivity; a credal set of the paper's four beliefs |
| 22 | [Which entomological uncertainty matters for malaria control](22_epidemiology_kim_malaria/22_epidemiology_kim_malaria.md) | Kim et al. 2017, doi:10.1111/risa.12606; USD per person over ten years (the paper's valuation) | `evpi` on four pairs of parameters; `evppi` of each parameter of a pair; overlap and complementarity of partial values |
| 30 | [Epidemiological and operational uncertainty in Ebola control](30_epidemiology_li_ebola/30_epidemiology_li_ebola.md) | Li et al. 2019, doi:10.1098/rspb.2019.0774; cases in a simulated population of 10,000 | a payoff table of 999 states (37 models × 27 cost combinations) read from the fixture; `evpi`; `evppi` by model, by cost combination and by each intervention's cost function; non-additivity; `MinimaxRegret` beside expected value |
| 31 | [Vaccinating before the efficacy is known](31_epidemiology_atkins_vaccination/31_epidemiology_atkins_vaccination.md) | Atkins et al. 2020, doi:10.1016/j.jtbi.2020.110380; cost in units of one infection, and days of outbreak (the paper's weights) | an SEIR model with vaccination, integrated in the vignette; a two-decision problem as `adaptive_management` with `transitions` carrying the first action; the paper's passive and active approaches as `passive` and `active`; how many monitored vaccinations make starting worthwhile |

Their conventions differ from those of the tutorials in two ways: **the unit is
the paper's own** (individuals or another count where the paper has no valuation, money
where the paper supplies one, always labelled as the paper's), and **no cost is
invented**: where a paper gives no cost of information, the vignette reports the
break-even cost. The baseline is the paper's no-intervention or status-quo action, or
its least intervention where it has neither (vignette 16).

## Conventions of the examples

- **In vignettes 1 to 15, utility is counted in individuals, never in money**: the number of animals of the
  focal species. For an infectious disease there are two counts, uninfected individuals
  (higher is better) and infected individuals (lower is better), combined as
  uninfected − w × infected and shown at w = 0, at w = 1 and as a sweep. Infected but
  living animals are not counted as population.
- **Every vignette has a no-intervention action**, sets it as the `baseline` of the
  `DecisionProblem`, and reports each action's gain over it.
- **All numbers of vignettes 1 to 15 are illustrative.** They are not data and not estimates for any real
  system, and each vignette says so. Vignette 14 reads a cached table of runs of a
  published model; the numbers read from that table are labelled as such, and
  everything computed from them is illustrative.
- **Vignettes 11 to 13 and 15 share a toy simulator**, `_toy_pond.jl`. **Vignette 14 needs
  `Chytridiomycosis.jl`**, which the vignette environment finds through a `[sources]` URL; the
  package itself does not depend on it. Vignette 12 loads GAM.jl for the optional regression backend.

## Rendering: the code runs once

```sh
julia scripts/render_vignettes.jl                # from the package root: all of them
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
