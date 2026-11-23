# DecisionAnalysis.jl

Standard decision analysis: payoff tables, beliefs, rules, Savage regret, experiments and the value of
information, exact on finite problems and estimated, with separate standard errors, from draws and
simulator runs. The Catlab-free core of the decision layer of the ecorecipes ecosystem.

## Place in the ecosystem

Model packages produce outcomes; this package analyses the payoff table they give. It depends on
FiniteKernels (finite spaces, kernels and states) through a `[sources]` path to the sibling directory
`../FiniteKernels.jl`, on `JuMP` (to state linear programmes; never on a solver), and on `LinearAlgebra`,
`Random`, `Statistics` and `Tables`. FiniteKernels brings
ResilienceCore and StructuredPopulationCore by URL; Pkg follows the sources of a path dependency, so they
are not listed here. The planned `CategoricalDecisionAnalysis.jl` sits one level up, imports Catlab
generics, forwards to this package and re-exports its names (ADR 0009). No model package depends on this
one: a model package writes a table of runs (or the user gives methods to the adapter verbs), and any
package-specific adapter is a weak-dependency extension on this side. `GAM` (ecorecipes/GAM.jl, by URL
source) is a weak dependency for `GAMRegression` (`ext/DecisionAnalysisGAMExt.jl`) and a test dependency;
`Chytridiomycosis` appears only in the vignette environment, by path.

## Invariants that must not be broken

- **No Catlab, GATlab or ACSets, ever**, in `[deps]`, `[extras]` or the docs environment.
- Never define or import `compose`, `otimes`, `id`, `mcopy`, `delete`, `braid`, `dom` or `codom`, and never
  export a bare `regret` or `regrets` (`CategoricalInfluenceDiagrams` exports `regrets` for a different
  quantity). `test/test_docstrings.jl` checks all of these.
- **Layouts.** `PayoffTable.Q[a, θ]`: rows = actions, columns = states. `Experiment.likelihood[z, θ] =
  P(z | θ)`: rows = signals, columns = states, columns sum to one. A garbling `G[z′, z]` is
  column-stochastic and acts as `G * L`.
- **The value-of-information definitions are a contract with the Lean development** and are written at the
  top of `src/voi.jl`: `value(e) = Σ_z max_a Σ_θ p(θ) L[z, θ] Q[a, θ]` (joint form, no division, so a signal
  of probability zero contributes zero), `V0 = max_a Σ_θ p(θ) Q[a, θ]`, `EVSI = value(e) − V0`,
  `EVPI = Σ_θ p(θ) max_a Q[a, θ] − V0`, `EVPPI = Σ_cells max_a Σ_{θ ∈ cell} p(θ) Q[a, θ] − V0`, and regret
  `R[a, θ] = max_a′ Q[a′, θ] − Q[a, θ]`. Change neither side alone.
- **The schemas `decision-certificate/1`, `garbling-witness/1`, `e-admissible-witness/1`,
  `not-e-admissible-witness/1` and `not-garbling-witness/1` and the fixture
  file `test/fixtures/worked_examples.json` are the same kind of contract.** The schemas are written down in
  `docs/src/certificates.md` and checked by `proofs/` (`lake exe check_certificate`). A witness is verified
  in exact arithmetic before it is written. A change of meaning needs a new schema string.
- Exact in, exact out. No function returns a floating-point answer for a `Rational` problem; tolerances
  default to zero for exact element types. The built-in linear-programme solver works in `Rational{BigInt}`
  for every input.
- **Optimisation goes through JuMP, and no solver is a hard dependency.** Every linear programme is written
  once, in `src/programs.jl`, as an exported builder that returns a `DecisionProgram` (a
  `JuMP.GenericModel{T}`, `T = Rational{BigInt}` for exact problems, with its variable and constraint
  references), with the programme in mathematical notation and its variable and constraint counts in the
  docstring. Never assemble a programme by hand elsewhere. `solve_program` has three routes:
  `optimizer = nothing` (default) feeds `JuMP.lp_matrix_data` to the internal simplex of `src/lp.jl`, which
  is the fallback for small problems; an exact optimiser (e.g. `CDDLib.Optimizer{Rational{BigInt}}`) solves
  the rational model; a floating-point optimiser (e.g. `HiGHS.Optimizer`) solves a `Float64` copy, and the
  primal and dual solutions are rounded and checked exactly. Solvers (CDDLib, HiGHS) appear only in the
  test extras and the vignette environment.
- **The solver is never trusted.** Whatever route produced it, a witness is verified in exact arithmetic
  before it is returned or exported, and an optimum found in floating point for an exact problem is returned
  only when certified exactly; otherwise `ExactVerificationError` is thrown. Yes/no answers say whether they
  are exact (`details.exact` of an `EAdmissible` result, `GarblingResult.exact`).
- Robust value of information is returned as computed, never clamped at zero. The rule-specific value and
  the EVPI envelope are separate fields of `RobustVOI`.
- E-admissibility is decided by linear programme over the credal set. Enumerating extreme points is wrong
  and a test (`test/test_credal.jl`) shows it.
- An operation that is not defined for a belief form throws `BeliefFormError`; it never guesses a prior.
  `Hurwicz` is over states and `AlphaMaxmin` over priors, with opposite conventions for `α`, both
  documented.
- Ties: every tied action is returned, in table order, and `first_choice` is the first. Nothing is random.
- A baseline changes no score, choice or value; it only adds the comparison with no intervention. Under a
  `CredalSet` or a `ScenarioSet`, `expected_gain_over_baseline` returns the named tuple `(lower, upper)`.
- `information_can_change_decision(problem, e)` is equivalent to `evsi(problem, e) > 0` (Lean theorem
  `evsi_eq_zero_iff`); a property test holds the two together. Minimax regret depends on the menu of
  actions (`minimaxRegret_violates_iia`); the rule's result (`details.note`) and the report say so.
- Types are immutable. Constructors copy their arrays and check them.
- **Sample-based estimators estimate the quantities defined in `src/voi.jl`**; they define nothing new.
  Every one has the form `Σ_i w_i max_a g[a, i] − max_a Σ_i w_i Q[a, i]` with `g` the fitted conditional
  means (top of `src/voi_sample.jl`), and each is tested against exact enumeration on a finite problem
  turned into a sample and against the closed forms of the R `voi` normal model (analytic EVPI 0.4798107).
- **Monte Carlo error is a separate field, never folded into a value**: `PayoffTable.standard_errors` and
  `.replicates`, and `VOIEstimate.standard_error` and `.bias`. A table with one run in a cell is flagged
  (`single_replicate`) and gets no invented standard error.
- **Selection bias is stated as proved, no more.** With noisy cells or sampled draws the act-now value,
  the perfect-information value and the value with information are each biased upwards, and the chosen
  action is on average worth less than its reported score; with exact payoffs and a sampled belief EVPI is
  biased downwards; in general **the sign of the bias of EVPI is indefinite** (Lean
  `evpi_bias_sign_indefinite`). Never write that noise "biases EVPI upwards".
- **No monetary costs anywhere.** Utility is counted in individuals; `enbs` on an estimate takes a cost in
  individuals and is for additive payoffs only.
- A `SampleBelief` is a single prior over its draws (weights normalised, exactly for integer and rational
  weights; default `1 // n`). State-wise rules on it are flagged `details.sample_dependent`; `Laplace`,
  `total_variation`, `bayes_invert` and the certificate exports throw `BeliefFormError`.
- `OutcomeTable.values[action, draw, replicate, outcome]`. Failed runs are kept and counted
  (`run_counts`, `failed_runs`) and left out of means. `outcomes` seeds the run of `(draw, replicate)`
  identically for every action (common random numbers); every random number is a seeded `Xoshiro`.

## Layout

- `src/errors.jl`: the root `DecisionAnalysisError` and every exception under it, with `showerror`.
- `src/lp.jl`: internal dense two-phase simplex in exact rational arithmetic (Bland's rule); the fallback
  solver, for small problems.
- `src/programs.jl`: `DecisionProgram` and the builders `eadmissibility_model`, `garbling_model`,
  `deficiency_model`, `evpi_envelope_model`, and `mixed_dominance_model` and `separating_payoff_model`
  for certified "no" answers (JuMP models).
- `src/solve.jl`: `solve_program` and `ProgramSolution`: the three routes, rounding, and the exact
  primal-dual check.
- `src/beliefs.jl`: `PointBelief`, `FiniteBelief`, `SampleBelief`, `CredalSet`, `ScenarioSet`; label and
  tolerance helpers.
- `src/payoff.jl`: `PayoffTable` (with optional standard errors and replicate counts), arithmetic on tables
  with matching labels, `combine_payoffs`, `individuals_payoff`, `infection_payoff`.
- `src/adapters.jl`: the method-less adapter verbs `with_parameters`, `apply_action`, `simulate`, `outcome`,
  and `parameter_draws`, `draw_weights`.
- `src/outcomes.jl`: `OutcomeTable` (from any Tables.jl table of runs) and the driver `outcomes`.
- `src/utilities.jl`: `IndividualsUtility`, `InfectionUtility`, `FunctionUtility`, `payoff_table`,
  `crn_correlation`; the count utilities are the implementation behind the matrix helpers of `payoff.jl`.
- `src/problem.jl`: `DecisionProblem`, feasibility filter, baseline and gains over it,
  `restrict_scenarios`, and the internal `_View` that aggregators act on.
- `src/aggregators.jl`: the ten aggregators and `aggregate`.
- `src/regret.jl`: regret tables (`RegretTable`, a labelled matrix) and profiles.
- `src/rules.jl`: transforms, `RobustnessMetric`, selectors, `DecisionRule`, the named rules, the
  set-valued rules, `DecisionResult`, `decide`.
- `src/experiments.jl`: `Experiment`, `bayes_invert`, `garble`, `is_garbling_of`, `garbling_result`,
  `deficiency`.
- `src/voi.jl`: `evpi`, `evppi`, `evsi`, `enbs`, `value_with_information`,
  `information_can_change_decision`, `robust_evpi`, `robust_evsi`.
- `src/voi_sample.jl`: `VOIEstimate`, the methods (`SampleAverage`, `Grouping`, `StrongOakley`,
  `Sadatsafavi`, `SplineRegression`, `GAMRegression`), the studies (`SummaryStatistics`,
  `ImportanceSampling`, `GaussianApproximation`), the method-taking `evpi`, `evppi`, `evsi`, `enbs`,
  `variance_components`, `expected_utility_estimates`, `expected_gain_estimates`, `conditional_problem`.
  A regression backend subtypes `AbstractRegression` and implements `fitted_conditional_means`.
- `src/sensitivity.jl`: `policy_regions`, `one_way_thresholds`, `weight_sweep`, `rank_reversals`.
- `ext/DecisionAnalysisGAMExt.jl`: `fitted_conditional_means` for `GAMRegression`.
- `src/fidelity.jl`: `total_variation`, `decision_loss_bound`.
- `src/report.jl`, `src/show.jl`: `DecisionReport` (named experiments, partitions, objectives), its
  Tables.jl rows, and printing. A display is a list of blocks (titles, grids, paragraphs, lists) rendered as
  plain text within 80 columns, Markdown or HTML, for `PayoffTable`, `RegretTable`, `DecisionResult` and
  `DecisionReport`.
- `src/certificates.jl`: `export_decision_certificate`, `export_garbling_witness`,
  `export_eadmissible_witness`, `export_not_eadmissible_witness`, `export_not_garbling_witness` and their
  hand-written JSON encoder.
- `proofs/`: the Lean 4 / Mathlib project `DecisionAnalysisProofs` and its `check_certificate` executable;
  see `proofs/README.md`.
- `test/helpers.jl`: the fixture reader and the slow oracles; `test/test_*.jl`: the suite, seeded
  `Random.Xoshiro` only; `test/test_programs.jl`: model shapes and the three routes agreeing (CDDLib and
  HiGHS are test dependencies); `test/test_lean_mirror.jl`: Julia mirrors of Lean lemmas; `test/fixtures/worked_examples.json`: the three worked examples.
- `docs/`: Documenter manual (`index.md`, `certificates.md`, the API reference split over `api/*.md` so that
  no page nears Documenter's 200 KiB limit, and `tutorials/`, the synced vignettes).
- `test/test_sample.jl`, `test_outcomes.jl`, `test_voi_sample.jl`, `test_sensitivity.jl`: the sample-based
  layer; `test_outcomes.jl` defines the toy simulator that the driver is tested on, and
  `test_voi_sample.jl` checks `MissingExtensionError` before it loads GAM.
- `vignettes/NN_name/NN_name.qmd`: fourteen Quarto vignettes (Julia engine), basic to advanced (1 to 10 the
  finite core, 11 to 13 sample-based analysis on the toy simulator `vignettes/_toy_pond.jl`, 14 the Bd
  application on the cached Drawert et al. 2017 table of `../Chytridiomycosis.jl`), each rendered to
  `.md`, `.html` and `.pdf` beside it; `vignettes/_common.jl` (plot helpers and a small Markdown table for
  ad hoc summaries; package objects display themselves), `vignettes/_pandoc/`
  (HTML style, LaTeX preamble). Vignette rules: utility is counted in individuals, never money (for a disease,
  uninfected − w × infected, shown at w = 0 and 1); every vignette has a no-intervention action as the
  `baseline` and reports gains over it; examples are ecological and every number is labelled illustrative
  (in vignette 14, every number not read from the cached table, and nothing is presented as a published
  result).
- `scripts/render_vignettes.jl` (run-once render) and `scripts/sync_vignettes.jl` (copy to `docs/src/tutorials`).

## Commands

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # the test suite; the only run that counts
julia --project=docs docs/make.jl --no-deploy                    # build the docs locally
DECISIONANALYSIS_LEAN=1 julia --project -e 'using Pkg; Pkg.test()'   # also run exported files through the Lean checker (needs lake and a built proofs/)
julia scripts/render_vignettes.jl [NN ...] [--no-execute]         # vignettes: code runs once (quarto to gfm), then pandoc makes HTML and PDF from the .md
julia scripts/sync_vignettes.jl [--check]                         # copy the rendered Markdown into docs/src/tutorials
```

## Files not to edit by hand

- `Manifest.toml` is generated and ignored.
- `docs/build/` is generated.
- `docs/src/tutorials/` is generated by `scripts/sync_vignettes.jl`.
- `vignettes/*/*.md`, `*.html`, `*.pdf` and `*_files/` are rendered output; edit the `.qmd` and re-render. Never
  add `html` or `pdf` to `vignettes/_quarto.yml` or a vignette's front matter (Quarto would re-run the code per
  format).
- `test/fixtures/worked_examples.json` is shared with the Lean development. Edit it only together with
  that side, and keep every number in it equal to what `lab/wiki/references/computations/worked_examples.py`
  prints.

## Style

JuliaFormatter `yas`; `snake_case` functions, `CamelCase` types, a leading underscore for internals; British
spelling in prose and American in identifiers; no emojis. A docstring on every exported name, enforced by
`test/test_docstrings.jl`. Typed exceptions in `src/errors.jl` under `DecisionAnalysisError`, carrying the
offending names (ADR 0013); invalid arguments and keywords raise `ArgumentError`; never a bare `error()`;
typed errors from FiniteKernels pass through unchanged; another package's type is a code span, never
`@ref`. Every fast path is tested against a slow oracle in `test/helpers.jl` that shares no code with it.
