# DecisionAnalysis.jl

Standard decision analysis: payoff tables, beliefs, rules, Savage regret, experiments and the value of
information, exact on finite problems and estimated, with separate standard errors, from draws and
simulator runs. The Catlab-free core of the decision layer of the ecorecipes ecosystem.

## Place in the ecosystem

Model packages produce outcomes; this package analyses the payoff table they give. It depends on
FiniteKernels (finite spaces, kernels and states) through a `[sources]` URL
(the Lean project in `proofs/` still path-requires the sibling clone `../FiniteKernels.jl`), on `JuMP` (to state linear programmes; never on a solver), and on `LinearAlgebra`,
`Logging`, `Random`, `Statistics` and `Tables`. FiniteKernels brings
ResilienceCore and StructuredPopulationCore through its own `[sources]`; Pkg follows the sources of a URL
dependency from Julia 1.13, which is why 1.13 is the minimum and why neither is listed here (on 1.12 both
had to be, and installing this package by URL failed).
Keep every `[sources]` entry a URL, in this file and in the docs, vignette and test environments. The planned `CategoricalDecisionAnalysis.jl` sits one level up, imports Catlab
generics, forwards to this package and re-exports its names (ADR 0009). No model package depends on this
one: a model package writes a table of runs (or the user gives methods to the adapter verbs). An adapter
for one model package is a weak-dependency extension inside that model package, triggered by this one;
generic bridges (GAM, InfluenceDiagrams) are extensions on this side. `GAM` (ecorecipes/GAM.jl, by URL
source) is a weak dependency for `GAMRegression` (`ext/DecisionAnalysisGAMExt.jl`) and a test dependency;
`Chytridiomycosis` appears only in the vignette environment, by URL (with DynamicalResilience, which it needs). `InfluenceDiagrams`
(ecorecipes/InfluenceDiagrams.jl) is a weak dependency for `normal_form`
(`ext/DecisionAnalysisInfluenceDiagramsExt.jl`) and **nothing more**: it brings GATlab and ACSets (not
Catlab), so it is in `[weakdeps]` only, never in `[extras]` or the docs or vignette environments, and its
tests have an environment of their own, `test/influence_diagrams/`, which lists it and its three sibling
packages by URL and runs only with `DECISIONANALYSIS_ID=1`.

## Invariants that must not be broken

- **No Catlab, GATlab or ACSets, ever**, in `[deps]`, `[extras]` or the docs environment, directly or
  through a dependency listed there (`test/test_bridge.jl` checks the direct case). The only place where
  GATlab and ACSets are installed is the gated environment `test/influence_diagrams/`.
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
  `evppi` without a method chooses by the focal parameters as `voi` does: `Grouping` when some is not
  numeric and none is floating-point, `SplineRegression` for one to four (and for a floating-point
  parameter named with a non-numeric one), `GaussianProcessRegression` from five (`_GP_FROM`).
- **`SplineRegression()` chooses the size of its basis from the draws** (`knots = :auto`): a constant
  (no dependence, value exactly zero), polynomials up to the degree, then one to four knots, by
  generalised cross-validation on the payoff differences from the act-now action; an integer `knots`
  fixes the basis. A fixed limit by draws per coefficient was tried and is wrong in both directions
  (`validation/README.md`): the overfitting depends on the noise of the payoffs, not on the sample size
  alone. `details.degree`, `.knots` and `.gcv` record the choice; a constant adds a note to the report.
- **No information is worth more than perfect information, and an estimate that says otherwise is
  flagged, not capped.** Every estimate made from fitted conditional means carries
  `details.sample_evpi`, `details.policy_gain` and `details.above_sample_evpi`; the flag (value above the
  EVPI of the same sample by more than twice the standard error) logs a warning and adds a note to the
  report. Cell-mean methods never raise it. `Grouping()` without `bins` throws on a floating-point
  parameter that makes most draws cells of their own.
- **The R package `voi` is GPL-3 and this package is MIT: nothing is translated from its source.** The
  methods that follow it (`docs/src/api/methods.md` has the table of correspondences) are written from the
  published papers and its documented behaviour, and compared with it numerically through
  `validation/voi_extra_reference.R` and `test/fixtures/voi/voi_extra_reference.json`. Not available:
  the INLA/SPDE Gaussian process (INLA is an R package) and BART. Two deliberate differences:
  `population_value` at a zero discount rate is value × population × horizon (the limit of the formula;
  `pop_voi` returns value × population), and `MARSRegression` takes knots from a grid of quantiles.
- **A model-based estimator needs the function the table was made from.** `MomentMatching` and
  `evppi_nested` take `payoffs(θ)`; for `MomentMatching` it must return the payoffs of the feasible actions
  in the problem's order and reproduce the payoff table, and the non-focal parameters are drawn from the
  draws independently of the focal ones. Its standard error covers the draws and the nested data sets
  (jackknife), not the approximation.
- **A non-linear utility is applied to each run before anything is averaged** (`_run_utilities`), and
  its expected utility is not a count of individuals; `certainty_equivalent` returns to individuals. A
  test holds `InfectionUtility` equal to the `AdditiveUtility` with unscaled counts.
- **The standard error of an estimate covers every source of its Monte Carlo error.** For
  `ImportanceSampling` it is `√(datasets² + draws²)` (delta method for the draws), with both parts and the
  posterior effective sample sizes in `details`; the calibration numbers quoted in the docstrings of
  `ImportanceSampling` and `GaussianApproximation` are measured, and `test/test_standard_errors.jl` holds
  the standard errors against the spread over independent repetitions.
- **Cross-language fixtures are contracts too.** `test/fixtures/voi/` and `test/fixtures/validation/` are
  written by the scripts of `validation/` (R `voi` and `cvar`; Python with scipy; Python 2.7 with
  `improb`) and carry their
  provenance. Never edit them by hand: re-run the script and report any number that moved. The test
  suite reads them and needs neither R nor Python; the chemotherapy data of `voi` are not copied here
  (money appears only as that validation target), and their comparison runs with `DECISIONANALYSIS_R=1`.
- **Monte Carlo error is a separate field, never folded into a value**: `PayoffTable.standard_errors` and
  `.replicates`, and `VOIEstimate.standard_error` and `.bias`. A table with one run in a cell is flagged
  (`single_replicate`) and gets no invented standard error.
- **Selection bias is stated as proved, no more.** With noisy cells or sampled draws the act-now value,
  the perfect-information value and the value with information are each biased upwards, and the chosen
  action is on average worth less than its reported score; with exact payoffs and a sampled belief EVPI is
  biased downwards; in general **the sign of the bias of EVPI is indefinite** (Lean
  `evpi_bias_sign_indefinite`). Never write that noise "biases EVPI upwards".
- **No monetary costs in the package's own examples.** Utility is counted in individuals; `enbs` on an
  estimate takes a cost in individuals and is for additive payoffs only. Money appears only as a published
  figure in its paper's own unit: the `voi` validation target, and the replication fixtures and vignettes
  whose source values outcomes in money (Shea et al. 2014, Luhede et al. 2024, Chadès et al. 2008, de Vries et al. 2021).
- **Replication fixtures are contracts with the literature.** `test/fixtures/replications/*.json` hold the
  small input tables of published analyses, the published numbers, and for each its DOI, table or figure,
  licence and how it was transcribed; `resolution` fields record why a number of ours differs from the
  paper's. Change a target only against the paper itself, never to make a test pass. `two_stage_value` is
  derived from the definitions of `src/voi.jl` and has no Lean counterpart; `hypothesis_partition` only
  builds a partition for `evppi`.
- A `SampleBelief` is a single prior over its draws (weights normalised, exactly for integer and rational
  weights; default `1 // n`). State-wise rules on it are flagged `details.sample_dependent`; `Laplace`,
  `total_variation`, `bayes_invert` and the certificate exports throw `BeliefFormError`.
- `OutcomeTable.values[action, draw, replicate, outcome]`. Failed runs are kept and counted
  (`run_counts`, `failed_runs`) and left out of means. `outcomes` seeds the run of `(draw, replicate)`
  identically for every action (common random numbers); every random number is a seeded `Xoshiro`. With
  `run_seed = (θ, r) -> integer` the seed of a run comes from the model's own convention instead of
  `seed`, so that new runs share random numbers with runs stored elsewhere; `combine_actions` joins the
  tables of the same draws and replicates (it cannot check the seeds, and its docstring says so).

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
  and `parameter_draws`, `draw_weights`. The driver calls `outcome(prob, solution)`, whose only method here is
  the fallback to `outcome(solution)`; a model package gives the two-argument form a method on its problem type
  when its solution is of a type it does not own.
- `src/outcomes.jl`: `OutcomeTable` (from any Tables.jl table of runs) and the driver `outcomes`.
- `src/utilities.jl`: `IndividualsUtility`, `InfectionUtility`, `FunctionUtility`, `payoff_table`,
  `crn_correlation`; the count utilities are the implementation behind the matrix helpers of `payoff.jl`.
  Then risk attitude (`LinearUtility`, `ExponentialUtility`, `PowerUtility`, `inverse_utility`,
  `certainty_equivalent`, `risk_premium`) and the multi-attribute `AdditiveUtility` and
  `MultiplicativeUtility`.
- `src/problem.jl`: `DecisionProblem`, feasibility filter, baseline and gains over it,
  `restrict_scenarios`, and the internal `_View` that aggregators act on.
- `src/aggregators.jl`: the ten aggregators and `aggregate`.
- `src/regret.jl`: regret tables (`RegretTable`, a labelled matrix) and profiles.
- `src/rules.jl`: transforms, `RobustnessMetric`, selectors, `DecisionRule`, the named rules, the
  set-valued rules, `DecisionResult`, `decide`.
- `src/experiments.jl`: `Experiment`, `gaussian_experiment`, `bayes_invert`, `garble`, `is_garbling_of`, `garbling_result`,
  `deficiency`.
- `src/voi.jl`: `evpi`, `evppi`, `hypothesis_partition`, `two_stage_value`, `evsi`, `enbs`,
  `value_with_information`,
  `information_can_change_decision`, `robust_evpi`, `robust_evsi` (`RobustVOI.upper_prior`), and the
  stub of `normal_form`, whose method is in `ext/DecisionAnalysisInfluenceDiagramsExt.jl`.
- `src/voi_sample.jl`: `VOIEstimate`, the methods (`SampleAverage`, `Grouping`, `StrongOakley`,
  `Sadatsafavi`, `SplineRegression`, `GAMRegression`), the studies (`SummaryStatistics`,
  `ImportanceSampling`, `GaussianApproximation`), the method-taking `evpi`, `evppi`, `evsi`, `enbs`,
  `variance_components`, `expected_utility_estimates`, `expected_gain_estimates`, `conditional_problem`.
  A regression backend subtypes `AbstractRegression` and implements `fitted_conditional_means`.
- `src/gaussian_process.jl`: `GaussianProcessRegression`, a small internal Gaussian process (linear mean,
  squared-exponential kernel, nugget, hyperparameters by marginal likelihood with a deterministic
  Nelder-Mead search, seeded subsamples), for five or more focal parameters.
- `src/mars.jl`: `MARSRegression`, multivariate adaptive regression splines written from Friedman (1991):
  greedy pairs of hinges at a grid of weighted quantiles, pruned by generalised cross-validation, one basis
  for all actions.
- `src/studies.jl`: the built-in study designs `BinomialStudy`, `TwoArmBinomialStudy`, `NormalStudy` with
  `simulate_study`, `study_loglikelihood`, `study_posterior` (conjugate), and the constructors of
  `SummaryStatistics` and `ImportanceSampling` from a design. Small internal samplers (binomial, gamma,
  beta), so that no distributions package is needed.
- `src/voi_model.jl`: the estimators that run the model: `evppi_nested` (two-level Monte Carlo) and
  `MomentMatching` with its `evsi` method.
- `src/design.jl`: `voi_profile`, `expected_variance_reduction`, `optimal_design`, and `value_of_waiting`
  (act now against wait, learn and act late, with a second payoff table for late action on the same
  horizon; `value = evsi_delayed − delay_cost`; backed by `proofs/DecisionAnalysisProofs/Finite/Waiting.lean`).
- `src/adaptive.jl`: `adaptive_management`, the finite-horizon recursion over unnormalised beliefs for an
  unknown state with action-dependent experiments; the state is fixed, or moves by `transitions`
  (`T[θ′, θ]`, column-stochastic like a likelihood; `signal_of` says whether a signal is about the state
  before or after the move). Returns `no_learning ≤ passive ≤ active ≤ perfect` and the active policy
  tree; the order is proved in `proofs/DecisionAnalysisProofs/Finite/Adaptive.lean` and, for a moving
  state, `Finite/AdaptiveDynamic.lean`. **With a moving state `no_learning` is the best fixed plan,
  `passive` replans each period and takes the plan's first action, and `perfect` is the state seen
  every period**; a passive manager that took the best action for the current period alone could fall
  below the fixed plan, so do not define it that way. The file is built from top-level functions over
  a small model type (`_Sequential`), one way of holding the value for each method (`_ByBeliefs`,
  `_BySets`) and one for each way of finding the fixed plan; **do not go back to nested closures**: an
  inner function that assigns a name also used by the enclosing function writes to the outer variable,
  which gave wrong values here once and in `evsi(::MomentMatching)` once. Two exact methods: `:beliefs` (default, every reachable belief once) and, for two
  states only, `:vectors` (the value function as the upper envelope of lines, one set per number of
  periods left, returned as `value_function`; backed by `Finite/ValueVectors.lean`), which reaches long
  horizons when the state moves. For three or more states exact pruning needs a linear programme per
  vector and was slower than `:beliefs` on every problem tried, so it is refused, not approximated.
  For a fixed state with repeated trials the lines are as many as the beliefs: use `:beliefs`. The
  passive manager is always found by following its beliefs (`passive=false` skips it). A third method,
  `:points`, is for any number of states and is **not exact**: it backs the value function up at the
  beliefs of a grid and returns `active` as a lower bound (the value of a policy that can be carried out)
  and `active_upper` as an upper bound (the "sawtooth" interpolation of a convex function), both proved
  in `Finite/ValueVectors.lean`. Never present its `active` as the value; report the pair. With it the
  fixed plan and the passive manager are computed only where the state is fixed or moves in the same way
  whatever is done, and are `nothing` otherwise.
- `src/survey.jl`: `survey_or_act` (act now or survey first, by value per unit of expected cost, after
  Raymond et al. 2020; values are gains over the baseline, the baseline costs zero, a tie means act now,
  and the survey policy is the best over all policies unless one is imposed) and `select_under_budget`
  (ranking by value per cost; a heuristic, and the budget holds in expected cost only). Backed by
  `proofs/DecisionAnalysisProofs/Finite/SurveyOrAct.lean`.
- `src/estimation.jl`: value of information for estimation from a `SampleBelief` alone
  (`expected_variance_reduction` with a study, `variance_reduction_table`,
  `equivalent_binomial_sample_size`), `population_value`, `enbs_curve`, `optimal_sample_size`, and the
  checks `regression_check` and `compare_regressions` (cross-validation by zero weights).
- `src/scenarios.jl`: scenario designs after Runge et al. (2024): `ScenarioAxis` (a decision or an
  uncertainty, stated, never guessed), `ScenarioDesign` with its class and checked purpose, `scenarios`,
  `design_actions`, `design_draws`, `design_belief` (a `ScenarioSet` unless every uncertainty axis has
  probabilities), `DecisionProblem(design, table, utility)`, `fractional_design`, `factor_effects`,
  `scenario_forecast` and `decision_robustness`.
- `src/sensitivity.jl`: `policy_regions`, `one_way_thresholds`, `weight_sweep`, `rank_reversals`.
- `ext/DecisionAnalysisGAMExt.jl`: `fitted_conditional_means` for `GAMRegression`.
- `src/fidelity.jl`: `total_variation`, `decision_loss_bound`, and the constants `payoff_range`,
  `largest_action_range`, `largest_regret` with `deficiency_value_bound`.
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
  `test_voi_sample.jl` checks `MissingExtensionError` before it loads GAM. `test_utilities.jl` (risk
  attitude and multi-attribute utilities), `test_gaussian_process.jl`, `test_standard_errors.jl`
  (standard errors against repetitions; the exhaustive Sadatsafavi search), `test_design.jl`, `test_survey.jl`, `test_adaptive.jl`,
  `test_additions.jl` (separating problem, upper-EVPI prior, the maximality shortcut, sharper
  constants), `test_voi_methods.jl` (the methods that follow `voi`: closed forms of its normal model,
  explicit loops, and the fixture `voi_extra_reference.json`), `test_validation.jl` and `test_validation_improb.jl` (the cross-language fixtures),
  `test_replications.jl` (published analyses, from `test/fixtures/replications/`) and `test_bridge.jl` (the
  influence-diagram extension; its real tests are `test/influence_diagrams/runtests.jl`).
- `validation/`: the scripts that write the cross-language fixtures (`voi_reference.R`,
  `cvar_reference.R`, `chemo_data.R`, `finite_reference.py`, `improb_reference.py`, `make_sample.jl`),
  `compare_voi.jl`, which prints our estimates beside those of `voi`, `requirements.txt`,
  `requirements-py27.txt` and a README with the versions used and the findings. The Python environments
  are `validation/.venv` and, for `improb` (Python 2.7), `validation/.venv27` (both ignored).
- `vignettes/NN_name/NN_name.qmd`: thirty-four Quarto vignettes (Julia engine): tutorials 1 to 20, case studies in ecology 21 to 28 and in human epidemiology 29 to 34, so that each group is a contiguous block (a new tutorial or ecology case study means renumbering what follows). Tutorials 1 to 15 run basic to advanced (1 to 10 the
  finite core, 11 to 13 sample-based analysis on the toy simulator `vignettes/_toy_pond.jl`, 14 the Bd
  application on the cached Drawert et al. 2017 table of `../Chytridiomycosis.jl`, 15 risk attitude and
  multi-attribute utilities on the toy simulator), 16 is the ecological example of the methods that follow `voi` and of scenario designs (a stage-structured frog population), 17 `survey_or_act` on synthetic parcels, 18 `value_of_waiting` on an invented outbreak, 19 `adaptive_management` on an invented colony, 20 the same with a state that moves (`transitions`, and the `:vectors` and `:points` methods) on an invented reserve, and fourteen case studies, each a replication of a published analysis, grouped by field in the
  directory name (`NN_ecology_*`, which includes diseases of animals: 21 Shea et al. 2014, 22 Runge et al.
  2011, 23 Li et al. 2021, 24 Luhede et al. 2024, 25 Christensen et al. 2026, 26 Bal et al. 2018, 27 Chadès
  et al. 2008, 28 Regan et al. 2006 with Rout et al. 2009; `NN_epidemiology_*`, human: 29 Eze et al. 2023, 30 Kim et al. 2017, 31 Li et al. 2019, which reads its 999-state table from the test fixture by a line pattern, so keep one row per line there; 32 Atkins et al. 2020, whose epidemic model is rebuilt in the vignette and in `test/test_replications.jl`; 33 de Vries et al. 2021; 34 Shea et al. 2023, which also reads its table from the fixture line by line, and whose value-of-information part is ours, not the paper's). Their titles do not
  say "replication"; the opening section "About this case study" does. `docs/make.jl` lists them under
  "Case studies", by field, from the directory name), each rendered to
  `.md`, `.html` and `.pdf` beside it; `vignettes/_common.jl` (plot helpers and a small Markdown table for
  ad hoc summaries; package objects display themselves), `vignettes/_pandoc/`
  (HTML style, LaTeX preamble). Vignette rules: utility is in the example's own unit, individuals where no
  valuation exists (for a disease, uninfected − w × infected, shown at w = 0 and 1) and money where the
  source provides one, with the unit and its source stated plainly and no cost ever invented (where a paper
  gives no cost of information, report the break-even cost); every vignette has a no-intervention or
  status-quo action as the `baseline` and reports gains over it; in vignettes 1 to 15 examples are
  ecological and every number is labelled illustrative (in vignette 14, every number not read from the
  cached table, and nothing is presented as a published result); in the case studies every input is a
  table printed in the paper or its supplement, every result is shown beside the published one, and what
  was not reproduced is said, with the reason.
- `scripts/render_vignettes.jl` (run-once render) and `scripts/sync_vignettes.jl` (copy to `docs/src/tutorials`).

## Commands

```sh
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'   # the test suite; the only run that counts
julia --project=docs docs/make.jl --no-deploy                    # build the docs locally
DECISIONANALYSIS_LEAN=1 julia --project -e 'using Pkg; Pkg.test()'   # also run exported files through the Lean checker (needs lake and a built proofs/)
DECISIONANALYSIS_R=1 julia --project -e 'using Pkg; Pkg.test()'      # also compare with voi's chemotherapy example (needs Rscript and the R package voi)
DECISIONANALYSIS_ID=1 julia --project -e 'using Pkg; Pkg.test()'     # also run the influence-diagram bridge tests in their own environment
Rscript validation/voi_reference.R && Rscript validation/cvar_reference.R   # rewrite the R fixtures
Rscript validation/voi_extra_reference.R                                    # the fixture of the later methods (needs earth)
validation/.venv/bin/python validation/finite_reference.py           # rewrite the Python fixture (see validation/README.md)
validation/.venv27/bin/python validation/improb_reference.py         # rewrite the improb fixture (Python 2.7; after finite_reference.py)
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
- `test/fixtures/voi/` and `test/fixtures/validation/` are written by the scripts of `validation/`.
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
