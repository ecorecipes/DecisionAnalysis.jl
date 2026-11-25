# DecisionAnalysis.jl

[![Docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://ecorecipes.github.io/DecisionAnalysis.jl/dev/)
[![Build Status](https://github.com/ecorecipes/DecisionAnalysis.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ecorecipes/DecisionAnalysis.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Standard decision analysis on the output of any model, with no category theory required: a payoff table,
a belief about its states, rules for choosing, Savage regret, and the value of information. Finite
problems are exact: give the package `Rational` inputs and every number it returns is a rational. Model
output held as draws and simulator runs is analysed by sample-based estimators of the same quantities,
each reported with its Monte Carlo standard error in a separate field.

The package depends on [FiniteKernels.jl](https://github.com/ecorecipes/FiniteKernels.jl),
[JuMP](https://jump.dev) (to state its linear programmes; no solver is a dependency), `LinearAlgebra`,
`Random`, `Statistics` and `Tables`, and on no Catlab, GATlab or ACSets. [GAM.jl](https://github.com/ecorecipes/GAM.jl)
is an optional regression backend and [InfluenceDiagrams.jl](https://github.com/ecorecipes/InfluenceDiagrams.jl)
an optional source of problems, each loaded as a weak-dependency extension.

## Features

- **Payoff tables and beliefs.** `PayoffTable` holds the conditional expected utility `Q[a, θ]` (rows =
  actions, columns = states) with any real element type. A belief is a `PointBelief`, a `FiniteBelief` (a
  probability vector, also built from a FiniteKernels state), a `CredalSet` (the convex hull of a list of
  priors) or a `ScenarioSet` (possible states, no probabilities). `DecisionProblem(payoff, belief)` pairs
  them, with an optional feasibility filter on the actions and an optional `baseline` action.
- **Utilities counted in individuals.** Tables with the same labels add, subtract and scale (`+`, `-`,
  scalar `*`), and `combine_payoffs(t1 => w1, t2 => w2, ...)` is their weighted sum, so a payoff table is
  built from tables of outcome counts without assembling a matrix by hand. `individuals_payoff(counts)` is
  the utility that counts individuals; `infection_payoff(; uninfected, infected, weight = 1)` is
  uninfected − weight × infected.
- **Comparison with no intervention.** Name the business-as-usual action as the `baseline` of a problem;
  `gain_over_baseline` gives the table of differences from it and `expected_gain_over_baseline` each
  action's expected gain (under a credal set, its lower and upper expected gain; under a scenario set, its
  minimum and maximum gain), and a report says for every rule whether it recommends the baseline.
- **Aggregators.** `Expectation`, `WorstCase`, `BestCase`, `Hurwicz(α)` (over states), `Laplace`,
  `LowerExpectation`, `UpperExpectation`, `AlphaMaxmin(α)` (over priors), `CVaR(α)`, `Quantile(α)`.
- **Regret.** `regret_table`, `relative_regret_table` (matrices that carry their labels, `RegretTable`),
  `expected_regret`, `max_regret`.
- **Rules.** `decide(problem, rule)` with `MaxExpectedUtility`, `Maximin`, `Maximax`, `MinimaxRegret`,
  `GammaMaximin`, `Satisfice(threshold)`, and the set-valued `EAdmissible` (a linear programme over the
  credal set, not an enumeration of its extreme points), `Maximal`, `IntervalDominance` and `ParetoFront`
  (over a vector of problems, or a named collection of payoff tables sharing one belief).
  A scalar rule is an aggregator, an optional transform and a selector (`DecisionRule`); the same pieces
  without the selector are a `RobustnessMetric`. Ties are all returned, and the first choice is the tied
  action that comes first in the table. Minimax regret depends on the menu of actions, and its result
  says so.
- **Experiments.** `Experiment(L)` with `L[z, θ] = P(z | θ)` (rows = signals, columns = states), also built
  from a `FiniteKernel`; `bayes_invert`, `garble`, `is_garbling_of` (returns a witness matrix),
  `garbling_result` (the same with a statement of whether the answer is exact) and `deficiency`, the last
  three by linear programme.
- **Value of information.** `evpi`, `evppi` (over a partition of the states; `hypothesis_partition` for the
  value of confirming or refuting one hypothesis), `two_stage_value` (an adaptive policy that chooses a first
  stage, learns, and chooses again), `evsi` (`gaussian_experiment` for a measurement with normal error), `enbs`,
  `value_with_information`, and `information_can_change_decision` (true exactly when EVSI is positive: no
  single action is optimal after every signal); under a credal set `robust_evpi` and `robust_evsi`, which keep the
  rule-specific value and the EVPI envelope apart and never clamp a negative value.
- **Linear programmes through JuMP.** Each programme is written once as a `JuMP.GenericModel` by an
  exported builder (`eadmissibility_model`, `garbling_model`, `deficiency_model`, `evpi_envelope_model`),
  and the functions that use them take an `optimizer` keyword. Three routes solve the same model: the
  built-in exact simplex (the default, no solver needed, for small problems), an exact optimiser such as
  `CDDLib.Optimizer{Rational{BigInt}}`, or a floating-point optimiser such as `HiGHS.Optimizer`, whose
  solution is rounded to rationals and verified exactly (`ExactVerificationError` if it does not verify).
  The solver is never trusted: every witness is re-checked in exact arithmetic before it is returned.
- **Sample beliefs and outcome tables.** `SampleBelief` holds draws of named parameters (any Tables.jl
  table) with weights, and is a single prior: expectation, CVaR, quantiles, regret and EVPI apply to it,
  worst and best cases over the draws are flagged as sample-dependent, and operations with no meaning for
  a sample throw `BeliefFormError`. `OutcomeTable` holds model output by draw × action × replicate × named
  outcome, built from any table of runs by naming its columns (failed runs are kept and counted) or by
  `outcomes(prob, actions, draws; replicates, seed)`, which drives a model through four adapter verbs
  (`with_parameters`, `apply_action`, `simulate`, `outcome`) with common random numbers across actions.
- **Utilities on outcomes, in individuals.** `IndividualsUtility`, `InfectionUtility(; uninfected, infected,
  weight)` and `FunctionUtility`; `payoff_table(outcomes, utility)` gives the conditional expected utility
  of each cell with its standard error and replicate count in separate fields, and with `baseline` the
  paired gains over no intervention. A table built from one run per cell is flagged. No monetary costs.
- **Risk attitude and several attributes.** `LinearUtility`, `ExponentialUtility` (constant absolute risk
  aversion) and `PowerUtility` (constant relative risk aversion) on a named outcome, optionally scaled to
  [0, 1] between a worst and a best level; `certainty_equivalent`, `risk_premium` and `inverse_utility`;
  `AdditiveUtility` and the Keeney-Raiffa `MultiplicativeUtility` over named attributes with weights. A
  non-linear utility is applied to every run before anything is averaged; its expected utility is no
  longer a count of individuals, and the certainty equivalent returns to individuals.
  `InfectionUtility` is the additive utility with unscaled counts.
- **Sample-based value of information.** `evpi`, `evppi` and `evsi` called with a method return a
  `VOIEstimate`: the value, its standard error, and the two selection biases, never folded together.
  EVPI by sample average or by all-parameter regression, with `variance_components` for the split between
  parameter and simulation variance; EVPPI by `Grouping` (discrete parameters), `StrongOakley` and
  `Sadatsafavi` (one parameter, no regression), `SplineRegression` (built-in additive regression spline),
  `GAMRegression` (GAM.jl extension) and `GaussianProcessRegression` (built in, for five or more focal
  parameters); without a method `evppi` chooses by the focal parameters, as the R package `voi` does.
  EVSI by `SummaryStatistics` (regression on summaries), `ImportanceSampling` (from a log-likelihood,
  with the posterior effective sample sizes and a standard error that covers both the simulated data
  sets and the draws) and `GaussianApproximation` (curves over sample size).
  Every estimator is tested against exact enumeration on finite problems and against closed forms.
- **Outcomes, estimation and design.** `voi_profile` gives the value of an experiment outcome by outcome
  (with its standard deviation and the probability that the study changes nothing);
  `expected_variance_reduction` is the value of information for estimating a quantity;
  `optimal_design` compares candidate experiments, studies or sample sizes by EVSI, or by net benefit in
  individuals where payoffs are additive.
- **Survey or act, under a budget.** `survey_or_act` compares acting on a management unit now with
  surveying it first, by value per unit of expected cost, when actions and the survey both cost money
  (after Raymond et al. 2020); `select_under_budget` then chooses units for a fixed budget.
- **Waiting to learn.** `value_of_waiting` compares acting now with waiting, learning and acting late,
  given the payoffs of late action on the same outcome horizon; it splits the value into what the
  information is worth to the late decision and what the delay costs, for one or several lengths of wait.
- **Sequential adaptive management.** `adaptive_management` values a decision repeated over a horizon
  when each action also reveals something: no learning, passive (update, but never act in order to
  learn), active (the best policy, by exact recursion over beliefs) and perfect information, with the
  active policy as a tree.
- **Influence diagrams.** With InfluenceDiagrams.jl loaded, `normal_form(model)` turns a single-decision
  influence diagram into a `DecisionProblem` and its information into an `Experiment`; the diagram's
  structural values of information are then `evpi` and `evsi` on the table.
- **Checked against other software.** The scripts in [`validation/`](validation/README.md) compare the
  package with the R packages `voi` and `cvar` and with independent Python code and scipy's `linprog`;
  the fixtures they write are read by the test suite, which needs neither R nor Python.
- **Sensitivity.** `policy_regions` and `one_way_thresholds` (exact switch points of a weight, such as the
  weight on infected individuals), `weight_sweep`, `rank_reversals`.
- **Fidelity.** `total_variation` and `decision_loss_bound(B, ε) = 2Bε`, with the sharper constants
  `largest_action_range` and `largest_regret` (`deficiency_value_bound`) beside `payoff_range`.
- **Reports and certificates.** `DecisionReport` puts every applicable rule side by side with the regret
  profile, EVPI, the EVSI of several named experiments (and whether each can change the decision), the
  EVPPI of named partitions, the robust sets, an optional Pareto section over several objectives, the gain
  over the baseline (a range under an imprecise belief) and the assumptions. It prints as aligned tables
  that fit 80 columns and is a Tables.jl table. Payoff tables, regret tables, rule results and reports
  also render as Markdown and HTML tables, so they display as tables in Quarto and notebooks.
  `export_decision_certificate` writes a rational problem and its claimed numbers as JSON for the Lean
  checker in `proofs/`; `export_garbling_witness` and `export_eadmissible_witness` write the witnesses that
  the linear programmes find, checked exactly before they are written.

## Installation

The ecosystem packages are not registered; they are found through `[sources]` URLs, so one
command installs the package:

```julia
using Pkg
Pkg.add(url="https://github.com/ecorecipes/DecisionAnalysis.jl")
```

To work on the package, clone it and run the suite. Only the Lean project in `proofs/` needs
`FiniteKernels.jl` cloned beside it:

```sh
git clone https://github.com/ecorecipes/FiniteKernels.jl
git clone https://github.com/ecorecipes/DecisionAnalysis.jl
cd DecisionAnalysis.jl
julia --project -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

## Quick start

```julia
using DecisionAnalysis

payoff = PayoffTable([:do_nothing, :raise_levee, :relocate_assets],
                     [:low, :moderate, :high, :extreme],
                     [0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48])
prior = [1 // 2, 3 // 10, 3 // 20, 1 // 20]
problem = DecisionProblem(payoff, prior; baseline=:do_nothing)

decide(problem, MaxExpectedUtility()).chosen   # [:do_nothing]
decide(problem, Maximin()).chosen              # [:relocate_assets]
decide(problem, MinimaxRegret()).chosen        # [:raise_levee]
evpi(problem)                                  # 207//20
expected_gain_over_baseline(problem)           # [0, -19//4, -57//2]

gauge = Experiment([9//10 3//5 1//5 1//20; 1//10 2//5 4//5 19//20];
                   signals=[:no_warning, :warning])
evsi(problem, gauge)                           # 97//20
noisy = garble(gauge, [4//5 1//5; 1//5 4//5])
evsi(problem, noisy)                           # 49//25, lower: a garbling is worth less
is_garbling_of(noisy, gauge)                   # the witness matrix

ensemble = CredalSet(payoff.states, [prior, [7 // 20, 3 // 10, 1 // 4, 1 // 10],
                                     [3 // 5, 1 // 4, 3 // 25, 3 // 100]])
robust = DecisionProblem(payoff, ensemble)
decide(robust, GammaMaximin()).chosen          # [:raise_levee]
decide(robust, EAdmissible()).chosen           # [:do_nothing, :raise_levee]

DecisionReport(problem; credal=ensemble, experiments=(; gauge, noisy))

# The same question by another route: any JuMP optimiser, exact or floating point.
# using CDDLib, HiGHS
# decide(robust, EAdmissible(); optimizer=CDDLib.Optimizer{Rational{BigInt}})
# decide(robust, EAdmissible(); optimizer=HiGHS.Optimizer)   # rounded, then verified exactly
solve_program(eadmissibility_model(robust, :raise_levee)).objective   # 0//1: E-admissible
information_can_change_decision(problem, gauge)                       # true: EVSI > 0
```

## Vignettes

Twenty vignettes in [`vignettes/`](vignettes/). The first fifteen go from a first decision table to
Lean-checked certificates, sample-based analysis and an application to chytridiomycosis,
on ecological examples whose outcomes are counted in individuals and always compared with no intervention
(all numbers illustrative). Vignettes 16 to 20 and 22 to 24 are replications of published analyses, listed after the
table. Each is rendered to Markdown, HTML and PDF beside its `.qmd`, with the code run
once: `julia scripts/render_vignettes.jl` (see [`vignettes/README.md`](vignettes/README.md)). The Markdown
is also published as the Tutorials of the documentation, copied by `julia scripts/sync_vignettes.jl`.

| | Vignette | Topics |
|---|---|---|
| 1 | [A first decision table](vignettes/01_first_decision_table/01_first_decision_table.md) | Actions, states, a belief, expected number of individuals, the baseline and gain over it |
| 2 | [Decision rules side by side](vignettes/02_decision_rules/02_decision_rules.md) | Expected utility, maximin, maximax, Hurwicz, Laplace, minimax regret, satisficing |
| 3 | [Regret](vignettes/03_regret/03_regret.md) | Regret tables, the identity with EVPI, menu-dependence of minimax regret |
| 4 | [Experiments and the value of sample information](vignettes/04_experiments_and_evsi/04_experiments_and_evsi.md) | Likelihoods, Bayesian inversion, EVSI, zero-value information, net benefit |
| 5 | [Partial information](vignettes/05_partial_information/05_partial_information.md) | EVPPI over partitions of the states |
| 6 | [Comparing experiments](vignettes/06_comparing_experiments/06_comparing_experiments.md) | Garbling, deficiency, incomparable designs |
| 7 | [Uncertain beliefs](vignettes/07_uncertain_beliefs/07_uncertain_beliefs.md) | Credal sets, scenario sets, robust rules, robust value of information |
| 8 | [Several objectives](vignettes/08_several_objectives/08_several_objectives.md) | The Pareto front, infected against uninfected counts |
| 9 | [The decision report](vignettes/09_decision_report/09_decision_report.md) | `DecisionReport` and its Tables.jl form |
| 10 | [Exact arithmetic and certificates](vignettes/10_exact_arithmetic_and_certificates/10_exact_arithmetic_and_certificates.md) | Rational problems, certificates and witnesses, the Lean checker |
| 11 | [Sample beliefs and outcome tables](vignettes/11_sample_beliefs_and_outcome_tables/11_sample_beliefs_and_outcome_tables.md) | Draws, a toy simulator behind the adapter verbs, common random numbers, payoffs with standard errors |
| 12 | [EVPPI from a sample](vignettes/12_evppi_methods/12_evppi_methods.md) | Grouping, Strong and Oakley, Sadatsafavi, regression splines, GAM.jl, a Gaussian process for five or more parameters; a check against closed forms |
| 13 | [EVSI from a sample](vignettes/13_evsi_methods/13_evsi_methods.md) | Regression on summaries, importance sampling with its effective sample size, the Gaussian approximation, net benefit in animals, `optimal_design` |
| 14 | [Bd epizootic response](vignettes/14_bd_epizootic_response/14_bd_epizootic_response.md) | The Drawert et al. (2017) model from a cached table of runs: gains, rules, groups, EVPI, EVPPI and the EVSI of a survey |
| 15 | [Risk attitude and several attributes](vignettes/15_risk_attitude_and_attributes/15_risk_attitude_and_attributes.md) | Certainty equivalents and risk premiums in animals, exponential and power utilities on simulator runs, additive and multiplicative utilities over uninfected and infected counts |
| 21 | [Which vital rate to measure](vignettes/21_estimation_studies_and_scenarios/21_estimation_studies_and_scenarios.md) | Value of information for a model that estimates (variance removed, by input and output), the EVSI of standard study designs by regression, Gaussian approximation and moment matching, a check by two-level Monte Carlo, the sample size with the largest net benefit, and scenario designs (value of information and sensitivity designs, a half fraction), after Jackson et al. (2021, 2022) and Runge et al. (2024) |
| 25 | [Survey first or act now, under a budget](vignettes/25_survey_or_act/25_survey_or_act.md) | `survey_or_act` and `select_under_budget` on synthetic parcels, after Raymond et al. (2020): the closed form of the rule, never, always and mixed surveying under a budget, expected against largest cost, more than one species |
| 26 | [Act now, or wait and learn?](vignettes/26_value_of_waiting/26_value_of_waiting.md) | `value_of_waiting` on an invented outbreak in a herd: the value of a test result to the late decision, the cost of the delay, the best length of wait, and why EVSI with the delay ignored overstates |
| 27 | [Learning while managing](vignettes/27_adaptive_management/27_adaptive_management.md) | `adaptive_management` on an invented bird colony: no learning, passive and active adaptive management and perfect information over several years, the value of updating against the value of probing, the policy tree, and how both depend on the horizon and the prior |

### Replications of published analyses

Five vignettes reproduce value-of-information analyses from the epidemiological and ecological literature
from the tables printed in the papers, show each result beside the published one, and say where the two
differ and why. The unit is the paper's own (a score, a count of beds, a growth rate, or the authors'
valuation in money); the baseline is the paper's status quo or least intervention. The same comparisons,
with the provenance of every number, are `test/fixtures/replications/*.json` and
`test/test_replications.jl`.

| | Vignette | Paper | What is reproduced, and what is not |
|---|---|---|---|
| 16 | [Adaptive culling for foot-and-mouth disease](vignettes/16_replication_shea_fmd/16_replication_shea_fmd.md) | Shea et al. 2014, *PLOS Biology* | EVPI of 45.9 and 61.2 million pounds once a misprint in the paper's Table 2 is corrected; the two-stage adaptive policy and the third of the EVPI it recovers (`two_stage_value`). The measles case study is not replicated. |
| 17 | [Learning by doing for whooping cranes](vignettes/17_replication_runge_cranes/17_replication_runge_cranes.md) | Runge et al. 2011, *Biological Conservation* | EVPI 0.117 (19.9%), the value of testing each hypothesis (`hypothesis_partition`), and the EVSI of trying a strategy and observing it with error (`gaussian_experiment`), which agrees with the paper's figure. |
| 18 | [ICU beds, and EVPXI against EVPPI](vignettes/18_replication_eze_icu/18_replication_eze_icu.md) | Eze et al. 2023, *BMC Health Services Research* | The EVPI curves and the authors' notebook; not the article's Table 5 nor two numbers in its text. Three different "partial" values on one table. |
| 19 | [Biological and operational uncertainty in weed control](vignettes/19_replication_li_thistle/19_replication_li_thistle.md) | Li et al. 2021, *Journal of Applied Ecology* | Table 1 (EVPI 0.15) and the matrix models; the joint analysis is **not** reproduced, and the vignette shows where it parts. Partial values do not add. |
| 20 | [The value of imperfect water-quality monitoring](vignettes/20_replication_luhede_monitoring/20_replication_luhede_monitoring.md) | Luhede et al. 2024, *Ecological Economics* | The perfect-information results exactly; the value of imperfect monitoring is quoted, not recomputed (its likelihood is unpublished), and set against the EVSI and break-even cost of monitoring programmes. |
| 22 | [Which entomological uncertainty matters for malaria control](vignettes/22_replication_kim_malaria/22_replication_kim_malaria.md) | Kim et al. 2017, *Risk Analysis* | The lognormal column of Table V (EVPI up to 8.4 dollars per person) from the printed payoff table; the uniform column is **not** reproducible (its probabilities are not printed). The value of each parameter of a pair alone is added. |
| 23 | [Recovery planning for a turtle](vignettes/23_replication_christensen_turtle/23_replication_christensen_turtle.md) | Christensen et al. 2026, *Conservation Science and Practice* | All twelve rows of Table 3 from the supplement's model outputs and value functions; the concave value function as printed does not give the paper's own tables, and a fitted one does. |
| 24 | [Which species to monitor, and how](vignettes/24_replication_bal_monitoring/24_replication_bal_monitoring.md) | Bal et al. 2018, *Methods in Ecology and Evolution* | The model rebuilt from the paper: every cell of Table S2 (EVPI, and the EVSI of surveillance and of targeted monitoring with detection errors) and the headline numbers of both case studies; the figures of the simulated systems are not. |

## Conventions

- Payoffs are utilities: higher is better. A regret table is a loss, and rules on it minimise.
- `Q[a, θ]`: rows are actions, columns are states. `L[z, θ]`: rows are signals, columns are states, so
  columns sum to one. A garbling `G[z′, z]` is column-stochastic and acts as `G * L`.
- Total variation is half the ℓ¹ distance; the `B` of `decision_loss_bound` is the range of the utility.
- `Hurwicz(α)` puts `α` on the best state (some texts put it on the worst: use `1 − α`);
  `AlphaMaxmin(α)` puts `α` on the lower expectation. `CVaR(α)` is the mean utility in the worst `α`
  share, the negative of the expected shortfall of loss-based software.
- State-wise rules ignore probabilities and range over every state of the table unless the belief is a
  `ScenarioSet` or a `PointBelief`.

## Relation to other packages

`DecisionProblems`, `DecisionNetworks`, `POMDPs` and `DecisionProgramming` in the General registry cover
sequential problems, influence diagrams and their solvers. None covers the payoff-table and
value-of-information scope here. Within the ecorecipes ecosystem, `InfluenceDiagrams.jl` solves diagrams
for maximum expected utility and `CategoricalInfluenceDiagrams.jl` composes them; this package analyses the
table such a model produces. It exports no `regret` or `regrets` (the latter is a best-response gap in
`CategoricalInfluenceDiagrams.jl`).

## Not in this version

Discounting, moment-matching EVSI, multilevel Monte Carlo, influence diagrams with more than one
decision, and anything sequential. See the roadmap in the lab wiki.

## Licence

MIT.
