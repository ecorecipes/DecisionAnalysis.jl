# DecisionAnalysis.jl

[![Build Status](https://github.com/ecorecipes/DecisionAnalysis.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ecorecipes/DecisionAnalysis.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Standard decision analysis on the output of any model, with no category theory required: a payoff table,
a belief about its states, rules for choosing, Savage regret, and the value of information. Version 0.1 is
the finite core, and it is exact: give it `Rational` inputs and every number it returns is a rational.

The package depends on [FiniteKernels.jl](https://github.com/ecorecipes/FiniteKernels.jl),
[JuMP](https://jump.dev) (to state its linear programmes; no solver is a dependency), `LinearAlgebra`,
`Random`, `Statistics` and `Tables`, and on no Catlab, GATlab or ACSets.

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
- **Value of information.** `evpi`, `evppi` (over a partition of the states), `evsi`, `enbs`,
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
- **Fidelity.** `total_variation` and `decision_loss_bound(B, ε) = 2Bε`.
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

The ecosystem packages are not registered. Clone this package beside `FiniteKernels.jl`, which it finds
through a `[sources]` path:

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

Ten vignettes in [`vignettes/`](vignettes/) go from a first decision table to Lean-checked certificates,
on ecological examples whose outcomes are counted in individuals and always compared with no intervention
(all numbers illustrative). Each is rendered to Markdown, HTML and PDF beside its `.qmd`, with the code run
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

## Conventions

- Payoffs are utilities: higher is better. A regret table is a loss, and rules on it minimise.
- `Q[a, θ]`: rows are actions, columns are states. `L[z, θ]`: rows are signals, columns are states, so
  columns sum to one. A garbling `G[z′, z]` is column-stochastic and acts as `G * L`.
- Total variation is half the ℓ¹ distance; the `B` of `decision_loss_bound` is the range of the utility.
- `Hurwicz(α)` puts `α` on the best state; `AlphaMaxmin(α)` puts `α` on the lower expectation.
- State-wise rules ignore probabilities and range over every state of the table unless the belief is a
  `ScenarioSet` or a `PointBelief`.

## Relation to other packages

`DecisionProblems`, `DecisionNetworks`, `POMDPs` and `DecisionProgramming` in the General registry cover
sequential problems, influence diagrams and their solvers. None covers the payoff-table and
value-of-information scope here. Within the ecorecipes ecosystem, `InfluenceDiagrams.jl` solves diagrams
for maximum expected utility and `CategoricalInfluenceDiagrams.jl` composes them; this package analyses the
table such a model produces. It exports no `regret` or `regrets` (the latter is a best-response gap in
`CategoricalInfluenceDiagrams.jl`).

## Not in version 0.1

Sample-based beliefs and Monte Carlo value of information, outcome tables and model adapters, utility
functions and multi-attribute utility, decision sensitivity, and anything sequential. See the roadmap in
the lab wiki.

## Licence

MIT.
