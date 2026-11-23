# DecisionAnalysis.jl

[![Build Status](https://github.com/ecorecipes/DecisionAnalysis.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/ecorecipes/DecisionAnalysis.jl/actions/workflows/CI.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Standard decision analysis on the output of any model, with no category theory required: a payoff table,
a belief about its states, rules for choosing, Savage regret, and the value of information. Version 0.1 is
the finite core, and it is exact: give it `Rational` inputs and every number it returns is a rational.

The package depends on [FiniteKernels.jl](https://github.com/ecorecipes/FiniteKernels.jl), `LinearAlgebra`,
`Random`, `Statistics` and `Tables`, and on no Catlab, GATlab or ACSets.

## Features

- **Payoff tables and beliefs.** `PayoffTable` holds the conditional expected utility `Q[a, θ]` (rows =
  actions, columns = states) with any real element type. A belief is a `PointBelief`, a `FiniteBelief` (a
  probability vector, also built from a FiniteKernels state), a `CredalSet` (the convex hull of a list of
  priors) or a `ScenarioSet` (possible states, no probabilities). `DecisionProblem(payoff, belief)` pairs
  them, with an optional feasibility filter on the actions and an optional `baseline` action.
- **Comparison with no intervention.** Name the business-as-usual action as the `baseline` of a problem;
  `gain_over_baseline` gives the table of differences from it and `expected_gain_over_baseline` each
  action's expected gain, and a report says for every rule whether it recommends the baseline.
- **Aggregators.** `Expectation`, `WorstCase`, `BestCase`, `Hurwicz(α)` (over states), `Laplace`,
  `LowerExpectation`, `UpperExpectation`, `AlphaMaxmin(α)` (over priors), `CVaR(α)`, `Quantile(α)`.
- **Regret.** `regret_table`, `relative_regret_table`, `expected_regret`, `max_regret`.
- **Rules.** `decide(problem, rule)` with `MaxExpectedUtility`, `Maximin`, `Maximax`, `MinimaxRegret`,
  `GammaMaximin`, `Satisfice(threshold)`, and the set-valued `EAdmissible` (a linear programme over the
  credal set, not an enumeration of its extreme points), `Maximal`, `IntervalDominance` and `ParetoFront`.
  A scalar rule is an aggregator, an optional transform and a selector (`DecisionRule`); the same pieces
  without the selector are a `RobustnessMetric`. Ties are all returned, and the first choice is the tied
  action that comes first in the table.
- **Experiments.** `Experiment(L)` with `L[z, θ] = P(z | θ)` (rows = signals, columns = states), also built
  from a `FiniteKernel`; `bayes_invert`, `garble`, `is_garbling_of` (returns a witness matrix) and
  `deficiency`, the last two by exact linear programme.
- **Value of information.** `evpi`, `evppi` (over a partition of the states), `evsi`, `enbs`,
  `value_with_information`; under a credal set `robust_evpi` and `robust_evsi`, which keep the
  rule-specific value and the EVPI envelope apart and never clamp a negative value.
- **Fidelity.** `total_variation` and `decision_loss_bound(B, ε) = 2Bε`.
- **Reports and certificates.** `DecisionReport` puts every applicable rule side by side with the regret
  profile, EVPI, the robust sets and the assumptions, prints as a table and is a Tables.jl table.
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

DecisionReport(problem; credal=ensemble, experiment=gauge)
```

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
