# DecisionAnalysis.jl

Standard decision analysis on finite problems, with no category theory required: a
payoff table, a belief about its states, rules for choosing, Savage regret, and the
value of information. Version 0.1 is finite and exact: give it `Rational` inputs and
every number it returns is a rational.

| object | meaning |
|---|---|
| [`PayoffTable`](@ref) | `Q[a, θ]`, the conditional expected utility of action `a` in state `θ` |
| [`FiniteBelief`](@ref), [`CredalSet`](@ref), [`ScenarioSet`](@ref), [`PointBelief`](@ref) | what is known about the state |
| [`DecisionProblem`](@ref) | a table, a belief, a feasibility filter and an optional baseline |
| [`decide`](@ref) | apply a rule and get scores and the chosen actions |
| [`Experiment`](@ref) | `L[z, θ] = P(z \| θ)`, an information source |
| [`evpi`](@ref), [`evppi`](@ref), [`evsi`](@ref), [`enbs`](@ref) | the value of information |
| [`DecisionReport`](@ref) | every applicable rule side by side |

## Quick start

```jldoctest quick
julia> using DecisionAnalysis

julia> payoff = PayoffTable([:do_nothing, :raise_levee, :relocate_assets],
                            [:low, :moderate, :high, :extreme],
                            [0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]);

julia> problem = DecisionProblem(payoff, [1 // 2, 3 // 10, 3 // 20, 1 // 20];
                                 baseline=:do_nothing);

julia> first_choice(decide(problem, MaxExpectedUtility()))
:do_nothing

julia> first_choice(decide(problem, Maximin()))
:relocate_assets

julia> first_choice(decide(problem, MinimaxRegret()))
:raise_levee

julia> evpi(problem)
207//20
```

Expected utility favours doing nothing, the worst case favours relocating, and
regret favours the levee: the decision is contested, and perfect information would
be worth 10.35. A river gauge is an [`Experiment`](@ref); a noisier gauge is a
[`garble`](@ref)d one and is worth less.

```jldoctest quick
julia> gauge = Experiment([9//10 3//5 1//5 1//20; 1//10 2//5 4//5 19//20];
                          signals=[:no_warning, :warning]);

julia> evsi(problem, gauge)
97//20

julia> evsi(problem, garble(gauge, [4//5 1//5; 1//5 4//5]))
49//25
```

With a set of priors instead of one, the rules over a [`CredalSet`](@ref) apply.

```jldoctest quick
julia> ensemble = CredalSet(payoff.states, [[1 // 2, 3 // 10, 3 // 20, 1 // 20],
                                            [7 // 20, 3 // 10, 1 // 4, 1 // 10],
                                            [3 // 5, 1 // 4, 3 // 25, 3 // 100]]);

julia> robust = DecisionProblem(payoff, ensemble);

julia> decide(robust, GammaMaximin()).chosen
1-element Vector{Symbol}:
 :raise_levee

julia> decide(robust, EAdmissible()).chosen
2-element Vector{Symbol}:
 :do_nothing
 :raise_levee
```

`DecisionReport(problem; credal=ensemble, experiment=gauge)` prints all of this as
one table, with the regret profile, the gain of each action over the baseline and
the assumptions under which the numbers hold.

## Conventions

- **Payoffs are utilities**: higher is better. Costs enter as negative payoffs. A
  regret table is a loss, and a rule on it minimises.
- **Layouts.** `Q[a, θ]`: rows are actions, columns are states. `L[z, θ]`: rows are
  signals, columns are states, so every column sums to one. A garbling `G[z′, z]` is
  column-stochastic and acts as `G * L`.
- **The payoff table holds conditional expected utility.** A stochastic simulator
  must be averaged inside each cell; the standard error can be stored beside the
  table and is never folded into a payoff.
- **Which states a rule sees.** Rules that ignore probabilities (maximin, maximax,
  Laplace, Hurwicz, minimax regret) range over every state of the table, unless the
  belief is a [`ScenarioSet`](@ref) or a [`PointBelief`](@ref).
- **Two α conventions.** [`Hurwicz`](@ref) mixes the best and worst payoff over
  states and puts `α` on the best. [`AlphaMaxmin`](@ref) mixes the lower and upper
  expectation over priors and puts `α` on the lower. Both follow their literature.
- **Ties.** Every tied action is returned, in table order; [`first_choice`](@ref) is
  the first. Ties are exact for rational problems and within a relative tolerance in
  floating point.
- **Total variation** is half the ℓ¹ distance, and the `B` of
  [`decision_loss_bound`](@ref) is the range of the utility.
- **What is learned is named.** [`evppi`](@ref) takes the partition of the states
  whose cell is learned; [`evsi`](@ref) takes the experiment. `EVSI ≤ EVPI` is a
  statement about the same states and a single prior.
- **Robust value of information may be negative** and is returned as computed.
- **[`enbs`](@ref) is for additive payoffs only.**

## Linear programmes

E-admissibility, the garbling test, deficiency and the upper EVPI envelope are
linear programmes. The package solves them itself, with a dense simplex method in
exact rational arithmetic, so the answers are exact statements ("this action is
E-admissible", "the deficiency is zero") and no solver has to be installed. Inputs
in floating point are solved as the exact rationals they are. The solver is meant for
the small problems of a decision table, not for large programmes.

## Scope of version 0.1

Finite problems, exactly. Sample-based beliefs and Monte Carlo value of information,
outcome tables and model adapters, utility functions, decision sensitivity and
sequential problems are planned for later versions.
