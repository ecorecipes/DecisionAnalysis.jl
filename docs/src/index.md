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
| [`information_can_change_decision`](@ref) | whether an experiment can change what is done (exactly when its EVSI is positive) |
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

`DecisionReport(problem; credal=ensemble, experiments=(; gauge))` prints all of this
as aligned tables that fit 80 columns (and as Markdown or HTML tables in a notebook),
with the regret profile, the gain of each action over the baseline, whether each
experiment can change the decision, and the assumptions under which the numbers hold.

## Utilities counted in individuals

A payoff table can be assembled from tables of outcome counts with `+`, `-` and
scalar `*` on tables with matching labels, or with [`combine_payoffs`](@ref).
[`individuals_payoff`](@ref) is the utility that counts individuals and
[`infection_payoff`](@ref) is uninfected − weight × infected. Several objectives
kept apart go to [`ParetoFront`](@ref) as a named collection of tables sharing one
belief.

```jldoctest
julia> using DecisionAnalysis

julia> uninfected = PayoffTable([:nothing, :baths], [:absent, :present], [500 100; 470 250]);

julia> infected = PayoffTable([:nothing, :baths], [:absent, :present], [0 250; 0 120]);

julia> infection_payoff(; uninfected, infected, weight=1)
PayoffTable{Int64}: 2 actions × 2 states
  action   absent  present
  -------  ------  -------
  nothing     500     -150
  baths       470      130

julia> decide((; uninfected, fewer_infected=-infected), [1 // 2, 1 // 2], ParetoFront()).chosen
1-element Vector{Symbol}:
 :baths
```

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
- **Minimax regret depends on the menu of actions.** Adding or removing an action,
  even one that is never chosen, can change its choice; the result of
  [`MinimaxRegret`](@ref) and the report say so.
- **A baseline under an imprecise belief gives a range.**
  [`expected_gain_over_baseline`](@ref) returns `(lower, upper)` for a credal set
  (the range of the expected gain) and for a scenario set (the range of the gain).
- **[`enbs`](@ref) is for additive payoffs only.**

## Linear programmes

E-admissibility, the garbling test, deficiency and the upper EVPI envelope are
linear programmes. Each is written once, as a [JuMP](https://jump.dev) model, by an
exported builder: [`eadmissibility_model`](@ref), [`garbling_model`](@ref),
[`deficiency_model`](@ref) and [`evpi_envelope_model`](@ref). A builder returns a
[`DecisionProgram`](@ref) holding a `JuMP.GenericModel{T}` with `T = Rational{BigInt}`
for an exact problem, so the model can be inspected, extended or solved by hand.

The functions that use them take an `optimizer` keyword, and there are three routes
to the same model.

| `optimizer` | what happens | answers |
|---|---|---|
| `nothing` (the default) | the built-in dense simplex in exact rational arithmetic, fed from the model's matrix form (`JuMP.lp_matrix_data`) | exact; no solver needs to be installed; for small problems |
| an exact optimiser, e.g. `CDDLib.Optimizer{Rational{BigInt}}` | the rational model is solved by it | exact |
| a floating-point optimiser, e.g. `HiGHS.Optimizer` | a `Float64` copy is solved, the solution is rounded to rationals and **verified exactly** | exact when verified; otherwise [`ExactVerificationError`](@ref) |

JuMP is a dependency of the package; no solver is. Install CDDLib, HiGHS or any
other JuMP solver yourself to use the second or third route.

**The solver is never trusted.** Whatever route produced it, a witness (the prior
under which an action is optimal, the garbling matrix) is checked in exact
arithmetic before it is returned or exported, and an optimal value found in floating
point (a deficiency, the EVPI envelope) is returned only if the rounded primal and
dual solutions certify it exactly. A yes-or-no answer says how sure it is:
`decide(problem, EAdmissible()).details.exact` and
[`garbling_result`](@ref)`.exact` are `true` for an exact statement and `false` for
one that holds to the tolerance of a floating-point optimiser (a "no" from HiGHS,
for example). Inputs in floating point are solved as the exact rationals they are
by the default route, and the answers are then numerical.

```jldoctest quick
julia> program = eadmissibility_model(robust, :raise_levee);

julia> program.model isa DecisionAnalysis.JuMP.GenericModel{Rational{BigInt}}
true

julia> solution = solve_program(program);

julia> solution.status, solution.objective, solution.exact
(:optimal, 0//1, true)
```

## Scope of version 0.1

Finite problems, exactly. Sample-based beliefs and Monte Carlo value of information,
outcome tables and model adapters, utility functions, decision sensitivity and
sequential problems are planned for later versions.
