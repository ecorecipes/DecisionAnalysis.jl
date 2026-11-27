# DecisionAnalysis.jl

Standard decision analysis on finite problems: a
payoff table, a belief about its states, rules for choosing, Savage regret, and the
value of information. Finite problems are exact: give the package `Rational` inputs
and every number it returns is a rational. Draws and simulator runs are analysed by
sample-based estimators of the same quantities, each with its Monte Carlo standard
error in a separate field.

The **tutorials** build up from a first decision table to value of information from
model runs, starting with
[a first decision table](tutorials/01_first_decision_table.md). The **case studies**
reproduce published analyses in ecology and in human epidemiology. Both are in the
menu, followed by the certificates and the API reference.

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
| [`SampleBelief`](@ref), [`OutcomeTable`](@ref) | draws of named parameters with weights; model output by draw × action × replicate |
| [`payoff_table`](@ref) | outcomes and a utility counted in individuals, to a payoff table with standard errors |
| [`VOIEstimate`](@ref) | a sample-based value of information with its standard error and selection biases |

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

## From model runs to a decision

A model package writes a table of runs, one row per run. Naming its columns gives an
[`OutcomeTable`](@ref); a utility counted in individuals gives a payoff table whose
standard errors are kept apart; and [`evpi`](@ref), [`evppi`](@ref) and [`evsi`](@ref)
called with a method return estimates with their errors.

```jldoctest
julia> using DecisionAnalysis

julia> runs = [(; draw=d, action=a, replicate=r, site=isodd(d) ? "pond" : "stream",
                uninfected=30 + 4d + r + (a == "treat") * (isodd(d) ? 9 : -3), infected=r)
               for d in 1:6 for a in ("none", "treat") for r in 1:3];

julia> table = OutcomeTable(runs; outcomes=[:uninfected, :infected], parameters=[:site]);

julia> problem = DecisionProblem(table, InfectionUtility(; weight=1); baseline=:none);

julia> expected_gain_over_baseline(problem)
2-element Vector{Rational{Int64}}:
 0
 3

julia> estimate = evppi(problem, :site);

julia> (estimate.method, estimate.value)
(:Grouping, 3//2)
```

Knowing the site is worth a frog and a half here: treatment gains nine at a pond and
loses three at a stream. See the tutorials on sample beliefs, EVPPI and EVSI, and the
[Sample-based analysis](api/samples.md) reference.

## Risk attitude, several attributes, and checks against other software

A count of individuals is a linear, risk-neutral utility. [`ExponentialUtility`](@ref)
and [`PowerUtility`](@ref) give one named outcome a risk attitude, and
[`AdditiveUtility`](@ref) and [`MultiplicativeUtility`](@ref) combine several
attributes scaled to `[0, 1]`. A non-linear utility is applied to every run before
anything is averaged; its expected utility is no longer a count of individuals, and
[`certainty_equivalent`](@ref) returns to individuals. See the
[reference](api/utilities.md) and the tutorial on risk attitude.

The estimators are compared with the R packages `voi` and `cvar`, and the finite rules
and linear programmes with independent Python code and scipy's `linprog`. The scripts,
the versions and every number are in the `validation/` directory of the repository; the
fixtures they write are read by the test suite, which needs neither R nor Python.

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
  states and puts `α` on the best (texts that call `α` the coefficient of pessimism
  put it on the worst: use `1 − α`). [`AlphaMaxmin`](@ref) mixes the lower and upper
  expectation over priors and puts `α` on the lower. Both follow their literature.
- **Tail risk is of utilities.** [`CVaR`](@ref) is the mean payoff in the worst `α`
  share and [`Quantile`](@ref) the lower quantile: the negatives of the expected
  shortfall and value at risk of software that works with losses.
- **Ties.** Every tied action is returned, in table order; [`first_choice`](@ref) is
  the first. Ties are exact for rational problems and within a relative tolerance in
  floating point.
- **Total variation** is half the ℓ¹ distance, and the `B` of
  [`decision_loss_bound`](@ref) is the range of the utility.
- **What is learned is named.** [`evppi`](@ref) takes the partition of the states
  whose cell is learned; [`evsi`](@ref) takes the experiment. `EVSI ≤ EVPI` is a
  statement about the same states and a single prior.
- **Monte Carlo error is a separate field.** A sample-based estimate
  ([`VOIEstimate`](@ref)) carries its value, its standard error and its selection
  biases side by side; none is folded into another. Selection bias belongs to the
  best reported value and the chosen action; the sign of the bias of EVPI is not
  determined in general.
- **No monetary costs.** Utility is counted in individuals of the focal species.
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
