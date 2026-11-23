# The decision report: every applicable rule side by side.

"""
    DecisionReport(problem::DecisionProblem; credal=nothing, experiment=nothing,
                   experiments=nothing, partitions=nothing, objectives=nothing,
                   pareto=nothing, hurwicz=nothing, assumptions=String[],
                   optimizer=nothing, parameters=nothing,
                   regression=SplineRegression(), seed=1, noise_correlation=0)

Everything the package can say about one problem, side by side, so that a decision
that is contested between rules is seen to be contested. Which parts are filled
depends on the belief:

- State-wise rules, always: [`Maximin`](@ref), [`Maximax`](@ref), Laplace,
  [`MinimaxRegret`](@ref), and Hurwicz when the optimism index `hurwicz` is given;
  the [`max_regret`](@ref) profile.
- With a single prior ([`FiniteBelief`](@ref), [`SampleBelief`](@ref) or
  [`PointBelief`](@ref)):
  [`MaxExpectedUtility`](@ref), the [`expected_regret`](@ref) profile, [`evpi`](@ref),
  the [`evsi`](@ref) of every experiment given, and the [`evppi`](@ref) of every
  partition given.
- With a credal set, either as the belief of the problem or passed as `credal`
  beside a single prior: [`GammaMaximin`](@ref), the robust sets
  [`EAdmissible`](@ref), [`Maximal`](@ref) and [`IntervalDominance`](@ref), and
  [`robust_evpi`](@ref).

**Sample-based problems.** When the belief is a [`SampleBelief`](@ref), every number
under the single prior is an estimate, and the report adds a Monte Carlo section in
which **standard errors are shown beside the values and never folded into them**:
the expected utility of each action with its standard error
([`expected_utility_estimates`](@ref)) and, with a baseline, the expected gain with
its own ([`expected_gain_estimates`](@ref)); the EVPI with its standard error and
the two selection biases ([`VOIEstimate`](@ref)); when the payoff table has standard
errors, the split of each action's variance between parameters and simulation
([`variance_components`](@ref)) and the all-parameter regression estimate of EVPI
with `regression` (skipped, with a note, when the draws carry no parameters or are
too few for it; `regression = nothing` turns it off); and the EVPPI of each
parameter named in `parameters` (names of columns of the draws, or tuples of names)
by the default method of [`evppi`](@ref). A table built from one run per cell is
flagged ([`single_replicate`](@ref)). The worst and best cases are over the sampled
draws and say so; Laplace is left out, since an unweighted mean over draws ignores
their weights. `seed` seeds the bias simulations, and `noise_correlation` is the
correlation of the cell noise between actions used for the cell-noise bias (zero by
default; pass [`crn_correlation`](@ref) for a table simulated with common random
numbers).

**Experiments and partitions.** `experiments` is a named collection of
[`Experiment`](@ref)s (a named tuple, a dictionary, or `name => experiment` pairs);
`experiment` is a single one, reported under the name `experiment`, and may be given
as well. For each the report gives the EVSI, its cost, the net value `EVSI − cost`
([`enbs`](@ref) with `population = 1`, meaningful for additive payoffs only) and
whether the experiment can change the decision
([`information_can_change_decision`](@ref)); an experiment that cannot is worth
nothing, and the report says so. `partitions` is a named collection of partitions
of the states (each a vector with one entry per state, as for [`evppi`](@ref)); for
each the report gives the EVPPI. Both need a single prior
([`BeliefFormError`](@ref) otherwise).

**Several objectives.** `objectives` is a named collection of payoff tables on the
actions and states of the problem, one per objective, all utilities (higher is
better). The report then has a multi-objective section: the [`ParetoFront`](@ref)
of the feasible actions under the belief of the problem. `pareto` is the rule to
use; by default the objectives are scored by expectation under a single prior, by
lower expectation over a credal set and by worst case over a scenario set.

**Baseline.** When the problem has a baseline action (no intervention), the report
states it in its header, gives every action's gain over it, and says for every rule
whether the action chosen is the baseline. Under a single prior the gain is the
expected gain; under a credal set it is the range of the expected gain over the
set, and under a scenario set the range of the gain over the scenarios (see
[`expected_gain_over_baseline`](@ref)). When there is no baseline, the assumptions
say so.

**Minimax regret depends on the menu of actions**, and the report says so under the
first choices: see [`MinimaxRegret`](@ref).

`optimizer` is passed to the linear programmes behind the robust sets and the EVPI
envelope (see [`solve_program`](@ref)); by default they are solved exactly by the
built-in simplex.

Fields:

- `problem`.
- `results`: the scalar rules, a vector of [`DecisionResult`](@ref); `robust`: the
  set-valued rules.
- `max_regret`, `expected_regret`.
- `expected_gain`: the expected gain over the baseline under a single prior;
  `gain_range`: the named tuple `(lower, upper)` for a credal set or a scenario set.
- `evpi`; `evsi`: the EVSI of the first experiment given.
- `experiments`: one named tuple `(name, evsi, cost, net, can_change_decision)` per
  experiment; `partitions`: one named tuple `(name, evppi)` per partition.
- `robust_evpi`.
- `pareto`: the result of the Pareto front, when `objectives` were given.
- `assumptions`: the stated assumptions under which the numbers hold, to which the
  strings in the `assumptions` keyword are appended.
- `monte_carlo`: for a sample-based problem, the named tuple `(; expected_utility,
  expected_gain, evpi, evpi_regression, evppi, variance, single_replicate)` of the
  estimates described above (`nothing` where one does not apply).

A field that does not apply is `nothing` (or empty).

A report prints as text that fits 80 columns: tables of actions against rules (in
blocks of a few rules when there are many), with aligned columns. In a notebook or
a Quarto document it renders as Markdown or HTML tables. It is also a Tables.jl
row table with one row per feasible action: the action, each scalar rule's score
and whether it chose the action, the regret profile, the gain over the baseline
when there is one (`expected_gain_over_baseline`, or `gain_over_baseline_lower` and
`gain_over_baseline_upper`), membership of each robust set, and of the Pareto front.
"""
struct DecisionReport{P<:DecisionProblem}
    problem::P
    results::Vector{DecisionResult}
    robust::Vector{DecisionResult}
    max_regret::Vector
    expected_regret::Union{Nothing,Vector}
    expected_gain::Union{Nothing,Vector}
    gain_range::Union{Nothing,NamedTuple}
    evpi::Union{Nothing,Real}
    evsi::Union{Nothing,Real}
    experiments::Vector{NamedTuple}
    partitions::Vector{NamedTuple}
    robust_evpi::Union{Nothing,RobustVOI}
    pareto::Union{Nothing,DecisionResult}
    assumptions::Vector{String}
    monte_carlo::Union{Nothing,NamedTuple}
end

function _has_prior(p::DecisionProblem)
    return p.belief isa Union{FiniteBelief,SampleBelief,PointBelief}
end

function _default_pareto(::Union{FiniteBelief,SampleBelief,PointBelief})
    return ParetoFront(Expectation())
end
_default_pareto(::CredalSet) = ParetoFront(LowerExpectation())
_default_pareto(::ScenarioSet) = ParetoFront(WorstCase())

function DecisionReport(problem::DecisionProblem;
                        credal::Union{Nothing,CredalSet}=nothing,
                        experiment::Union{Nothing,Experiment}=nothing,
                        experiments=nothing, partitions=nothing, objectives=nothing,
                        pareto::Union{Nothing,ParetoFront}=nothing,
                        hurwicz::Union{Nothing,Real}=nothing,
                        assumptions::AbstractVector{<:AbstractString}=String[],
                        optimizer=nothing, parameters=nothing,
                        regression::Union{Nothing,AbstractRegression}=SplineRegression(),
                        seed::Integer=1, noise_correlation::Real=0)
    belief = problem.belief
    sampled = belief isa SampleBelief
    parameters === nothing || sampled ||
        throw(BeliefFormError(:DecisionReport, nameof(typeof(belief)),
                              "`parameters` asks for the EVPPI of named parameters " *
                              "of the draws and needs a SampleBelief; use " *
                              "`partitions` for a finite belief"))
    belief isa CredalSet && credal !== nothing &&
        throw(ArgumentError("the problem already has a credal set as its belief; do " *
                            "not pass `credal` as well"))
    belief isa ScenarioSet && credal !== nothing &&
        throw(ArgumentError("`credal` accompanies a single prior, not a scenario set"))
    pareto === nothing || objectives !== nothing ||
        throw(ArgumentError("`pareto` needs `objectives`"))
    single = _has_prior(problem)
    named_experiments = Pair{Symbol,Experiment}[]
    experiment === nothing || push!(named_experiments, :experiment => experiment)
    experiments === nothing ||
        append!(named_experiments, _named_items(experiments, Experiment, "experiments"))
    allunique(first.(named_experiments)) ||
        throw(ArgumentError("the names of the experiments must be distinct, got " *
                            "$(first.(named_experiments))"))
    named_partitions = partitions === nothing ? Pair{Symbol,AbstractVector}[] :
                       _named_items(partitions, AbstractVector, "partitions")
    isempty(named_experiments) || single ||
        throw(BeliefFormError(:DecisionReport, nameof(typeof(belief)),
                              "the EVSI of an experiment needs a single prior; use " *
                              "robust_evsi for a credal set"))
    isempty(named_partitions) || single ||
        throw(BeliefFormError(:DecisionReport, nameof(typeof(belief)),
                              "the EVPPI of a partition needs a single prior"))
    robust_problem = belief isa CredalSet ? problem :
                     credal === nothing ? nothing :
                     DecisionProblem(problem.payoff, credal, problem.feasible,
                                     problem.baseline)

    rules = AbstractRule[]
    single && push!(rules, MaxExpectedUtility())
    push!(rules, Maximin(), Maximax())
    sampled || push!(rules, DecisionRule(Laplace(); name=:Laplace))
    hurwicz === nothing || push!(rules, DecisionRule(Hurwicz(hurwicz); name=:Hurwicz))
    push!(rules, MinimaxRegret())
    results = DecisionResult[decide(problem, r) for r in rules]
    robust = DecisionResult[]
    robust_value = nothing
    if robust_problem !== nothing
        push!(results, decide(robust_problem, GammaMaximin()))
        push!(robust, decide(robust_problem, EAdmissible(); optimizer))
        for r in (Maximal(), IntervalDominance())
            push!(robust, decide(robust_problem, r))
        end
        robust_value = robust_evpi(robust_problem; optimizer)
    end

    experiment_rows = NamedTuple[]
    for (name, e) in named_experiments
        value = evsi(problem, e)
        push!(experiment_rows,
              (; name, evsi=value, cost=e.cost, net=value - e.cost,
               can_change_decision=information_can_change_decision(problem, e)))
    end
    partition_rows = NamedTuple[(; name, evppi=evppi(problem, part))
                                for (name, part) in named_partitions]

    front = nothing
    if objectives !== nothing
        rule = pareto === nothing ? _default_pareto(belief) : pareto
        front = decide(objectives, belief, rule; feasible=problem.feasible,
                       baseline=problem.baseline)
    end

    notes = ["Payoffs are conditional expected utilities of each action in each state; " *
             "higher is better.",
             "State-wise rules (maximin, maximax, Laplace, Hurwicz, minimax regret) " *
             "ignore probabilities and range over " *
             (belief isa ScenarioSet ? "the states of the scenario set." :
              belief isa PointBelief ? "the one known state." :
              sampled ?
              "the draws in the sample, whatever their weights, so their scores " *
              "depend on which draws were sampled; Laplace is not shown for a sample." :
              "every state of the table, including any of probability zero."),
             "Regret is measured against the best feasible action in each state, so it " *
             "changes when the set of actions changes. Minimax regret therefore " *
             "depends on the menu of actions: adding or removing an action, even one " *
             "that is never chosen, can change its choice.",
             "Tied actions are all reported; the first choice is the tied action that " *
             "comes first in the payoff table."]
    count(problem.feasible) == length(problem.feasible) ||
        push!(notes,
              "Infeasible actions are excluded throughout: " *
              join(problem.payoff.actions[.!problem.feasible], ", ") * ".")
    single &&
        push!(notes,
              "Expected utility, expected regret and EVPI are under the single stated " *
              "prior; EVPI is the value of learning the state exactly and at no cost.")
    isempty(experiment_rows) ||
        push!(notes,
              "EVSI is gross of the cost of the experiment and assumes the stated " *
              "likelihood. The net value is EVSI minus the cost, which is meaningful " *
              "only when payoffs and costs are in the same additive units. An " *
              "experiment can change the decision exactly when its EVSI is positive.")
    isempty(partition_rows) ||
        push!(notes,
              "EVPPI is the value of learning which cell of a partition of the states " *
              "holds, and nothing more, at no cost.")
    if robust_problem !== nothing
        k = length(robust_problem.belief.points)
        push!(notes,
              "The credal set is the convex hull of $k stated prior" *
              (k == 1 ? "" : "s") * "; E-admissibility is decided over the whole hull.",
              "The rule-specific robust EVPI is under Γ-maximin; the envelope is the " *
              "range of the single-prior EVPI over the credal set.")
    end
    front === nothing ||
        push!(notes,
              "The Pareto front keeps the actions that no other action matches on " *
              "every objective and beats on one; each objective is summarised over " *
              "the states by " * string(nameof(typeof(front.rule.aggregator))) * ".")
    if problem.baseline === nothing
        push!(notes,
              "No baseline action (no intervention) is set, so nothing here is " *
              "expressed as a gain over business as usual.")
    else
        push!(notes,
              "The baseline (no intervention) is $(problem.baseline)" *
              (problem.baseline in feasible_actions(problem) ? "" :
               ", which is not feasible") *
              "; gains are payoffs minus the payoff of the baseline in the same state" *
              (single ? "." :
               ", and no expected gain is shown because the belief has no single " *
               "prior: the range of the " *
               (belief isa CredalSet ? "expected gain over the credal set" :
                "gain over the scenarios") * " is shown in its place."))
    end
    monte_carlo = nothing
    if sampled
        monte_carlo = _monte_carlo(problem, parameters, regression, seed, notes,
                                   noise_correlation)
    end
    append!(notes, String.(assumptions))

    gain = problem.baseline === nothing ? nothing : expected_gain_over_baseline(problem)
    return DecisionReport(problem, results, robust, max_regret(problem),
                          single ? expected_regret(problem) : nothing,
                          single ? gain : nothing, single ? nothing : gain,
                          single ? evpi(problem) : nothing,
                          isempty(experiment_rows) ? nothing : first(experiment_rows).evsi,
                          experiment_rows, partition_rows, robust_value, front, notes,
                          monte_carlo)
end

# The Monte Carlo section of a sample-based report; appends its assumptions to `notes`.
function _monte_carlo(problem::DecisionProblem, parameters, regression, seed::Integer,
                      notes::Vector{String}, noise_correlation::Real)
    belief = problem.belief
    noisy = problem.payoff.standard_errors !== nothing
    flagged = single_replicate(problem.payoff)
    push!(notes,
          "The belief is a sample of $(length(belief)) draws (effective sample size " *
          "$(round(Float64(effective_sample_size(belief)); sigdigits=4))): expected " *
          "utility, regret and every value of information are estimates. Standard " *
          "errors are shown beside them and are not folded into any value; they " *
          "treat the draws as a random sample of the belief.")
    flagged &&
        push!(notes,
              "Some cells of the payoff table hold a single simulator run, not a " *
              "conditional expectation: the act-now value and the value with " *
              "information are each biased upwards by the simulation noise, by " *
              "amounts that cannot be estimated from one run, and the action chosen " *
              "is on average worth less than its reported score.")
    regressed = nothing
    if regression !== nothing && (noisy || flagged)
        if isempty(belief.draws)
            push!(notes,
                  "The draws carry no parameters, so the all-parameter regression " *
                  "estimate of EVPI, which smooths over simulation noise, is not shown.")
        elseif regression isa SplineRegression &&
               10 * size(_design(regression, belief.draws, belief.weights), 2) >
               length(belief)
            push!(notes,
                  "There are too few draws for the all-parameter regression estimate " *
                  "of EVPI with the default regression (fewer than ten draws per " *
                  "coefficient), so it is not shown; pass a smaller `regression`.")
        else
            regressed = evpi(problem, regression; seed)
            push!(notes,
                  "The all-parameter regression estimate of EVPI replaces the noisy " *
                  "cells by fitted conditional means, so simulation noise is smoothed " *
                  "and not maximised over; it is the estimate to prefer when cells " *
                  "are noisy.")
        end
    end
    named = parameters === nothing ? [] :
            parameters isa Union{Symbol,AbstractString} ? [parameters] : collect(parameters)
    partial = NamedTuple[(; name=join(_focal_names(f), ", "),
                          estimate=evppi(problem,
                                         f isa AbstractString ? Symbol(f) :
                                         f isa Symbol ? f : Tuple(_focal_names(f));
                                         seed)) for f in named]
    push!(notes,
          "Selection bias (the optimiser's curse) is reported, not removed. It " *
          "belongs to the best reported value and the chosen action: a maximum of " *
          "estimates overstates, on average, what the chosen action is worth. The " *
          "act-now value and the value with perfect information are each biased " *
          "upwards (by sampled draws and by noisy cells); EVPI is their difference, " *
          "so the sign of its bias is not determined in general. With exact payoffs " *
          "and sampled draws, the plain EVPI is biased downwards.")
    return (; expected_utility=expected_utility_estimates(problem),
            expected_gain=problem.baseline === nothing ? nothing :
                          expected_gain_estimates(problem),
            evpi=evpi(problem, SampleAverage(); seed, noise_correlation),
            evpi_regression=regressed,
            evppi=partial, variance=noisy ? variance_components(problem) : nothing,
            single_replicate=flagged)
end

# Tables.jl: one row per feasible action.
function _report_rows(r::DecisionReport)
    actions = feasible_actions(r.problem)
    names = Symbol[:action]
    columns = Any[actions]
    for res in r.results
        name = _rule_name(res.rule)
        push!(names, name, Symbol(name, :_chosen))
        push!(columns, res.scores, [a in res.chosen for a in actions])
    end
    push!(names, :max_regret)
    push!(columns, r.max_regret)
    if r.expected_regret !== nothing
        push!(names, :expected_regret)
        push!(columns, r.expected_regret)
    end
    if r.monte_carlo !== nothing
        push!(names, :expected_utility_standard_error)
        push!(columns, [e.standard_error for e in r.monte_carlo.expected_utility])
    end
    if r.expected_gain !== nothing
        push!(names, :expected_gain_over_baseline)
        push!(columns, r.expected_gain)
        if r.monte_carlo !== nothing
            push!(names, :expected_gain_standard_error)
            push!(columns, [e.standard_error for e in r.monte_carlo.expected_gain])
        end
    end
    if r.gain_range !== nothing
        push!(names, :gain_over_baseline_lower, :gain_over_baseline_upper)
        push!(columns, r.gain_range.lower, r.gain_range.upper)
    end
    for res in r.robust
        push!(names, _rule_name(res.rule))
        push!(columns, [a in res.chosen for a in actions])
    end
    if r.pareto !== nothing
        push!(names, :ParetoFront)
        push!(columns, [a in r.pareto.chosen for a in actions])
    end
    return [NamedTuple{Tuple(names)}(Tuple(c[i] for c in columns))
            for i in eachindex(actions)]
end

Tables.istable(::Type{<:DecisionReport}) = true
Tables.rowaccess(::Type{<:DecisionReport}) = true
Tables.rows(r::DecisionReport) = _report_rows(r)
