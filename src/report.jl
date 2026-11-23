# The decision report: every applicable rule side by side.

"""
    DecisionReport(problem::DecisionProblem; credal=nothing, experiment=nothing,
                   experiments=nothing, partitions=nothing, objectives=nothing,
                   pareto=nothing, hurwicz=nothing, assumptions=String[],
                   optimizer=nothing)

Everything the package can say about one problem, side by side, so that a decision
that is contested between rules is seen to be contested. Which parts are filled
depends on the belief:

- State-wise rules, always: [`Maximin`](@ref), [`Maximax`](@ref), Laplace,
  [`MinimaxRegret`](@ref), and Hurwicz when the optimism index `hurwicz` is given;
  the [`max_regret`](@ref) profile.
- With a single prior ([`FiniteBelief`](@ref) or [`PointBelief`](@ref)):
  [`MaxExpectedUtility`](@ref), the [`expected_regret`](@ref) profile, [`evpi`](@ref),
  the [`evsi`](@ref) of every experiment given, and the [`evppi`](@ref) of every
  partition given.
- With a credal set, either as the belief of the problem or passed as `credal`
  beside a single prior: [`GammaMaximin`](@ref), the robust sets
  [`EAdmissible`](@ref), [`Maximal`](@ref) and [`IntervalDominance`](@ref), and
  [`robust_evpi`](@ref).

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
end

_has_prior(p::DecisionProblem) = p.belief isa Union{FiniteBelief,PointBelief}

_default_pareto(::Union{FiniteBelief,PointBelief}) = ParetoFront(Expectation())
_default_pareto(::CredalSet) = ParetoFront(LowerExpectation())
_default_pareto(::ScenarioSet) = ParetoFront(WorstCase())

function DecisionReport(problem::DecisionProblem;
                        credal::Union{Nothing,CredalSet}=nothing,
                        experiment::Union{Nothing,Experiment}=nothing,
                        experiments=nothing, partitions=nothing, objectives=nothing,
                        pareto::Union{Nothing,ParetoFront}=nothing,
                        hurwicz::Union{Nothing,Real}=nothing,
                        assumptions::AbstractVector{<:AbstractString}=String[],
                        optimizer=nothing)
    belief = problem.belief
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
    push!(rules, Maximin(), Maximax(), DecisionRule(Laplace(); name=:Laplace))
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
    append!(notes, String.(assumptions))

    gain = problem.baseline === nothing ? nothing : expected_gain_over_baseline(problem)
    return DecisionReport(problem, results, robust, max_regret(problem),
                          single ? expected_regret(problem) : nothing,
                          single ? gain : nothing, single ? nothing : gain,
                          single ? evpi(problem) : nothing,
                          isempty(experiment_rows) ? nothing : first(experiment_rows).evsi,
                          experiment_rows, partition_rows, robust_value, front, notes)
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
    if r.expected_gain !== nothing
        push!(names, :expected_gain_over_baseline)
        push!(columns, r.expected_gain)
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
