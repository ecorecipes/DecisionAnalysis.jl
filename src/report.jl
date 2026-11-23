# The decision report: every applicable rule side by side.

"""
    DecisionReport(problem::DecisionProblem; credal=nothing, experiment=nothing,
                   hurwicz=nothing, assumptions=String[])

Everything the package can say about one problem, side by side, so that a decision
that is contested between rules is seen to be contested. Which parts are filled
depends on the belief:

- State-wise rules, always: [`Maximin`](@ref), [`Maximax`](@ref), Laplace,
  [`MinimaxRegret`](@ref), and Hurwicz when the optimism index `hurwicz` is given;
  the [`max_regret`](@ref) profile.
- With a single prior ([`FiniteBelief`](@ref) or [`PointBelief`](@ref)):
  [`MaxExpectedUtility`](@ref), the [`expected_regret`](@ref) profile, [`evpi`](@ref),
  and the [`evsi`](@ref) of `experiment` when one is given.
- With a credal set, either as the belief of the problem or passed as `credal`
  beside a single prior: [`GammaMaximin`](@ref), the robust sets
  [`EAdmissible`](@ref), [`Maximal`](@ref) and [`IntervalDominance`](@ref), and
  [`robust_evpi`](@ref).

When the problem has a baseline action (no intervention), the report states it in
its header, gives every action's expected gain over it (under a single prior), and
says for every rule whether the action chosen is the baseline. When it has none, the
assumptions say so.

Fields: `problem`, `results` (the scalar rules, a vector of
[`DecisionResult`](@ref)), `robust` (the set-valued rules), `max_regret`,
`expected_regret`, `expected_gain` (over the baseline), `evpi`, `evsi`,
`robust_evpi` (each `nothing` when not applicable) and `assumptions`, the stated assumptions under which the numbers hold,
to which the strings in the `assumptions` keyword are appended.

A report prints as a table of actions against rules, and is a Tables.jl row table
with one row per feasible action: the action, each scalar rule's score and whether
it chose the action, the regret profile, the expected gain over the baseline when
there is one, and membership of each robust set.
"""
struct DecisionReport{P<:DecisionProblem}
    problem::P
    results::Vector{DecisionResult}
    robust::Vector{DecisionResult}
    max_regret::Vector
    expected_regret::Union{Nothing,Vector}
    expected_gain::Union{Nothing,Vector}
    evpi::Union{Nothing,Real}
    evsi::Union{Nothing,Real}
    robust_evpi::Union{Nothing,RobustVOI}
    assumptions::Vector{String}
end

_has_prior(p::DecisionProblem) = p.belief isa Union{FiniteBelief,PointBelief}

function DecisionReport(problem::DecisionProblem;
                        credal::Union{Nothing,CredalSet}=nothing,
                        experiment::Union{Nothing,Experiment}=nothing,
                        hurwicz::Union{Nothing,Real}=nothing,
                        assumptions::AbstractVector{<:AbstractString}=String[])
    belief = problem.belief
    belief isa CredalSet && credal !== nothing &&
        throw(ArgumentError("the problem already has a credal set as its belief; do " *
                            "not pass `credal` as well"))
    belief isa ScenarioSet && credal !== nothing &&
        throw(ArgumentError("`credal` accompanies a single prior, not a scenario set"))
    single = _has_prior(problem)
    experiment === nothing || single ||
        throw(BeliefFormError(:DecisionReport, nameof(typeof(belief)),
                              "the EVSI of an experiment needs a single prior; use " *
                              "robust_evsi for a credal set"))
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
        for r in (EAdmissible(), Maximal(), IntervalDominance())
            push!(robust, decide(robust_problem, r))
        end
        robust_value = robust_evpi(robust_problem)
    end

    notes = ["Payoffs are conditional expected utilities of each action in each state; " *
             "higher is better.",
             "State-wise rules (maximin, maximax, Laplace, Hurwicz, minimax regret) " *
             "ignore probabilities and range over " *
             (belief isa ScenarioSet ? "the states of the scenario set." :
              belief isa PointBelief ? "the one known state." :
              "every state of the table, including any of probability zero."),
             "Regret is measured against the best feasible action in each state, so it " *
             "changes when the set of actions changes.",
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
    experiment === nothing ||
        push!(notes,
              "EVSI is gross of the cost of the experiment and assumes the stated " *
              "likelihood.")
    if robust_problem !== nothing
        k = length(robust_problem.belief.points)
        push!(notes,
              "The credal set is the convex hull of $k stated prior" *
              (k == 1 ? "" : "s") * "; E-admissibility is decided over the whole hull.",
              "The rule-specific robust EVPI is under Γ-maximin; the envelope is the " *
              "range of the single-prior EVPI over the credal set.")
    end
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
               ", and no expected gain is shown because the belief has no single prior."))
    end
    append!(notes, String.(assumptions))

    return DecisionReport(problem, results, robust, max_regret(problem),
                          single ? expected_regret(problem) : nothing,
                          single && problem.baseline !== nothing ?
                          expected_gain_over_baseline(problem) : nothing,
                          single ? evpi(problem) : nothing,
                          experiment === nothing ? nothing : evsi(problem, experiment),
                          robust_value, notes)
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
    for res in r.robust
        push!(names, _rule_name(res.rule))
        push!(columns, [a in res.chosen for a in actions])
    end
    return [NamedTuple{Tuple(names)}(Tuple(c[i] for c in columns))
            for i in eachindex(actions)]
end

Tables.istable(::Type{<:DecisionReport}) = true
Tables.rowaccess(::Type{<:DecisionReport}) = true
Tables.rows(r::DecisionReport) = _report_rows(r)
