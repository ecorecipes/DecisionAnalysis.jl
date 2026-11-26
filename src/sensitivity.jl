# Decision sensitivity: how the choice moves along one weight or parameter.

"""
    policy_regions(base, direction; from=0, to=1) -> Vector{NamedTuple}

The **policy regions** of a one-parameter family of problems: the intervals of the
parameter `t` on which each action maximises expected utility, when the payoff is

    Q(t) = Q_base + t · Q_direction,    from ≤ t ≤ to.

`base` is a [`DecisionProblem`](@ref) with a single prior (a finite or sample belief)
and `direction` a [`PayoffTable`](@ref) with the same labels. Expected utility is
linear in `t` for every action, so the regions are found exactly, as the upper
envelope of straight lines: the result is exact for exact inputs, and nothing is
gridded. One `(; from, to, actions)` per region, in order, where `actions` are the
feasible actions that are optimal throughout the region (several when they tie
everywhere on it).

This is the weight sweep of the infection utility: with `base` the problem on the
uninfected counts and `direction` minus the infected counts, `t` is the weight `w`
in `uninfected − w × infected`. The interior boundaries are the
[`one_way_thresholds`](@ref).

```jldoctest
julia> uninfected = PayoffTable([:none, :treat], [:low, :high], [90 40; 80 70]);

julia> infected = PayoffTable([:none, :treat], [:low, :high], [10 50; 15 20]);

julia> p = DecisionProblem(uninfected, [1 // 2, 1 // 2]; baseline=:none);

julia> policy_regions(p, -infected; from=0, to=2)
1-element Vector{@NamedTuple{from::Rational{Int64}, to::Rational{Int64}, actions::Vector{Symbol}}}:
 (from = 0, to = 2, actions = [:treat])
```
"""
function policy_regions(base::DecisionProblem, direction::PayoffTable; from::Real=0,
                        to::Real=1)
    from < to || throw(ArgumentError("need from < to, got from = $from and to = $to"))
    _check_same_labels(base.payoff, direction)
    p = _prior(base, :policy_regions)
    α = _feasible_Q(base) * p
    β = _feasible_rows(base, direction.Q) * p
    actions = feasible_actions(base)
    T = typeof((one(eltype(α)) + one(eltype(β)) * from) / one(eltype(β)))
    function tol(t)
        return _is_exact(T) ? zero(T) :
               sqrt(eps(float(T))) * max(one(T), maximum(abs, α .+ t .* β))
    end
    best_at(t) = (v=α .+ t .* β; findall(>=(maximum(v) - tol(t)), v))
    # Candidate breakpoints: every crossing of two lines inside the interval.
    points = T[from, to]
    for i in eachindex(α), j in (i + 1):length(α)
        β[i] == β[j] && continue
        t = (α[j] - α[i]) / (β[i] - β[j])
        from < t < to && push!(points, t)
    end
    sort!(unique!(points))
    regions = @NamedTuple{from::T, to::T, actions::Vector{Symbol}}[]
    for k in 1:(length(points) - 1)
        lo, hi = points[k], points[k + 1]
        # Lines optimal at the midpoint are optimal on the whole open interval,
        # because no two lines cross inside it.
        best = actions[best_at((lo + hi) / 2)]
        if !isempty(regions) && regions[end].actions == best
            regions[end] = (; from=regions[end].from, to=hi, actions=best)
        else
            push!(regions, (; from=lo, to=hi, actions=best))
        end
    end
    return regions
end

"""
    one_way_thresholds(base, direction; from=0, to=1) -> Vector{NamedTuple}

The values of the parameter `t` at which the action that maximises expected utility
changes, for the one-parameter family of [`policy_regions`](@ref): one
`(; threshold, below, above)` per switch, with the optimal actions on either side.
An empty result says that one choice holds over the whole range, so the decision
is insensitive to the parameter there.
"""
function one_way_thresholds(base::DecisionProblem, direction::PayoffTable; kw...)
    regions = policy_regions(base, direction; kw...)
    return [(; threshold=regions[k].to, below=regions[k].actions,
             above=regions[k + 1].actions) for k in 1:(length(regions) - 1)]
end

"""
    weight_sweep(build, weights; rule=MaxExpectedUtility()) -> Vector{NamedTuple}

Apply `rule` to the problem `build(w)` for every `w` in `weights` and tabulate the
result: one `(; weight, first_choice, chosen, scores, ranking)` per weight, where
`ranking` lists the feasible actions from best to worst under the rule's score (ties
in table order). `build` is any function from a weight to a
[`DecisionProblem`](@ref), so the sweep works for any utility and any rule, for
example

    weight_sweep(w -> DecisionProblem(outcomes, InfectionUtility(; weight=w);
                                      baseline=:none), 0:0.25:2)

The result is a Tables.jl row table. For expected utility that is linear in the
weight, [`policy_regions`](@ref) gives the switch points exactly instead of on a
grid. See [`rank_reversals`](@ref) for where the order of the actions changes.
"""
function weight_sweep(build, weights; rule::AbstractRule=MaxExpectedUtility())
    return map(collect(weights)) do w
        problem = build(w)
        problem isa DecisionProblem ||
            throw(ArgumentError("`build` must return a DecisionProblem, got " *
                                "$(typeof(problem))"))
        result = decide(problem, rule)
        result.scores isa AbstractVector ||
            throw(ArgumentError("weight_sweep needs a rule that scores every action"))
        order = sortperm(result.scores; rev=result.sense === :max, alg=MergeSort)
        return (; weight=w, first_choice=first_choice(result), chosen=result.chosen,
                scores=result.scores, ranking=result.actions[order])
    end
end

"""
    rank_reversals(sweep) -> Vector{NamedTuple}
    rank_reversals(a::DecisionProblem, b::DecisionProblem; aggregator=Expectation())

**Rank reversals**: pairs of actions whose order differs between two analyses.

The first form takes the result of [`weight_sweep`](@ref) and compares each weight
with the next: one `(; from, to, actions, top_changed)` per pair of actions that
swap between the two weights, where `top_changed` says whether the first choice
itself changed.

The second form compares two problems with the same feasible actions (two
utilities, two beliefs, a count and a fraction): one `(; actions, top_changed)` per
pair of actions that the aggregator ranks strictly one way in `a` and strictly the
other way in `b`. An empty result says that the two analyses order the actions
alike, ties aside.
"""
function rank_reversals(sweep::AbstractVector{<:NamedTuple})
    out = @NamedTuple{from::Any, to::Any, actions::Tuple{Symbol,Symbol},
                      top_changed::Bool}[]
    for k in 1:(length(sweep) - 1)
        x, y = sweep[k], sweep[k + 1]
        position = Dict(a => i for (i, a) in enumerate(y.ranking))
        all(a -> haskey(position, a), x.ranking) &&
        length(x.ranking) == length(y.ranking) ||
            throw(ArgumentError("the feasible actions change along the sweep"))
        for i in eachindex(x.ranking), j in (i + 1):length(x.ranking)
            a, b = x.ranking[i], x.ranking[j]
            position[a] > position[b] &&
                push!(out,
                      (; from=x.weight, to=y.weight, actions=(a, b),
                       top_changed=x.first_choice != y.first_choice))
        end
    end
    return out
end

function rank_reversals(a::DecisionProblem, b::DecisionProblem;
                        aggregator::AbstractAggregator=Expectation())
    actions = feasible_actions(a)
    actions == feasible_actions(b) ||
        throw(LabelMismatchError("feasible actions of the two problems", actions,
                                 feasible_actions(b)))
    sa, sb = aggregate(aggregator, a), aggregate(aggregator, b)
    top_changed = actions[argmax(sa)] != actions[argmax(sb)]
    return [(; actions=(actions[i], actions[j]), top_changed)
            for i in eachindex(actions)
            for j in (i + 1):length(actions)
            if (sa[i] > sa[j] && sb[i] < sb[j]) || (sa[i] < sa[j] && sb[i] > sb[j])]
end
