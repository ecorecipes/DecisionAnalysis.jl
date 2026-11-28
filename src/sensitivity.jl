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
    T = _div_type(promote_type(eltype(α), eltype(β), typeof(from), typeof(to)),
                  promote_type(eltype(β), Int))
    function tol(t)
        return _is_exact(T) ? zero(T) :
               sqrt(eps(float(T))) * max(one(T), maximum(abs, α .+ t .* β))
    end
    best_at(t) = (v=α .+ t .* β; findall(>=(maximum(v) - tol(t)), v))
    # Candidate breakpoints: every crossing of two lines inside the interval.
    points = T[from, to]
    for i in eachindex(α), j in (i + 1):length(α)
        β[i] == β[j] && continue
        t = _div(α[j] - α[i], β[i] - β[j])
        from < t < to && push!(points, t)
    end
    sort!(unique!(points))
    regions = @NamedTuple{from::T, to::T, actions::Vector{Symbol}}[]
    for k in 1:(length(points) - 1)
        lo, hi = points[k], points[k + 1]
        # Lines optimal at the midpoint are optimal on the whole open interval,
        # because no two lines cross inside it.
        best = actions[best_at(_div(lo + hi, 2))]
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
                scores=result.scores, ranking=result.actions[order], sense=result.sense)
    end
end

"""
    rank_reversals(sweep) -> Vector{NamedTuple}
    rank_reversals(a::DecisionProblem, b::DecisionProblem; aggregator=Expectation())

**Rank reversals**: pairs of actions whose order differs between two analyses.

The first form takes the result of [`weight_sweep`](@ref) and compares each weight
with the next: one `(; from, to, actions, top_changed)` per pair of actions that
strictly swap between the two weights (ties aside), where `top_changed` says whether
the first choice itself changed.

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
        sense_x = get(x, :sense, :max)
        sense_y = get(y, :sense, :max)
        sx = sort(collect(x.scores); rev=sense_x === :max)
        sy = sort(collect(y.scores); rev=sense_y === :max)
        tol_x = _tolerance(sx, 0, _default_rtol(eltype(sx)))
        tol_y = _tolerance(sy, 0, _default_rtol(eltype(sy)))
        for i in eachindex(x.ranking), j in (i + 1):length(x.ranking)
            a, b = x.ranking[i], x.ranking[j]
            pi, pj = position[a], position[b]
            pi > pj || continue
            # Require a strict gap in both rows so ties do not register as reversals.
            gap_x = sense_x === :max ? sx[i] - sx[j] : sx[j] - sx[i]
            gap_y = sense_y === :max ? sy[pj] - sy[pi] : sy[pi] - sy[pj]
            if gap_x > tol_x && gap_y > tol_y
                push!(out,
                      (; from=x.weight, to=y.weight, actions=(a, b),
                       top_changed=x.first_choice != y.first_choice))
            end
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
    ta = _tolerance(sa, 0, _default_rtol(eltype(sa)))
    tb = _tolerance(sb, 0, _default_rtol(eltype(sb)))
    top_a = findfirst(>=(maximum(sa) - ta), sa)
    top_b = findfirst(>=(maximum(sb) - tb), sb)
    top_changed = actions[top_a] != actions[top_b]
    return [(; actions=(actions[i], actions[j]), top_changed)
            for i in eachindex(actions)
            for j in (i + 1):length(actions)
            if (sa[i] - sa[j] > ta && sb[j] - sb[i] > tb) ||
        (sa[j] - sa[i] > ta && sb[i] - sb[j] > tb)]
end

"""
    two_way_policy_regions(base::DecisionProblem, dir1::PayoffTable, dir2::PayoffTable;
                           range1=(0, 1), range2=(0, 1), grid=(25, 25)) -> NamedTuple

Two-way sensitivity map of the optimal action under [`MaxExpectedUtility`](@ref) for the
two-parameter family

    Q(t₁, t₂) = Q_base + t₁ · Q_dir1 + t₂ · Q_dir2,

evaluated over `range1 × range2` (each a `(from, to)` tuple or explicit grid vector;
`grid = (n1, n2)` sets the resolution when ranges are tuples).

Returns `(; t1, t2, action_grid, score_grid, actions, shares, rows)`: `action_grid[i, j]`
is the first optimal action at `(t1[i], t2[j])`, `shares` gives the fraction of grid
points won by each action, and `rows` is a Tables.jl-compatible vector of
`(; t1, t2, action, tied, score)` records.
"""
function two_way_policy_regions(base::DecisionProblem, dir1::PayoffTable,
                                dir2::PayoffTable; range1=(0, 1), range2=(0, 1),
                                grid::Tuple{Integer,Integer}=(25, 25))
    _check_same_labels(base.payoff, dir1)
    _check_same_labels(base.payoff, dir2)
    p = _prior(base, :two_way_policy_regions)
    α = _feasible_Q(base) * p
    β1 = _feasible_rows(base, dir1.Q) * p
    β2 = _feasible_rows(base, dir2.Q) * p
    actions = feasible_actions(base)
    function _make_axis(r, n::Integer, label::AbstractString)
        if r isa Tuple && length(r) == 2
            n >= 2 || throw(ArgumentError("grid size for $label must be at least 2"))
            r[1] < r[2] || throw(ArgumentError("$label needs from < to, got $r"))
            return collect(range(Float64(r[1]), Float64(r[2]); length=Int(n)))
        end
        v = collect(r)
        isempty(v) && throw(ArgumentError("$label cannot be empty"))
        return v
    end
    t1 = _make_axis(range1, grid[1], "range1")
    t2 = _make_axis(range2, grid[2], "range2")
    n1, n2 = length(t1), length(t2)
    action_grid = Matrix{Symbol}(undef, n1, n2)
    score_grid = Matrix{Float64}(undef, n1, n2)
    counts = Dict{Symbol,Int}(a => 0 for a in actions)
    rows = NamedTuple[]
    sizehint!(rows, n1 * n2)
    for j in 1:n2, i in 1:n1
        v = Float64.(α .+ t1[i] .* β1 .+ t2[j] .* β2)
        tol = _tolerance(v, 0, _default_rtol(Float64))
        top = maximum(v)
        best = actions[findall(>=(top - tol), v)]
        winner = first(best)
        action_grid[i, j] = winner
        score_grid[i, j] = top
        counts[winner] += 1
        push!(rows, (; t1=t1[i], t2=t2[j], action=winner, tied=best, score=top))
    end
    total_pts = n1 * n2
    shares = NamedTuple{Tuple(actions)}(Tuple(counts[a] / total_pts for a in actions))
    return (; t1, t2, action_grid, score_grid, actions, shares, rows)
end

"""
    decision_curve(outcomes::AbstractVector{<:Real}, predictions::AbstractVector{<:Real};
                   thresholds=0.05:0.05:0.95, weights=nothing) -> Vector{NamedTuple}

**Decision curve analysis** (Vickers and Elkin 2006, *Medical Decision Making* 26:565):
evaluates the net benefit of acting whenever predicted event risk `predictions[i] ≥ p_t`
across risk thresholds `p_t ∈ (0, 1)`, against the two reference strategies **treat all**
and **treat none**:

    NB(p_t) = TP(p_t) / N − (FP(p_t) / N) · p_t / (1 − p_t).

`outcomes` must be binary (`0` or `1`, or `Bool`) and `predictions` must lie in `[0, 1]`.
Returns a Tables.jl-compatible vector of named tuples `(; threshold, net_benefit,
treat_all, treat_none, net_benefit_over_all, interventions_avoided)` — exact when
`outcomes`, `predictions`, `thresholds` and `weights` are exact rationals/integers.
"""
function decision_curve(outcomes::AbstractVector{<:Real},
                        predictions::AbstractVector{<:Real};
                        thresholds=0.05:0.05:0.95,
                        weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    n = length(outcomes)
    n >= 1 || throw(ArgumentError("outcomes cannot be empty"))
    length(predictions) == n ||
        throw(ArgumentError("outcomes ($n) and predictions ($(length(predictions))) must " *
                            "have the same length"))
    all(y -> y == 0 || y == 1, outcomes) ||
        throw(ArgumentError("outcomes must be binary (0 or 1)"))
    all(p -> isfinite(p) && 0 <= p <= 1, predictions) ||
        throw(ArgumentError("predictions must lie in [0, 1]"))
    ts = collect(thresholds)
    isempty(ts) && throw(ArgumentError("thresholds cannot be empty"))
    all(t -> isfinite(t) && 0 < t < 1, ts) ||
        throw(ArgumentError("thresholds must lie strictly in (0, 1)"))
    w = weights === nothing ? fill(1 // n, n) : _normalise_weights(collect(weights))
    length(w) == n || throw(ArgumentError("$(length(w)) weights for $n observations"))
    y_int = [y == 1 ? 1 : 0 for y in outcomes]
    prevalence = sum(w[i] for i in 1:n if y_int[i] == 1; init=zero(eltype(w)))
    return map(ts) do pt
        odds = _div(pt, one(pt) - pt)
        tp = sum(w[i] for i in 1:n if predictions[i] >= pt && y_int[i] == 1;
                 init=zero(eltype(w)))
        fp = sum(w[i] for i in 1:n if predictions[i] >= pt && y_int[i] == 0;
                 init=zero(eltype(w)))
        nb = tp - fp * odds
        nb_all = prevalence - (one(prevalence) - prevalence) * odds
        nb_none = zero(nb)
        over_all = nb - nb_all
        avoided = _div(over_all, odds)
        return (; threshold=pt, net_benefit=nb, treat_all=nb_all, treat_none=nb_none,
                net_benefit_over_all=over_all, interventions_avoided=avoided)
    end
end
