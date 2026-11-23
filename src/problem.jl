# The decision problem: a payoff table, a belief and a feasibility filter.

"""
    DecisionProblem(payoff::PayoffTable, belief; feasible=nothing, baseline=nothing)
    DecisionProblem(payoff::PayoffTable, probabilities::AbstractVector; feasible=nothing,
                    baseline=nothing)

A decision problem under uncertainty: choose a row of `payoff` given `belief` about
its columns. `belief` is a [`PointBelief`](@ref), a [`FiniteBelief`](@ref), a
[`SampleBelief`](@ref), a [`CredalSet`](@ref) or a [`ScenarioSet`](@ref); a plain
probability vector is read as a `FiniteBelief` over the states of the table.

The states of a `FiniteBelief`, `SampleBelief` or `CredalSet` must equal the states of the table, in
order; the state of a `PointBelief` and the states of a `ScenarioSet` must be among
them. A mismatch throws [`LabelMismatchError`](@ref).

`feasible` filters the actions: `nothing` keeps all, a `Bool` vector keeps the rows
marked `true`, a collection of labels keeps those actions, and a function is called
on each action label. Infeasible actions are ignored by everything: they receive no
score, cannot be chosen, and do not enter regret or any value of information. A
filter that leaves no action throws [`NoFeasibleActionError`](@ref).

`baseline` names the action that stands for no intervention (business as usual), so
that every analysis can be read against it: see [`gain_over_baseline`](@ref) and
[`expected_gain_over_baseline`](@ref), and the baseline section of a
[`DecisionReport`](@ref). It must be one of the actions of the table, otherwise
[`InvalidBaselineError`](@ref) is thrown; it need not be feasible (business as usual
may be what is no longer allowed). Setting a baseline changes no score, no choice and
no value of information.

```jldoctest
julia> t = PayoffTable([:wait, :act], [:low, :high], [0 -100; -30 -30]);

julia> p = DecisionProblem(t, [3 // 5, 2 // 5]);

julia> first_choice(decide(p, MaxExpectedUtility()))
:act
```
"""
struct DecisionProblem{T<:Real,B<:AbstractBelief,S}
    payoff::PayoffTable{T,S}
    belief::B
    feasible::Vector{Bool}
    baseline::Union{Nothing,Symbol}
    function DecisionProblem(payoff::PayoffTable{T,S}, belief::B, feasible::Vector{Bool},
                             baseline::Union{Nothing,Symbol}=nothing) where {T,S,
                                                                             B<:AbstractBelief}
        _check_belief(payoff, belief)
        baseline === nothing || baseline in payoff.actions ||
            throw(InvalidBaselineError(baseline, payoff.actions))
        length(feasible) == size(payoff, 1) ||
            throw(ArgumentError("the feasibility mask has length $(length(feasible)) " *
                                "for $(size(payoff, 1)) actions"))
        any(feasible) ||
            throw(NoFeasibleActionError("the feasibility filter leaves none of the " *
                                        "actions $(payoff.actions)"))
        return new{T,B,S}(payoff, belief, feasible, baseline)
    end
end

function DecisionProblem(payoff::PayoffTable, belief::AbstractBelief; feasible=nothing,
                         baseline::Union{Nothing,Symbol,AbstractString}=nothing)
    return DecisionProblem(payoff, belief, _feasibility_mask(payoff, feasible),
                           baseline === nothing ? nothing : Symbol(baseline))
end

function DecisionProblem(payoff::PayoffTable, probabilities::AbstractVector{<:Real};
                         feasible=nothing, baseline=nothing)
    length(probabilities) == size(payoff, 2) ||
        throw(InvalidBeliefError("$(length(probabilities)) probabilities for the " *
                                 "$(size(payoff, 2)) states of the payoff table"))
    return DecisionProblem(payoff, FiniteBelief(payoff.states, probabilities); feasible,
                           baseline)
end

_feasibility_mask(payoff::PayoffTable, ::Nothing) = fill(true, size(payoff, 1))
_feasibility_mask(::PayoffTable, mask::AbstractVector{Bool}) = collect(Bool, mask)
function _feasibility_mask(payoff::PayoffTable, f::Function)
    return Bool[f(a) for a in payoff.actions]
end
function _feasibility_mask(payoff::PayoffTable, keep)
    wanted = _labels(keep)
    for a in wanted
        a in payoff.actions ||
            throw(ArgumentError("feasible action $a is not among the actions " *
                                "$(payoff.actions)"))
    end
    return Bool[a in wanted for a in payoff.actions]
end

function _check_belief(payoff::PayoffTable, b::Union{FiniteBelief,SampleBelief,CredalSet})
    b.states == payoff.states ||
        throw(LabelMismatchError("states of the belief against the payoff table",
                                 payoff.states, b.states))
    return nothing
end
function _check_belief(payoff::PayoffTable, b::Union{PointBelief,ScenarioSet})
    issubset(state_labels(b), payoff.states) ||
        throw(LabelMismatchError("states of the belief against the payoff table",
                                 payoff.states, state_labels(b)))
    return nothing
end

state_labels(p::DecisionProblem) = p.payoff.states
action_labels(p::DecisionProblem) = p.payoff.actions

"""
    feasible_actions(p::DecisionProblem) -> Vector{Symbol}

The labels of the actions that pass the feasibility filter, in row order. Scores,
regret tables and chosen sets all refer to these actions.
"""
feasible_actions(p::DecisionProblem) = p.payoff.actions[p.feasible]

"""
    feasible_payoff(p::DecisionProblem) -> PayoffTable

The payoff table restricted to the feasible actions.
"""
feasible_payoff(p::DecisionProblem) = _subtable(p.payoff, findall(p.feasible), :)

"""
    restrict_scenarios(p::DecisionProblem, states) -> DecisionProblem

The problem with only the given states kept, in table order: the scenario-subset step
of a [`RobustnessMetric`](@ref). The belief follows: a [`FiniteBelief`](@ref) and
every point of a [`CredalSet`](@ref) are conditioned on the subset (an `ArgumentError`
if a member gives it probability zero), a [`ScenarioSet`](@ref) is intersected with
it, and a [`PointBelief`](@ref) must lie in it.
"""
function restrict_scenarios(p::DecisionProblem, states)
    wanted = _labels(states)
    for s in wanted
        s in p.payoff.states ||
            throw(ArgumentError("state $s is not among the states $(p.payoff.states)"))
    end
    cols = findall(in(wanted), p.payoff.states)
    isempty(cols) && throw(ArgumentError("the scenario subset is empty"))
    payoff = _subtable(p.payoff, :, cols)
    return DecisionProblem(payoff, _restrict_belief(p.belief, p.payoff.states, cols),
                           p.feasible, p.baseline)
end

function _condition(probabilities::AbstractVector, cols)
    mass = sum(probabilities[cols])
    iszero(mass) &&
        throw(ArgumentError("a belief gives the scenario subset probability zero and " *
                            "cannot be conditioned on it"))
    return probabilities[cols] ./ mass
end
function _restrict_belief(b::FiniteBelief, states, cols)
    return FiniteBelief(states[cols], _condition(b.probabilities, cols))
end
function _restrict_belief(b::SampleBelief, states, cols)
    return SampleBelief(states[cols], map(c -> c[cols], b.draws),
                        _condition(b.weights, cols))
end
function _restrict_belief(b::CredalSet, states, cols)
    return CredalSet(states[cols], [_condition(q, cols) for q in b.points])
end
function _restrict_belief(b::ScenarioSet, states, cols)
    kept = [s for s in states[cols] if s in b.states]
    isempty(kept) &&
        throw(ArgumentError("the scenario subset does not meet the scenario set"))
    return ScenarioSet(kept)
end
function _restrict_belief(b::PointBelief, states, cols)
    b.state in states[cols] ||
        throw(ArgumentError("the scenario subset excludes the known state $(b.state)"))
    return b
end

function _baseline_row(p::DecisionProblem, operation::Symbol)
    p.baseline === nothing && throw(MissingBaselineError(operation))
    return findfirst(==(p.baseline), p.payoff.actions)
end

"""
    gain_over_baseline(p::DecisionProblem) -> PayoffTable

The payoff table of differences from the baseline action,
`G[a, θ] = Q[a, θ] − Q[baseline, θ]`, for the feasible actions: what each action
gains over no intervention in each state. The element type is that of the payoffs,
so rational tables stay exact; standard errors are not carried over, because the
error of a difference depends on how the cells were simulated. The row of the
baseline, when it is feasible, is zero.

Subtracting the same number from every action in a state changes no regret, no
Bayes action and no value of information, so a problem restated in gains over the
baseline has the same [`regret_table`](@ref), [`evpi`](@ref) and [`evsi`](@ref).

Throws [`MissingBaselineError`](@ref) when the problem has no baseline.

```jldoctest
julia> t = PayoffTable([:wait, :act], [:low, :high], [0 -100; -30 -30]);

julia> p = DecisionProblem(t, [3 // 5, 2 // 5]; baseline=:wait);

julia> gain_over_baseline(p).Q
2×2 Matrix{Int64}:
   0   0
 -30  70

julia> expected_gain_over_baseline(p)
2-element Vector{Rational{Int64}}:
  0
 10
```
"""
function gain_over_baseline(p::DecisionProblem)
    row = _baseline_row(p, :gain_over_baseline)
    rows = findall(p.feasible)
    G = p.payoff.Q[rows, :] .- p.payoff.Q[row:row, :]
    return PayoffTable(p.payoff.actions[rows], p.payoff.states, G)
end

"""
    expected_gain_over_baseline(p::DecisionProblem)

The gain of every feasible action over the baseline action (no intervention), in
[`feasible_actions`](@ref) order. What is returned depends on the belief.

- **A single prior** ([`FiniteBelief`](@ref), [`SampleBelief`](@ref) or
  [`PointBelief`](@ref)): a vector, the
  expected gain `Σ_θ p(θ) (Q[a, θ] − Q[baseline, θ])` of each action, which is its
  expected utility minus that of no intervention.
- **A [`CredalSet`](@ref)**: a named tuple `(lower = ..., upper = ...)` of two
  vectors, the smallest and the largest expected gain of each action over the set.
  The expected gain is linear in the prior, so both are attained at generating
  points. The two ends are in general attained at different priors for different
  actions; `lower[a] > 0` says that `a` beats the baseline under every prior in the
  set.
- **A [`ScenarioSet`](@ref)**: a named tuple `(lower = ..., upper = ...)` of the
  minimum and the maximum of the gain `Q[a, θ] − Q[baseline, θ]` over the states of
  the set. No probabilities are involved, so these are gains, not expected gains.

The row of the baseline, when it is feasible, is zero in every case. Throws
[`MissingBaselineError`](@ref) when the problem has no baseline.

```jldoctest
julia> t = PayoffTable([:wait, :act], [:low, :high], [0 -100; -30 -30]);

julia> credal = CredalSet([:low, :high], [[3 // 5, 2 // 5], [9 // 10, 1 // 10]]);

julia> expected_gain_over_baseline(DecisionProblem(t, credal; baseline=:wait))
(lower = Rational{Int64}[0, -20], upper = Rational{Int64}[0, 10])
```
"""
function expected_gain_over_baseline(p::DecisionProblem)
    row = _baseline_row(p, :expected_gain_over_baseline)
    G = p.payoff.Q[p.feasible, :] .- p.payoff.Q[row:row, :]
    return _expected_gain(p.belief, p, G)
end

function _expected_gain(::Union{FiniteBelief,SampleBelief,PointBelief}, p::DecisionProblem,
                        G)
    return G * _prior(p, :expected_gain_over_baseline)
end
function _expected_gain(b::CredalSet, ::DecisionProblem, G)
    E = reduce(hcat, [G * q for q in b.points])
    return (lower=vec(minimum(E; dims=2)), upper=vec(maximum(E; dims=2)))
end
function _expected_gain(b::ScenarioSet, p::DecisionProblem, G)
    cols = findall(in(b.states), p.payoff.states)
    return (lower=vec(minimum(view(G, :, cols); dims=2)),
            upper=vec(maximum(view(G, :, cols); dims=2)))
end

# Internal view of a problem: the payoff matrix of the feasible actions over the
# columns that the belief ranges over, and the prior vectors over those columns
# (`nothing` for a scenario set, which has none).
struct _View{M<:AbstractMatrix,P}
    Q::M
    priors::P
    form::Symbol
end

function _view(p::DecisionProblem{T,<:FiniteBelief}) where {T}
    return _View(p.payoff.Q[p.feasible, :], [p.belief.probabilities], :FiniteBelief)
end
function _view(p::DecisionProblem{T,<:SampleBelief}) where {T}
    return _View(p.payoff.Q[p.feasible, :], [p.belief.weights], :SampleBelief)
end
function _view(p::DecisionProblem{T,<:CredalSet}) where {T}
    return _View(p.payoff.Q[p.feasible, :], p.belief.points, :CredalSet)
end
function _view(p::DecisionProblem{T,ScenarioSet}) where {T}
    cols = findall(in(p.belief.states), p.payoff.states)
    return _View(p.payoff.Q[p.feasible, cols], nothing, :ScenarioSet)
end
function _view(p::DecisionProblem{T,PointBelief}) where {T}
    cols = findall(==(p.belief.state), p.payoff.states)
    return _View(p.payoff.Q[p.feasible, cols], [[1]], :PointBelief)
end

# The single prior of a problem, over all states of the table; throws for a form that
# has none. `operation` names the caller in the error.
function _prior(p::DecisionProblem{T,<:FiniteBelief}, ::Symbol) where {T}
    return p.belief.probabilities
end
function _prior(p::DecisionProblem{T,<:SampleBelief}, ::Symbol) where {T}
    return p.belief.weights
end
function _prior(p::DecisionProblem{T,PointBelief}, ::Symbol) where {T}
    return Int[s == p.belief.state for s in p.payoff.states]
end
function _prior(p::DecisionProblem{T,<:CredalSet}, operation::Symbol) where {T}
    throw(BeliefFormError(operation, :CredalSet,
                          "it needs a single prior; use robust_evpi or robust_evsi, " *
                          "or build the problem from one member of the set"))
end
function _prior(p::DecisionProblem{T,ScenarioSet}, operation::Symbol) where {T}
    throw(BeliefFormError(operation, :ScenarioSet,
                          "a scenario set carries no probabilities; supply a " *
                          "FiniteBelief"))
end
