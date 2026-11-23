# Savage regret. No function here is called `regret` or `regrets`:
# CategoricalInfluenceDiagrams exports `regrets` for a best-response gap, which is a
# different quantity.

# R[a, θ] = max_a′ M[a′, θ] − M[a, θ]
_regret_matrix(M::AbstractMatrix) = maximum(M; dims=1) .- M

function _relative_regret_matrix(M::AbstractMatrix)
    best = maximum(M; dims=1)
    j = findfirst(iszero, best)
    j === nothing ||
        throw(ArgumentError("relative regret is undefined: the best payoff in column " *
                            "$(j[2]) of the table is zero"))
    return (best .- M) ./ abs.(best)
end

"""
    RegretTable{T} <: AbstractMatrix{T}

A regret table with its labels, as returned by [`regret_table`](@ref) and
[`relative_regret_table`](@ref). It is a matrix (rows = actions, columns = states):
it can be indexed, compared with a plain matrix and used in arithmetic like one.
The labels are in the fields `actions` and `states`, the entries in `R`, and
`relative` says whether it is a relative regret table. It prints as a grid of
actions against states, and renders as a Markdown or HTML table in a notebook or a
Quarto document.
"""
struct RegretTable{T<:Real} <: AbstractMatrix{T}
    actions::Vector{Symbol}
    states::Vector{Symbol}
    R::Matrix{T}
    relative::Bool
end

Base.size(t::RegretTable) = size(t.R)
Base.getindex(t::RegretTable, i::Int, j::Int) = t.R[i, j]
Base.IndexStyle(::Type{<:RegretTable}) = IndexCartesian()

state_labels(t::RegretTable) = t.states
action_labels(t::RegretTable) = t.actions

function _regret_labels(problem::DecisionProblem)
    states = problem.payoff.states
    b = problem.belief
    cols = b isa ScenarioSet ? findall(in(b.states), states) :
           b isa PointBelief ? findall(==(b.state), states) : eachindex(states)
    return feasible_actions(problem), states[cols]
end

"""
    regret_table(payoff::PayoffTable) -> RegretTable
    regret_table(problem::DecisionProblem) -> RegretTable

The Savage regret table `R[a, θ] = max_a′ Q[a′, θ] − Q[a, θ]`: what is lost in state
`θ` by taking `a` instead of the best action for that state. Rows are actions and
columns are states, as in the payoff table. Every entry is nonnegative and every
column has a zero. The result is a [`RegretTable`](@ref): a matrix that carries its
labels.

For a problem, the maximum runs over the feasible actions only and the rows are the
[`feasible_actions`](@ref); for a [`ScenarioSet`](@ref) or a [`PointBelief`](@ref)
the columns are the states the belief allows. Regret depends on the set of actions
compared: adding or removing an action can change every entry of a column.

```jldoctest
julia> R = regret_table(PayoffTable([:wait, :act], [:low, :high], [0 -1000; -200 -200]));

julia> R == [0 800; 200 0]
true

julia> R
RegretTable{Int64}: regret, 2 actions × 2 states
  action  low  high
  ------  ---  ----
  wait      0   800
  act     200     0
```
"""
function regret_table(payoff::PayoffTable)
    return RegretTable(payoff.actions, payoff.states, _regret_matrix(payoff.Q), false)
end
function regret_table(problem::DecisionProblem)
    actions, states = _regret_labels(problem)
    return RegretTable(actions, states, _regret_matrix(_view(problem).Q), false)
end

"""
    relative_regret_table(payoff::PayoffTable) -> RegretTable
    relative_regret_table(problem::DecisionProblem) -> RegretTable

Regret as a share of the best achievable payoff in each state,
`R[a, θ] / |max_a′ Q[a′, θ]|`, with the same layout and the same restrictions as
[`regret_table`](@ref). It is defined only when no state has a best payoff of zero;
otherwise an `ArgumentError` is thrown. Unlike regret, it is not invariant under
adding a constant to the payoffs.
"""
function relative_regret_table(payoff::PayoffTable)
    return RegretTable(payoff.actions, payoff.states, _relative_regret_matrix(payoff.Q),
                       true)
end
function relative_regret_table(problem::DecisionProblem)
    actions, states = _regret_labels(problem)
    return RegretTable(actions, states, _relative_regret_matrix(_view(problem).Q), true)
end

"""
    expected_regret(problem::DecisionProblem) -> Vector

The expected regret (expected opportunity loss) of every feasible action under the
single prior of the problem, `Σ_θ p(θ) R[a, θ]`. Its minimum equals [`evpi`](@ref)
and is attained exactly at the actions that maximise expected utility. Needs a
[`FiniteBelief`](@ref) or a [`PointBelief`](@ref).
"""
function expected_regret(problem::DecisionProblem)
    v = _view(problem)
    return _scores(Expectation(), _regret_matrix(v.Q), v, :min)
end

"""
    max_regret(problem::DecisionProblem) -> Vector

The largest regret of every feasible action over the states the belief allows (every
state of the table unless the belief is a [`ScenarioSet`](@ref) or a
[`PointBelief`](@ref)). Minimising it is [`MinimaxRegret`](@ref).
"""
function max_regret(problem::DecisionProblem)
    v = _view(problem)
    return _scores(WorstCase(), _regret_matrix(v.Q), v, :min)
end
