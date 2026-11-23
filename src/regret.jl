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
    regret_table(payoff::PayoffTable) -> Matrix
    regret_table(problem::DecisionProblem) -> Matrix

The Savage regret table `R[a, θ] = max_a′ Q[a′, θ] − Q[a, θ]`: what is lost in state
`θ` by taking `a` instead of the best action for that state. Rows are actions and
columns are states, as in the payoff table. Every entry is nonnegative and every
column has a zero.

For a problem, the maximum runs over the feasible actions only and the rows are the
[`feasible_actions`](@ref); for a [`ScenarioSet`](@ref) or a [`PointBelief`](@ref)
the columns are the states the belief allows. Regret depends on the set of actions
compared: adding or removing an action can change every entry of a column.

```jldoctest
julia> regret_table(PayoffTable([0 -1000; -200 -200]))
2×2 Matrix{Int64}:
   0  800
 200    0
```
"""
regret_table(payoff::PayoffTable) = _regret_matrix(payoff.Q)
regret_table(problem::DecisionProblem) = _regret_matrix(_view(problem).Q)

"""
    relative_regret_table(payoff::PayoffTable) -> Matrix
    relative_regret_table(problem::DecisionProblem) -> Matrix

Regret as a share of the best achievable payoff in each state,
`R[a, θ] / |max_a′ Q[a′, θ]|`, with the same layout and the same restrictions as
[`regret_table`](@ref). It is defined only when no state has a best payoff of zero;
otherwise an `ArgumentError` is thrown. Unlike regret, it is not invariant under
adding a constant to the payoffs.
"""
relative_regret_table(payoff::PayoffTable) = _relative_regret_matrix(payoff.Q)
function relative_regret_table(problem::DecisionProblem)
    return _relative_regret_matrix(_view(problem).Q)
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
