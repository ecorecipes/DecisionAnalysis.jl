# Model approximation: how much a decision can lose when the belief it was made under
# is off by a known amount.

"""
    total_variation(p, q)

The total-variation distance between two probability vectors, or two
[`FiniteBelief`](@ref)s over the same states:

    TV(p, q) = (1/2) Σ_θ |p(θ) − q(θ)| = max_A |p(A) − q(A)|.

The convention is **half** the ℓ¹ distance, so the result lies in `[0, 1]`. This is
the `ε` of [`decision_loss_bound`](@ref) and the distance inside
[`deficiency`](@ref).

```jldoctest
julia> total_variation([1 // 2, 1 // 2], [1 // 4, 3 // 4])
1//4
```
"""
function total_variation(p::AbstractVector{<:Real}, q::AbstractVector{<:Real})
    isempty(p) && throw(ArgumentError("the vectors cannot be empty"))
    length(p) == length(q) ||
        throw(ArgumentError("the vectors have lengths $(length(p)) and $(length(q))"))
    return _div(sum(abs(a - b) for (a, b) in zip(p, q)), 2)
end

function total_variation(p::FiniteBelief, q::FiniteBelief)
    p.states == q.states ||
        throw(LabelMismatchError("states of two beliefs", p.states, q.states))
    return total_variation(p.probabilities, q.probabilities)
end

function total_variation(p::SampleBelief, ::SampleBelief)
    return throw(BeliefFormError(:total_variation, :SampleBelief,
                                 "two samples of a continuous distribution share no draw, so " *
                                 "the distance between their empirical distributions says " *
                                 "nothing about the distributions"))
end

"""
    decision_loss_bound(B, ε)

The bound `2 B ε` on the expected utility lost by deciding under an approximate
belief. Conventions, which fix the constant:

- `ε` is a **total-variation** distance, half the ℓ¹ distance
  ([`total_variation`](@ref)), between the belief `q` used to decide and the true
  belief `p`.
- `B = sup u − inf u` is the **range** of the utility: the largest payoff in the
  table minus the smallest (not the largest absolute payoff).

Then for every action `|E_p u(a) − E_q u(a)| ≤ B ε`, and so the action that is best
under `q` falls short of the best under `p`, in expectation under `p`, by at most
`2 B ε`. With `ε` an ℓ¹ distance the bound would read `B ε`; with `B` a bound on
`|u|` it would read `4 B ε`.

The bound also holds with the sharper constant `B = max_a (max_θ Q[a, θ] − min_θ
Q[a, θ])`, the largest range of **one action's** payoffs across states, which is
never more than the range of the table (proved in the Lean development):
[`largest_action_range`](@ref) computes it, and `decision_loss_bound(x, ε)` on a
payoff table or a problem uses it.

```jldoctest
julia> decision_loss_bound(150, 1 // 100)
3//1
```
"""
function decision_loss_bound(B::Real, ε::Real)
    B >= 0 || throw(ArgumentError("the utility range B must be nonnegative, got $B"))
    ε >= 0 || throw(ArgumentError("the distance ε must be nonnegative, got $ε"))
    return 2 * B * ε
end

_constant_matrix(t::PayoffTable) = t.Q
_constant_matrix(p::DecisionProblem) = _feasible_Q(p)

"""
    payoff_range(x)

The range of the payoffs of a [`PayoffTable`](@ref) or of the feasible actions of a
[`DecisionProblem`](@ref): the largest payoff minus the smallest. It is the constant
`B` in the plain statements of [`decision_loss_bound`](@ref) and of the value bound of
[`deficiency`](@ref); [`largest_action_range`](@ref) and [`largest_regret`](@ref) are
the sharper constants, never larger than it.

```jldoctest
julia> payoff_range(PayoffTable([0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]))
150
```
"""
function payoff_range(x::Union{PayoffTable,DecisionProblem})
    lo, hi = extrema(_constant_matrix(x))
    return hi - lo
end

"""
    largest_action_range(x)

The largest range of **one action's** payoffs across the states,
`max_a (max_θ Q[a, θ] − min_θ Q[a, θ])`, of a [`PayoffTable`](@ref) or of the feasible
actions of a [`DecisionProblem`](@ref). It is the sharper constant for
[`decision_loss_bound`](@ref): the expected utility of an action moves by at most its
own range times the total-variation distance between two beliefs, whatever the other
actions pay (proved in the Lean development). It is never more than
[`payoff_range`](@ref).

```jldoctest
julia> largest_action_range(PayoffTable([0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]))
150

julia> largest_action_range(PayoffTable([0 10; 100 105]))    # the range of the table is 105
10
```
"""
function largest_action_range(x::Union{PayoffTable,DecisionProblem})
    Q = _constant_matrix(x)
    return maximum(maximum(view(Q, a, :)) - minimum(view(Q, a, :)) for a in axes(Q, 1))
end

"""
    largest_regret(x)

The largest entry of the [`regret_table`](@ref) of a [`PayoffTable`](@ref) or of the
feasible actions of a [`DecisionProblem`](@ref): the most that any action falls short
of the best action in any state. It is the sharper constant for the value bound of
[`deficiency`](@ref) (see [`deficiency_value_bound`](@ref)), and is never more than
[`payoff_range`](@ref) (proved in the Lean development). Like every regret it depends
on the menu of actions.

```jldoctest
julia> largest_regret(PayoffTable([0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]))
102
```
"""
function largest_regret(x::Union{PayoffTable,DecisionProblem})
    return maximum(_regret_matrix(_constant_matrix(x)))
end

"""
    decision_loss_bound(x, ε)

The bound of [`decision_loss_bound`](@ref)`(B, ε)` with the sharper constant read
from a [`PayoffTable`](@ref) or a [`DecisionProblem`](@ref):
`2 · largest_action_range(x) · ε` ([`largest_action_range`](@ref)).

```jldoctest
julia> decision_loss_bound(PayoffTable([0 10; 100 105]), 1 // 10)
2//1
```
"""
function decision_loss_bound(x::Union{PayoffTable,DecisionProblem}, ε::Real)
    return decision_loss_bound(largest_action_range(x), ε)
end

"""
    deficiency_value_bound(x, δ)

The bound `largest_regret(x) · δ` on how much more an experiment `e′` can be worth
than an experiment `e` in the decision problem `x` (a [`PayoffTable`](@ref) or a
[`DecisionProblem`](@ref)), when `δ = deficiency(e, e′)` is Le Cam's deficiency of `e`
with respect to `e′` ([`deficiency`](@ref)):

    value(e′) − value(e) ≤ largest_regret(x) · δ(e, e′)     for every prior.

It uses the sharper constant [`largest_regret`](@ref) in place of the range of the
table (proved in the Lean development). With `δ = 0`, `e′` is a garbling of `e` and
the bound is Blackwell's theorem.

```jldoctest
julia> e = Experiment([4//5 1//5; 1//5 4//5]);

julia> noisy = garble(e, [3//4 1//4; 1//4 3//4]);

julia> deficiency_value_bound(PayoffTable([0 -100; -30 -30]), deficiency(noisy, e))
21//2
```
"""
function deficiency_value_bound(x::Union{PayoffTable,DecisionProblem}, δ::Real)
    δ >= 0 || throw(ArgumentError("the deficiency δ must be nonnegative, got $δ"))
    return largest_regret(x) * δ
end
