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
    length(p) == length(q) ||
        throw(ArgumentError("the vectors have lengths $(length(p)) and $(length(q))"))
    return sum(abs(a - b) for (a, b) in zip(p, q)) / 2
end

function total_variation(p::FiniteBelief, q::FiniteBelief)
    p.states == q.states ||
        throw(LabelMismatchError("states of two beliefs", p.states, q.states))
    return total_variation(p.probabilities, q.probabilities)
end

function total_variation(p::SampleBelief, ::SampleBelief)
    throw(BeliefFormError(:total_variation, :SampleBelief,
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
never more than the range of the table (proved in the Lean development).

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
