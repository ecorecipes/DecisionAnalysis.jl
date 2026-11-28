# Aggregators collapse the uncertainty in one row of a payoff table to one number.
#
# Internally an aggregator acts on a matrix `M` (rows = actions, columns = the states
# the belief ranges over) and on `priors`, the list of prior vectors over those columns
# (`nothing` for a scenario set). Everything is written for utilities (higher is
# better); a loss table such as regret is aggregated as `-aggregate(-M)`, which turns
# "worst" into "largest" without a second implementation.

"""
    AbstractAggregator

Supertype of the aggregators: [`Expectation`](@ref), [`WorstCase`](@ref),
[`BestCase`](@ref), [`Hurwicz`](@ref), [`Laplace`](@ref),
[`LowerExpectation`](@ref), [`UpperExpectation`](@ref), [`AlphaMaxmin`](@ref),
[`CVaR`](@ref) and [`Quantile`](@ref). An aggregator turns each action's payoffs
across states into one score; see [`aggregate`](@ref).

They fall into three families. State-wise aggregators (`WorstCase`, `BestCase`,
`Hurwicz`, `Laplace`) ignore probabilities and range over the states the belief
allows: the states of a [`ScenarioSet`](@ref), the one state of a
[`PointBelief`](@ref), and **every** state of the table for a [`FiniteBelief`](@ref)
or a [`CredalSet`](@ref), including states of probability zero, and every draw of a
[`SampleBelief`](@ref) (so their value depends on which draws were sampled; `Laplace`
is not defined for a sample). Single-prior aggregators (`Expectation`, `CVaR`,
`Quantile`) need a `FiniteBelief`, a `SampleBelief` or a `PointBelief`. Prior-set aggregators (`LowerExpectation`, `UpperExpectation`,
`AlphaMaxmin`) range over the points of a `CredalSet` and reduce to the expectation
for a single prior. An aggregator applied to a form it is not defined for throws
[`BeliefFormError`](@ref).
"""
abstract type AbstractAggregator end

"""
    Expectation()

Expected payoff under a single prior (a [`FiniteBelief`](@ref), a
[`SampleBelief`](@ref) or a [`PointBelief`](@ref)): `Σ_θ p(θ) Q[a, θ]`.
"""
struct Expectation <: AbstractAggregator end

"""
    WorstCase()

The smallest payoff over the states the belief allows (Wald). Applied to a regret
table it is the largest regret.
"""
struct WorstCase <: AbstractAggregator end

"""
    BestCase()

The largest payoff over the states the belief allows.
"""
struct BestCase <: AbstractAggregator end

function _unit_interval(α::Real, what::AbstractString)
    0 <= α <= 1 || throw(ArgumentError("$what must lie in [0, 1], got $α"))
    return α
end
function _half_open_interval(α::Real, what::AbstractString)
    0 < α <= 1 || throw(ArgumentError("$what must lie in (0, 1], got $α"))
    return α
end

"""
    Hurwicz(α)

The Hurwicz criterion **over states**: `α · best + (1 − α) · worst`, where `α` in
`[0, 1]` is the index of optimism. `Hurwicz(0)` is [`WorstCase`](@ref) and
`Hurwicz(1)` is [`BestCase`](@ref). Probabilities are ignored.

This is not [`AlphaMaxmin`](@ref), which mixes the lower and upper **expectations
over a set of priors** and, following its own literature, puts `α` on the lower one.

**Which way `α` runs differs between texts.** Here, as in Hurwicz's optimism index, it
weights the best outcome. Several textbooks and software packages call `α` the
coefficient of pessimism and put it on the worst outcome; a value from such a source
is `Hurwicz(1 − α)` here. The independent Python reference of `validation/` uses the
convention of this package and says so in its fixture.
"""
struct Hurwicz{T<:Real} <: AbstractAggregator
    α::T
    Hurwicz{T}(α::T) where {T<:Real} = new{T}(_unit_interval(α, "the optimism index α"))
end
Hurwicz(α::T) where {T<:Real} = Hurwicz{T}(α)

"""
    Laplace()

The unweighted mean payoff over the states the belief allows (the principle of
insufficient reason). Probabilities are ignored.
"""
struct Laplace <: AbstractAggregator end

"""
    LowerExpectation()

The smallest expected payoff over a [`CredalSet`](@ref), `min_p Σ_θ p(θ) Q[a, θ]`.
The expectation is linear in the prior, so the minimum over the set is attained at
one of its generating points. Maximising it is [`GammaMaximin`](@ref).
"""
struct LowerExpectation <: AbstractAggregator end

"""
    UpperExpectation()

The largest expected payoff over a [`CredalSet`](@ref), attained at one of its
generating points.
"""
struct UpperExpectation <: AbstractAggregator end

"""
    AlphaMaxmin(α)

The α-maxmin criterion **over priors**: `α · lower + (1 − α) · upper` expectation
over a [`CredalSet`](@ref), where `α` in `[0, 1]` is the weight on the lower
expectation (the index of ambiguity aversion). `AlphaMaxmin(1)` is
[`LowerExpectation`](@ref) and `AlphaMaxmin(0)` is [`UpperExpectation`](@ref).

This is not [`Hurwicz`](@ref), which mixes the best and worst payoffs **over states**
and puts `α` on the best one. The two share a name in parts of the literature.
"""
struct AlphaMaxmin{T<:Real} <: AbstractAggregator
    α::T
    function AlphaMaxmin{T}(α::T) where {T<:Real}
        return new{T}(_unit_interval(α, "the ambiguity-aversion index α"))
    end
end
AlphaMaxmin(α::T) where {T<:Real} = AlphaMaxmin{T}(α)

"""
    CVaR(α)

Conditional value at risk of the payoff at level `α` in `(0, 1]`: the expected payoff
in the worst `α` share of the probability mass,
`(1/α) ∫₀^α q(u) du` with `q` the quantile function of the payoff under the prior.
`CVaR(1)` is the [`Expectation`](@ref) and the limit `α → 0` is the worst case over
the support. Needs a single prior. Applied to a regret table it is the expected regret
in the worst `α` share.

**Conventions differ.** This is the tail mean of a **utility** at the tail
probability `α`, so a larger value is better. Software for financial risk works with
losses and reports a positive number for a bad tail: the R package `cvar` gives
`ES(q, p_loss = α) = −CVaR(α)` and `VaR = −`[`Quantile`](@ref)`(α)`, and some texts
index the level by the confidence `1 − α`. It equals the optimum of Rockafellar and
Uryasev's programme `max_t t − E[(t − X)⁺] / α`. Both are checked in
`test/test_validation.jl` against `cvar` and against scipy's `linprog`.
"""
struct CVaR{T<:Real} <: AbstractAggregator
    α::T
    CVaR{T}(α::T) where {T<:Real} = new{T}(_half_open_interval(α, "the tail level α"))
end
CVaR(α::T) where {T<:Real} = CVaR{T}(α)

"""
    Quantile(α)

The lower `α`-quantile of the payoff under the prior, for `α` in `(0, 1]`: the
smallest payoff `x` with `P(payoff ≤ x) ≥ α`. `Quantile(1)` is the best payoff on the
support. Needs a single prior. Applied to a regret table the bad tail is the upper
one, so the result is the largest regret `x` with `P(regret ≥ x) ≥ α`.
"""
struct Quantile{T<:Real} <: AbstractAggregator
    α::T
    function Quantile{T}(α::T) where {T<:Real}
        return new{T}(_half_open_interval(α, "the level α"))
    end
end
Quantile(α::T) where {T<:Real} = Quantile{T}(α)

"""
    SmoothAmbiguity(ambiguity_aversion::Real; weights=nothing)
    SmoothAmbiguity(phi::Function; inverse=identity, weights=nothing)

The **smooth ambiguity** aggregator of Klibanoff, Marinacci and Mukerji (2005,
*Econometrica* 73:1849) over a [`CredalSet`](@ref) (or a single prior, where it reduces
to [`Expectation`](@ref)):

    S(a) = φ⁻¹(Σ_k μ_k φ(E_{p_k}[Q_a])),

where `p₁, …, p_K` are the generating priors of the credal set and `μ` are second-order
weights (uniform `1 / K` when `weights = nothing`).

- With a real parameter `η = ambiguity_aversion`, uses the constant absolute ambiguity
  aversion (exponential) form `φ(v) = −exp(−η v) / η`, returning the certainty-equivalent
  score `−(1 / η) log Σ_k μ_k exp(−η E_{p_k}[Q_a])` via shift-invariant log-sum-exp (`η > 0`
  is ambiguity-averse, `η = 0` is ambiguity-neutral, `η < 0` is ambiguity-seeking). As
  `η → +∞` it approaches [`LowerExpectation`](@ref).
- With a custom strictly increasing function `phi` and optional `inverse`, evaluates
  `inverse(Σ_k μ_k phi(E_{p_k}[Q_a]))`.
"""
struct SmoothAmbiguity{F,I,W} <: AbstractAggregator
    phi::F
    inverse::I
    weights::W
end
function SmoothAmbiguity(η::Real; weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    isfinite(η) ||
        throw(ArgumentError("ambiguity_aversion must be finite, got $η"))
    w = weights === nothing ? nothing : _normalise_weights(collect(weights))
    return SmoothAmbiguity(Float64(η), nothing, w)
end
function SmoothAmbiguity(phi::Function; inverse::Function=identity,
                         weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    w = weights === nothing ? nothing : _normalise_weights(collect(weights))
    return SmoothAmbiguity(phi, inverse, w)
end

# Internals
# ---------

_rows(M::AbstractMatrix) = (view(M, i, :) for i in axes(M, 1))

function _single_prior(agg::AbstractAggregator, v::_View)
    v.priors === nothing &&
        throw(BeliefFormError(nameof(typeof(agg)), v.form,
                              "a scenario set carries no probabilities; use a " *
                              "state-wise aggregator or supply a FiniteBelief"))
    length(v.priors) == 1 ||
        throw(BeliefFormError(nameof(typeof(agg)), v.form,
                              "it needs a single prior; use LowerExpectation, " *
                              "UpperExpectation or AlphaMaxmin over the set"))
    return only(v.priors)
end

function _prior_set(agg::AbstractAggregator, v::_View)
    v.priors === nothing &&
        throw(BeliefFormError(nameof(typeof(agg)), v.form,
                              "a scenario set carries no probabilities; use a " *
                              "state-wise aggregator or supply a CredalSet"))
    return v.priors
end

# Expected payoffs: rows = actions, columns = priors.
_expectations(M::AbstractMatrix, priors) = reduce(hcat, [M * q for q in priors])

_aggregate(agg::Expectation, M, v::_View) = M * _single_prior(agg, v)
_aggregate(::WorstCase, M, ::_View) = [minimum(r) for r in _rows(M)]
_aggregate(::BestCase, M, ::_View) = [maximum(r) for r in _rows(M)]
function _aggregate(agg::Hurwicz, M, ::_View)
    return [agg.α * maximum(r) + (1 - agg.α) * minimum(r) for r in _rows(M)]
end
function _aggregate(agg::Laplace, M, v::_View)
    v.form === :SampleBelief &&
        throw(BeliefFormError(:Laplace, v.form,
                              "an unweighted mean over the draws ignores their weights; " *
                              "use Expectation"))
    return [_mean(r) for r in _rows(M)]
end
# Integer payoffs keep an exact mean.
_mean(r::AbstractVector{<:Integer}) = sum(r) // length(r)
_mean(r::AbstractVector) = mean(r)
function _aggregate(agg::LowerExpectation, M, v::_View)
    return vec(minimum(_expectations(M, _prior_set(agg, v)); dims=2))
end
function _aggregate(agg::UpperExpectation, M, v::_View)
    return vec(maximum(_expectations(M, _prior_set(agg, v)); dims=2))
end
function _aggregate(agg::AlphaMaxmin, M, v::_View)
    E = _expectations(M, _prior_set(agg, v))
    return vec(agg.α .* minimum(E; dims=2) .+ (1 - agg.α) .* maximum(E; dims=2))
end

function _cvar(x::AbstractVector, p::AbstractVector, α::Real)
    order = sortperm(x)
    remaining = α * one(eltype(p))
    total = zero(promote_type(eltype(x), eltype(p), typeof(α)))
    for i in order
        mass = min(p[i], remaining)
        total += mass * x[i]
        remaining -= mass
        remaining > 0 || break
    end
    # In floating point the masses can fall short of α by rounding; divide by what was
    # actually accumulated.
    return _div(total, α - max(remaining, zero(remaining)))
end

function _quantile(x::AbstractVector, p::AbstractVector, α::Real)
    order = sortperm(x)
    cumulative = zero(eltype(p))
    for i in order
        cumulative += p[i]
        cumulative >= α && return x[i]
    end
    # Rounding left the cumulative mass just below α: the answer is the largest payoff
    # with positive probability.
    return x[order[findlast(i -> p[i] > 0, order)]]
end

function _aggregate(agg::CVaR, M, v::_View)
    q = _single_prior(agg, v)
    return [_cvar(r, q, agg.α) for r in _rows(M)]
end
function _aggregate(agg::Quantile, M, v::_View)
    q = _single_prior(agg, v)
    return [_quantile(r, q, agg.α) for r in _rows(M)]
end

function _aggregate(agg::SmoothAmbiguity, M, v::_View)
    priors = _prior_set(agg, v)
    E = _expectations(M, priors)
    na, K = size(E)
    μ = if agg.weights === nothing
        fill(1.0 / K, K)
    else
        length(agg.weights) == K ||
            throw(ArgumentError("SmoothAmbiguity received $(length(agg.weights)) " *
                                "second-order weights for $K generating priors"))
        Float64.(agg.weights)
    end
    if agg.phi isa Float64
        η = agg.phi
        if iszero(η)
            return [sum(μ[k] * Float64(E[a, k]) for k in 1:K) for a in 1:na]
        end
        return map(1:na) do a
            row = Float64[Float64(E[a, k]) for k in 1:K]
            α = -η
            xstar = α < 0 ? minimum(row[k] for k in 1:K if μ[k] > 0) :
                    maximum(row[k] for k in 1:K if μ[k] > 0)
            total = sum(μ[k] * exp(α * (row[k] - xstar)) for k in 1:K if μ[k] > 0)
            return xstar + log(total) / α
        end
    end
    return [agg.inverse(sum(μ[k] * agg.phi(E[a, k]) for k in 1:K)) for a in 1:na]
end

# `sense` is :max for a utility table and :min for a loss table.
function _scores(agg::AbstractAggregator, M, v::_View, sense::Symbol)
    return sense === :max ? _aggregate(agg, M, v) : -_aggregate(agg, -M, v)
end

"""
    aggregate(aggregator, problem::DecisionProblem) -> Vector

The score of every feasible action of `problem` under `aggregator`, in
[`feasible_actions`](@ref) order. Payoffs are utilities, so a higher score is better.

```jldoctest
julia> t = PayoffTable([:wait, :act], [:low, :high], [0 -100; -30 -30]);

julia> p = DecisionProblem(t, [3 // 5, 2 // 5]);

julia> aggregate(Expectation(), p)
2-element Vector{Rational{Int64}}:
 -40
 -30

julia> aggregate(WorstCase(), p)
2-element Vector{Int64}:
 -100
  -30
```
"""
function aggregate(agg::AbstractAggregator, p::DecisionProblem)
    v = _view(p)
    return _aggregate(agg, v.Q, v)
end
