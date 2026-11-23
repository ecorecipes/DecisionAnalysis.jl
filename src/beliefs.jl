# The four finite belief forms of version 0.1, and the small helpers on labels and
# probability vectors that every other file uses.

# Labels
# ------

_labels(v) = Symbol[Symbol(x) for x in v]
_default_labels(prefix::AbstractString, n::Integer) = [Symbol(prefix, i) for i in 1:n]

function _unique_labels(v, what::AbstractString)
    out = _labels(v)
    allunique(out) || throw(ArgumentError("$what labels must be distinct, got $out"))
    return out
end

# Arithmetic
# ----------

_is_exact(::Type{<:Integer}) = true
_is_exact(::Type{<:Rational}) = true
_is_exact(::Type) = false

# Tolerance on "sums to one" and "is nonnegative": none in exact arithmetic.
_default_atol(::Type{T}) where {T} = _is_exact(T) ? 0 : 1e-8
# Relative tolerance on ties between scores: none in exact arithmetic.
_default_rtol(::Type{T}) where {T<:AbstractFloat} = sqrt(eps(T))
_default_rtol(::Type) = 0

# `nothing` if `p` is a probability vector within `atol`, otherwise the reason.
function _probability_defect(p::AbstractVector, atol::Real)
    isempty(p) && return "is empty"
    for (i, v) in enumerate(p)
        isfinite(v) || return "has a non-finite entry at index $i"
        v < -atol && return "has the negative entry $v at index $i"
    end
    s = sum(p)
    abs(s - 1) <= atol || return "sums to $s, not one"
    return nothing
end

# Belief types
# ------------

"""
    AbstractBelief

Supertype of the belief forms: [`PointBelief`](@ref), [`FiniteBelief`](@ref),
[`CredalSet`](@ref) and [`ScenarioSet`](@ref). A belief says what is known about the
state; which rules and which value-of-information quantities are defined depends on
its form, and an operation that is not defined for a form throws
[`BeliefFormError`](@ref) instead of guessing.
"""
abstract type AbstractBelief end

"""
    PointBelief(state)

The state is known: all mass on the one labelled `state` (a symbol or a string).
Every aggregator returns the payoff in that state, and the value of further
information is zero.

```jldoctest
julia> PointBelief(:high).state
:high
```
"""
struct PointBelief <: AbstractBelief
    state::Symbol
    PointBelief(state::Union{Symbol,AbstractString}) = new(Symbol(state))
end

"""
    FiniteBelief(states, probabilities; atol)
    FiniteBelief(probabilities; atol)
    FiniteBelief(k::FiniteKernel)

A probability vector over labelled states. `probabilities` must be finite,
nonnegative and sum to one, exactly for `Rational` or integer entries and within
`atol` (default `1e-8`) for floating point; otherwise [`InvalidBeliefError`](@ref) is
thrown. Without `states` the labels are `:s1, :s2, ...`, which are also the default
state labels of a [`PayoffTable`](@ref).

The third form takes a state of FiniteKernels (a `FiniteKernel` with an empty
domain). A one-axis space keeps its labels; the joint states of a several-axis space
are labelled by joining the axis labels with `_`, in `joint_states` order.

```jldoctest
julia> b = FiniteBelief([:low, :high], [3 // 5, 2 // 5]);

julia> probabilities(b)
2-element Vector{Rational{Int64}}:
 3//5
 2//5
```
"""
struct FiniteBelief{T<:Real} <: AbstractBelief
    states::Vector{Symbol}
    probabilities::Vector{T}
    function FiniteBelief{T}(states::Vector{Symbol}, probabilities::Vector{T};
                             atol::Real=_default_atol(T)) where {T<:Real}
        allunique(states) ||
            throw(InvalidBeliefError("state labels must be distinct, got $states"))
        length(states) == length(probabilities) ||
            throw(InvalidBeliefError("$(length(states)) state labels for " *
                                     "$(length(probabilities)) probabilities"))
        defect = _probability_defect(probabilities, atol)
        defect === nothing ||
            throw(InvalidBeliefError("the probability vector $defect"))
        return new{T}(states, probabilities)
    end
end

function FiniteBelief(states, probabilities::AbstractVector{T}; kw...) where {T<:Real}
    return FiniteBelief{T}(_labels(states), collect(probabilities); kw...)
end

function FiniteBelief(probabilities::AbstractVector{<:Real}; kw...)
    return FiniteBelief(_default_labels("s", length(probabilities)), probabilities; kw...)
end

function FiniteBelief(k::FiniteKernel; kw...)
    ndims(k.dom) == 0 ||
        throw(ArgumentError("a FiniteBelief needs a state (a kernel with an empty " *
                            "domain), got a kernel with domain axes $(axis_names(k.dom))"))
    X = k.codom
    states = ndims(X) == 1 ? copy(only(factors(X)).labels) :
             [Symbol(join(s, "_")) for s in joint_states(X)]
    return FiniteBelief(states, vec(k.table); kw...)
end

"""
    CredalSet(states, points; atol)
    CredalSet(points; atol)
    CredalSet(beliefs::AbstractVector{<:FiniteBelief})

A closed convex set of probability vectors over labelled states, given by a finite
list of generating points; the set is their convex hull. The points are meant to be
the extreme points, but a redundant generator (one inside the hull of the others)
changes no result. Each point is checked like a [`FiniteBelief`](@ref).

Γ-maximin, maximality and interval dominance need only these points. E-admissibility
does not: an action can be optimal at an interior prior and at no generator, so
[`EAdmissible`](@ref) solves a linear programme over the hull.

```jldoctest
julia> c = CredalSet([:low, :high], [[1 // 2, 1 // 2], [1 // 4, 3 // 4]]);

julia> length(extreme_points(c))
2
```
"""
struct CredalSet{T<:Real} <: AbstractBelief
    states::Vector{Symbol}
    points::Vector{Vector{T}}
    function CredalSet{T}(states::Vector{Symbol}, points::Vector{Vector{T}};
                          atol::Real=_default_atol(T)) where {T<:Real}
        allunique(states) ||
            throw(InvalidBeliefError("state labels must be distinct, got $states"))
        isempty(points) &&
            throw(InvalidBeliefError("a credal set needs at least one point"))
        for (k, p) in enumerate(points)
            length(p) == length(states) ||
                throw(InvalidBeliefError("point $k of the credal set has length " *
                                         "$(length(p)) for $(length(states)) states"))
            defect = _probability_defect(p, atol)
            defect === nothing ||
                throw(InvalidBeliefError("point $k of the credal set $defect"))
        end
        return new{T}(states, points)
    end
end

function CredalSet(states, points::AbstractVector{<:AbstractVector{<:Real}}; kw...)
    isempty(points) && throw(InvalidBeliefError("a credal set needs at least one point"))
    T = mapreduce(eltype, promote_type, points)
    return CredalSet{T}(_labels(states), [collect(T, p) for p in points]; kw...)
end

function CredalSet(points::AbstractVector{<:AbstractVector{<:Real}}; kw...)
    isempty(points) && throw(InvalidBeliefError("a credal set needs at least one point"))
    return CredalSet(_default_labels("s", length(first(points))), points; kw...)
end

function CredalSet(beliefs::AbstractVector{<:FiniteBelief}; kw...)
    isempty(beliefs) && throw(InvalidBeliefError("a credal set needs at least one point"))
    states = first(beliefs).states
    for b in beliefs
        b.states == states ||
            throw(LabelMismatchError("states of the members of a credal set", states,
                                     b.states))
    end
    return CredalSet(states, [b.probabilities for b in beliefs]; kw...)
end

"""
    ScenarioSet(states)

A set of states held to be possible, with no probabilities attached. State-wise
aggregators ([`WorstCase`](@ref), [`BestCase`](@ref), [`Hurwicz`](@ref),
[`Laplace`](@ref)) and regret range over exactly these states; an expectation over a
scenario set throws [`BeliefFormError`](@ref).

```jldoctest
julia> ScenarioSet([:low, :high]).states
2-element Vector{Symbol}:
 :low
 :high
```
"""
struct ScenarioSet <: AbstractBelief
    states::Vector{Symbol}
    function ScenarioSet(states)
        labels = _labels(states)
        isempty(labels) &&
            throw(InvalidBeliefError("a scenario set needs at least one state"))
        allunique(labels) ||
            throw(InvalidBeliefError("state labels must be distinct, got $labels"))
        return new(labels)
    end
end

# Accessors
# ---------

"""
    state_labels(x) -> Vector{Symbol}

The state labels of a belief, a [`PayoffTable`](@ref), a [`DecisionProblem`](@ref) or
an [`Experiment`](@ref), in order. For an experiment built without state labels the
result is `nothing`.
"""
state_labels(b::PointBelief) = [b.state]
state_labels(b::FiniteBelief) = b.states
state_labels(b::CredalSet) = b.states
state_labels(b::ScenarioSet) = b.states

"""
    probabilities(b::FiniteBelief) -> Vector

The probability vector of a finite belief, in [`state_labels`](@ref) order.
"""
probabilities(b::FiniteBelief) = b.probabilities

"""
    extreme_points(c::CredalSet) -> Vector{Vector}

The generating points of a credal set, each a probability vector in
[`state_labels`](@ref) order.
"""
extreme_points(c::CredalSet) = c.points

function Base.:(==)(a::FiniteBelief, b::FiniteBelief)
    return a.states == b.states && a.probabilities == b.probabilities
end
Base.hash(b::FiniteBelief, h::UInt) = hash(b.probabilities, hash(b.states, hash(:FB, h)))
Base.:(==)(a::CredalSet, b::CredalSet) = a.states == b.states && a.points == b.points
Base.hash(b::CredalSet, h::UInt) = hash(b.points, hash(b.states, hash(:CS, h)))
Base.:(==)(a::ScenarioSet, b::ScenarioSet) = a.states == b.states
Base.hash(b::ScenarioSet, h::UInt) = hash(b.states, hash(:SS, h))

"""
    random_belief(rng, n; T=Float64, states) -> FiniteBelief

A random belief over `n` states with full support, for tests and examples. With a
`Rational` element type `T` the probabilities are exact (integer weights from `1:9`,
normalised); otherwise they are normalised uniform draws.
"""
function random_belief(rng::AbstractRNG, n::Integer; T::Type{<:Real}=Float64,
                       states=_default_labels("s", n))
    return FiniteBelief(states, _random_probabilities(rng, T, n))
end

function _random_probabilities(rng::AbstractRNG, ::Type{T}, n::Integer) where {T<:Real}
    if T <: Rational
        w = rand(rng, 1:9, n)
        return T[T(x, sum(w)) for x in w]
    end
    w = rand(rng, T, n) .+ T(0.05)
    return w ./ sum(w)
end

"""
    random_credal_set(rng, n, k; T=Float64, states) -> CredalSet

A credal set over `n` states generated by `k` random points, each drawn as in
[`random_belief`](@ref).
"""
function random_credal_set(rng::AbstractRNG, n::Integer, k::Integer;
                           T::Type{<:Real}=Float64, states=_default_labels("s", n))
    k >= 1 || throw(ArgumentError("a credal set needs at least one point, got k = $k"))
    return CredalSet(states, [_random_probabilities(rng, T, n) for _ in 1:k])
end
