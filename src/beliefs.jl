# The belief forms (four finite ones and the sample), and the small helpers on labels and
# probability vectors that every other file uses.

# Labels
# ------

_labels(v) = Symbol[Symbol(x) for x in v]
_default_labels(prefix::AbstractString, n::Integer) = [Symbol(prefix, i) for i in 1:n]

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
[`SampleBelief`](@ref), [`CredalSet`](@ref) and [`ScenarioSet`](@ref). A belief says what is known about the
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

The third form is available when FiniteKernels is loaded (`using FiniteKernels`); it
takes a state of that package (a `FiniteKernel` with an empty domain). A one-axis space keeps its labels; the joint states of a several-axis space
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
    probabilities(b::SampleBelief) -> Vector

The probability vector of a finite belief, in [`state_labels`](@ref) order; for a
[`SampleBelief`](@ref), the normalised weights of the draws.
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

# Sample beliefs
# --------------

# Weights as a probability vector: exact for integer and rational weights.
function _normalise_weights(w::AbstractVector{<:Real})
    isempty(w) && throw(InvalidBeliefError("a sample needs at least one draw"))
    for (i, v) in enumerate(w)
        isfinite(v) || throw(InvalidBeliefError("the weight of draw $i is $v, which is " *
                                                "not finite"))
        v >= 0 ||
            throw(InvalidBeliefError("the weight of draw $i is $v, which is negative"))
    end
    total = sum(w)
    total > 0 || throw(InvalidBeliefError("the weights of a sample sum to zero"))
    return _is_exact(eltype(w)) ? [v // total for v in w] : float.(w) ./ float(total)
end

# A Tables.jl table as a named tuple of plain vectors.
function _column_table(table)
    table isa NamedTuple && isempty(table) && return (;)
    Tables.istable(table) ||
        throw(ArgumentError("expected a Tables.jl table (for example a named tuple of " *
                            "vectors or a vector of named tuples), got $(typeof(table))"))
    columns = Tables.columntable(table)
    return map(collect, columns)
end

# The column `name` of a table of draws and the table without it.
function _take_column(columns::NamedTuple, name::Symbol, what::AbstractString)
    haskey(columns, name) ||
        throw(ArgumentError("the draws have no column $name to use as $what; " *
                            "columns: $(collect(keys(columns)))"))
    return columns[name], Base.structdiff(columns, NamedTuple{(name,)})
end

"""
    SampleBelief(draws; weights=nothing, labels=nothing)
    SampleBelief(n::Integer; weights=nothing, labels=nothing)

A belief held as a sample: `draws` is a Tables.jl table with one row per draw and
one column per **named parameter** (a named tuple of vectors, a vector of named
tuples, a data frame), and `weights` gives each draw's weight. The states of the
belief are the draws, so a [`PayoffTable`](@ref) used with it has one column per draw,
in row order. The second form is a sample of `n` draws with no recorded parameters.

- `weights`: `nothing` (equal weights, the exact `1 // n`), a vector of nonnegative
  numbers, or the name of a column of `draws` that holds them (the column is then
  taken out of the parameters). Weights are normalised to sum to one: exactly for
  integer or rational weights, in floating point otherwise. A negative or
  non-finite weight, or weights that sum to zero, throw [`InvalidBeliefError`](@ref).
- `labels`: the state labels of the draws: `nothing` (`:s1, :s2, ...`, the default
  state labels of a payoff table), a vector, or
  the name of a column of `draws` that identifies them (taken out of the parameters).

**What a sample supports.** A sample is a single prior, the weighted empirical
distribution of the draws, so [`Expectation`](@ref), [`CVaR`](@ref),
[`Quantile`](@ref), regret, [`evpi`](@ref) and [`evppi`](@ref) by partition apply to it
as they do to a [`FiniteBelief`](@ref), and they are **estimates** of the quantities
for the distribution the draws came from. [`WorstCase`](@ref), [`BestCase`](@ref) and
[`Hurwicz`](@ref) range over the draws that happen to be in the sample; the results
of rules built on them say so (`details.sample_dependent`), because a larger sample
can only make the worst case worse. Operations that have no meaning for a sample
throw [`BeliefFormError`](@ref): [`Laplace`](@ref) (an unweighted mean over draws
ignores the weights; use the expectation), [`total_variation`](@ref),
[`bayes_invert`](@ref) and the certificate exports (a certificate states exact
numbers for a finite problem; convert with `FiniteBelief(sample)` if the draws are
the whole population and not a sample of it).

The estimators with standard errors are [`evpi`](@ref), [`evppi`](@ref) and
[`evsi`](@ref) called with a method ([`SampleAverage`](@ref),
[`SplineRegression`](@ref), ...). See also [`parameter_draws`](@ref),
[`draw_weights`](@ref) and [`effective_sample_size`](@ref).

```jldoctest
julia> b = SampleBelief((; growth=[0.1, 0.3, 0.2], weight=[1, 1, 2]); weights=:weight);

julia> probabilities(b)
3-element Vector{Rational{Int64}}:
 1//4
 1//4
 1//2

julia> effective_sample_size(b)
8//3
```
"""
struct SampleBelief{T<:Real,C<:NamedTuple} <: AbstractBelief
    states::Vector{Symbol}
    draws::C
    weights::Vector{T}
    function SampleBelief(states::Vector{Symbol}, draws::C,
                          weights::Vector{T}) where {T<:Real,C<:NamedTuple}
        n = length(weights)
        length(states) == n ||
            throw(InvalidBeliefError("$(length(states)) labels for $n draws"))
        allunique(states) ||
            throw(InvalidBeliefError("the labels of the draws must be distinct"))
        for (name, column) in pairs(draws)
            length(column) == n ||
                throw(InvalidBeliefError("parameter $name has $(length(column)) values " *
                                         "for $n draws"))
        end
        defect = _probability_defect(weights, _default_atol(T))
        defect === nothing ||
            throw(InvalidBeliefError("the weight vector of the sample $defect"))
        return new{T,C}(states, draws, weights)
    end
end

function SampleBelief(draws; weights=nothing, labels=nothing)
    all_columns = _column_table(draws)
    w, without_weights = weights isa Symbol ?
                         _take_column(all_columns, weights, "weights") :
                         (weights, all_columns)
    l, columns = labels isa Symbol ? _take_column(without_weights, labels, "labels") :
                 (labels, without_weights)
    n = w !== nothing ? length(w) :
        l !== nothing ? length(l) :
        isempty(columns) ?
        throw(ArgumentError("draws with no column: give the number of draws, " *
                            "SampleBelief(n)")) : length(first(columns))
    return _sample_belief(columns, n, w, l)
end

function SampleBelief(n::Integer; weights=nothing, labels=nothing)
    n >= 1 || throw(InvalidBeliefError("a sample needs at least one draw"))
    return _sample_belief((;), n, weights, labels)
end

function _sample_belief(columns::NamedTuple, n::Integer, weights, labels)
    w = weights === nothing ? fill(1 // n, n) : _normalise_weights(collect(weights))
    states = labels === nothing ? _default_labels("s", n) : _labels(labels)
    return SampleBelief(states, columns, w)
end

"""
    FiniteBelief(b::SampleBelief)

The weighted empirical distribution of a sample as a finite belief over its draws.
Use it when the draws are the whole population of states (so that nothing is being
estimated), for example to export a certificate.
"""
FiniteBelief(b::SampleBelief) = FiniteBelief(b.states, b.weights)

state_labels(b::SampleBelief) = b.states
probabilities(b::SampleBelief) = b.weights
Base.length(b::SampleBelief) = length(b.weights)

function Base.:(==)(a::SampleBelief, b::SampleBelief)
    return a.states == b.states && a.weights == b.weights && isequal(a.draws, b.draws)
end
function Base.hash(b::SampleBelief, h::UInt)
    return hash(b.draws, hash(b.weights, hash(b.states, hash(:SB, h))))
end

"""
    effective_sample_size(b::SampleBelief)
    effective_sample_size(weights::AbstractVector)

Kish's effective sample size of a weighted sample, `(Σ w)² / Σ w²`: the number of
equally weighted draws that would give a mean the same variance. It equals the
number of draws for equal weights and falls towards one as the weights concentrate.
"""
effective_sample_size(b::SampleBelief) = effective_sample_size(b.weights)
function effective_sample_size(weights::AbstractVector{<:Real})
    isempty(weights) && throw(ArgumentError("no weights"))
    return sum(weights)^2 / sum(abs2, weights)
end
