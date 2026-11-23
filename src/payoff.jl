# The payoff table: the seam between model packages and the decision layer, and between
# Julia and Lean.

"""
    PayoffTable(actions, states, Q; standard_errors=nothing)
    PayoffTable(Q; actions, states, standard_errors=nothing)

The payoff table of a decision problem: `Q[a, θ]` is the **conditional expected
utility** of action `a` in state `θ`, with rows indexed by `actions` and columns by
`states` (symbols or strings; defaults `:a1, :a2, ...` and `:s1, :s2, ...`). Higher is
better; costs enter as negative payoffs.

The element type is generic, so a table of `Rational{BigInt}` is analysed exactly.
`standard_errors`, when given, is a matrix of the same size holding the Monte Carlo
standard error of each cell (a stochastic simulator must be averaged inside each
cell); it is carried along and never folded into a payoff.

Throws [`InvalidPayoffError`](@ref) for an empty table, a non-finite entry, or sizes
that do not match the labels. The matrix is copied.

```jldoctest
julia> t = PayoffTable([:wait, :act], [:low, :high], [0 -100; -30 -30]);

julia> size(t)
(2, 2)
```
"""
struct PayoffTable{T<:Real,S<:Union{Nothing,Matrix{<:Real}}}
    actions::Vector{Symbol}
    states::Vector{Symbol}
    Q::Matrix{T}
    standard_errors::S
    function PayoffTable{T,S}(actions::Vector{Symbol}, states::Vector{Symbol},
                              Q::Matrix{T},
                              standard_errors::S) where {T<:Real,
                                                         S<:Union{Nothing,Matrix{<:Real}}}
        (isempty(actions) || isempty(states)) &&
            throw(InvalidPayoffError("a payoff table needs at least one action and one state"))
        size(Q) == (length(actions), length(states)) ||
            throw(InvalidPayoffError("the payoff matrix has size $(size(Q)) for " *
                                     "$(length(actions)) actions (rows) and " *
                                     "$(length(states)) states (columns)"))
        allunique(actions) ||
            throw(InvalidPayoffError("action labels must be distinct, got $actions"))
        allunique(states) ||
            throw(InvalidPayoffError("state labels must be distinct, got $states"))
        i = findfirst(!isfinite, Q)
        i === nothing ||
            throw(InvalidPayoffError("the payoff of action $(actions[i[1]]) in state " *
                                     "$(states[i[2]]) is $(Q[i]), which is not finite"))
        if standard_errors !== nothing
            size(standard_errors) == size(Q) ||
                throw(InvalidPayoffError("standard errors have size " *
                                         "$(size(standard_errors)), the payoff matrix " *
                                         "$(size(Q))"))
            all(x -> isfinite(x) && x >= 0, standard_errors) ||
                throw(InvalidPayoffError("standard errors must be finite and nonnegative"))
        end
        return new{T,S}(actions, states, Q, standard_errors)
    end
end

function PayoffTable(actions, states, Q::AbstractMatrix{T};
                     standard_errors::Union{Nothing,AbstractMatrix{<:Real}}=nothing) where {T<:Real}
    se = standard_errors === nothing ? nothing : Matrix(standard_errors)
    return PayoffTable{T,typeof(se)}(_labels(actions), _labels(states), Matrix{T}(Q), se)
end

function PayoffTable(Q::AbstractMatrix{<:Real};
                     actions=_default_labels("a", size(Q, 1)),
                     states=_default_labels("s", size(Q, 2)), standard_errors=nothing)
    return PayoffTable(actions, states, Q; standard_errors)
end

Base.size(t::PayoffTable) = size(t.Q)
Base.size(t::PayoffTable, d::Integer) = size(t.Q, d)
Base.eltype(::Type{<:PayoffTable{T}}) where {T} = T
Base.eltype(::PayoffTable{T}) where {T} = T

function Base.:(==)(a::PayoffTable, b::PayoffTable)
    return a.actions == b.actions && a.states == b.states && a.Q == b.Q &&
           a.standard_errors == b.standard_errors
end
function Base.hash(t::PayoffTable, h::UInt)
    return hash(t.standard_errors, hash(t.Q, hash(t.states, hash(t.actions, hash(:PT, h)))))
end

state_labels(t::PayoffTable) = t.states

"""
    action_labels(x) -> Vector{Symbol}

The action labels of a [`PayoffTable`](@ref), or of a [`DecisionProblem`](@ref) (all
of them; see [`feasible_actions`](@ref) for those that pass its filter), in row order.
"""
action_labels(t::PayoffTable) = t.actions

# The table restricted to some rows and columns, standard errors included.
function _subtable(t::PayoffTable, rows, cols)
    se = t.standard_errors === nothing ? nothing : t.standard_errors[rows, cols]
    return PayoffTable(t.actions[rows], t.states[cols], t.Q[rows, cols];
                       standard_errors=se)
end

"""
    random_payoff_table(rng, m, n; T=Float64, actions, states) -> PayoffTable

A random table of `m` actions by `n` states, for tests and examples. With a `Rational`
element type `T` the entries are integers from `-20:20` (so ties do occur and
arithmetic is exact); otherwise they are standard normal draws.
"""
function random_payoff_table(rng::AbstractRNG, m::Integer, n::Integer;
                             T::Type{<:Real}=Float64, actions=_default_labels("a", m),
                             states=_default_labels("s", n))
    Q = T <: Rational ? T.(rand(rng, -20:20, m, n)) : randn(rng, T, m, n)
    return PayoffTable(actions, states, Q)
end
