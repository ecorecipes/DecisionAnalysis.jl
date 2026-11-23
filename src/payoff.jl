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

**Arithmetic.** Tables with the same action and state labels, in the same order, can
be added and subtracted (`t + u`, `t - u`, `-t`), and a table can be multiplied or
divided by a number (`c * t`, `t * c`, `t / c`); the result is a `PayoffTable` with
the same labels, and mismatched labels throw [`LabelMismatchError`](@ref). This is
how a utility is assembled from tables of outcome counts: see
[`combine_payoffs`](@ref), [`individuals_payoff`](@ref) and
[`infection_payoff`](@ref). Standard errors are not carried through a sum or a
difference (the error of a sum depends on how the cells were simulated); a multiple
keeps them, scaled.

**Display.** A table prints as a grid of actions against states; in a notebook or a
Quarto document it renders as a Markdown or HTML table.

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

# Arithmetic on tables with the same labels
# ----------------------------------------

function _check_same_labels(a::PayoffTable, b::PayoffTable)
    a.actions == b.actions ||
        throw(LabelMismatchError("actions of two payoff tables", a.actions, b.actions))
    a.states == b.states ||
        throw(LabelMismatchError("states of two payoff tables", a.states, b.states))
    return nothing
end

function Base.:+(a::PayoffTable, b::PayoffTable)
    _check_same_labels(a, b)
    return PayoffTable(a.actions, a.states, a.Q + b.Q)
end
function Base.:-(a::PayoffTable, b::PayoffTable)
    _check_same_labels(a, b)
    return PayoffTable(a.actions, a.states, a.Q - b.Q)
end
function Base.:-(a::PayoffTable)
    return PayoffTable(a.actions, a.states, -a.Q;
                       standard_errors=a.standard_errors)
end
function Base.:*(c::Real, a::PayoffTable)
    se = a.standard_errors === nothing ? nothing : abs(c) .* a.standard_errors
    return PayoffTable(a.actions, a.states, c .* a.Q; standard_errors=se)
end
Base.:*(a::PayoffTable, c::Real) = c * a
function Base.:/(a::PayoffTable, c::Real)
    iszero(c) && throw(ArgumentError("a payoff table cannot be divided by zero"))
    se = a.standard_errors === nothing ? nothing : a.standard_errors ./ abs(c)
    return PayoffTable(a.actions, a.states, a.Q ./ c; standard_errors=se)
end

"""
    combine_payoffs(terms::Pair{<:PayoffTable,<:Real}...) -> PayoffTable

The weighted sum of payoff tables with the same labels: `combine_payoffs(t1 => w1,
t2 => w2, ...)` is the table `w1 · t1 + w2 · t2 + ...`. It is the way to build a
utility from tables of outcome counts without assembling the matrix by hand; a
negative weight makes an outcome a cost. The element type follows the tables and
the weights, so rational tables with rational or integer weights stay exact.
Mismatched labels throw [`LabelMismatchError`](@ref).

```jldoctest
julia> uninfected = PayoffTable([:nothing, :treat], [:absent, :present], [500 100; 470 250]);

julia> infected = PayoffTable([:nothing, :treat], [:absent, :present], [0 250; 0 120]);

julia> combine_payoffs(uninfected => 1, infected => -1 // 2).Q
2×2 Matrix{Rational{Int64}}:
 500  -25
 470  190
```
"""
function combine_payoffs(terms::Pair{<:PayoffTable,<:Real}...)
    isempty(terms) && throw(ArgumentError("combine_payoffs needs at least one table"))
    return reduce(+, (w * t for (t, w) in terms))
end

# A named collection: a named tuple, a dictionary, or a collection of `name => item`
# pairs, each item of type `T`. `what` names the items in error messages.
function _named_items(items, ::Type{T}, what::AbstractString) where {T}
    out = Pair{Symbol,T}[]
    for (name, item) in pairs(items)
        if item isa Pair
            name, item = item
        end
        item isa T ||
            throw(ArgumentError("expected named $what (a named tuple, a dictionary or " *
                                "`name => item` pairs), got an entry of type " *
                                "$(typeof(item))"))
        push!(out, Symbol(name) => item)
    end
    allunique(first.(out)) ||
        throw(ArgumentError("the names of the $what must be distinct, got " *
                            "$(first.(out))"))
    return out
end

function _named_tables(tables)
    out = _named_items(tables, PayoffTable, "payoff tables")
    isempty(out) && throw(ArgumentError("no payoff table was given"))
    return out
end

function _named_table(tables, name)
    named = _named_tables(tables)
    i = findfirst(p -> first(p) == Symbol(name), named)
    i === nothing &&
        throw(ArgumentError("there is no outcome table named $name among " *
                            "$(first.(named))"))
    return last(named[i])
end

function _count_table(table::PayoffTable, what::AbstractString)
    i = findfirst(<(0), table.Q)
    i === nothing ||
        throw(InvalidPayoffError("the $what count of action $(table.actions[i[1]]) in " *
                                 "state $(table.states[i[2]]) is $(table.Q[i]); a count " *
                                 "cannot be negative"))
    return table
end

"""
    individuals_payoff(counts::PayoffTable) -> PayoffTable
    individuals_payoff(outcomes, name) -> PayoffTable

The utility that counts individuals: the payoff of an action in a state is the
(expected) number of individuals of the focal population, so the payoff table is the
table of counts itself, checked to be nonnegative ([`InvalidPayoffError`](@ref)
otherwise). Expected utility is then an expected number of individuals, and every
value of information is read in individuals.

The second form picks the table called `name` from `outcomes`, a named collection of
outcome count tables: a named tuple, a dictionary, or a vector of `name => table`
pairs.

```jldoctest
julia> counts = PayoffTable([:nothing, :fence], [:dry, :wet], [40 90; 60 85]);

julia> individuals_payoff((; toadlets=counts), :toadlets) == counts
true
```
"""
individuals_payoff(counts::PayoffTable) = _count_table(counts, "individuals")
individuals_payoff(outcomes, name) = individuals_payoff(_named_table(outcomes, name))

"""
    infection_payoff(; uninfected::PayoffTable, infected::PayoffTable, weight=1)
    infection_payoff(outcomes; uninfected=:uninfected, infected=:infected, weight=1)

The utility for an infectious disease, from two tables of counts with the same
labels:

    payoff = uninfected − weight × infected.

Uninfected individuals count for one each and every infected individual counts
against, with `weight ≥ 0` the one free parameter: `weight = 0` counts only the
uninfected, and `weight = 1` sets an infected individual against an uninfected one.
The weight is a value judgement, so show results at `0`, at `1`, and as a sweep.
Whether infected but living individuals also count towards the population is a
modelling choice made when the count tables are built, not here.

The second form takes a named collection of outcome count tables (a named tuple, a
dictionary, or `name => table` pairs) and the names of the two tables in it.

Both counts must be nonnegative ([`InvalidPayoffError`](@ref) otherwise) and share
their labels ([`LabelMismatchError`](@ref) otherwise); a negative `weight` is an
`ArgumentError`. The result is exact for exact counts and an exact weight.

```jldoctest
julia> uninfected = PayoffTable([:nothing, :treat], [:absent, :present], [500 100; 470 250]);

julia> infected = PayoffTable([:nothing, :treat], [:absent, :present], [0 250; 0 120]);

julia> infection_payoff(; uninfected, infected).Q
2×2 Matrix{Int64}:
 500  -150
 470   130

julia> infection_payoff((; uninfected, infected); weight=0) == uninfected
true
```
"""
function infection_payoff(; uninfected::PayoffTable, infected::PayoffTable, weight::Real=1)
    weight >= 0 || throw(ArgumentError("the weight must be nonnegative, got $weight"))
    return _count_table(uninfected, "uninfected") -
           weight * _count_table(infected, "infected")
end
function infection_payoff(outcomes; uninfected=:uninfected, infected=:infected,
                          weight::Real=1)
    return infection_payoff(; uninfected=_named_table(outcomes, uninfected),
                            infected=_named_table(outcomes, infected), weight)
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
