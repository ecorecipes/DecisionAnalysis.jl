# Utilities on outcome tables: counts of individuals, never money. A utility maps the
# named outcomes of one run to a number; `payoff_table` averages it inside each cell.

"""
    AbstractUtility

Supertype of the utilities that [`payoff_table`](@ref) applies to an
[`OutcomeTable`](@ref): [`IndividualsUtility`](@ref), [`InfectionUtility`](@ref) and
[`FunctionUtility`](@ref). A utility is callable on the named outcomes of one run:
`u((; uninfected=40, infected=3))`. Utility is counted in individuals of the focal
species; there are no monetary costs.
"""
abstract type AbstractUtility end

"""
    IndividualsUtility(name)

The utility that counts individuals: the value of the outcome called `name` (the
number of the focal species at the evaluation time). It is linear, so expected
utility is an expected number of individuals and every value of information is read
in individuals. The count must not be negative.

```jldoctest
julia> IndividualsUtility(:frogs)((; frogs=12, eggs=300))
12
```
"""
struct IndividualsUtility <: AbstractUtility
    name::Symbol
    IndividualsUtility(name::Union{Symbol,AbstractString}) = new(Symbol(name))
end

"""
    InfectionUtility(; uninfected=:uninfected, infected=:infected, weight=1)

The utility for an infectious disease, from two counts of one run:

    utility = uninfected − weight × infected.

`uninfected` and `infected` name the outcomes that hold the two counts. Uninfected
individuals count for one each and every infected individual counts against, with
`weight ≥ 0` the one free parameter: `weight = 0` counts only the uninfected and
`weight = 1` sets an infected individual against an uninfected one. The weight is a
value judgement, so show results at `0`, at `1` and as a sweep
([`policy_regions`](@ref)). Whether infected but living individuals also count
towards the population is a modelling choice: to count every survivor, use
[`IndividualsUtility`](@ref) on the number alive.

This is the utility behind [`infection_payoff`](@ref), which applies it to tables of
counts.

```jldoctest
julia> InfectionUtility(; weight=1 // 2)((; uninfected=40, infected=6))
37//1
```
"""
struct InfectionUtility{W<:Real} <: AbstractUtility
    uninfected::Symbol
    infected::Symbol
    weight::W
    function InfectionUtility(uninfected::Symbol, infected::Symbol,
                              weight::W) where {W<:Real}
        weight >= 0 || throw(ArgumentError("the weight must be nonnegative, got $weight"))
        return new{W}(uninfected, infected, weight)
    end
end
function InfectionUtility(; uninfected=:uninfected, infected=:infected, weight::Real=1)
    return InfectionUtility(Symbol(uninfected), Symbol(infected), weight)
end

"""
    FunctionUtility(f; name=:utility)

A utility given by a function of the named outcomes of one run:
`f(o::NamedTuple) -> Real`, for example
`FunctionUtility(o -> o.survivors / o.initial)` for the surviving fraction. Passing a
plain function to [`payoff_table`](@ref) wraps it in this type. Keep the result in
units of individuals (or a fraction of them); the package has no monetary costs.
"""
struct FunctionUtility{F} <: AbstractUtility
    f::F
    name::Symbol
end
function FunctionUtility(f; name::Union{Symbol,AbstractString}=:utility)
    return FunctionUtility(f,
                           Symbol(name))
end

# A linear utility as `outcome name => coefficient` terms, counts being nonnegative.
_terms(u::IndividualsUtility) = [u.name => 1]
_terms(u::InfectionUtility) = [u.uninfected => 1, u.infected => -u.weight]
_count_words(::IndividualsUtility, ::Symbol) = "individuals"
function _count_words(u::InfectionUtility, name::Symbol)
    return name == u.uninfected ? "uninfected" :
           "infected"
end

function _outcome_value(o::NamedTuple, name::Symbol)
    haskey(o, name) ||
        throw(InvalidOutcomeError("there is no outcome named $name among $(keys(o))"))
    return o[name]
end

function (u::Union{IndividualsUtility,InfectionUtility})(o::NamedTuple)
    return sum(_terms(u)) do (name, coefficient)
        v = _outcome_value(o, name)
        v >= 0 || throw(InvalidOutcomeError("the $(_count_words(u, name)) count is $v; " *
                                            "a count cannot be negative"))
        return coefficient * v
    end
end
(u::FunctionUtility)(o::NamedTuple) = u.f(o)

# The same utility applied to tables of counts: the implementation behind
# `individuals_payoff` and `infection_payoff` on payoff tables.
function _linear_payoff(u::Union{IndividualsUtility,InfectionUtility}, table_of)
    return reduce(+,
                  (coefficient * _count_table(table_of(name), _count_words(u, name))
                   for (name, coefficient) in _terms(u)))
end

# Utility of every usable run: U[action, draw, replicate] and the mask of usable runs.
function _run_utilities(t::OutcomeTable, u::Union{IndividualsUtility,InfectionUtility})
    terms = _terms(u)
    index = map(terms) do (name, _)
        k = findfirst(==(name), t.outcomes)
        k === nothing &&
            throw(InvalidOutcomeError("there is no outcome named $name among " *
                                      "$(t.outcomes)"))
        return k
    end
    ok = t.status .== _OK
    for ((name, _), k) in zip(terms, index)
        slice = view(t.values, :, :, :, k)
        i = findfirst(i -> ok[i] && slice[i] < 0, CartesianIndices(ok))
        i === nothing ||
            throw(InvalidOutcomeError("the $(_count_words(u, name)) count of action " *
                                      "$(t.actions[i[1]]) at draw $(t.draws[i[2]]) is " *
                                      "$(slice[i]); a count cannot be negative"))
    end
    U = sum(coefficient .* view(t.values, :, :, :, k)
            for ((_, coefficient), k) in zip(terms, index))
    return U, ok
end

function _run_utilities(t::OutcomeTable, u::FunctionUtility)
    ok = t.status .== _OK
    names = Tuple(t.outcomes)
    raw = Array{Any}(undef, size(ok))
    fill!(raw, 0)
    T = Union{}
    for i in CartesianIndices(ok)
        ok[i] || continue
        v = u.f(NamedTuple{names}(Tuple(view(t.values, i, :))))
        (v isa Real && isfinite(v)) ||
            throw(InvalidOutcomeError("the utility of action $(t.actions[i[1]]) at draw " *
                                      "$(t.draws[i[2]]), replicate $(t.replicates[i[3]]) " *
                                      "is $(repr(v)), not a finite number"))
        raw[i] = v
        T = promote_type(T, typeof(v))
    end
    T === Union{} && throw(InvalidOutcomeError("the table has no usable run"))
    return Array{T,3}(raw), ok
end

_exact_mean(x::AbstractVector{<:Union{Integer,Rational}}) = sum(x) // length(x)
_exact_mean(x::AbstractVector) = sum(x) / length(x)

"""
    payoff_table(outcomes::OutcomeTable, utility; baseline=nothing) -> PayoffTable

The payoff table of an outcome table under a utility: `Q[a, θ]` is the **conditional
expected utility** of action `a` at draw `θ`, the mean of the utility over the usable
replicates of the cell. `utility` is an [`AbstractUtility`](@ref)
([`IndividualsUtility`](@ref), [`InfectionUtility`](@ref), [`FunctionUtility`](@ref))
or a plain function of the named outcomes of one run.

The utility is applied to each run and then averaged, so a non-linear utility is
handled correctly (the utility of the mean outcome is not the mean utility).

**Monte Carlo error is kept apart.** The result carries `standard_errors`, the
standard error of each cell mean (sample standard deviation over √n), and
`replicates`, the number of runs averaged; neither is folded into a payoff. A cell
with one run has no standard error: the table then has `standard_errors = nothing`
and is flagged by [`single_replicate`](@ref). Runs with a failing status are left
out, and a cell with no usable run throws [`InvalidOutcomeError`](@ref).

**Gains over no intervention.** With `baseline` (an action label), the table holds
the gain of each action over the baseline, `Q[a, θ] − Q[baseline, θ]`, computed
replicate by replicate from the runs that both have (the same replicate uses the same
random numbers under [`outcomes`](@ref)), so its standard errors are those of the
paired differences and are smaller than independent runs would give. The baseline
row is zero. Regret and every value of information are the same for the gains as
for the payoffs.

Exact in, exact out: integer or rational utilities give rational payoffs; standard
errors are floating point.

```jldoctest
julia> runs = [(; draw=d, action=a, replicate=r, uninfected=20d + r, infected=(a == "none") * 4r)
               for d in 1:2 for a in ("none", "treat") for r in 1:2];

julia> t = OutcomeTable(runs; outcomes=[:uninfected, :infected]);

julia> payoff_table(t, InfectionUtility(; weight=1); baseline=:none).Q
2×2 Matrix{Rational{Int64}}:
 0  0
 6  6
```
"""
function payoff_table(t::OutcomeTable, u::AbstractUtility; baseline=nothing)
    U, ok = _run_utilities(t, u)
    if baseline !== nothing
        b = findfirst(==(Symbol(baseline)), t.actions)
        b === nothing && throw(InvalidBaselineError(Symbol(baseline), t.actions))
        ok = ok .& view(ok, b:b, :, :)
        U = U .- view(U, b:b, :, :)
    end
    na, nd = length(t.actions), length(t.draws)
    counts = [count(view(ok, a, d, :)) for a in 1:na, d in 1:nd]
    i = findfirst(iszero, counts)
    i === nothing ||
        throw(InvalidOutcomeError("action $(t.actions[i[1]]) at draw $(t.draws[i[2]]) " *
                                  "has no usable run" *
                                  (baseline === nothing ? "" :
                                   " shared with the baseline") *
                                  ", so its conditional expected utility cannot be " *
                                  "estimated"))
    cell(a, d) = U[a, d, findall(view(ok, a, d, :))]
    Q = [_exact_mean(cell(a, d)) for a in 1:na, d in 1:nd]
    se = nothing
    if all(>=(2), counts)
        se = [_standard_error(cell(a, d)) for a in 1:na, d in 1:nd]
    end
    return PayoffTable(t.actions, t.draws, Q; standard_errors=se, replicates=counts)
end
payoff_table(t::OutcomeTable, f; kw...) = payoff_table(t, FunctionUtility(f); kw...)

function _standard_error(x::AbstractVector)
    n = length(x)
    m = sum(Float64, x) / n
    return sqrt(sum(v -> (Float64(v) - m)^2, x) / (n - 1) / n)
end

individuals_payoff(t::OutcomeTable, name) = payoff_table(t, IndividualsUtility(name))
function infection_payoff(t::OutcomeTable; uninfected=:uninfected, infected=:infected,
                          weight::Real=1)
    return payoff_table(t, InfectionUtility(; uninfected, infected, weight))
end

"""
    DecisionProblem(outcomes::OutcomeTable, utility; feasible=nothing, baseline=nothing)

The decision problem of an outcome table under a utility: the payoff table
[`payoff_table`](@ref)`(outcomes, utility)` with the [`SampleBelief`](@ref) of the
table (its draws and their weights). `baseline` names the no-intervention action, as
for any problem; the payoffs are kept as payoffs, not as gains.
"""
function DecisionProblem(t::OutcomeTable, utility; feasible=nothing, baseline=nothing)
    return DecisionProblem(payoff_table(t, utility), SampleBelief(t); feasible, baseline)
end

"""
    crn_correlation(outcomes::OutcomeTable, utility) -> Float64

The achieved correlation between actions of the simulation noise within a cell: the
pooled correlation, over draws and pairs of actions, of the utilities of the same
replicate under two actions (deviations from each cell's mean). Under common random
numbers ([`outcomes`](@ref)) it is positive, and the closer to one, the more of the
noise cancels in a comparison of actions; independent runs give about zero. Pass it
as `noise_correlation` to [`evpi`](@ref) with [`SampleAverage`](@ref) and to
[`DecisionReport`](@ref), so that the cell-noise bias is not overstated. `NaN` when
there is no noise to correlate.
"""
function crn_correlation(t::OutcomeTable, utility)
    u = utility isa AbstractUtility ? utility : FunctionUtility(utility)
    U, ok = _run_utilities(t, u)
    covariance = 0.0
    scale = 0.0
    for d in eachindex(t.draws), a in eachindex(t.actions), b in (a + 1):length(t.actions)
        both = findall(view(ok, a, d, :) .& view(ok, b, d, :))
        length(both) >= 2 || continue
        x = Float64.(U[a, d, both])
        y = Float64.(U[b, d, both])
        x .-= sum(x) / length(x)
        y .-= sum(y) / length(y)
        covariance += dot(x, y)
        scale += sqrt(sum(abs2, x) * sum(abs2, y))
    end
    return iszero(scale) ? NaN : covariance / scale
end
