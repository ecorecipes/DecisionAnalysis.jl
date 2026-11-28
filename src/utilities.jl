# Utilities on outcome tables: counts of individuals, never money. A utility maps the
# named outcomes of one run to a number; `payoff_table` averages it inside each cell.

"""
    AbstractUtility

Supertype of the utilities that [`payoff_table`](@ref) applies to an
[`OutcomeTable`](@ref). The count utilities [`IndividualsUtility`](@ref) and
[`InfectionUtility`](@ref) are linear, so their expected utility is an expected number
of individuals. The risk-attitude utilities [`LinearUtility`](@ref),
[`ExponentialUtility`](@ref) and [`PowerUtility`](@ref) act on one named outcome, and
[`AdditiveUtility`](@ref) and [`MultiplicativeUtility`](@ref) combine several
attributes; [`FunctionUtility`](@ref) wraps any function. A utility is callable on the
named outcomes of one run: `u((; uninfected=40, infected=3))`. Outcomes are counts of
individuals of the focal species; there are no monetary costs.
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

# Any other utility is applied run by run: a non-linear utility of the mean outcome is
# not the mean utility, so nothing is averaged before the utility is taken.
function _run_utilities(t::OutcomeTable, u::AbstractUtility)
    ok = t.status .== _OK
    names = Tuple(t.outcomes)
    U = nothing
    for i in CartesianIndices(ok)
        ok[i] || continue
        v = u(NamedTuple{names}(Tuple(view(t.values, i, :))))
        (v isa Real && isfinite(v)) ||
            throw(InvalidOutcomeError("the utility of action $(t.actions[i[1]]) at draw " *
                                      "$(t.draws[i[2]]), replicate $(t.replicates[i[3]]) " *
                                      "is $(repr(v)), not a finite number"))
        if U === nothing
            T = typeof(v) === Bool ? Int : typeof(v)
            U = zeros(T, size(ok))
            U[i] = v
        elseif v isa eltype(U)
            U[i] = v
        else
            T = promote_type(eltype(U), typeof(v))
            T === Bool && (T = Int)
            U = Array{T,3}(U)
            U[i] = v
        end
    end
    U === nothing && throw(InvalidOutcomeError("the table has no usable run"))
    return U, ok
end

_exact_mean_div(s::Union{Integer,Rational}, n::Int) = s // n
_exact_mean_div(s::Real, n::Int) = s / n
_exact_mean_type(::Type{T}) where {T<:Union{Integer,Rational}} = typeof(zero(T) // 1)
_exact_mean_type(::Type{T}) where {T<:Real} = typeof(zero(T) / 1)
_exact_mean(x::AbstractVector{<:Union{Integer,Rational}}) = sum(x) // length(x)
_exact_mean(x::AbstractVector) = sum(x) / length(x)

"""
    payoff_table(outcomes::OutcomeTable, utility; baseline=nothing) -> PayoffTable

The payoff table of an outcome table under a utility: `Q[a, θ]` is the **conditional
expected utility** of action `a` at draw `θ`, the mean of the utility over the usable
replicates of the cell. `utility` is an [`AbstractUtility`](@ref)
([`IndividualsUtility`](@ref), [`InfectionUtility`](@ref), the risk-attitude utilities
[`LinearUtility`](@ref), [`ExponentialUtility`](@ref) and [`PowerUtility`](@ref), the
multi-attribute [`AdditiveUtility`](@ref) and [`MultiplicativeUtility`](@ref),
[`FunctionUtility`](@ref)) or a plain function of the named outcomes of one run.

**The utility is applied to each run, before anything is averaged over replicates.**
A non-linear utility of the mean outcome is not the mean utility (Jensen's
inequality), so averaging the outcomes first would be wrong for every utility but a
linear one. With a non-linear utility the payoffs, and every expected utility and
value of information computed from them, are in units of utility and **no longer
counts of individuals**; [`certainty_equivalent`](@ref) takes an expected utility back
to individuals.

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
    raw, usable = _run_utilities(t, u)
    b = baseline === nothing ? nothing : findfirst(==(Symbol(baseline)), t.actions)
    baseline !== nothing && b === nothing &&
        throw(InvalidBaselineError(Symbol(baseline), t.actions))
    # Relative to the baseline: differences within a run, where both runs are usable.
    U = b === nothing ? raw : raw .- view(raw, b:b, :, :)
    ok = b === nothing ? usable : usable .& view(usable, b:b, :, :)
    na, nd, nr = size(U)
    counts = Matrix{Int}(undef, na, nd)
    min_count = nr
    @inbounds for d in 1:nd, a in 1:na
        c = 0
        for r in 1:nr
            c += ok[a, d, r]
        end
        counts[a, d] = c
        c < min_count && (min_count = c)
    end
    if min_count == 0
        i = findfirst(iszero, counts)
        throw(InvalidOutcomeError("action $(t.actions[i[1]]) at draw $(t.draws[i[2]]) " *
                                  "has no usable run" *
                                  (baseline === nothing ? "" :
                                   " shared with the baseline") *
                                  ", so its conditional expected utility cannot be " *
                                  "estimated"))
    end
    need_se = min_count >= 2
    TU = eltype(U) === Bool ? Int : eltype(U)
    TQ = _exact_mean_type(TU)
    Q = Matrix{TQ}(undef, na, nd)
    se = need_se ? Matrix{Float64}(undef, na, nd) : nothing
    @inbounds for d in 1:nd, a in 1:na
        c = counts[a, d]
        s = zero(TU)
        first_v = zero(TU)
        has_first = false
        all_same = true
        for r in 1:nr
            ok[a, d, r] || continue
            v = TU(U[a, d, r])
            if !has_first
                first_v = v
                has_first = true
            elseif v != first_v
                all_same = false
            end
            s += v
        end
        Q[a, d] = _exact_mean_div(s, c)
        if se !== nothing
            if all_same
                se[a, d] = 0.0
            else
                m = Float64(s) / c
                ssq = 0.0
                for r in 1:nr
                    ok[a, d, r] || continue
                    diff = Float64(U[a, d, r]) - m
                    ssq += diff * diff
                end
                se[a, d] = sqrt(ssq / (c - 1) / c)
            end
        end
    end
    nc = nothing
    if se !== nothing && na >= 2
        ρ = _crn_correlation(U, ok)
        isfinite(ρ) && (nc = clamp(ρ, -1.0, 1.0))
    end
    return PayoffTable(t.actions, t.draws, Q; standard_errors=se, replicates=counts,
                       noise_correlation=nc)
end
payoff_table(t::OutcomeTable, f; kw...) = payoff_table(t, FunctionUtility(f); kw...)

function _crn_correlation(U::AbstractArray{<:Real,3}, ok::AbstractArray{Bool,3})
    na, nd, nr = size(ok)
    covariance = 0.0
    scale = 0.0
    x = Float64[]
    y = Float64[]
    sizehint!(x, nr)
    sizehint!(y, nr)
    @inbounds for d in 1:nd, a in 1:na, b in (a + 1):na
        empty!(x)
        empty!(y)
        for r in 1:nr
            if ok[a, d, r] && ok[b, d, r]
                push!(x, Float64(U[a, d, r]))
                push!(y, Float64(U[b, d, r]))
            end
        end
        m = length(x)
        m >= 2 || continue
        mx = sum(x) / m
        my = sum(y) / m
        sxy = 0.0
        sxx = 0.0
        syy = 0.0
        for i in 1:m
            dx = x[i] - mx
            dy = y[i] - my
            sxy += dx * dy
            sxx += dx * dx
            syy += dy * dy
        end
        covariance += sxy
        scale += sqrt(sxx * syy)
    end
    return iszero(scale) ? NaN : covariance / scale
end

function _standard_error(x::AbstractVector)
    n = length(x)
    allequal(x) && return 0.0    # equal runs (a deterministic model): exactly zero, no rounding
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
    return _crn_correlation(U, ok)
end

# Risk attitude: utilities of one named outcome
# ---------------------------------------------

function _check_scale(worst, best)
    (worst === nothing) == (best === nothing) ||
        throw(ArgumentError("give both `worst` and `best`, or neither"))
    worst === nothing && return nothing, nothing
    (worst isa Real && best isa Real && isfinite(worst) && isfinite(best)) ||
        throw(ArgumentError("`worst` and `best` must be finite numbers, got $worst and " *
                            "$best"))
    worst != best ||
        throw(ArgumentError("`worst` and `best` must differ, got $worst twice"))
    return promote(worst, best)
end

_exact_ratio(a::Union{Integer,Rational}, b::Union{Integer,Rational}) = a // b
_exact_ratio(a::Real, b::Real) = a / b

"""
    LinearUtility(name; worst=nothing, best=nothing)

The risk-neutral utility of the outcome called `name`: `u(x) = x`. With `worst` and
`best` it is scaled to `[0, 1]`,

    u(x) = (x − worst) / (best − worst),

so that `u(worst) = 0` and `u(best) = 1`; `best < worst` describes an outcome that
counts against (infected individuals). The scaled form is what
[`AdditiveUtility`](@ref) and [`MultiplicativeUtility`](@ref) combine.

Linear utility is the reference case of risk attitude: the
[`certainty_equivalent`](@ref) of a lottery is its mean and the
[`risk_premium`](@ref) is zero. Unscaled, expected utility is an expected number of
individuals; scaled, it is a fraction of the way from `worst` to `best`. Exact in,
exact out. Unlike [`IndividualsUtility`](@ref) it accepts any real outcome.

```jldoctest
julia> u = LinearUtility(:frogs; worst=0, best=50); u((; frogs=20))
2//5

julia> certainty_equivalent(u, [10, 30])
20//1
```
"""
struct LinearUtility{B<:Union{Nothing,Real}} <: AbstractUtility
    name::Symbol
    worst::B
    best::B
end
function LinearUtility(name::Union{Symbol,AbstractString}; worst=nothing, best=nothing)
    return LinearUtility(Symbol(name), _check_scale(worst, best)...)
end

"""
    ExponentialUtility(name; risk_aversion, worst=nothing, best=nothing)

The utility with **constant absolute risk aversion** `r = risk_aversion` (per
individual) on the outcome called `name`:

    u(x) = (1 − exp(−r x)) / r,        and u(x) = x at r = 0.

`r > 0` is risk-averse (a sure number of individuals is preferred to a lottery with
the same mean), `r < 0` risk-seeking, and `1 / r` is the risk tolerance in
individuals. The attitude does not depend on the level of the outcome: adding the
same number to every result of a lottery adds it to the certainty equivalent.

With `worst` and `best` the utility is scaled to `[0, 1]` and measured along the
signed improvement `d = σ (x − worst)` from the worst level towards the best (with
`σ = +1` when `best > worst` and `σ = −1` when `best < worst`),
`u(x) = v(d) / v(|best − worst|)` with `v` the function above, so `best < worst`
describes an outcome that counts against, and `r > 0` is then averse to risk in the
improvement.

**Expected utility under this utility is not a count of individuals.** Use
[`certainty_equivalent`](@ref) to return to individuals, and [`risk_premium`](@ref)
for what the risk costs.

```jldoctest
julia> u = ExponentialUtility(:frogs; risk_aversion=0.05);

julia> round(certainty_equivalent(u, [0, 40]); digits=3)   # the mean is 20
11.324
```
"""
struct ExponentialUtility{B<:Union{Nothing,Real}} <: AbstractUtility
    name::Symbol
    risk_aversion::Float64
    worst::B
    best::B
end
function ExponentialUtility(name::Union{Symbol,AbstractString}; risk_aversion::Real,
                            worst=nothing, best=nothing)
    isfinite(risk_aversion) ||
        throw(ArgumentError("the risk aversion must be finite, got $risk_aversion"))
    return ExponentialUtility(Symbol(name), Float64(risk_aversion),
                              _check_scale(worst, best)...)
end

"""
    PowerUtility(name; risk_aversion, worst=nothing, best=nothing)

The utility with **constant relative risk aversion** `ρ = risk_aversion` on the
outcome called `name`, a positive number of individuals:

    u(x) = x^(1 − ρ) / (1 − ρ),        and u(x) = log(x) at ρ = 1.

`ρ = 0` is risk-neutral, `ρ > 0` risk-averse and `ρ < 0` risk-seeking. The attitude
is to proportional changes: multiplying every result of a lottery by the same factor
multiplies the certainty equivalent by it. The outcome must be nonnegative, and
positive when `ρ ≥ 1` (a count of zero then has utility `−∞`), otherwise
[`InvalidOutcomeError`](@ref) is thrown.

With `worst` and `best` (`0 ≤ worst < best`, and `worst > 0` when `ρ ≥ 1`) the
utility is scaled to `[0, 1]`: `u(x) = (v(x) − v(worst)) / (v(best) − v(worst))`
with `v` the function above. An outcome that counts against has no power utility
here; use [`LinearUtility`](@ref) or [`ExponentialUtility`](@ref) for it.

**Expected utility under this utility is not a count of individuals**; see
[`certainty_equivalent`](@ref).

```jldoctest
julia> u = PowerUtility(:frogs; risk_aversion=0.5);

julia> round(certainty_equivalent(u, [0, 40]); digits=6)     # the mean is 20
10.0
```
"""
struct PowerUtility{B<:Union{Nothing,Real}} <: AbstractUtility
    name::Symbol
    risk_aversion::Float64
    worst::B
    best::B
end
function PowerUtility(name::Union{Symbol,AbstractString}; risk_aversion::Real,
                      worst=nothing, best=nothing)
    isfinite(risk_aversion) ||
        throw(ArgumentError("the risk aversion must be finite, got $risk_aversion"))
    worst, best = _check_scale(worst, best)
    if worst !== nothing
        0 <= worst < best ||
            throw(ArgumentError("a scaled PowerUtility needs 0 ≤ worst < best, got " *
                                "worst = $worst and best = $best"))
        risk_aversion < 1 || worst > 0 ||
            throw(ArgumentError("with risk aversion ≥ 1 the utility of zero is −∞, so " *
                                "`worst` must be positive"))
    end
    return PowerUtility(Symbol(name), Float64(risk_aversion), worst, best)
end

const _AttributeUtility = Union{LinearUtility,ExponentialUtility,PowerUtility}

# +1 when more of the outcome is better, −1 when it counts against.
_direction(u::_AttributeUtility) = u.worst === nothing || u.best > u.worst ? 1 : -1

# The unscaled utility v and its inverse.
_natural(::LinearUtility, x::Real) = x
_natural_inverse(::LinearUtility, y::Real) = y
function _natural(u::ExponentialUtility, x::Real)
    r = u.risk_aversion
    return iszero(r) ? float(x) : -expm1(-r * x) / r
end
function _natural_inverse(u::ExponentialUtility, y::Real)
    r = u.risk_aversion
    iszero(r) && return float(y)
    r * y < 1 ||
        throw(ArgumentError("no outcome has utility $y under this exponential utility " *
                            "(its utilities are below $(1 / r))"))
    return -log1p(-r * y) / r
end
function _natural(u::PowerUtility, x::Real)
    ρ = u.risk_aversion
    (x > 0 || (x == 0 && ρ < 1)) ||
        throw(InvalidOutcomeError("the outcome $(u.name) is $x; a power utility with " *
                                  "risk aversion $ρ needs a " *
                                  (ρ < 1 ? "nonnegative" : "positive") * " outcome"))
    return isone(ρ) ? log(x) : float(x)^(1 - ρ) / (1 - ρ)
end
function _natural_inverse(u::PowerUtility, y::Real)
    ρ = u.risk_aversion
    isone(ρ) && return exp(y)
    ok = ρ < 1 ? (1 - ρ) * y >= 0 : (1 - ρ) * y > 0
    ok || throw(ArgumentError("no outcome has utility $y under this power utility"))
    return ((1 - ρ) * y)^(1 / (1 - ρ))
end

# Linear and exponential utilities are scaled along the improvement from `worst`.
function _value(u::Union{LinearUtility,ExponentialUtility}, x::Real)
    u.worst === nothing && return _natural(u, x)
    σ = _direction(u)
    u isa LinearUtility && return _exact_ratio(σ * (x - u.worst), σ * (u.best - u.worst))
    return _natural(u, σ * (x - u.worst)) / _natural(u, σ * (u.best - u.worst))
end
function _value(u::PowerUtility, x::Real)
    v = _natural(u, x)
    u.worst === nothing && return v
    lo = _natural(u, u.worst)
    return (v - lo) / (_natural(u, u.best) - lo)
end

(u::_AttributeUtility)(x::Real) = _value(u, x)
(u::_AttributeUtility)(o::NamedTuple) = _value(u, _outcome_value(o, u.name))

# Fast path for single-attribute utilities: reads the one named column directly with no
# per-run NamedTuple allocation.
function _run_utilities(t::OutcomeTable, u::_AttributeUtility)
    k = findfirst(==(u.name), t.outcomes)
    k === nothing &&
        throw(InvalidOutcomeError("there is no outcome named $(u.name) among $(t.outcomes)"))
    ok = t.status .== _OK
    slice = view(t.values, :, :, :, k)
    first_i = findfirst(ok)
    first_i === nothing && throw(InvalidOutcomeError("the table has no usable run"))
    v1 = _value(u, slice[first_i])
    (v1 isa Real && isfinite(v1)) ||
        throw(InvalidOutcomeError("the utility of action $(t.actions[first_i[1]]) at draw " *
                                  "$(t.draws[first_i[2]]), replicate " *
                                  "$(t.replicates[first_i[3]]) is $(repr(v1)), not a " *
                                  "finite number"))
    T = typeof(v1) === Bool ? Int : typeof(v1)
    U = zeros(T, size(ok))
    @inbounds for i in CartesianIndices(ok)
        ok[i] || continue
        v = _value(u, slice[i])
        (v isa Real && isfinite(v)) ||
            throw(InvalidOutcomeError("the utility of action $(t.actions[i[1]]) at draw " *
                                      "$(t.draws[i[2]]), replicate $(t.replicates[i[3]]) " *
                                      "is $(repr(v)), not a finite number"))
        U[i] = v
    end
    return U, ok
end

"""
    inverse_utility(utility, value)

The outcome whose utility is `value`: the inverse of a single-attribute utility
([`LinearUtility`](@ref), [`ExponentialUtility`](@ref), [`PowerUtility`](@ref),
[`IndividualsUtility`](@ref)). Applied to an expected utility it gives the
[`certainty_equivalent`](@ref), in individuals. Throws `ArgumentError` when no
outcome has that utility.

```jldoctest
julia> u = PowerUtility(:frogs; risk_aversion=0.5); inverse_utility(u, u(16))
16.0
```
"""
inverse_utility(::IndividualsUtility, value::Real) = value
function inverse_utility(u::Union{LinearUtility,ExponentialUtility}, value::Real)
    u.worst === nothing && return _natural_inverse(u, value)
    σ = _direction(u)
    span = σ * (u.best - u.worst)
    u isa LinearUtility && return u.worst + σ * value * span
    return u.worst + σ * _natural_inverse(u, value * _natural(u, span))
end
function inverse_utility(u::PowerUtility, value::Real)
    u.worst === nothing && return _natural_inverse(u, value)
    lo = _natural(u, u.worst)
    return _natural_inverse(u, lo + value * (_natural(u, u.best) - lo))
end
function inverse_utility(u::AbstractUtility, ::Real)
    return throw(ArgumentError("$(nameof(typeof(u))) is not a utility of a single outcome, so " *
                               "it has no inverse and no certainty equivalent"))
end

_direction(::AbstractUtility) = 1
_lottery_value(u::_AttributeUtility, x::Real) = _value(u, x)
_lottery_value(::IndividualsUtility, x::Real) = x
function _lottery_value(u::AbstractUtility, ::Real)
    return inverse_utility(u, 0)          # throws the explanatory ArgumentError
end

function _lottery_weights(lottery::AbstractVector, weights)
    isempty(lottery) && throw(ArgumentError("the lottery has no outcome"))
    weights === nothing && return fill(1 // length(lottery), length(lottery))
    length(weights) == length(lottery) ||
        throw(ArgumentError("$(length(weights)) weights for $(length(lottery)) outcomes"))
    return _normalise_weights(collect(weights))
end

# Shift-invariant certainty equivalent for ExponentialUtility via log-sum-exp, avoiding
# expm1 saturation on large outcome scales.
function _exp_certainty_equivalent(u::ExponentialUtility, xs::AbstractVector{<:Real},
                                   ws::AbstractVector{<:Real})
    r = u.risk_aversion
    iszero(r) && return sum(Float64(ws[i]) * Float64(xs[i]) for i in eachindex(xs))
    α = -r * _direction(u)
    xstar = α < 0 ? minimum(Float64(xs[i]) for i in eachindex(xs) if ws[i] > 0) :
            maximum(Float64(xs[i]) for i in eachindex(xs) if ws[i] > 0)
    total = sum(Float64(ws[i]) * exp(α * (Float64(xs[i]) - xstar))
                for i in eachindex(xs) if ws[i] > 0)
    return xstar + log(total) / α
end

"""
    certainty_equivalent(utility, lottery::AbstractVector; weights=nothing)
    certainty_equivalent(outcomes::OutcomeTable, utility) -> Vector{NamedTuple}

The **certainty equivalent** of a lottery over one outcome: the sure number of
individuals that is exactly as good as the lottery,

    CE = u⁻¹(E u(X)).

`lottery` holds the possible outcomes and `weights` their probabilities (equal when
`nothing`; normalised). `utility` is a single-attribute utility
([`LinearUtility`](@ref), [`ExponentialUtility`](@ref), [`PowerUtility`](@ref),
[`IndividualsUtility`](@ref)). An expected utility under a non-linear utility is not
a number of individuals; the certainty equivalent **returns it to individuals**, and
is below the mean for a risk-averse utility ([`risk_premium`](@ref)). To convert an
expected utility that is already computed, use [`inverse_utility`](@ref).

With an [`OutcomeTable`](@ref) the lottery of each action is its runs: a draw with
its weight, then a usable replicate of that cell with equal chances, which is the
lottery whose expected utility is the action's expected payoff under
[`payoff_table`](@ref). The utility is applied to every run before averaging. One
`(; action, expected_utility, certainty_equivalent, expected_outcome, risk_premium)`
per action is returned.

A multi-attribute utility has no certainty equivalent (there is no single outcome to
return to) and throws `ArgumentError`.

```jldoctest
julia> u = ExponentialUtility(:frogs; risk_aversion=0.05);

julia> round(certainty_equivalent(u, [0, 40]; weights=[1, 3]); digits=3)   # the mean is 30
20.911
```
"""
function certainty_equivalent(u::AbstractUtility, lottery::AbstractVector{<:Real};
                              weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    w = _lottery_weights(lottery, weights)
    u isa ExponentialUtility && return _exp_certainty_equivalent(u, lottery, w)
    return inverse_utility(u,
                           sum(w[i] * _lottery_value(u, lottery[i])
                               for i in eachindex(lottery)))
end

"""
    risk_premium(utility, lottery::AbstractVector; weights=nothing)

The **risk premium** of a lottery over one outcome: how many individuals of expected
outcome the decision-maker would give up to be rid of the risk,

    risk premium = E X − CE,

with `CE` the [`certainty_equivalent`](@ref). It is zero for a linear utility,
positive for a risk-averse one and negative for a risk-seeking one. For an outcome
that counts against (`best < worst`) the sign is turned, `CE − E X`, so that a
positive premium still means risk aversion. It is in individuals.

```jldoctest
julia> round(risk_premium(PowerUtility(:frogs; risk_aversion=0.5), [0, 40]); digits=6)
10.0
```
"""
function risk_premium(u::AbstractUtility, lottery::AbstractVector{<:Real};
                      weights::Union{Nothing,AbstractVector{<:Real}}=nothing)
    w = _lottery_weights(lottery, weights)
    ce = certainty_equivalent(u, lottery; weights)
    return _direction(u) * (sum(w[i] * lottery[i] for i in eachindex(lottery)) - ce)
end

function certainty_equivalent(t::OutcomeTable, u::AbstractUtility)
    u isa Union{_AttributeUtility,IndividualsUtility} || inverse_utility(u, 0)
    U, ok = _run_utilities(t, u)
    k = findfirst(==(u.name), t.outcomes)
    w = SampleBelief(t).weights
    return map(enumerate(t.actions)) do (a, action)
        eu = zero(promote_type(eltype(U), eltype(w)))
        ex = zero(promote_type(eltype(t.values), eltype(w)))
        xs = u isa ExponentialUtility ? Float64[] : nothing
        ws = u isa ExponentialUtility ? Float64[] : nothing
        for d in eachindex(t.draws)
            runs = findall(view(ok, a, d, :))
            isempty(runs) &&
                throw(InvalidOutcomeError("action $action at draw $(t.draws[d]) has no " *
                                          "usable run"))
            eu += w[d] * _exact_mean(U[a, d, runs])
            ex += w[d] * _exact_mean(t.values[a, d, runs, k])
            if xs !== nothing
                per = Float64(w[d]) / length(runs)
                for r in runs
                    push!(xs, Float64(t.values[a, d, r, k]))
                    push!(ws, per)
                end
            end
        end
        ce = xs !== nothing ? _exp_certainty_equivalent(u, xs, ws) : inverse_utility(u, eu)
        return (; action, expected_utility=eu, certainty_equivalent=ce,
                expected_outcome=ex, risk_premium=_direction(u) * (ex - ce))
    end
end

# Multi-attribute utilities
# -------------------------

function _attribute_pairs(pairs)
    isempty(pairs) && throw(ArgumentError("give at least one attribute"))
    components = map(pairs) do pair
        u = first(pair)
        return u isa AbstractUtility ? u :
               u isa Union{Symbol,AbstractString} ? LinearUtility(u) : FunctionUtility(u)
    end
    weights = map(last, pairs)
    all(k -> k isa Real && isfinite(k), weights) ||
        throw(ArgumentError("the weights must be finite numbers, got $(collect(weights))"))
    return Tuple(components), Tuple(weights)
end

"""
    AdditiveUtility(utility => weight, ...)
    AdditiveUtility(utilities; weights)

The **additive multi-attribute utility** over named outcomes of one run:

    U(x) = Σ_i k_i u_i(x_i),

with `u_i` the utility of attribute `i` (a [`LinearUtility`](@ref),
[`ExponentialUtility`](@ref), [`PowerUtility`](@ref) or any other utility of one run; a
bare outcome name stands for `LinearUtility(name)`) and `k_i` its weight. The
additive form assumes that preferences over lotteries on one attribute do not depend
on the levels of the others, and that only the marginal distributions of the
attributes matter (additive independence; Keeney and Raiffa 1976).

In the standard use every `u_i` is scaled to `[0, 1]` (`worst` and `best` given) and
the weights are nonnegative and sum to one: `U` then lies in `[0, 1]`, and `k_i` is
the utility of moving attribute `i` from its worst to its best level with the others
at their worst. The weights are value judgements; show how the choice moves with them
([`weight_sweep`](@ref), [`policy_regions`](@ref)). Unscaled utilities and any real
weights are also accepted, which gives the count utilities as special cases:
`AdditiveUtility(:uninfected => 1, :infected => -w)` is
[`InfectionUtility`](@ref)`(; weight=w)`.

Used with [`payoff_table`](@ref) it is applied to each run before averaging. With
scaled attributes **expected utility is not a count of individuals**, and a
multi-attribute utility has no [`certainty_equivalent`](@ref). Exact in, exact out
when the attribute utilities are linear.

```jldoctest
julia> u = AdditiveUtility(LinearUtility(:uninfected; worst=0, best=50) => 3 // 4,
                           LinearUtility(:infected; worst=50, best=0) => 1 // 4);

julia> u((; uninfected=40, infected=10))
4//5

julia> AdditiveUtility(:uninfected => 1, :infected => -1 // 2)((; uninfected=40, infected=6))
37//1
```
"""
struct AdditiveUtility{U<:Tuple,W<:Tuple} <: AbstractUtility
    utilities::U
    weights::W
end
AdditiveUtility(pairs::Pair...) = AdditiveUtility(_attribute_pairs(pairs)...)
function AdditiveUtility(utilities::Union{Tuple,AbstractVector}; weights)
    length(utilities) == length(weights) ||
        throw(ArgumentError("$(length(weights)) weights for $(length(utilities)) " *
                            "attributes"))
    return AdditiveUtility((u => k for (u, k) in zip(utilities, weights))...)
end
function (u::AdditiveUtility)(o::NamedTuple)
    return sum(k * c(o) for (c, k) in zip(u.utilities, u.weights))
end

# Elementary symmetric polynomials e_0, e_1, ..., e_n of k.
function _elementary_symmetric(k::Vector{Float64})
    n = length(k)
    e = zeros(n + 1)
    e[1] = 1.0
    for ki in k
        for m in n:-1:1
            e[m + 1] += ki * e[m]
        end
    end
    return e
end

# Evaluate g(K) = (Π_i (1 + K k_i) - 1 - K) / K = (Σ k_i - 1) + Σ_{m=2}^n e_m K^{m-1}.
function _eval_interaction_poly(c0::Float64, e::Vector{Float64}, K::Float64)
    acc = e[end]
    for m in (length(e) - 2):-1:2
        acc = acc * K + e[m + 1]
    end
    return c0 + K * acc
end

# The interaction constant K of the multiplicative form: the root of
# 1 + K = Π (1 + K k_i) other than zero, in (−1, 0) when Σ k_i > 1 and positive when
# Σ k_i < 1, by dividing out the trivial root K = 0.
function _interaction_constant(weights)
    total = sum(weights)
    isone(total) && return zero(total)
    k = Float64.(collect(weights))
    length(k) == 2 && return (1.0 - k[1] - k[2]) / (k[1] * k[2])
    e = _elementary_symmetric(k)
    c0 = sum(k) - 1.0
    if total > 1
        lo, hi = -1.0, 0.0
        _eval_interaction_poly(c0, e, lo) >= 0 && return -1.0
    else
        lo, hi = 0.0, 1.0
        while _eval_interaction_poly(c0, e, hi) < 0 && isfinite(hi)
            hi *= 2
        end
    end
    for _ in 1:200
        mid = (lo + hi) / 2
        (mid == lo || mid == hi) && break
        if _eval_interaction_poly(c0, e, mid) < 0
            lo = mid
        else
            hi = mid
        end
    end
    return (lo + hi) / 2
end

"""
    MultiplicativeUtility(utility => weight, ...)
    MultiplicativeUtility(utilities; weights)

The **multiplicative multi-attribute utility** of Keeney and Raiffa (1976) over
named outcomes of one run:

    1 + K U(x) = Π_i (1 + K k_i u_i(x_i)),

with every attribute utility `u_i` scaled to `[0, 1]` (give `worst` and `best`), and
weights `0 < k_i < 1`. `k_i` is the utility of attribute `i` at its best with all the
others at their worst. The interaction constant `K > −1` is the nonzero root of
`1 + K = Π_i (1 + K k_i)`, found by bisection and stored in the field
`interaction`; `U` lies in `[0, 1]`.

- `Σ k_i = 1`: `K = 0` and the form reduces to the [`AdditiveUtility`](@ref).
- `Σ k_i > 1`: `−1 < K < 0`, the attributes are substitutes (doing well on one makes
  up for another);
- `Σ k_i < 1`: `K > 0`, the attributes are complements (doing well on all of them
  together is worth more than the sum).

The form assumes mutual utility independence of the attributes, which is weaker than
the additive independence of the additive form. An attribute utility outside
`[0, 1]` in a run (an outcome beyond `worst` or `best`) throws
[`InvalidOutcomeError`](@ref). Used with [`payoff_table`](@ref) it is applied to each
run before averaging; **expected utility is not a count of individuals**, and there
is no [`certainty_equivalent`](@ref).

```jldoctest
julia> u = MultiplicativeUtility(LinearUtility(:uninfected; worst=0, best=50) => 0.6,
                                 LinearUtility(:infected; worst=50, best=0) => 0.6);

julia> round(u.interaction; digits=4)
-0.5556

julia> round(u((; uninfected=50, infected=50)); digits=4)   # one attribute at its best
0.6

julia> round(u((; uninfected=50, infected=0)); digits=4)    # both
1.0
```
"""
struct MultiplicativeUtility{U<:Tuple,W<:Tuple,K<:Real} <: AbstractUtility
    utilities::U
    weights::W
    interaction::K
end
function MultiplicativeUtility(pairs::Pair...)
    utilities, weights = _attribute_pairs(pairs)
    length(utilities) >= 2 ||
        throw(ArgumentError("a multiplicative utility needs at least two attributes"))
    all(k -> 0 < k < 1, weights) ||
        throw(ArgumentError("the weights of a multiplicative utility must lie strictly " *
                            "between 0 and 1, got $(collect(weights))"))
    for u in utilities
        u isa _AttributeUtility && u.worst === nothing &&
            throw(ArgumentError("the utility of $(u.name) is not scaled to [0, 1]; " *
                                "give it `worst` and `best`"))
    end
    return MultiplicativeUtility(utilities, weights, _interaction_constant(weights))
end
function MultiplicativeUtility(utilities::Union{Tuple,AbstractVector}; weights)
    length(utilities) == length(weights) ||
        throw(ArgumentError("$(length(weights)) weights for $(length(utilities)) " *
                            "attributes"))
    return MultiplicativeUtility((u => k for (u, k) in zip(utilities, weights))...)
end
function (u::MultiplicativeUtility)(o::NamedTuple)
    values = map(u.utilities) do c
        v = c(o)
        -1e-12 <= v <= 1 + 1e-12 ||
            throw(InvalidOutcomeError("an attribute utility is $v, outside [0, 1]: an " *
                                      "outcome lies beyond the `worst` or `best` of its " *
                                      "utility"))
        return v
    end
    K = u.interaction
    iszero(K) && return sum(k * v for (k, v) in zip(u.weights, values))
    return (prod(1 + K * k * v for (k, v) in zip(u.weights, values)) - 1) / K
end
