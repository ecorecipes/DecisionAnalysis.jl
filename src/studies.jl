# Standard study designs: what is observed, how to simulate it, its likelihood and its
# conjugate analysis. One description serves the three sample-based routes to EVSI
# (summary statistics, importance sampling, moment matching). These are the built-in
# studies of the R package `voi` ("binary", "trial_binary", "normal_known").

"""
    AbstractStudyDesign

Supertype of the built-in study designs: [`BinomialStudy`](@ref),
[`TwoArmBinomialStudy`](@ref) and [`NormalStudy`](@ref). A design names the parameters
of the draws that it observes and has methods of [`simulate_study`](@ref),
[`study_loglikelihood`](@ref) and [`study_posterior`](@ref). It is turned into a study
for [`evsi`](@ref) by `SummaryStatistics(belief, design; seed)`,
`ImportanceSampling(belief, design; seed)` or `MomentMatching(design; payoffs, prior)`.
A design of one's own subtypes it and gives the three functions methods, or is passed
to those studies as plain functions.
"""
abstract type AbstractStudyDesign end

"""
    BinomialStudy(parameter; n)

A study that observes a yes-or-no outcome on `n` individuals: the count `x1` of
successes is binomial with probability the value of `parameter` (a column of the
draws, a probability). In ecology: animals found infected among `n` examined, marked
animals alive at the next visit among `n` marked, sites occupied among `n` surveyed
(when detection is perfect). Its conjugate analysis has a `Beta(a, b)` prior, given as
`prior = (; a, b)`.
"""
struct BinomialStudy <: AbstractStudyDesign
    parameter::Symbol
    n::Int
    function BinomialStudy(parameter; n::Integer)
        n >= 1 || throw(ArgumentError("the sample size n must be at least 1, got $n"))
        return new(Symbol(parameter), n)
    end
end

"""
    TwoArmBinomialStudy(first, second; n)

A two-arm study with a yes-or-no outcome, `n` individuals **in each arm**: the counts
`x1` and `x2` are binomial with the probabilities named by `first` and `second`. An
experiment with treated and control ponds, or a comparison of survival in two
habitats. Its conjugate analysis has independent Beta priors,
`prior = (; a1, b1, a2, b2)`.
"""
struct TwoArmBinomialStudy <: AbstractStudyDesign
    parameters::NTuple{2,Symbol}
    n::Int
    function TwoArmBinomialStudy(first, second; n::Integer)
        n >= 1 || throw(ArgumentError("the sample size n must be at least 1, got $n"))
        Symbol(first) == Symbol(second) &&
            throw(ArgumentError("the two arms must have different parameters"))
        return new((Symbol(first), Symbol(second)), n)
    end
end

"""
    NormalStudy(parameter; n, sd=1)

A study of `n` measurements, normal about the value of `parameter` with a **known**
standard deviation `sd`: the data reduce to their mean `xbar`, normal with standard
deviation `sd / √n`. Its conjugate analysis has a normal prior,
`prior = (; mean, sd)`, worth `(sd of the data / sd of the prior)²` observations.
"""
struct NormalStudy <: AbstractStudyDesign
    parameter::Symbol
    n::Int
    sd::Float64
    function NormalStudy(parameter; n::Integer, sd::Real=1)
        n >= 1 || throw(ArgumentError("the sample size n must be at least 1, got $n"))
        (isfinite(sd) && sd > 0) ||
            throw(ArgumentError("sd must be positive and finite, got $sd"))
        return new(Symbol(parameter), n, Float64(sd))
    end
end

_study_parameters(s::Union{BinomialStudy,NormalStudy}) = [s.parameter]
_study_parameters(s::TwoArmBinomialStudy) = collect(s.parameters)

function _probability(θ, name::Symbol)
    haskey(θ, name) || throw(ArgumentError("the draw has no parameter $name"))
    p = getproperty(θ, name)
    (p isa Real && 0 <= p <= 1) ||
        throw(ArgumentError("the parameter $name must be a probability, got $(repr(p))"))
    return Float64(p)
end

# Exact Binomial(n, p) sampler in O(1) expected time: inverse-CDF (BINV) when
# min(np, n(1-p)) < 10 and Kachitvichyanukul & Schmeiser (1988) BTPE otherwise.
function _binomial(rng::AbstractRNG, n::Int, p::Float64)
    (n <= 0 || p <= 0.0) && return 0
    p >= 1.0 && return n
    q = p <= 0.5 ? p : 1.0 - p
    r = 1.0 - q
    np = n * q
    if np < 10.0
        s = q / r
        a = (n + 1) * s
        pk = exp(n * log1p(-q))
        u = rand(rng)
        y = 0
        while u > pk && y < n && pk > 0.0
            u -= pk
            y += 1
            pk *= (a / y - s)
        end
        return p <= 0.5 ? y : n - y
    end
    # BTPE (Kachitvichyanukul & Schmeiser 1988, ACM TOMS 14:216)
    fm = np + q
    m = floor(Int, fm)
    npq = np * r
    p1 = floor(2.195 * sqrt(npq) - 4.6 * r) + 0.5
    xm = m + 0.5
    xl = xm - p1
    xr = xm + p1
    c = 0.134 + 20.5 / (15.3 + m)
    al = (fm - xl) / (fm - xl * q)
    laml = al * (1.0 + 0.5 * al)
    ar = (xr - fm) / (xr * r)
    lamr = ar * (1.0 + 0.5 * ar)
    p2 = p1 * (1.0 + 2.0 * c)
    p3 = p2 + c / laml
    p4 = p3 + c / lamr
    s = q / r
    a_step = (n + 1) * s
    log_s = log(s)
    lg_mode = loggamma(m + 1.0) + loggamma(n - m + 1.0)
    k_limit = floor(Int, npq / 2.0) - 1
    while true
        u = rand(rng) * p4
        v = rand(rng)
        if u <= p1
            y = floor(Int, xm - p1 * v + u)
            return p <= 0.5 ? y : n - y
        elseif u <= p2
            x = xl + (u - p1) / c
            v = v * c + 1.0 - abs(m - x + 0.5) / p1
            v > 1.0 && continue
            y = floor(Int, x)
        elseif u <= p3
            y = floor(Int, xl + log(v) / laml)
            y < 0 && continue
            v *= (u - p2) * laml
        else
            y = floor(Int, xr - log(v) / lamr)
            y > n && continue
            v *= (u - p3) * lamr
        end
        k = abs(y - m)
        if k <= 20 || k >= k_limit
            f = 1.0
            if m < y
                for i in (m + 1):y
                    f *= (a_step / i - s)
                end
            elseif m > y
                for i in (y + 1):m
                    f /= (a_step / i - s)
                end
            end
            v <= f && return p <= 0.5 ? y : n - y
        else
            v <= 0.0 && return p <= 0.5 ? y : n - y
            lv = log(v)
            rho = (k / npq) * ((k * (k / 3.0 + 0.625) + 1.0 / 6.0) / npq + 0.5)
            t = -k * k / (2.0 * npq)
            lv < t - rho && return p <= 0.5 ? y : n - y
            lv > t + rho && continue
            log_ratio = lg_mode - loggamma(y + 1.0) - loggamma(n - y + 1.0) +
                        (y - m) * log_s
            lv <= log_ratio && return p <= 0.5 ? y : n - y
        end
    end
end

# log(Gamma(shape, 1)) by Marsaglia and Tsang (2000), sampling in log space for shape < 1
# so small shapes never underflow to 0 / 0 = NaN in `_beta`.
function _log_gamma(rng::AbstractRNG, shape::Float64)
    if shape < 1
        u = rand(rng)
        while iszero(u)
            u = rand(rng)
        end
        return _log_gamma(rng, shape + 1.0) + log(u) / shape
    end
    d = shape - 1 / 3
    c = 1 / sqrt(9d)
    while true
        z = randn(rng)
        v = (1 + c * z)^3
        v > 0 || continue
        u = rand(rng)
        u > 0 || continue
        log(u) < z^2 / 2 + d - d * v + d * log(v) && return log(d * v)
    end
end

_gamma(rng::AbstractRNG, shape::Float64) = exp(_log_gamma(rng, shape))

function _beta(rng::AbstractRNG, a::Real, b::Real)
    (isfinite(a) && isfinite(b) && a > 0 && b > 0) ||
        throw(ArgumentError("Beta parameters must be positive and finite, got ($a, $b)"))
    lx = _log_gamma(rng, Float64(a))
    ly = _log_gamma(rng, Float64(b))
    m = max(lx, ly)
    return exp(lx - (m + log(exp(lx - m) + exp(ly - m))))
end

"""
    simulate_study(rng, design::AbstractStudyDesign, θ) -> NamedTuple

One data set of the study under the parameters `θ` (a named tuple, one row of the
draws), as its sufficient statistics: `(; x1)` for a [`BinomialStudy`](@ref),
`(; x1, x2)` for a [`TwoArmBinomialStudy`](@ref), `(; xbar)` for a
[`NormalStudy`](@ref). These are also the summary statistics that the regression
method regresses on.
"""
function simulate_study(rng::AbstractRNG, s::BinomialStudy, θ)
    return (;
            x1=_binomial(rng, s.n,
                         _probability(θ,
                                      s.parameter)))
end
function simulate_study(rng::AbstractRNG, s::TwoArmBinomialStudy, θ)
    return (; x1=_binomial(rng, s.n, _probability(θ, s.parameters[1])),
            x2=_binomial(rng, s.n, _probability(θ, s.parameters[2])))
end
function simulate_study(rng::AbstractRNG, s::NormalStudy, θ)
    haskey(θ, s.parameter) ||
        throw(ArgumentError("the draw has no parameter $(s.parameter)"))
    return (; xbar=Float64(getproperty(θ, s.parameter)) + s.sd / sqrt(s.n) * randn(rng))
end

_xlogy(x::Real, y::Real) = iszero(x) ? 0.0 : x * log(y)
_binomial_loglikelihood(x::Real, n::Int, p::Float64) = _xlogy(x, p) + _xlogy(n - x, 1 - p)

"""
    study_loglikelihood(design::AbstractStudyDesign, data, θ) -> Float64

The log-likelihood of a data set of [`simulate_study`](@ref) under the parameters `θ`,
up to a constant that does not depend on `θ` (the binomial coefficient is left out).
"""
function study_loglikelihood(s::BinomialStudy, data, θ)
    return _binomial_loglikelihood(data.x1, s.n,
                                   _probability(θ,
                                                s.parameter))
end
function study_loglikelihood(s::TwoArmBinomialStudy, data, θ)
    return _binomial_loglikelihood(data.x1, s.n, _probability(θ, s.parameters[1])) +
           _binomial_loglikelihood(data.x2, s.n, _probability(θ, s.parameters[2]))
end
function study_loglikelihood(s::NormalStudy, data, θ)
    return -s.n * (data.xbar - Float64(getproperty(θ, s.parameter)))^2 / (2 * s.sd^2)
end

function _prior_fields(prior, names, design)
    (prior isa NamedTuple && all(k -> haskey(prior, k), names)) ||
        throw(ArgumentError("the conjugate analysis of a $(nameof(typeof(design))) " *
                            "needs `prior = (; $(join(names, ", ")))`"))
    return prior
end

"""
    study_posterior(rng, design::AbstractStudyDesign, data, prior; draws=1000) -> NamedTuple

A sample of `draws` values from the posterior of the design's parameters given one
data set, by its conjugate analysis: a named tuple of vectors, one per parameter.
`prior` is `(; a, b)` for a [`BinomialStudy`](@ref) (a Beta prior),
`(; a1, b1, a2, b2)` for a [`TwoArmBinomialStudy`](@ref), and `(; mean, sd)` for a
[`NormalStudy`](@ref). The prior must be the distribution the draws of the parameter
came from, or [`evsi`](@ref) by moment matching values a study against one belief and
analyses it with another.
"""
function study_posterior(rng::AbstractRNG, s::BinomialStudy, data, prior;
                         draws::Integer=1000)
    p = _prior_fields(prior, (:a, :b), s)
    return NamedTuple{(s.parameter,)}(([_beta(rng, p.a + data.x1, p.b + s.n - data.x1)
                                        for _ in 1:draws],))
end
function study_posterior(rng::AbstractRNG, s::TwoArmBinomialStudy, data, prior;
                         draws::Integer=1000)
    p = _prior_fields(prior, (:a1, :b1, :a2, :b2), s)
    return NamedTuple{s.parameters}(([_beta(rng, p.a1 + data.x1, p.b1 + s.n - data.x1)
                                      for _ in 1:draws],
                                     [_beta(rng, p.a2 + data.x2, p.b2 + s.n - data.x2)
                                      for _ in 1:draws]))
end
function study_posterior(rng::AbstractRNG, s::NormalStudy, data, prior; draws::Integer=1000)
    p = _prior_fields(prior, (:mean, :sd), s)
    p.sd > 0 || throw(ArgumentError("the prior sd must be positive, got $(p.sd)"))
    precision = 1 / p.sd^2 + s.n / s.sd^2
    centre = (p.mean / p.sd^2 + s.n * data.xbar / s.sd^2) / precision
    return NamedTuple{(s.parameter,)}((centre .+ randn(rng, draws) ./ sqrt(precision),))
end

_design_belief(b::SampleBelief) = b
_design_belief(p::DecisionProblem) = (_sample_problem(p, :evsi); p.belief)

"""
    SummaryStatistics(belief, design::AbstractStudyDesign; seed, regression=SplineRegression())

The regression study of a built-in design: one data set of the design is simulated
from every draw of `belief` (a [`SampleBelief`](@ref), or a sample problem), with a
`Xoshiro(seed)`, and its sufficient statistics are the summaries.
"""
function SummaryStatistics(belief::Union{SampleBelief,DecisionProblem},
                           design::AbstractStudyDesign;
                           seed::Union{Nothing,Integer}=nothing,
                           regression::AbstractRegression=SplineRegression())
    seed === nothing && throw(ArgumentError("give `seed`"))
    rng = Xoshiro(seed)
    rows = Tables.rowtable(_design_belief(belief).draws)
    return SummaryStatistics([simulate_study(rng, design, θ) for θ in rows]; regression)
end

"""
    ImportanceSampling(belief, design::AbstractStudyDesign; seed, datasets=nothing)

The importance-sampling study of a built-in design: data sets simulated from the
draws of `belief` and their log-likelihood under every draw, as
[`importance_loglikelihoods`](@ref) builds them.
"""
function ImportanceSampling(belief::Union{SampleBelief,DecisionProblem},
                            design::AbstractStudyDesign;
                            seed::Union{Nothing,Integer}=nothing,
                            datasets::Union{Nothing,Integer}=nothing)
    return importance_loglikelihoods((rng, θ) -> simulate_study(rng, design, θ),
                                     (data, θ) -> study_loglikelihood(design, data, θ),
                                     _design_belief(belief); datasets, seed)
end
