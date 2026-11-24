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
        sd > 0 || throw(ArgumentError("sd must be positive, got $sd"))
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

_binomial(rng::AbstractRNG, n::Int, p::Float64) = count(_ -> rand(rng) < p, 1:n)

# Gamma(shape, 1) by Marsaglia and Tsang (2000), with the boost for shape < 1.
function _gamma(rng::AbstractRNG, shape::Float64)
    shape < 1 && return _gamma(rng, shape + 1) * rand(rng)^(1 / shape)
    d = shape - 1 / 3
    c = 1 / sqrt(9d)
    while true
        z = randn(rng)
        v = (1 + c * z)^3
        v > 0 || continue
        log(rand(rng)) < z^2 / 2 + d - d * v + d * log(v) && return d * v
    end
end

function _beta(rng::AbstractRNG, a::Real, b::Real)
    (a > 0 && b > 0) ||
        throw(ArgumentError("Beta parameters must be positive, got ($a, $b)"))
    x = _gamma(rng, Float64(a))
    return x / (x + _gamma(rng, Float64(b)))
end

"""
    simulate_study(rng, design::AbstractStudyDesign, θ) -> NamedTuple

One data set of the study under the parameters `θ` (a named tuple, one row of the
draws), as its sufficient statistics: `(; x1)` for a [`BinomialStudy`](@ref),
`(; x1, x2)` for a [`TwoArmBinomialStudy`](@ref), `(; xbar)` for a
[`NormalStudy`](@ref). These are also the summary statistics that the regression
method regresses on.
"""
simulate_study(rng::AbstractRNG, s::BinomialStudy, θ) =
    (; x1=_binomial(rng, s.n, _probability(θ, s.parameter)))
simulate_study(rng::AbstractRNG, s::TwoArmBinomialStudy, θ) =
    (; x1=_binomial(rng, s.n, _probability(θ, s.parameters[1])),
     x2=_binomial(rng, s.n, _probability(θ, s.parameters[2])))
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
study_loglikelihood(s::BinomialStudy, data, θ) =
    _binomial_loglikelihood(data.x1, s.n, _probability(θ, s.parameter))
study_loglikelihood(s::TwoArmBinomialStudy, data, θ) =
    _binomial_loglikelihood(data.x1, s.n, _probability(θ, s.parameters[1])) +
    _binomial_loglikelihood(data.x2, s.n, _probability(θ, s.parameters[2]))
study_loglikelihood(s::NormalStudy, data, θ) =
    -s.n * (data.xbar - Float64(getproperty(θ, s.parameter)))^2 / (2 * s.sd^2)

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
function study_posterior(rng::AbstractRNG, s::BinomialStudy, data, prior; draws::Integer=1000)
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
