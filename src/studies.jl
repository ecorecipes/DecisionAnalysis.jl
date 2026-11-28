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

"""
    CountStudy(parameter; n, dispersion=nothing)

A study of `n` independent count observations (transects, quadrats, traps, host
examinations) with mean count given by `parameter` (a nonnegative column of the draws).

- `dispersion = nothing` (default): `n` i.i.d. `Poisson(λ)` counts. The total count
  `total ~ Poisson(n λ)` is a sufficient statistic and has a conjugate `Gamma(shape, rate)`
  analysis (`prior = (; shape, rate)`).
- `dispersion = k` (`k > 0`): `n` i.i.d. overdispersed Negative Binomial counts with
  mean `λ` and variance `λ + λ² / k` (size parameter `k`). The total count
  `total ~ NegativeBinomial(mean = n λ, size = n k)` is a sufficient statistic for
  [`SummaryStatistics`](@ref) and [`ImportanceSampling`](@ref).
"""
struct CountStudy <: AbstractStudyDesign
    parameter::Symbol
    n::Int
    dispersion::Union{Nothing,Float64}
    function CountStudy(parameter; n::Integer, dispersion::Union{Nothing,Real}=nothing)
        n >= 1 || throw(ArgumentError("the sample size n must be at least 1, got $n"))
        if dispersion !== nothing
            (isfinite(dispersion) && dispersion > 0) ||
                throw(ArgumentError("dispersion must be positive and finite, got $dispersion"))
        end
        return new(Symbol(parameter), Int(n),
                   dispersion === nothing ? nothing : Float64(dispersion))
    end
end

"""
    OccupancyStudy(occupancy, detection; sites, visits)

An imperfect-detection occupancy survey (MacKenzie et al. 2002, *Ecology* 83:2248):
`sites` independent sites, each visited `visits` times. A site is occupied with
probability `ψ = θ[occupancy]` and, when occupied, the species or pathogen is detected on
each visit independently with probability `p = θ[detection]`.

The pair `(; sites_detected, total_detections)` is a two-dimensional exact sufficient
statistic for any `sites` and `visits`, so the design works directly with both
[`SummaryStatistics`](@ref) and [`ImportanceSampling`](@ref) to trade off surveying more
sites against more repeat visits per site.
"""
struct OccupancyStudy <: AbstractStudyDesign
    occupancy::Symbol
    detection::Symbol
    sites::Int
    visits::Int
    function OccupancyStudy(occupancy, detection; sites::Integer, visits::Integer)
        sites >= 1 || throw(ArgumentError("sites must be at least 1, got $sites"))
        visits >= 1 || throw(ArgumentError("visits must be at least 1, got $visits"))
        Symbol(occupancy) == Symbol(detection) &&
            throw(ArgumentError("occupancy and detection must name different parameters"))
        return new(Symbol(occupancy), Symbol(detection), Int(sites), Int(visits))
    end
end

"""
    CaptureRecaptureStudy(abundance, detection; occasions=2)

A closed-population capture-recapture survey (Otis et al. 1978 model `M₀`; Lincoln-Petersen
when `occasions = 2`) over `occasions ≥ 2` sampling occasions. `abundance` names the
nonnegative population size `N` in the draws; `detection` either names the per-occasion
capture probability `p` in the draws or fixes a known probability `0 ≤ p ≤ 1`.

Returns the exact sufficient statistics `(; unique_captured, total_captures)` (with
recaptures equal to `total_captures - unique_captured` for `occasions = 2`), supporting
[`SummaryStatistics`](@ref) and [`ImportanceSampling`](@ref).
"""
struct CaptureRecaptureStudy{D<:Union{Symbol,Float64}} <: AbstractStudyDesign
    abundance::Symbol
    detection::D
    occasions::Int
    function CaptureRecaptureStudy(abundance, detection; occasions::Integer=2)
        occasions >= 2 ||
            throw(ArgumentError("occasions must be at least 2, got $occasions"))
        if detection isa Real
            (isfinite(detection) && 0 <= detection <= 1) ||
                throw(ArgumentError("fixed detection probability must lie in [0, 1], got " *
                                    "$detection"))
            return new{Float64}(Symbol(abundance), Float64(detection), Int(occasions))
        end
        det = Symbol(detection)
        det == Symbol(abundance) &&
            throw(ArgumentError("abundance and detection must name different parameters"))
        return new{Symbol}(Symbol(abundance), det, Int(occasions))
    end
end

"""
    MvNormalStudy(parameters; n, covariance)

A study of `n` independent multivariate normal measurements about the vector of focal
`parameters`, with known positive-definite observation `covariance` matrix `Σ`. The
sufficient statistic is the sample mean vector `(; xbar_<p1>, xbar_<p2>, ...)`, normal
with covariance `Σ / n`. Its conjugate analysis takes `prior = (; mean, covariance)`.
"""
struct MvNormalStudy <: AbstractStudyDesign
    parameters::Vector{Symbol}
    n::Int
    covariance::Matrix{Float64}
    chol_L::Matrix{Float64}
    precision::Matrix{Float64}
    function MvNormalStudy(parameters; n::Integer, covariance::AbstractMatrix{<:Real})
        n >= 1 || throw(ArgumentError("the sample size n must be at least 1, got $n"))
        pars = _labels(parameters)
        d = length(pars)
        d >= 1 || throw(ArgumentError("name at least one parameter"))
        allunique(pars) || throw(ArgumentError("parameters must be distinct, got $pars"))
        size(covariance) == (d, d) ||
            throw(ArgumentError("covariance must be $(d)×$(d), got $(size(covariance))"))
        all(isfinite, covariance) ||
            throw(ArgumentError("covariance entries must be finite"))
        Σ = Matrix{Float64}(covariance)
        Σ = (Σ .+ Σ') ./ 2
        F = cholesky(Symmetric(Σ); check=false)
        issuccess(F) ||
            throw(ArgumentError("covariance matrix must be symmetric positive definite"))
        return new(pars, Int(n), Σ, Matrix(F.L), inv(F))
    end
end

_study_parameters(s::Union{BinomialStudy,NormalStudy,CountStudy}) = [s.parameter]
_study_parameters(s::TwoArmBinomialStudy) = collect(s.parameters)
_study_parameters(s::OccupancyStudy) = [s.occupancy, s.detection]
function _study_parameters(s::CaptureRecaptureStudy)
    return s.detection isa Symbol ? [s.abundance, s.detection] : [s.abundance]
end
_study_parameters(s::MvNormalStudy) = copy(s.parameters)

function _nonnegative_rate(θ, name::Symbol)
    haskey(θ, name) || throw(ArgumentError("the draw has no parameter $name"))
    v = getproperty(θ, name)
    (v isa Real && isfinite(v) && v >= 0) ||
        throw(ArgumentError("the parameter $name must be finite and nonnegative, got " *
                            "$(repr(v))"))
    return Float64(v)
end

# Exact O(1)-expected-time Poisson sampler: Knuth for μ < 30 and Hörmann (1993) PTRS
# transformed rejection with squeeze for μ >= 30.
function _poisson(rng::AbstractRNG, μ::Float64)
    μ <= 0.0 && return 0
    if μ < 30.0
        L = exp(-μ)
        k = 0
        p = 1.0
        while p > L
            k += 1
            p *= rand(rng)
        end
        return k - 1
    end
    s = sqrt(μ)
    b = 0.931 + 2.53 * s
    a = -0.059 + 0.02483 * b
    inv_alpha = 1.1239 + 1.1328 / (b - 3.4)
    v_r = 0.9277 - 3.6224 / (b - 2.0)
    log_mu = log(μ)
    while true
        u = rand(rng) - 0.5
        v = rand(rng)
        us = 0.5 - abs(u)
        k = floor(Int, (2.0 * a / us + b) * u + μ + 0.43)
        k < 0 && continue
        (us >= 0.07 && v <= v_r) && return k
        (us < 0.013 && v > us) && continue
        v <= 0.0 && return k
        log(v * inv_alpha / (a / (us * us) + b)) <= -μ + k * log_mu - loggamma(k + 1.0) &&
            return k
    end
end

# Exact zero-truncated Binomial(K, p) sampler given K >= 1, 0 < p <= 1.
function _zt_binomial(rng::AbstractRNG, K::Int, p::Float64)
    K == 1 && return 1
    p >= 1.0 && return K
    if K > 50 && K * p >= 1.0
        while true
            y = _binomial(rng, K, p)
            y > 0 && return y
        end
    end
    p_any = -expm1(K * log1p(-p))
    u = rand(rng) * p_any
    y = 1
    py = exp(log(K) + log(p) + (K - 1) * log1p(-p))
    s = p / (1.0 - p)
    while u > py && y < K && py > 0.0
        u -= py
        y += 1
        py *= ((K - y + 1) / y) * s
    end
    return y
end

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
function simulate_study(rng::AbstractRNG, s::CountStudy, θ)
    λ = _nonnegative_rate(θ, s.parameter)
    if s.dispersion === nothing
        return (; total=_poisson(rng, s.n * λ))
    end
    k = s.dispersion
    rate_mix = iszero(λ) ? 0.0 : _gamma(rng, s.n * k) * (λ / k)
    return (; total=_poisson(rng, rate_mix))
end
function simulate_study(rng::AbstractRNG, s::OccupancyStudy, θ)
    ψ = _probability(θ, s.occupancy)
    p = _probability(θ, s.detection)
    s_occ = _binomial(rng, s.sites, ψ)
    (s_occ == 0 || p == 0.0) && return (; sites_detected=0, total_detections=0)
    p >= 1.0 && return (; sites_detected=s_occ, total_detections=s.visits * s_occ)
    p_any = -expm1(s.visits * log1p(-p))
    s_det = _binomial(rng, s_occ, p_any)
    d_tot = 0
    for _ in 1:s_det
        d_tot += _zt_binomial(rng, s.visits, p)
    end
    return (; sites_detected=s_det, total_detections=d_tot)
end
function simulate_study(rng::AbstractRNG, s::CaptureRecaptureStudy, θ)
    N = round(Int, _nonnegative_rate(θ, s.abundance))
    p = s.detection isa Symbol ? _probability(θ, s.detection) : s.detection
    (N == 0 || p == 0.0) && return (; unique_captured=0, total_captures=0)
    p >= 1.0 && return (; unique_captured=N, total_captures=s.occasions * N)
    p_any = -expm1(s.occasions * log1p(-p))
    u = _binomial(rng, N, p_any)
    c = 0
    for _ in 1:u
        c += _zt_binomial(rng, s.occasions, p)
    end
    return (; unique_captured=u, total_captures=c)
end
function simulate_study(rng::AbstractRNG, s::MvNormalStudy, θ)
    d = length(s.parameters)
    μ = Vector{Float64}(undef, d)
    for (i, name) in enumerate(s.parameters)
        haskey(θ, name) || throw(ArgumentError("the draw has no parameter $name"))
        μ[i] = Float64(getproperty(θ, name))
    end
    z = randn(rng, d)
    xbar = μ .+ (s.chol_L * z) ./ sqrt(s.n)
    names = Tuple(Symbol(:xbar_, name) for name in s.parameters)
    return NamedTuple{names}(Tuple(xbar))
end

_xlogy(x::Real, y::Real) = iszero(x) ? 0.0 : x * log(y)
_binomial_loglikelihood(x::Real, n::Int, p::Float64) = _xlogy(x, p) + _xlogy(n - x, 1 - p)

"""
    study_loglikelihood(design::AbstractStudyDesign, data, θ) -> Float64

The log-likelihood of a data set of [`simulate_study`](@ref) under the parameters `θ`,
up to a constant that does not depend on `θ` (combinatorial constants independent of `θ`
are left out).
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
function study_loglikelihood(s::CountStudy, data, θ)
    λ = _nonnegative_rate(θ, s.parameter)
    tot = Float64(data.total)
    if s.dispersion === nothing
        return _xlogy(tot, λ) - s.n * λ
    end
    k = s.dispersion
    r = s.n * k
    return _xlogy(tot, λ) - (tot + r) * log(k + λ)
end
function study_loglikelihood(s::OccupancyStudy, data, θ)
    ψ = _probability(θ, s.occupancy)
    p = _probability(θ, s.detection)
    s_det = Int(data.sites_detected)
    d_tot = Int(data.total_detections)
    s_zero = s.sites - s_det
    p_zero = (1.0 - ψ) + ψ * (1.0 - p)^s.visits
    return _xlogy(s_zero, p_zero) + _xlogy(s_det, ψ) + _xlogy(d_tot, p) +
           _xlogy(s.visits * s_det - d_tot, 1.0 - p)
end
function study_loglikelihood(s::CaptureRecaptureStudy, data, θ)
    N = _nonnegative_rate(θ, s.abundance)
    p = s.detection isa Symbol ? _probability(θ, s.detection) : s.detection
    u = Int(data.unique_captured)
    c = Int(data.total_captures)
    N < u && return -Inf
    misses = s.occasions * N - c
    misses < 0 && return -Inf
    return loggamma(N + 1.0) - loggamma(N - u + 1.0) + _xlogy(c, p) +
           _xlogy(misses, 1.0 - p)
end
function study_loglikelihood(s::MvNormalStudy, data, θ)
    d = length(s.parameters)
    diff = Vector{Float64}(undef, d)
    for (i, name) in enumerate(s.parameters)
        haskey(θ, name) || throw(ArgumentError("the draw has no parameter $name"))
        xbi = Float64(getproperty(data, Symbol(:xbar_, name)))
        diff[i] = xbi - Float64(getproperty(θ, name))
    end
    return -0.5 * s.n * dot(diff, s.precision * diff)
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
function study_posterior(rng::AbstractRNG, s::CountStudy, data, prior; draws::Integer=1000)
    s.dispersion === nothing ||
        throw(ArgumentError("CountStudy with Negative Binomial overdispersion has no " *
                            "closed-form conjugate posterior; use SummaryStatistics, " *
                            "ImportanceSampling, or a custom posterior in MomentMatching"))
    p = _prior_fields(prior, (:shape, :rate), s)
    (isfinite(p.shape) && p.shape > 0 && isfinite(p.rate) && p.rate > 0) ||
        throw(ArgumentError("Gamma prior shape and rate must be positive and finite"))
    a_post = Float64(p.shape) + Float64(data.total)
    b_post = Float64(p.rate) + s.n
    return NamedTuple{(s.parameter,)}(([_gamma(rng, a_post) / b_post for _ in 1:draws],))
end
function study_posterior(rng::AbstractRNG, s::MvNormalStudy, data, prior;
                         draws::Integer=1000)
    p = _prior_fields(prior, (:mean, :covariance), s)
    d = length(s.parameters)
    μ0 = Float64.(collect(p.mean))
    length(μ0) == d ||
        throw(ArgumentError("prior mean must have length $d, got $(length(μ0))"))
    Σ0 = Matrix{Float64}(p.covariance)
    size(Σ0) == (d, d) ||
        throw(ArgumentError("prior covariance must be $(d)×$(d), got $(size(Σ0))"))
    F0 = cholesky(Symmetric((Σ0 .+ Σ0') ./ 2); check=false)
    issuccess(F0) || throw(ArgumentError("prior covariance must be positive definite"))
    Λ0 = inv(F0)
    xbar = Float64[Float64(getproperty(data, Symbol(:xbar_, name)))
                   for name in s.parameters]
    Λ_post = Λ0 .+ s.n .* s.precision
    Fp = cholesky(Symmetric((Λ_post .+ Λ_post') ./ 2))
    Σ_post = inv(Fp)
    μ_post = Σ_post * (Λ0 * μ0 .+ s.n .* (s.precision * xbar))
    Lp = Matrix(cholesky(Symmetric((Σ_post .+ Σ_post') ./ 2)).L)
    Z = randn(rng, d, draws)
    samples = μ_post .+ Lp * Z
    return NamedTuple{Tuple(s.parameters)}(Tuple(vec(samples[i, :]) for i in 1:d))
end
function study_posterior(::AbstractRNG, s::Union{OccupancyStudy,CaptureRecaptureStudy},
                         data, prior; draws::Integer=1000)
    return throw(ArgumentError("$(nameof(typeof(s))) has no closed-form conjugate " *
                               "posterior; use SummaryStatistics, ImportanceSampling, or " *
                               "a custom posterior in MomentMatching"))
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
