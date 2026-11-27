# Value of information that needs the model itself, not only a table of its runs: the
# two-level Monte Carlo estimate of EVPPI and the moment-matching estimate of EVSI.
# Both estimate the quantities defined at the top of `voi.jl`.

_payoff_vector(y::NamedTuple) = Float64[v for v in values(y)]
_payoff_vector(y::AbstractVector{<:Real}) = Float64.(y)
function _payoff_vector(y)
    return throw(ArgumentError("`payoffs(θ)` must return a vector or a named tuple of numbers, " *
                               "one per action; got $(typeof(y))"))
end

# E max(m + Z) − max(m) for Z ~ N(0, C), by simulation: by how much noise of covariance
# `C` in the means `m` inflates their maximum.
function _maximum_inflation(m::AbstractVector, C::AbstractMatrix, rng::AbstractRNG;
                            draws::Integer=64)
    length(m) == 1 && return 0.0
    root = _sqrt_psd(C)
    z = zeros(length(m))
    total = 0.0
    for _ in 1:draws
        randn!(rng, z)
        total += maximum(m .+ root * z)
    end
    return total / draws - maximum(m)
end

"""
    evppi_nested(payoffs, sample_focal, sample_rest; outer, inner, seed) -> VOIEstimate

The EVPPI of the focal parameters by **two-level (nested) Monte Carlo**, straight from
its definition: for each of `outer` values of the focal parameters, the expected
payoff of every action given them is the mean over `inner` values of the other
parameters, and

    EVPPI ≈ mean over outer of max_a (inner mean of a) − max_a (mean over all of a).

- `payoffs(θ)`: the payoffs of the actions at the parameters `θ`, a vector or a named
  tuple of numbers, in a fixed order. `θ` is the two samples merged into one named
  tuple.
- `sample_focal(rng)`: one value of the focal parameters, a named tuple.
- `sample_rest(rng, φ)`: one value of the other parameters **given** the focal ones
  `φ`, a named tuple. When the two groups are independent it ignores `φ`.

It needs `outer × inner` evaluations of the model and no regression, so it is the
reference to check a regression estimate against on a model that is cheap to run, and
too expensive otherwise ([`evppi`](@ref) with a method needs one sample). Every random
number comes from `Xoshiro(seed)`; `seed` is required.

The result is a [`VOIEstimate`](@ref). `standard_error` is that of the mean over the
outer values, with the inner means as they are. The estimate is **biased upwards for
any finite `inner`**: the maximum over actions of noisy inner means overstates the
maximum of the conditional expectations. `bias.cells` estimates that inflation from
the covariance of the inner means (the actions share their inner sample, so their
noise is correlated and inflates the maximum less than independent noise would), and
is reported, not removed; increase `inner` until it is small beside the estimate.
"""
function evppi_nested(payoffs, sample_focal, sample_rest; outer::Integer, inner::Integer,
                      seed::Union{Nothing,Integer}=nothing)
    seed === nothing && throw(ArgumentError("give `seed`"))
    outer >= 2 || throw(ArgumentError("outer must be at least 2, got $outer"))
    inner >= 2 || throw(ArgumentError("inner must be at least 2, got $inner"))
    rng = Xoshiro(seed)
    means = nothing
    inflation = 0.0
    for r in 1:outer
        φ = sample_focal(rng)
        φ isa NamedTuple ||
            throw(ArgumentError("`sample_focal(rng)` must return a named tuple, got " *
                                "$(typeof(φ))"))
        total = nothing
        products = nothing
        for _ in 1:inner
            rest = sample_rest(rng, φ)
            rest isa NamedTuple ||
                throw(ArgumentError("`sample_rest(rng, φ)` must return a named tuple, " *
                                    "got $(typeof(rest))"))
            y = _payoff_vector(payoffs(merge(rest, φ)))
            means === nothing && (means = zeros(length(y), outer))
            length(y) == size(means, 1) ||
                throw(ArgumentError("`payoffs` returned $(length(y)) payoffs in one " *
                                    "call and $(size(means, 1)) in another"))
            total === nothing &&
                ((total, products) = (zero(y), zeros(length(y), length(y))))
            total .+= y
            products .+= y .* y'
        end
        m = total ./ inner
        means[:, r] .= m
        # The inner means of the actions share their inner sample, so their noise is
        # correlated: the inflation of the maximum is simulated with their covariance.
        covariance = (products ./ inner .- m .* m') ./ (inner - 1)
        inflation += _maximum_inflation(m, covariance, rng)
    end
    w = fill(1 / outer, outer)
    v0, a = _act_now(means, w)
    gains = let means = means
        [maximum(view(means, :, r)) - means[a, r] for r in 1:outer]
    end
    value = sum(gains) / outer
    bias = (; act_now=_act_now_bias(means, w, rng), cells=max(inflation / outer, 0.0))
    details = (; act_now_value=v0, act_now_index=a, outer=Int(outer), inner=Int(inner),
               model_evaluations=Int(outer) * Int(inner))
    return VOIEstimate(:evppi, :NestedMonteCarlo, value, _weighted_mean_se(gains, w), bias,
                       details)
end

"""
    MomentMatching(focal; payoffs, simulate_data, posterior, nested=30, regression=nothing)
    MomentMatching(design::AbstractStudyDesign; payoffs, prior, nested=30,
                   posterior_draws=1000, regression=nothing)

A study valued by the **moment-matching** method of Heath, Manolopoulou and Baio
(2018, *Medical Decision Making* 38:163), for [`evsi`](@ref). It needs the model and a
Bayesian analysis of the study's data, and no summary statistic and no likelihood
over all the draws, so it serves a study whose analysis is a model fit (a
mark-recapture analysis, a hierarchical model).

The idea: the expected payoffs given the study's data are, to a good approximation,
the expected payoffs given the focal parameters (the fitted conditional means of
[`evppi`](@ref)) shrunk towards their mean, by as much as makes their variance equal to
the variance that the study removes, `Var(payoff) − E Var(payoff | data)`, estimated
by running the analysis at `nested` data sets only. Both variances are taken over
the same draws of the other parameters, so that their difference is not swamped by
the variance that the study cannot touch.

- `focal`: the parameters the study informs (columns of the draws).
- `payoffs(θ)`: the payoffs of the **feasible actions of the problem, in its order**, at
  a full set of parameters `θ` (a named tuple with every column of the draws). It
  must be the function the payoff table was made from; a table of noisy runs is not.
- `simulate_data(rng, θ)`: one data set of the study under `θ`.
- `posterior(rng, data)`: a sample from the posterior of the focal parameters given
  the data, a Tables.jl table with a column for each of them (any number of rows, a
  few hundred or more). The other parameters keep their prior: they are drawn from
  the draws at random, so they are taken to be independent of the focal ones.
- `nested`: the number of data sets analysed. Thirty to fifty are usually enough; the
  cost is `nested ×` (one analysis + the model twice at every posterior draw).
- `regression`: the method for the conditional means given the focal parameters, as
  for [`evppi`](@ref) (default: chosen by the focal parameters).

With a built-in design ([`BinomialStudy`](@ref), ...) the data, the analysis and the
focal parameters come from the design and its conjugate `prior`.

The nested data sets are simulated at draws stratified by the fitted advantage of
the most variable action (one draw at random in each of `nested` strata of equal
weight), so that they cover the range that matters and its tails. For several
actions the shrinkage is a matrix, which matches the covariance of the payoffs'
differences from the act-now action and not only their variances; of the matrices
that do, it is the one that moves the fitted values least (it is the identity when
the study removes all that the focal parameters explain). The standard error
combines the sampling of the draws with a jackknife over the nested data sets; the
approximation itself (payoffs given the data being a shrunk copy of payoffs given the
parameters) is a bias that no standard error covers. The method lives on variances:
when the payoffs are strongly skewed, or the focal parameters explain only a small
share of their variance, the variance removed is a small difference of two large and
noisy ones and needs many posterior draws and data sets (`details.variance_ratio`,
the share of the EVPPI variance removed, should lie between 0 and 1 and settle as
they grow); the regression on a summary statistic, where one exists, is then the
better estimator. If the estimated variance removed
exceeds what learning the focal parameters exactly would remove, it is capped there
(`details.capped`), since a study is worth no more than the EVPPI of what it informs.
"""
struct MomentMatching{P,S,A,M}
    focal::Vector{Symbol}
    payoffs::P
    simulate_data::S
    posterior::A
    nested::Int
    regression::M
    function MomentMatching(focal; payoffs, simulate_data, posterior, nested::Integer=30,
                            regression::Union{Nothing,AbstractVOIMethod}=nothing)
        nested >= 3 || throw(ArgumentError("nested must be at least 3, got $nested"))
        _require_fit(regression)
        names = _focal_names(focal)
        return new{typeof(payoffs),typeof(simulate_data),typeof(posterior),
                   typeof(regression)}(names, payoffs, simulate_data, posterior,
                                       Int(nested), regression)
    end
end
function MomentMatching(design::AbstractStudyDesign; payoffs, prior, nested::Integer=30,
                        posterior_draws::Integer=1000,
                        regression::Union{Nothing,AbstractVOIMethod}=nothing)
    posterior_draws >= 10 ||
        throw(ArgumentError("posterior_draws must be at least 10, got $posterior_draws"))
    return MomentMatching(_study_parameters(design); payoffs, nested, regression,
                          simulate_data=(rng, θ) -> simulate_study(rng, design, θ),
                          posterior=(rng, data) -> study_posterior(rng, design, data, prior;
                                                                   draws=posterior_draws))
end

# Symmetric square root, and the pseudo-inverse of one, of a covariance matrix.
function _sqrt_psd(C::AbstractMatrix; inverse::Bool=false)
    F = eigen(Symmetric((C .+ C') ./ 2))
    top = maximum(abs, F.values; init=0.0)
    scale = map(F.values) do λ
        λ > 1e-12 * top || return 0.0
        return inverse ? 1 / sqrt(λ) : sqrt(λ)
    end
    return F.vectors * (scale .* F.vectors')
end

function _weighted_covariance(D::AbstractMatrix, w::AbstractVector)
    centred = D .- D * w
    return (centred .* w') * centred'
end

# The estimate from the fitted contrasts `Gc` (actions × draws, differences from the
# act-now action), rescaled to the covariance `target`.
function _matched_value(Gc::AbstractMatrix, w::AbstractVector, target::AbstractMatrix,
                        fitted::AbstractMatrix)
    # The map between two covariances that moves the fitted values least: symmetric,
    # the identity when the two agree, and a plain scaling for one contrast.
    root, inverse_root = _sqrt_psd(fitted), _sqrt_psd(fitted; inverse=true)
    T = inverse_root * _sqrt_psd(root * target * root) * inverse_root
    centre = Gc * w
    matched = centre .+ T * (Gc .- centre)
    best = [maximum(view(matched, :, i)) for i in axes(matched, 2)]
    return dot(w, best) - maximum(centre), best
end

# The estimate for the covariance `target` that the study is expected to remove: the
# target made symmetric and positive semidefinite, capped at the fitted covariance
# (information cannot remove more than the EVPPI variance), and then matched. Returns
# the value, the best matched payoff of each draw, the variance removed, whether the
# cap applied, and the ratio before capping. A function of its own, not a closure: it is
# called again for each leave-one-out replicate, and must not write to the caller's
# variables.
function _matched_estimate(Gc::AbstractMatrix, w::AbstractVector, target::AbstractMatrix,
                           fitted::AbstractMatrix)
    F = eigen(Symmetric((target .+ target') ./ 2))
    removed = F.vectors * (max.(F.values, 0.0) .* F.vectors')
    ratio = tr(removed) / tr(fitted)
    capped = ratio > 1
    capped && (removed = removed ./ ratio)
    value, best = _matched_value(Gc, w, removed, fitted)
    return value, best, tr(removed), capped, ratio
end

function evsi(problem::DecisionProblem, study::MomentMatching; seed::Integer=1)
    Q, w = _sample_problem(problem, :evsi)
    belief = problem.belief
    columns = _focal_columns(belief, study.focal)
    method = _method_for(study.regression, columns)
    G, info = _fit_screened(method, columns, Q, w)
    wf = Float64.(w)
    Qf = Float64.(Q)
    v0, a = _act_now(Q, w)
    Gc = Float64.(G) .- permutedims(Float64.(G[a, :]))
    fitted = _weighted_covariance(Gc, wf)
    prior = _weighted_covariance(Qf .- permutedims(Qf[a, :]), wf)
    rng = Xoshiro(seed)
    bias = (; act_now=_act_now_bias(Q, w, rng), cells=nothing)
    base = (; act_now_value=v0, act_now_index=a, draws=length(w), focal=study.focal,
            nested=study.nested, fit=_method_name(method))
    if tr(fitted) <= 1e-12 * max(tr(prior), floatmin())
        # Learning the focal parameters exactly would change nothing.
        return VOIEstimate(:evsi, :MomentMatching, 0.0, 0.0, bias,
                           merge(base,
                                 (; variance_removed=0.0, evppi_variance=0.0,
                                  capped=false, model_evaluations=0)))
    end
    # Nested data sets at draws spread over the most variable fitted contrast.
    spread = [fitted[k, k] for k in axes(fitted, 1)]
    order = sortperm(view(Gc, argmax(spread), :))
    cumulative = cumsum(wf[order])
    # One draw at random within each stratum of equal weight (a midpoint would never
    # reach the tails, where the variance left by a study can be largest).
    picks = [order[min(searchsortedfirst(cumulative,
                                         (q - rand(rng)) / study.nested *
                                         cumulative[end]), length(order))]
             for q in 1:(study.nested)]
    rows = Tables.rowtable(belief.draws)
    all_cumulative = cumsum(wf)
    # For each nested data set, the covariance of the payoffs' differences over a set
    # of draws of the other parameters, before the study (focal values from the prior,
    # that is from other draws picked at random) minus after it (focal values from the
    # posterior). Both sides recombine the same draws of the other parameters with
    # focal values in the same way, so that their difference, the variance the study
    # removes, is neither swamped by the variance that no study of the focal
    # parameters touches nor biased by the accidents of how the draws were paired.
    function pick()
        return min(searchsortedfirst(all_cumulative, rand(rng) * all_cumulative[end]),
                   length(rows))
    end
    function contrasts(θ)
        y = _payoff_vector(study.payoffs(θ))
        length(y) == size(Q, 1) ||
            throw(ArgumentError("`payoffs` returned $(length(y)) payoffs for the " *
                                "$(size(Q, 1)) feasible actions of the problem"))
        return y .- y[a]
    end
    removed_covariances = Matrix{Float64}[]
    evaluations = 0
    for i in picks
        data = study.simulate_data(rng, rows[i])
        post = _column_table(study.posterior(rng, data))
        for name in study.focal
            haskey(post, name) ||
                throw(ArgumentError("`posterior` returned no column $name; columns: " *
                                    "$(collect(keys(post)))"))
        end
        S = length(first(post))
        S >= 2 || throw(ArgumentError("`posterior` returned $S draws; it needs many"))
        D = zeros(size(Q, 1), S)
        before = zeros(size(Q, 1), S)
        names = Tuple(study.focal)
        for s in 1:S
            other = rows[pick()]
            prior_focal = NamedTuple{names}(Tuple(getproperty(rows[pick()], name)
                                                  for name in names))
            posterior_focal = NamedTuple{names}(Tuple(post[name][s] for name in names))
            before[:, s] .= contrasts(merge(other, prior_focal))
            D[:, s] .= contrasts(merge(other, posterior_focal))
        end
        evaluations += 2 * S
        equal = fill(1 / S, S)
        push!(removed_covariances,
              (_weighted_covariance(before, equal) .- _weighted_covariance(D, equal)) .*
              (S / (S - 1)))
    end
    K = length(removed_covariances)
    removed_mean = sum(removed_covariances) ./ K
    value, best, removed, capped, ratio = _matched_estimate(Gc, wf, removed_mean, fitted)
    leave_one_out = [first(_matched_estimate(Gc, wf, (removed_mean .* K .- C) ./ (K - 1),
                                             fitted)) for C in removed_covariances]
    centre = sum(leave_one_out) / K
    nested_se = sqrt((K - 1) / K * sum(v -> (v - centre)^2, leave_one_out))
    draws_se = _weighted_mean_se(best, wf)
    details = merge(base,
                    (; variance_removed=removed, evppi_variance=tr(fitted), capped,
                     variance_ratio=ratio,
                     model_evaluations=evaluations,
                     draws_standard_error=draws_se,
                     nested_standard_error=nested_se), info)
    return VOIEstimate(:evsi, :MomentMatching, value, hypot(draws_se, nested_se), bias,
                       details)
end
