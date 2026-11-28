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
        m = nothing
        M2 = nothing
        for k in 1:inner
            rest = sample_rest(rng, φ)
            rest isa NamedTuple ||
                throw(ArgumentError("`sample_rest(rng, φ)` must return a named tuple, " *
                                    "got $(typeof(rest))"))
            y = _payoff_vector(payoffs(merge(rest, φ)))
            means === nothing && (means = zeros(length(y), outer))
            length(y) == size(means, 1) ||
                throw(ArgumentError("`payoffs` returned $(length(y)) payoffs in one " *
                                    "call and $(size(means, 1)) in another"))
            if m === nothing
                m = zero(y)
                M2 = zeros(length(y), length(y))
            end
            delta = y .- m
            m .+= delta ./ k
            delta2 = y .- m
            M2 .+= delta .* delta2'
        end
        means[:, r] .= m
        # The inner means of the actions share their inner sample, so their noise is
        # correlated: the inflation of the maximum is simulated with their covariance.
        covariance = M2 ./ (inner * (inner - 1))
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
    MultilevelMC(; tol, base_inner=2, pilot=200, min_level=2, max_level=10,
                 max_outer=200_000)

Antithetic coupled **multilevel Monte Carlo** configuration (Giles and Goda 2019,
*Operations Research* 67:1759) for nested [`evppi_nested`](@ref) and [`evsi`](@ref).

At level `ℓ = 0`, each outer sample averages `M₀ = base_inner` inner draws; at level
`ℓ ≥ 1`, each outer sample splits `M_ℓ = M₀ · 2^ℓ` inner draws into two independent
halves `A` and `B` of size `M_{ℓ-1}` and accumulates the antithetic difference

    Δ_ℓ = max_a Ȳ^{(ℓ)}_a − ½ (max_a Ȳ^{(A)}_a + max_a Ȳ^{(B)}_a) ≤ 0,

which cancels to zero whenever the same action wins in both halves. This gives variance
decay `O(2^{-3ℓ/2})` against cost `O(2^ℓ)`, reducing the complexity of reaching RMS error
`tol` from `O(tol⁻³)` for standard two-level Monte Carlo to `O(tol⁻²)`.

Levels are added adaptively from `min_level` up to `max_level` until both the Monte Carlo
standard error and the level-`L` bias indicator are below `tol / √2`.
"""
struct MultilevelMC
    tol::Float64
    base_inner::Int
    pilot::Int
    min_level::Int
    max_level::Int
    max_outer::Int
    function MultilevelMC(; tol::Real, base_inner::Integer=2, pilot::Integer=200,
                          min_level::Integer=2, max_level::Integer=10,
                          max_outer::Integer=200_000)
        (isfinite(tol) && tol > 0) ||
            throw(ArgumentError("tol must be positive and finite, got $tol"))
        base_inner >= 1 ||
            throw(ArgumentError("base_inner must be at least 1, got $base_inner"))
        pilot >= 20 || throw(ArgumentError("pilot must be at least 20, got $pilot"))
        min_level >= 1 ||
            throw(ArgumentError("min_level must be at least 1, got $min_level"))
        max_level >= min_level ||
            throw(ArgumentError("max_level must be at least min_level ($min_level), got " *
                                "$max_level"))
        max_outer >= pilot ||
            throw(ArgumentError("max_outer must be at least pilot ($pilot), got $max_outer"))
        return new(Float64(tol), Int(base_inner), Int(pilot), Int(min_level),
                   Int(max_level), Int(max_outer))
    end
end

# Accumulate `n_add` outer samples for MLMC level `lvl` (lvl = 0 uses M0 inner draws and
# stores mean payoff vectors for act-now determination; lvl >= 1 uses antithetic splitting
# of M = M0 * 2^lvl inner draws into two halves of M0 * 2^(lvl - 1)).
function _mlmc_add_level0!(rng::AbstractRNG, payoffs, sample_outer, sample_inner,
                           M0::Int, n_add::Int, means_acc::Vector{Vector{Float64}},
                           n_actions::Ref{Int})
    evals = 0
    for _ in 1:n_add
        φ = sample_outer(rng)
        m = nothing
        for k in 1:M0
            rest = sample_inner(rng, φ)
            y = _payoff_vector(payoffs(rest, φ))
            evals += 1
            if n_actions[] == 0
                n_actions[] = length(y)
            elseif length(y) != n_actions[]
                throw(ArgumentError("`payoffs` returned $(length(y)) payoffs in one " *
                                    "call and $(n_actions[]) in another"))
            end
            if m === nothing
                m = copy(y)
            else
                m .+= y
            end
        end
        m ./= M0
        push!(means_acc, m)
    end
    return evals
end

function _mlmc_add_coupled!(rng::AbstractRNG, payoffs, sample_outer, sample_inner,
                            half_m::Int, n_add::Int, count_ref::Ref{Int},
                            mean_ref::Ref{Float64}, m2_ref::Ref{Float64},
                            n_actions::Int)
    evals = 0
    sumA = Vector{Float64}(undef, n_actions)
    sumB = Vector{Float64}(undef, n_actions)
    inv_half = 1.0 / half_m
    inv_full = 0.5 * inv_half
    for _ in 1:n_add
        φ = sample_outer(rng)
        fill!(sumA, 0.0)
        fill!(sumB, 0.0)
        for _ in 1:half_m
            y = _payoff_vector(payoffs(sample_inner(rng, φ), φ))
            evals += 1
            @inbounds for a in 1:n_actions
                sumA[a] += y[a]
            end
        end
        for _ in 1:half_m
            y = _payoff_vector(payoffs(sample_inner(rng, φ), φ))
            evals += 1
            @inbounds for a in 1:n_actions
                sumB[a] += y[a]
            end
        end
        maxA = -Inf
        maxB = -Inf
        maxF = -Inf
        @inbounds for a in 1:n_actions
            va = sumA[a] * inv_half
            vb = sumB[a] * inv_half
            vf = (sumA[a] + sumB[a]) * inv_full
            va > maxA && (maxA = va)
            vb > maxB && (maxB = vb)
            vf > maxF && (maxF = vf)
        end
        Δ = maxF - 0.5 * (maxA + maxB)
        k = count_ref[] + 1
        count_ref[] = k
        d1 = Δ - mean_ref[]
        mean_ref[] += d1 / k
        m2_ref[] += d1 * (Δ - mean_ref[])
    end
    return evals
end

function _mlmc_level0_stats(means_acc::Vector{Vector{Float64}})
    N = length(means_acc)
    na = length(first(means_acc))
    grand = zeros(na)
    for m in means_acc
        grand .+= m
    end
    grand ./= N
    a0 = argmax(grand)
    v0 = grand[a0]
    mu = 0.0
    m2 = 0.0
    for (k, m) in enumerate(means_acc)
        g = maximum(m) - m[a0]
        d1 = g - mu
        mu += d1 / k
        m2 += d1 * (g - mu)
    end
    return mu, m2 / (N - 1), v0, a0
end

function _mlmc_nested(quantity::Symbol, payoffs_pair, sample_outer, sample_inner,
                      method::MultilevelMC; seed::Union{Nothing,Integer}=nothing)
    seed === nothing && throw(ArgumentError("give `seed`"))
    rng = Xoshiro(seed)
    M0 = method.base_inner
    L = method.min_level
    n_actions = Ref(0)
    means0 = Vector{Float64}[]
    counts_c = [Ref(0) for _ in 1:(method.max_level)]
    means_c = [Ref(0.0) for _ in 1:(method.max_level)]
    m2_c = [Ref(0.0) for _ in 1:(method.max_level)]
    total_evals = 0
    total_evals += _mlmc_add_level0!(rng, payoffs_pair, sample_outer, sample_inner, M0,
                                     method.pilot, means0, n_actions)
    for l in 1:L
        total_evals += _mlmc_add_coupled!(rng, payoffs_pair, sample_outer, sample_inner,
                                          M0 * (1 << (l - 1)), method.pilot, counts_c[l],
                                          means_c[l], m2_c[l], n_actions[])
    end
    converged = false
    ε = method.tol
    for _ in 1:25
        m0, v0_var, _, _ = _mlmc_level0_stats(means0)
        # Current level counts, variances and costs (in model evaluations per outer sample).
        n_cur = Int[length(means0); [counts_c[l][] for l in 1:L]]
        vars = Float64[max(v0_var, 0.0);
                       [max(m2_c[l][] / max(1, counts_c[l][] - 1), 0.0) for l in 1:L]]
        costs = Float64[Float64(M0); [Float64(M0 * (1 << l)) for l in 1:L]]
        sum_sqrt = sum(sqrt(vars[i] * costs[i]) for i in eachindex(vars))
        added_any = false
        if sum_sqrt > 0.0
            for idx in 1:(L + 1)
                target = ceil(Int, 2.0 / (ε * ε) * sqrt(vars[idx] / costs[idx]) * sum_sqrt)
                target = min(target, method.max_outer)
                need = target - n_cur[idx]
                if need > 0
                    added_any = true
                    if idx == 1
                        total_evals += _mlmc_add_level0!(rng, payoffs_pair, sample_outer,
                                                         sample_inner, M0, need, means0,
                                                         n_actions)
                    else
                        l = idx - 1
                        total_evals += _mlmc_add_coupled!(rng, payoffs_pair, sample_outer,
                                                          sample_inner, M0 * (1 << (l - 1)),
                                                          need, counts_c[l], means_c[l],
                                                          m2_c[l], n_actions[])
                    end
                end
            end
        end
        # Weak-error bias check (Giles 2015): |E[Δ_L]| ≈ |E[Y^(L) - Y]|.
        rem_bias = L >= 2 ? max(abs(means_c[L][]), 0.5 * abs(means_c[L - 1][])) :
                   abs(means_c[L][])
        if rem_bias <= ε / sqrt(2.0)
            added_any || (converged=true; break)
        elseif L < method.max_level
            L += 1
            total_evals += _mlmc_add_coupled!(rng, payoffs_pair, sample_outer, sample_inner,
                                              M0 * (1 << (L - 1)), method.pilot,
                                              counts_c[L], means_c[L], m2_c[L],
                                              n_actions[])
        else
            added_any || break
        end
    end
    m0, v0_var, act_v0, act_idx = _mlmc_level0_stats(means0)
    lvl_means = Float64[m0; [means_c[l][] for l in 1:L]]
    lvl_vars = Float64[max(v0_var, 0.0);
                       [max(m2_c[l][] / max(1, counts_c[l][] - 1), 0.0) for l in 1:L]]
    lvl_counts = Int[length(means0); [counts_c[l][] for l in 1:L]]
    inner_sizes = Int[M0 * (1 << l) for l in 0:L]
    value = sum(lvl_means)
    se = sqrt(sum(lvl_vars[i] / lvl_counts[i] for i in eachindex(lvl_means)))
    rem_bias = abs(lvl_means[end])
    # Recompute convergence cleanly from final stats.
    converged = converged || (se <= ε / sqrt(2.0) && rem_bias <= ε / sqrt(2.0))
    # Act-now selection bias estimated from level-0 inner-mean matrix.
    M0_mat = reduce(hcat, means0)
    w0 = fill(1.0 / length(means0), length(means0))
    bias = (; act_now=_act_now_bias(M0_mat, w0, rng), cells=rem_bias)
    details = (; act_now_value=act_v0, act_now_index=act_idx, levels=L + 1,
               inner_sizes, outer_counts=lvl_counts, level_means=lvl_means,
               level_variances=lvl_vars, model_evaluations=total_evals, tol=ε, converged)
    return VOIEstimate(quantity, :MultilevelMC, value, se, bias, details)
end

function evppi_nested(payoffs, sample_focal, sample_rest, method::MultilevelMC;
                      seed::Union{Nothing,Integer}=nothing)
    eval_pair(rest, φ) = payoffs(merge(rest, φ))
    return _mlmc_nested(:evppi, eval_pair, sample_focal, sample_rest, method; seed)
end

"""
    evsi(payoffs, sample_data, sample_posterior, method::MultilevelMC; seed) -> VOIEstimate
    evsi(problem::DecisionProblem, design::AbstractStudyDesign, method::MultilevelMC;
         payoffs, prior, seed) -> VOIEstimate

The expected value of sample information by **antithetic multilevel Monte Carlo**
([`MultilevelMC`](@ref), Giles and Goda 2019).

- In the model form, `sample_data(rng)` simulates one data set from the prior predictive
  distribution, `sample_posterior(rng, data)` draws one full parameter named tuple `θ`
  from the posterior given `data`, and `payoffs(θ)` returns the action payoffs.
- In the study-design form, `design` is a built-in [`AbstractStudyDesign`](@ref) with
  conjugate `prior` and `problem` provides the [`SampleBelief`](@ref) for the non-focal
  parameters (sampled independently of the focal parameters).
"""
function evsi(payoffs, sample_data, sample_posterior, method::MultilevelMC;
              seed::Union{Nothing,Integer}=nothing)
    eval_pair(θ, _) = payoffs(θ)
    return _mlmc_nested(:evsi, eval_pair, sample_data, sample_posterior, method; seed)
end
function evsi(problem::DecisionProblem, design::AbstractStudyDesign, method::MultilevelMC;
              payoffs, prior, seed::Union{Nothing,Integer}=nothing)
    _sample_problem(problem, :evsi)
    belief = problem.belief
    rows = Tables.rowtable(belief.draws)
    wf = Float64.(belief.weights)
    cumulative = cumsum(wf)
    n = length(rows)
    focal = _study_parameters(design)
    function _pick_row(rng)
        return rows[min(searchsortedfirst(cumulative, rand(rng) * cumulative[end]), n)]
    end
    sample_data(rng) = simulate_study(rng, design, _pick_row(rng))
    function sample_post(rng, data)
        post = study_posterior(rng, design, data, prior; draws=1)
        post_nt = NamedTuple{Tuple(focal)}(Tuple(first(post[nm]) for nm in focal))
        return merge(_pick_row(rng), post_nt)
    end
    eval_pair(θ, _) = payoffs(θ)
    return _mlmc_nested(:evsi, eval_pair, sample_data, sample_post, method; seed)
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
    removed_pos = F.vectors * (max.(F.values, 0.0) .* F.vectors')
    ratio = tr(removed_pos) / tr(fitted)
    Ff = eigen(Symmetric((fitted .+ fitted') ./ 2))
    keep = findall(>(1e-12 * max(1.0, maximum(Ff.values))), Ff.values)
    if isempty(keep)
        removed = zero(removed_pos)
        capped = tr(removed_pos) > 0
    else
        V = Ff.vectors[:, keep]
        root = sqrt.(Ff.values[keep])
        inv_root = 1.0 ./ root
        W = (inv_root .* V') * removed_pos * (V .* inv_root')
        Fw = eigen(Symmetric((W .+ W') ./ 2))
        capped = any(>(1.0), Fw.values)
        vals = clamp.(Fw.values, 0.0, 1.0)
        W_cap = Fw.vectors * (vals .* Fw.vectors')
        removed = (V .* root') * W_cap * (root .* V')
        removed = (removed .+ removed') ./ 2
    end
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
                                  capped=false, variance_ratio=0.0, model_evaluations=0,
                                  draws_standard_error=0.0, nested_standard_error=0.0),
                                 info))
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
            prior_row = rows[pick()]
            prior_focal = NamedTuple{names}(Tuple(getproperty(prior_row, name)
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
