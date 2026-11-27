# Gaussian-process regression for EVPPI with five or more focal parameters: a small
# internal implementation after Strong, Oakley and Brennan (2014), with no dependency
# beyond LinearAlgebra. It is one more way of obtaining the fitted conditional means
# g[a, i] of the estimator at the top of `voi_sample.jl`.

"""
    GaussianProcessRegression(; max_points=1500, hyper_points=500, seed=1,
                              min_nugget=1e-8, min_length_scale=1)

Regression by a **Gaussian process**, the method for five or more focal parameters
(Strong, Oakley and Brennan 2014, *Medical Decision Making* 34:311), built in with no
dependency beyond LinearAlgebra. Each action's payoff is modelled as

    Q[a, i] = β₀ + βᵀ x_i + f(x_i) + ε_i,

a linear mean, a Gaussian process `f` with the squared-exponential correlation
`c(x, x′) = exp(−Σ_j ((x_j − x′_j) / δ_j)²)`, one length scale `δ_j` per column, and
independent noise with variance `ν` times that of `f` (the **nugget**), which stands
for everything the columns do not explain: the other parameters and the simulation
noise of the cells. The columns are standardised first. `β` and the variance are
integrated out under flat priors, and the hyperparameters `δ` and `ν` are set, for
each action separately, by maximising the resulting marginal likelihood with a
Nelder-Mead search from a fixed start. The fitted conditional means are the
posterior mean of `β₀ + βᵀ x + f(x)` at every draw.

The cost of a Gaussian process grows with the cube of the number of points, so two
subsamples of the draws are used, both chosen from `seed` by a generator that gives
the same subsamples on every Julia version and platform:

- `hyper_points`: the number of draws used to estimate the hyperparameters;
- `max_points`: the largest number of draws the process is conditioned on. With more
  draws than this, the process is fitted to a subsample and **predicts the
  conditional means of all the draws**, so every draw still enters the estimate.

**The estimate depends on the subsamples, and its standard error does not include
that.** The dependence is small when the columns explain much of the payoff and can
be as large as the standard error when they explain little, because the length scales
are then poorly determined by a few hundred points. Compare two or three values of
`seed` to see it; a larger `hyper_points` reduces it at a cost that grows with the cube
(the default of 500 takes about three times as long as 250, the value Strong, Oakley
and Brennan use, and roughly halved the spread between seeds where it was largest).

A draw's weight divides its noise variance; draws of weight zero are not used in the
fit. `min_nugget` is the smallest nugget the search may choose (relative to the
variance of `f`); it also keeps the linear algebra stable.

**The length scales are bounded below**, at `min_length_scale` times the median
distance between neighbouring points of the subsample used for the hyperparameters
(in standardised coordinates). A process with shorter length scales is independent
noise at the points it is fitted to: the likelihood cannot tell it from the nugget,
but its fitted values would reproduce the payoffs, noise included, and the estimate
would drift up to the EVPI. With many parameters the points are far apart, so the
bound is what keeps the fit a smooth function of them; lower it only for payoffs known
to vary on a finer scale than the draws resolve. The columns must be numeric
(use [`Grouping`](@ref) for a discrete parameter); a constant column is dropped.

A Gaussian process does not reproduce cell means of discrete columns exactly, and it
is slower than [`SplineRegression`](@ref) for a few parameters; it is what
[`evppi`](@ref) uses by default from five numeric focal parameters on, where an
additive spline would miss the interactions and a tensor-product one has too many
coefficients.
"""
struct GaussianProcessRegression <: AbstractRegression
    max_points::Int
    hyper_points::Int
    seed::Int
    min_nugget::Float64
    min_length_scale::Float64
    function GaussianProcessRegression(; max_points::Integer=1500,
                                       hyper_points::Integer=500, seed::Integer=1,
                                       min_nugget::Real=1e-8, min_length_scale::Real=1)
        max_points >= 10 ||
            throw(ArgumentError("max_points must be at least 10, got $max_points"))
        hyper_points >= 10 ||
            throw(ArgumentError("hyper_points must be at least 10, got $hyper_points"))
        min_nugget > 0 ||
            throw(ArgumentError("min_nugget must be positive, got $min_nugget"))
        min_length_scale > 0 ||
            throw(ArgumentError("min_length_scale must be positive, got " *
                                "$min_length_scale"))
        return new(max_points, min(hyper_points, max_points), seed, Float64(min_nugget),
                   Float64(min_length_scale))
    end
end

# A seeded shuffle that is the same on every Julia version and platform. `shuffle!` is
# not: Julia may change how it draws from a generator between versions, and then a
# subsample chosen with it, and every estimate computed from that subsample, changes
# with no change to this package. SplitMix64 (Steele, Lea and Flood 2014) drives a
# Fisher-Yates shuffle; the index is the high word of a 128-bit product, whose
# departure from uniform is below 2^-64 times the length.
mutable struct _SplitMix64
    state::UInt64
    _SplitMix64(seed::Integer) = new(seed % UInt64)
end

function _next!(g::_SplitMix64)
    z = (g.state += 0x9e3779b97f4a7c15)
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return z ⊻ (z >> 31)
end

function _shuffle!(g::_SplitMix64, v::AbstractVector)
    for i in length(v):-1:2
        j = 1 + Int((UInt128(_next!(g)) * UInt128(i)) >> 64)
        v[i], v[j] = v[j], v[i]
    end
    return v
end

# Squared-exponential correlations between the rows of X and of Y, after dividing each
# column by its length scale.
function _gp_correlation(X::AbstractMatrix, Y::AbstractMatrix, δ::AbstractVector)
    Xs = X ./ δ'
    Ys = Y ./ δ'
    D = sum(abs2, Xs; dims=2) .+ sum(abs2, Ys; dims=2)' .- 2 .* (Xs * Ys')
    return exp.(.-max.(D, 0.0))
end

# The pieces of the fit at given hyperparameters: `nothing` when the covariance is not
# numerically positive definite. `noise[i]` multiplies the nugget of point i.
function _gp_pieces(X::AbstractMatrix, H::AbstractMatrix, y::AbstractVector,
                    noise::AbstractVector, δ::AbstractVector, ν::Real)
    A = _gp_correlation(X, X, δ)
    for i in axes(A, 1)
        A[i, i] += ν * noise[i]
    end
    F = cholesky(Symmetric(A); check=false)
    issuccess(F) || return nothing
    AiH = F \ H
    W = cholesky(Symmetric(H' * AiH); check=false)
    issuccess(W) || return nothing
    β = W \ (AiH' * y)
    α = F \ (y .- H * β)
    return (; F, W, β, α)
end

# Log marginal likelihood of (log δ, log ν) with β and the variance integrated out
# under flat priors, up to a constant; −Inf where the fit fails.
function _gp_log_likelihood(X, H, y, noise, logθ::AbstractVector, bounds)
    p = size(X, 2)
    all(i -> bounds[i][1] <= logθ[i] <= bounds[i][2], eachindex(logθ)) || return -Inf
    pieces = _gp_pieces(X, H, y, noise, exp.(view(logθ, 1:p)), exp(logθ[p + 1]))
    pieces === nothing && return -Inf
    n, q = size(H)
    s2 = dot(y .- H * pieces.β, pieces.α) / (n - q - 2)
    s2 > 0 || return -Inf
    return -logdet(pieces.F) / 2 - logdet(pieces.W) / 2 - (n - q) / 2 * log(s2)
end

# Nelder-Mead maximisation from `start` with the given initial step; deterministic.
function _nelder_mead(f, start::Vector{Float64}; step::Float64=0.7, evaluations::Int=400,
                      tolerance::Float64=1e-6)
    d = length(start)
    simplex = [copy(start) for _ in 0:d]
    for i in 1:d
        simplex[i + 1][i] += step
    end
    values = [f(x) for x in simplex]
    used = d + 1
    while used < evaluations
        order = sortperm(values; rev=true)
        simplex, values = simplex[order], values[order]
        isfinite(values[end]) && values[1] - values[end] < tolerance && break
        centroid = sum(simplex[1:d]) ./ d
        reflected = centroid .+ (centroid .- simplex[end])
        fr = f(reflected)
        used += 1
        if fr > values[1]
            expanded = centroid .+ 2 .* (centroid .- simplex[end])
            fe = f(expanded)
            used += 1
            simplex[end], values[end] = fe > fr ? (expanded, fe) : (reflected, fr)
        elseif fr > values[d]
            simplex[end], values[end] = reflected, fr
        else
            contracted = centroid .+ 0.5 .* (simplex[end] .- centroid)
            fc = f(contracted)
            used += 1
            if fc > values[end]
                simplex[end], values[end] = contracted, fc
            else
                for i in 2:(d + 1)
                    simplex[i] = simplex[1] .+ 0.5 .* (simplex[i] .- simplex[1])
                    values[i] = f(simplex[i])
                end
                used += d
            end
        end
    end
    best = argmax(values)
    return simplex[best], values[best]
end

# Median distance from a point to its nearest neighbour (standardised coordinates).
function _gp_spacing(X::AbstractMatrix)
    n = size(X, 1)
    D = sum(abs2, X; dims=2) .+ sum(abs2, X; dims=2)' .- 2 .* (X * X')
    nearest = [sqrt(max(minimum(D[i, j] for j in 1:n if j != i), 0.0)) for i in 1:n]
    positive = sort!(filter(>(0), nearest))
    return isempty(positive) ? 1.0 : positive[(length(positive) + 1) ÷ 2]
end

# Standardised numeric columns as a matrix, constant columns dropped.
function _gp_inputs(columns::NamedTuple, w::AbstractVector)
    for (name, x) in pairs(columns)
        _is_numeric(x) ||
            throw(ArgumentError("GaussianProcessRegression needs numeric columns; " *
                                "$name is not. Use Grouping() for a discrete parameter."))
    end
    wf = Float64.(w)
    kept = Vector{Float64}[]
    for x in columns
        xf = Float64.(x)
        centre = dot(wf, xf)
        spread = sqrt(dot(wf, (xf .- centre) .^ 2))
        spread > 0 && isfinite(spread) && push!(kept, (xf .- centre) ./ spread)
    end
    return isempty(kept) ? zeros(length(wf), 0) : reduce(hcat, kept)
end

function fitted_conditional_means(method::GaussianProcessRegression, columns::NamedTuple,
                                  Q::AbstractMatrix, w::AbstractVector)
    X = _gp_inputs(columns, w)
    n, p = size(X)
    wf = Float64.(w)
    Qf = Float64.(Q)
    means = Qf * wf
    G = repeat(means, 1, n)
    p == 0 && return G
    usable = findall(>(0), wf)
    rng = _SplitMix64(method.seed)
    fit = length(usable) <= method.max_points ? usable :
          sort!(_shuffle!(rng, copy(usable))[1:(method.max_points)])
    length(fit) > p + 3 ||
        throw(ArgumentError("a Gaussian process on $p columns needs more than " *
                            "$(p + 3) draws of positive weight, got $(length(fit))"))
    hyper = length(fit) <= method.hyper_points ? collect(eachindex(fit)) :
            sort!(_shuffle!(rng, collect(eachindex(fit)))[1:(method.hyper_points)])
    Xf = X[fit, :]
    Hf = hcat(ones(length(fit)), Xf)
    # A draw's weight divides its noise variance (relative to the mean weight).
    noise = (sum(wf[fit]) / length(fit)) ./ wf[fit]
    # A process whose length scales are below the spacing of its points is independent
    # noise at those points: the likelihood cannot tell it from the nugget, but its
    # fitted values would reproduce the payoffs, noise and all, and inflate the
    # estimate. The length scales are therefore bounded below by the typical distance
    # between neighbouring points of the subsample they are estimated on.
    spacing = _gp_spacing(view(Xf, hyper, :))
    floor_δ = log(method.min_length_scale * spacing)
    bounds = vcat(fill((floor_δ, max(6.0, floor_δ + 8.0)), p),
                  [(log(method.min_nugget), 6.0)])
    start = vcat(fill(max(log(sqrt(p)), floor_δ + log(2.0)), p), log(0.1))
    for a in axes(Qf, 1)
        raw = Qf[a, fit]
        centre = sum(raw) / length(raw)
        scale = sqrt(sum(abs2, raw .- centre) / length(raw))
        # A row that does not vary (the payoff of a reference action, say) is its mean.
        scale > 1e-12 * max(1.0, abs(centre)) || continue
        y = (raw .- centre) ./ scale
        function objective(θ)
            return _gp_log_likelihood(view(Xf, hyper, :), view(Hf, hyper, :),
                                      view(y, hyper), view(noise, hyper), θ, bounds)
        end
        θ, _ = _nelder_mead(objective, start; evaluations=150 * (p + 1))
        θ, _ = _nelder_mead(objective, θ; step=0.3, evaluations=100 * (p + 1))
        δ, ν = exp.(θ[1:p]), exp(θ[p + 1])
        pieces = _gp_pieces(Xf, Hf, y, noise, δ, ν)
        while pieces === nothing        # widen the nugget until the factorisation holds
            ν = max(10ν, 1e-6)
            pieces = _gp_pieces(Xf, Hf, y, noise, δ, ν)
        end
        for block in Iterators.partition(1:n, 2000)
            Xb = X[block, :]
            fitted = pieces.β[1] .+ Xb * pieces.β[2:end] .+
                     _gp_correlation(Xb, Xf, δ) * pieces.α
            G[a, block] .= centre .+ scale .* fitted
        end
        # The fitted means average to the row's mean, as every regression here does.
        G[a, :] .+= means[a] - dot(wf, view(G, a, :))
    end
    return G
end
