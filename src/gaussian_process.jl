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
box-constrained L-BFGS search using analytical gradients from a fixed start. The fitted
conditional means are the posterior mean of `β₀ + βᵀ x + f(x)` at every draw.

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

# In-place squared-exponential correlation matrix between rows of `X` and itself, storing
# the scaled inputs `X ./ δ'` in `Xs` and squared row norms in `norms`.
function _gp_correlation_self!(C::Matrix{Float64}, Xs::Matrix{Float64},
                               norms::Vector{Float64}, X::Matrix{Float64},
                               δ::AbstractVector{Float64})
    n, p = size(X)
    @inbounds for k in 1:p
        invδ = 1.0 / δ[k]
        for i in 1:n
            Xs[i, k] = X[i, k] * invδ
        end
    end
    fill!(norms, 0.0)
    @inbounds for k in 1:p, i in 1:n
        v = Xs[i, k]
        norms[i] += v * v
    end
    mul!(C, Xs, transpose(Xs))
    @inbounds for j in 1:n
        nj = norms[j]
        for i in 1:n
            if i == j
                C[i, i] = 1.0
            else
                d = norms[i] + nj - 2.0 * C[i, j]
                C[i, j] = exp(-max(d, 0.0))
            end
        end
    end
    return C
end

# Log marginal likelihood of `θ = (log δ, log ν)` with β and the variance integrated out
# under flat priors, and its analytical gradient written into `g_out` (when non-nothing).
# Returns `-Inf` when `θ` leaves `[lo, hi]` or the covariance is not numerically positive
# definite.
function _gp_ll_and_grad!(g_out::Union{Nothing,Vector{Float64}}, X::Matrix{Float64},
                          H::Matrix{Float64}, y::Vector{Float64}, noise::Vector{Float64},
                          θ::AbstractVector{Float64}, lo::Vector{Float64},
                          hi::Vector{Float64}, C::Matrix{Float64}, A::Matrix{Float64},
                          P::Matrix{Float64}, Xs::Matrix{Float64}, norms::Vector{Float64},
                          MX::Matrix{Float64}, msum::Vector{Float64})
    p = size(X, 2)
    n, q = size(H)
    n > q + 2 || return -Inf
    @inbounds for i in eachindex(θ)
        (lo[i] <= θ[i] <= hi[i]) || return -Inf
    end
    δ = Vector{Float64}(undef, p)
    @inbounds for k in 1:p
        δ[k] = exp(θ[k])
    end
    ν = exp(θ[p + 1])
    _gp_correlation_self!(C, Xs, norms, X, δ)
    copyto!(A, C)
    @inbounds for i in 1:n
        A[i, i] += ν * noise[i]
    end
    F = cholesky!(Symmetric(A); check=false)
    issuccess(F) || return -Inf
    AiH = F \ H
    W = cholesky!(Symmetric(H' * AiH); check=false)
    issuccess(W) || return -Inf
    β = W \ (AiH' * y)
    r = y .- H * β
    α = F \ r
    rα = dot(r, α)
    (rα > 0 && isfinite(rα)) || return -Inf
    s2 = rα / (n - q - 2)
    ll = -logdet(F) / 2 - logdet(W) / 2 - (n - q) / 2 * log(s2)
    isfinite(ll) || return -Inf
    g_out === nothing && return ll

    # P = A⁻¹ - A⁻¹ H W⁻¹ Hᵀ A⁻¹ via LAPACK `potri!` on the Cholesky factor.
    copyto!(P, F.factors)
    LAPACK.potri!(F.uplo, P)
    if F.uplo == 'U'
        @inbounds for j in 1:n, i in 1:(j - 1)
            P[j, i] = P[i, j]
        end
    else
        @inbounds for j in 1:n, i in (j + 1):n
            P[j, i] = P[i, j]
        end
    end
    V = AiH / W.U
    mul!(P, V, transpose(V), -1.0, 1.0)

    # dℓ = (1/2) tr((c_α α αᵀ - P) dA), with c_α = (n - q) / (rᵀ α).
    c_alpha = (n - q) / rα
    g_nugget = 0.0
    @inbounds for j in 1:n
        aj = c_alpha * α[j]
        acc_j = 0.0
        for i in 1:n
            mij = 0.5 * (aj * α[i] - P[i, j])
            if i == j
                g_nugget += mij * noise[i]
            end
            kij = mij * C[i, j]
            P[i, j] = kij
            acc_j += kij
        end
        msum[j] = acc_j
    end
    g_out[p + 1] = ν * g_nugget
    mul!(MX, P, Xs)
    @inbounds for k in 1:p
        s = 0.0
        for i in 1:n
            xik = Xs[i, k]
            s += msum[i] * xik * xik - xik * MX[i, k]
        end
        g_out[k] = 4.0 * s
    end
    return ll
end

# Box-constrained projected L-BFGS maximisation of `fg!(g, θ)` on `[lo, hi]`;
# deterministic, using two-metric projection on the free coordinates.
function _lbfgs_box(fg!, start::Vector{Float64}, lo::Vector{Float64}, hi::Vector{Float64};
                    max_iter::Int=60, mem::Int=7, gtol::Float64=1e-5, ftol::Float64=1e-7)
    d = length(start)
    x = clamp.(start, lo, hi)
    g_ll = zeros(Float64, d)
    ll = fg!(g_ll, x)
    if !isfinite(ll)
        # Fallback if the initial point is numerically indefinite: widen the nugget.
        for shift in (1.0, 2.5, 4.0)
            x[end] = min(hi[end], max(x[end], start[end] + shift))
            ll = fg!(g_ll, x)
            isfinite(ll) && break
        end
        isfinite(ll) || return x, -Inf
    end
    f = -ll
    g = .-g_ll
    best_x = copy(x)
    best_ll = ll

    S = Vector{Float64}[]
    Y = Vector{Float64}[]
    RHO = Float64[]
    free = trues(d)
    g_proj = zeros(Float64, d)
    q = zeros(Float64, d)
    dir = zeros(Float64, d)
    x_try = zeros(Float64, d)
    g_try_ll = zeros(Float64, d)
    g_try = zeros(Float64, d)
    alphas = zeros(Float64, mem)
    dx = zeros(Float64, d)

    for _ in 1:max_iter
        gnorm = 0.0
        @inbounds for i in 1:d
            at_lo = x[i] <= lo[i] + 1e-12 && g[i] > 0.0
            at_hi = x[i] >= hi[i] - 1e-12 && g[i] < 0.0
            if at_lo || at_hi
                free[i] = false
                g_proj[i] = 0.0
            else
                free[i] = true
                g_proj[i] = g[i]
                a = abs(g[i])
                a > gnorm && (gnorm = a)
            end
        end
        gnorm <= gtol && break

        # Two-loop L-BFGS on the free subspace.
        used_lbfgs = !isempty(S)
        if used_lbfgs
            copyto!(q, g_proj)
            kmem = length(S)
            for j in kmem:-1:1
                sj = S[j]
                yj = Y[j]
                acc = 0.0
                @inbounds for i in 1:d
                    free[i] && (acc += sj[i] * q[i])
                end
                aj = RHO[j] * acc
                alphas[j] = aj
                @inbounds for i in 1:d
                    free[i] && (q[i] -= aj * yj[i])
                end
            end
            slast, ylast = S[end], Y[end]
            sy = 0.0
            yy = 0.0
            @inbounds for i in 1:d
                if free[i]
                    sy += slast[i] * ylast[i]
                    yy += ylast[i] * ylast[i]
                end
            end
            if !(sy > 0.0 && yy > 0.0)
                sy = dot(slast, ylast)
                yy = dot(ylast, ylast)
            end
            gamma = (sy > 0.0 && yy > 0.0) ? clamp(sy / yy, 1e-4, 1e4) :
                    min(1.0, 1.0 / max(gnorm, 1.0))
            @inbounds for i in 1:d
                dir[i] = free[i] ? gamma * q[i] : 0.0
            end
            for j in 1:kmem
                sj = S[j]
                yj = Y[j]
                acc = 0.0
                @inbounds for i in 1:d
                    free[i] && (acc += yj[i] * dir[i])
                end
                beta = RHO[j] * acc
                diff = alphas[j] - beta
                @inbounds for i in 1:d
                    free[i] && (dir[i] += sj[i] * diff)
                end
            end
            @inbounds for i in 1:d
                dir[i] = -dir[i]
            end
            # Verify sufficient descent; reset history if the active set broke PD.
            gd = dot(g_proj, dir)
            dnorm2 = dot(dir, dir)
            gpnorm2 = dot(g_proj, g_proj)
            if !(gd < -1e-12 * sqrt(gpnorm2 * dnorm2))
                empty!(S)
                empty!(Y)
                empty!(RHO)
                used_lbfgs = false
            end
        end
        if !used_lbfgs
            scale = min(1.0, 1.0 / max(gnorm, 1.0))
            @inbounds for i in 1:d
                dir[i] = -scale * g_proj[i]
            end
        end
        maxstep = maximum(abs, dir)
        if maxstep > 5.0
            dir .*= 5.0 / maxstep
        end

        # Projected Armijo backtracking line search.
        step = 1.0
        accepted = false
        f_new = f
        for _ in 1:20
            @inbounds for i in 1:d
                xi = clamp(x[i] + step * dir[i], lo[i], hi[i])
                x_try[i] = xi
                dx[i] = xi - x[i]
            end
            slope = dot(g, dx)
            ll_try = fg!(g_try_ll, x_try)
            if isfinite(ll_try)
                if ll_try > best_ll
                    best_ll = ll_try
                    copyto!(best_x, x_try)
                end
                f_try = -ll_try
                if (slope < 0.0 && f_try <= f + 1e-4 * slope) || f_try < f
                    f_new = f_try
                    @inbounds for i in 1:d
                        g_try[i] = -g_try_ll[i]
                    end
                    accepted = true
                    break
                end
            end
            step *= 0.5
        end
        if !accepted
            if used_lbfgs
                empty!(S)
                empty!(Y)
                empty!(RHO)
                continue
            end
            break
        end

        sk = x_try .- x
        yk = g_try .- g
        sy = dot(sk, yk)
        ss = dot(sk, sk)
        if sy > 1e-12 * ss
            if length(S) == mem
                popfirst!(S)
                popfirst!(Y)
                popfirst!(RHO)
            end
            push!(S, sk)
            push!(Y, yk)
            push!(RHO, 1.0 / sy)
        end
        df = abs(f - f_new)
        copyto!(x, x_try)
        copyto!(g, g_try)
        f = f_new
        if df <= ftol * max(1.0, abs(f)) && maximum(abs, sk) <= 1e-5
            break
        end
    end
    return best_x, best_ll
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

# Drop columns of `Xf` that are constant or linearly dependent on earlier columns on the
# subsample, so `H = [1 Xf]` stays full column rank.
function _gp_full_rank_cols(Xf::AbstractMatrix{Float64})
    kept = Int[]
    basis = Vector{Float64}[]
    n_fit = size(Xf, 1)
    for j in axes(Xf, 2)
        col = view(Xf, :, j)
        v = col .- (sum(col) / n_fit)
        scale = sqrt(dot(v, v))
        scale > 0 || continue
        v ./= scale
        for _ in 1:2, b in basis
            v .-= dot(b, v) .* b
        end
        rest = sqrt(dot(v, v))
        if rest > 1e-8
            push!(basis, v ./ rest)
            push!(kept, j)
        end
    end
    return kept
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
    Xf_raw = X[fit, :]
    kept = _gp_full_rank_cols(Xf_raw)
    isempty(kept) && return G
    X_use = length(kept) < p ? X[:, kept] : X
    Xf = length(kept) < p ? Xf_raw[:, kept] : Xf_raw
    p_use = length(kept)
    n_hyper = max(method.hyper_points, p_use + 4)
    hyper = length(fit) <= n_hyper ? collect(eachindex(fit)) :
            sort!(_shuffle!(rng, collect(eachindex(fit)))[1:n_hyper])
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
    lo = vcat(fill(floor_δ, p_use), log(method.min_nugget))
    hi = vcat(fill(max(6.0, floor_δ + 8.0), p_use), 6.0)
    start = vcat(fill(max(log(sqrt(p_use)), floor_δ + log(2.0)), p_use), log(0.1))
    Xh = Xf[hyper, :]
    Hh = Hf[hyper, :]
    nh_len = length(hyper)
    noise_h = noise[hyper]
    C_buf = Matrix{Float64}(undef, nh_len, nh_len)
    A_buf = Matrix{Float64}(undef, nh_len, nh_len)
    P_buf = Matrix{Float64}(undef, nh_len, nh_len)
    Xs_buf = Matrix{Float64}(undef, nh_len, p_use)
    norms_buf = Vector{Float64}(undef, nh_len)
    MX_buf = Matrix{Float64}(undef, nh_len, p_use)
    msum_buf = Vector{Float64}(undef, nh_len)
    for a in axes(Qf, 1)
        raw = Qf[a, fit]
        centre = sum(raw) / length(raw)
        scale = sqrt(sum(abs2, raw .- centre) / length(raw))
        # A row that does not vary (the payoff of a reference action, say) is its mean.
        scale > 1e-12 * max(1.0, abs(centre)) || continue
        y = (raw .- centre) ./ scale
        yh = y[hyper]
        eval_fg!(g_out, θ_in) = _gp_ll_and_grad!(g_out, Xh, Hh, yh, noise_h, θ_in, lo, hi,
                                                 C_buf, A_buf, P_buf, Xs_buf, norms_buf,
                                                 MX_buf, msum_buf)
        θ, _ = _lbfgs_box(eval_fg!, start, lo, hi)
        δ, ν = exp.(θ[1:p_use]), exp(θ[p_use + 1])
        pieces = _gp_pieces(Xf, Hf, y, noise, δ, ν)
        for _ in 1:24
            pieces !== nothing && break
            ν = max(10ν, 1e-6)
            pieces = _gp_pieces(Xf, Hf, y, noise, δ, ν)
        end
        pieces === nothing && continue
        for block in Iterators.partition(1:n, 2000)
            Xb = X_use[block, :]
            fitted = pieces.β[1] .+ Xb * pieces.β[2:end] .+
                     _gp_correlation(Xb, Xf, δ) * pieces.α
            G[a, block] .= centre .+ scale .* fitted
        end
        # The fitted means average to the row's mean, as every regression here does.
        G[a, :] .+= means[a] - dot(wf, view(G, a, :))
    end
    return G
end
