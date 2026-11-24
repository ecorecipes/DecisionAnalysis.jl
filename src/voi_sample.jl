# Sample-based value of information. Every function here is an ESTIMATOR of a quantity
# defined at the top of `voi.jl` (the contract with the Lean development), for the
# distribution the draws of a `SampleBelief` came from. With conditional means
# g[a, i] ≈ E[Q[a, θ] | what is learned, at draw i] and weights w, all of them have the
# form
#
#   estimate = Σ_i w_i max_a g[a, i] − max_a Σ_i w_i Q[a, i],
#
# and differ only in how g is obtained. Monte Carlo error and bias are reported in
# separate fields and never folded into the value.

# Results and methods
# -------------------

const _Bias = @NamedTuple{act_now::Float64,cells::Union{Nothing,Float64}}

"""
    VOIEstimate

A sample-based estimate of a value of information, as returned by [`evpi`](@ref),
[`evppi`](@ref) and [`evsi`](@ref) when they are called with a method. Fields:

- `quantity`: `:evpi`, `:evppi` or `:evsi`; `method`: the name of the method.
- `value`: the estimate, in the units of the payoff (individuals).
- `standard_error`: the Monte Carlo standard error of `value` from the finite number
  of draws (and, for importance sampling, of simulated data sets as well: see
  [`ImportanceSampling`](@ref)), **kept apart and never folded into the value**. By default it is the linearised error of the
  weighted mean of the per-draw **policy gains** `Q[â_i, i] − Q[a*, i]`: what
  following the fitted policy (the action `â_i` with the largest fitted conditional
  mean at draw `i`) gains over the act-now action `a*`, evaluated on the payoffs
  themselves. To first order this is the error of the estimate including the
  uncertainty of the fit (for cell means it is exact, and the mean of the policy
  gains is the estimate). With `bootstrap = B` it is instead the standard deviation
  over `B` resamples of the draws with the fit redone each time.
- `bias`: a named tuple of the two selection biases that the optimiser's curse puts
  into the **two terms** of the estimate, each an estimate of an upward bias in one
  term (the expected maximum of unbiased estimates is at least the maximum of what
  they estimate), and again not subtracted:
  `act_now` is the upward bias of the act-now value `max_a Σ_i w_i Q[a, i]` from
  maximising sample means: the best reported value overstates, on average, what the
  chosen action is worth; `cells` is the upward bias of the perfect-information
  value `Σ_i w_i max_a Q[a, i]` from maximising noisy cell means (zero for a table
  with no standard errors, and `nothing` for methods that smooth over the noise).
  A value of information is the difference of two such terms, so **the sign of its
  bias is not determined in general**: `act_now` pulls it down and `cells` pulls a
  plain [`SampleAverage`](@ref) EVPI up. With exact cells only the first acts, and
  the plain EVPI of a sample is biased downwards. Both are plug-in normal
  approximations, simulated with a seeded `Xoshiro`; `cells` takes the noise of the
  actions in a cell to be independent unless a `noise_correlation` is given, and so
  overstates the bias under common random numbers (see [`crn_correlation`](@ref)).
- `details`: method-specific diagnostics (act-now value and action, effective sample
  sizes, residual variance, ...). For importance sampling the posterior effective
  sample sizes are part of the estimate and are printed with it. The estimates made
  from fitted conditional means also carry a **check of the fit**: `sample_evpi`,
  the plain EVPI of the same sample; `policy_gain`, the weighted mean of the policy
  gains, which is what following the fitted policy gains on the payoffs of the
  sample and is never above `sample_evpi`; and `above_sample_evpi`, `true` when
  `value` exceeds `sample_evpi` by more than twice the standard error. No
  information is worth more than perfect information, and a set of cell means
  respects that on every sample, so the flag is raised only by a regression whose
  fitted values are not conditional means (too few draws for its coefficients, or a
  misspecified form). The estimate is then wrong by more than its standard error
  says, a warning is logged, and the value is returned as computed, not capped.

`Float64(e)` and `e.value` give the number; arithmetic is deliberately not defined,
so that an error cannot be dropped by accident.
"""
struct VOIEstimate{T<:Real,D<:NamedTuple}
    quantity::Symbol
    method::Symbol
    value::T
    standard_error::Float64
    bias::_Bias
    details::D
    function VOIEstimate(quantity::Symbol, method::Symbol, value::T, standard_error::Real,
                         bias::NamedTuple, details::D) where {T<:Real,D<:NamedTuple}
        return new{T,D}(quantity, method, value, Float64(standard_error),
                        _Bias((bias.act_now, bias.cells)), details)
    end
end

Base.Float64(e::VOIEstimate) = Float64(e.value)

function Base.show(io::IO, e::VOIEstimate)
    print(io, "VOIEstimate(", e.quantity, " by ", e.method, ": ", _cell(e.value), " ± ",
          _cell(e.standard_error), " (standard error); act-now bias ",
          _cell(e.bias.act_now))
    e.bias.cells === nothing || print(io, ", cell-noise bias ", _cell(e.bias.cells))
    get(e.details, :above_sample_evpi, false) &&
        print(io, "; ABOVE the EVPI of the sample, ", _cell(e.details.sample_evpi),
              ": the fit is poor, do not use")
    haskey(e.details, :median_ess) &&
        print(io, "; posterior effective sample size: median ", _cell(e.details.median_ess),
              ", smallest ", _cell(e.details.minimum_ess), " of ", e.details.draws,
              " draws")
    return print(io, ")")
end

"""
    AbstractVOIMethod

Supertype of the methods that estimate a value of information from a
[`SampleBelief`](@ref): [`SampleAverage`](@ref), [`Grouping`](@ref),
[`StrongOakley`](@ref), [`Sadatsafavi`](@ref) and the regressions
([`AbstractRegression`](@ref): [`SplineRegression`](@ref), [`GAMRegression`](@ref),
[`GaussianProcessRegression`](@ref)).
"""
abstract type AbstractVOIMethod end

"""
    AbstractRegression <: AbstractVOIMethod

Supertype of the regression methods, which estimate the conditional mean of every
action's payoff given some columns (parameters or summary statistics) by regressing
the payoffs on them: [`SplineRegression`](@ref) (built in),
[`GAMRegression`](@ref) (extension, needs GAM.jl) and
[`GaussianProcessRegression`](@ref) (built in, for five or more columns). A regression smooths over the
simulation noise of the cells, so it is the estimator to use when cells are noisy.
"""
abstract type AbstractRegression <: AbstractVOIMethod end

"""
    SampleAverage()

The plain estimator: the definition applied to the sample as it stands. For
[`evpi`](@ref) it is `Σ_i w_i max_a Q[a, i] − max_a Σ_i w_i Q[a, i]`, the number that
`evpi(problem)` returns, now with its standard error and its two biases. It is right
for a table of exact conditional expectations; for a table of noisy cell means its
perfect-information term maximises over the noise and is biased upwards (see
[`VOIEstimate`](@ref)), and a regression is the better estimator.
"""
struct SampleAverage <: AbstractVOIMethod end

"""
    Grouping(; bins=nothing)

EVPPI for a discrete (grouping) variable: the conditional mean of each action's payoff
is its weighted mean within each cell, where a cell is one combination of values of
the focal columns. It is the estimator for a categorical parameter such as a
scenario label or a group, and it is `evppi(problem, partition)` with the partition
read from the draws. With `bins = k`, a single numeric focal parameter is first cut
into `k` bins of equal weight. Cells with few draws make the estimate noisy and
bias its value-with-information term upwards; the result reports the smallest cell.

Without `bins`, a floating-point focal parameter that takes a different value at
most draws is not a grouping variable: every draw would be a cell of its own and the
result would be the EVPI of the sample, whichever parameter was named. That throws
an `ArgumentError`; give `bins`, or use [`StrongOakley`](@ref),
[`Sadatsafavi`](@ref) or a regression.
"""
struct Grouping <: AbstractVOIMethod
    bins::Union{Nothing,Int}
    function Grouping(; bins::Union{Nothing,Integer}=nothing)
        bins === nothing || bins >= 1 ||
            throw(ArgumentError("bins must be at least 1, got $bins"))
        return new(bins)
    end
end

"""
    StrongOakley(; blocks=nothing)

Single-parameter EVPPI by the method of Strong and Oakley (2013, *Medical Decision
Making* 33:755): sort the draws by the focal parameter, cut them into `blocks`
consecutive blocks of equal weight, and take each action's mean payoff within a
block as its conditional mean there. No regression and no extra dependency. More
blocks follow the conditional mean more closely but average fewer draws each, which
biases the value-with-information term upwards; `blocks = nothing` uses `clamp(round(√n / 2), 2, 50)`.
"""
struct StrongOakley <: AbstractVOIMethod
    blocks::Union{Nothing,Int}
    function StrongOakley(; blocks::Union{Nothing,Integer}=nothing)
        blocks === nothing || blocks >= 1 ||
            throw(ArgumentError("blocks must be at least 1, got $blocks"))
        return new(blocks)
    end
end

"""
    Sadatsafavi(; separators=1)

Single-parameter EVPPI by the method of Sadatsafavi, Bansback, Zafari, Najafzadeh and
Marra (2013, *Value in Health* 16:438): sort the draws by the focal parameter and
find the `separators` cut points (one or two) that maximise the value of choosing
the best action separately in each segment. It is exact, as the sample grows, when
the best action changes at most `separators` times along the parameter. The search
is exhaustive: every position is tried for one cut, and every pair of positions for
two (in time proportional to draws × actions, by a running maximum), so the value
with two separators is never below the value with one. Choosing the cuts to maximise
the value biases the value-with-information term upwards in small samples.

**Checked against the R package `voi`** (1.0.3, method `"sal"`) on the same draws:
with one and with two separators the two agree to rounding (relative differences
below 1e-9) on the normal model of its vignette and on its chemotherapy example,
both of which have two actions. With more than two actions `voi` places the cuts
pair of actions by pair of actions, which need not give the best segmentation, so
this estimate can be larger than its value there. The fixtures and the numbers are
in `test/fixtures/voi` and `validation/`.
"""
struct Sadatsafavi <: AbstractVOIMethod
    separators::Int
    function Sadatsafavi(; separators::Integer=1)
        separators in (1, 2) ||
            throw(ArgumentError("separators must be 1 or 2, got $separators"))
        return new(separators)
    end
end

"""
    SplineRegression(; knots=4, degree=3, interactions=false)

The built-in regression: an additive regression spline fitted by weighted least
squares, with no dependency beyond LinearAlgebra. Each numeric column enters through
a truncated-power spline basis of the given `degree` with `knots` interior knots at
weighted quantiles; a non-numeric column, or a numeric one with no more distinct
values than the basis has functions, enters as a factor (one mean per level), so
that the fit reproduces cell means exactly when the columns are discrete. With
`interactions = true` the products of the basis functions of every pair of columns
are added (a tensor-product surface per pair), which needs many more draws.

It is meant for up to about four columns. There is no penalty and no smoothing
parameter: the flexibility is set by `knots`, and `SplineRegression(; knots=0,
degree=1)` is the linear metamodel of Jalal, Goldhaber-Fiebert and Kuntz (2015). For
penalised smooths use [`GAMRegression`](@ref), and for five or more columns
[`GaussianProcessRegression`](@ref).
"""
struct SplineRegression <: AbstractRegression
    knots::Int
    degree::Int
    interactions::Bool
    function SplineRegression(; knots::Integer=4, degree::Integer=3,
                              interactions::Bool=false)
        knots >= 0 || throw(ArgumentError("knots must be nonnegative, got $knots"))
        degree >= 1 || throw(ArgumentError("degree must be at least 1, got $degree"))
        return new(knots, degree, interactions)
    end
end

"""
    GAMRegression(; k=10, tensor=false)

Regression by a generalised additive model with penalised smooths, fitted by GAM.jl
(`github.com/ecorecipes/GAM.jl`): one smooth `s(x, k = k)` per numeric column, or a
tensor-product smooth of all of them with `tensor = true`; non-numeric columns enter
as factors. This is the regression that the R package `voi` uses by default for up to
four parameters (Strong, Oakley and Brennan 2014).

The method lives in a weak-dependency extension: load GAM.jl (`using GAM`) before
using it, otherwise [`MissingExtensionError`](@ref) is thrown. GAM.jl is not a
dependency of this package.
"""
struct GAMRegression <: AbstractRegression
    k::Int
    tensor::Bool
    function GAMRegression(; k::Integer=10, tensor::Bool=false)
        k >= 3 || throw(ArgumentError("k must be at least 3, got $k"))
        return new(k, tensor)
    end
end

_method_name(m::AbstractVOIMethod) = nameof(typeof(m))

# Helpers on samples
# ------------------

function _sample_problem(problem::DecisionProblem, operation::Symbol)
    problem.belief isa SampleBelief ||
        throw(BeliefFormError(operation, nameof(typeof(problem.belief)),
                              "the sample-based estimators need a SampleBelief; on a " *
                              "finite belief the value of information is exact " *
                              "(call the function without a method)"))
    return _feasible_Q(problem), problem.belief.weights
end

function _focal_names(focal)
    names = focal isa Union{Symbol,AbstractString} ? [Symbol(focal)] : _labels(focal)
    isempty(names) && throw(ArgumentError("name at least one focal parameter"))
    allunique(names) || throw(ArgumentError("a focal parameter is named twice: $names"))
    return names
end

function _focal_columns(belief::SampleBelief, focal)
    names = _focal_names(focal)
    for name in names
        haskey(belief.draws, name) ||
            throw(ArgumentError("the draws have no parameter $name; parameters: " *
                                "$(collect(keys(belief.draws)))"))
    end
    return NamedTuple{Tuple(names)}(Tuple(belief.draws[name] for name in names))
end

_is_numeric(column::AbstractVector) = eltype(column) <: Real && !(eltype(column) <: Bool)

function _single_numeric(columns::NamedTuple, method::AbstractVOIMethod)
    length(columns) == 1 ||
        throw(ArgumentError("$(_method_name(method)) is a single-parameter method; got " *
                            "$(length(columns)) focal parameters. Use a regression."))
    x = only(columns)
    _is_numeric(x) ||
        throw(ArgumentError("$(_method_name(method)) needs a numeric parameter; " *
                            "$(only(keys(columns))) is not. Use Grouping()."))
    return x
end

# Act-now value and the first action that attains it (ties: table order).
function _act_now(Q::AbstractMatrix, w::AbstractVector)
    means = Q * w
    return maximum(means), argmax(means)
end

# The estimate, from conditional means G[a, i], with the act-now value and action.
function _voi_from_fit(G::AbstractMatrix, Q::AbstractMatrix, w::AbstractVector)
    v0, a = _act_now(Q, w)
    best = [maximum(view(G, :, i)) for i in axes(G, 2)]
    return dot(w, best) - v0, v0, a
end

# Per-draw policy gains: the payoff of the action the fit prefers at each draw minus
# that of the act-now action. Their weighted mean is the estimate when the fit is a
# set of cell means, and their spread gives its first-order standard error, fit
# uncertainty included.
function _policy_gains(::AbstractVOIMethod, G::AbstractMatrix, Q::AbstractMatrix,
                       w::AbstractVector, a::Int)
    return [Float64(Q[argmax(view(G, :, i)), i]) - Float64(Q[a, i]) for i in axes(G, 2)]
end

# Linearised standard error of the weighted mean Σ w_i d_i of a self-normalised sample.
function _weighted_mean_se(d::AbstractVector, w::AbstractVector)
    m = sum(Float64(w[i]) * Float64(d[i]) for i in eachindex(d))
    return sqrt(sum(Float64(w[i])^2 * (Float64(d[i]) - m)^2 for i in eachindex(d)))
end

const _BIAS_DRAWS = 2000

# Upward bias of max_a (weighted sample mean of Q[a, ·]) under a normal approximation
# to the sampling distribution of the means: E max(m + Z) − max(m), Z ~ N(0, C).
function _act_now_bias(Q::AbstractMatrix, w::AbstractVector, rng::AbstractRNG)
    size(Q, 1) == 1 && return 0.0
    wf = Float64.(w)
    Qf = Float64.(Q)
    m = Qf * wf
    R = (Qf .- m) .* wf'
    F = eigen(Symmetric(R * R'))
    root = F.vectors * (sqrt.(max.(F.values, 0.0)) .* F.vectors')
    total = 0.0
    z = zeros(length(m))
    for _ in 1:_BIAS_DRAWS
        randn!(rng, z)
        total += maximum(m .+ root * z)
    end
    return max(total / _BIAS_DRAWS - maximum(m), 0.0)
end

# Upward bias of Σ_i w_i max_a Q̂[a, i] from normal noise in the cells, with a common
# correlation ρ between the actions of a cell. The shared part of the noise shifts
# every action alike and leaves the maximum's excess unchanged, so only the part
# scaled by √(1 − ρ) is simulated.
function _cell_noise_bias(Q::AbstractMatrix, se::AbstractMatrix, w::AbstractVector,
                          rng::AbstractRNG, ρ::Real=0)
    (size(Q, 1) == 1 || all(iszero, se) || ρ >= 1) && return 0.0
    se = sqrt(1 - ρ) .* se
    reps = clamp(200_000 ÷ size(Q, 2), 20, 400)
    z = zeros(size(Q, 1))
    total = 0.0
    for i in axes(Q, 2)
        q = Float64.(view(Q, :, i))
        s = Float64.(view(se, :, i))
        all(iszero, s) && continue
        acc = 0.0
        for _ in 1:reps
            randn!(rng, z)
            acc += maximum(q .+ s .* z)
        end
        total += Float64(w[i]) * (acc / reps - maximum(q))
    end
    return max(total, 0.0)
end

function _feasible_se(problem::DecisionProblem)
    se = problem.payoff.standard_errors
    return se === nothing ? nothing : se[problem.feasible, :]
end

# Standard deviation of `estimate(idx)` over `B` resamples of the draws.
function _bootstrap_se(estimate, n::Int, B::Int, rng::AbstractRNG)
    B >= 2 || throw(ArgumentError("bootstrap needs at least 2 resamples, got $B"))
    values = Float64[]
    for _ in 1:B
        push!(values, Float64(estimate(rand(rng, 1:n, n))))
    end
    m = sum(values) / B
    return sqrt(sum(v -> (v - m)^2, values) / (B - 1))
end

# Weighted quantiles of x at the probabilities ps (lower quantile of the weighted
# empirical distribution).
function _weighted_quantiles(x::AbstractVector, w::AbstractVector, ps)
    order = sortperm(x)
    cumulative = cumsum(Float64.(w[order]))
    total = cumulative[end]
    return [x[order[min(searchsortedfirst(cumulative, p * total - 1e-12 * total),
                        length(order))]] for p in ps]
end

# Conditional means by method
# ---------------------------

# Weighted mean of each row of Q within each cell; `cell[i]` is the cell of draw i.
function _cell_means(Q::AbstractMatrix, w::AbstractVector, cell::AbstractVector{Int},
                     ncells::Int)
    P = promote_type(eltype(Q), eltype(w))
    T = Base.promote_op(/, P, P)
    sums = zeros(T, size(Q, 1), ncells)
    mass = zeros(T, ncells)
    for i in axes(Q, 2)
        mass[cell[i]] += w[i]
        for a in axes(Q, 1)
            sums[a, cell[i]] += w[i] * Q[a, i]
        end
    end
    G = zeros(T, size(Q))
    for i in axes(Q, 2), a in axes(Q, 1)
        # A cell of zero weight contributes nothing whatever its mean is taken to be.
        G[a, i] = iszero(mass[cell[i]]) ? Q[a, i] : sums[a, cell[i]] / mass[cell[i]]
    end
    return G
end

function _cells(columns::NamedTuple)
    n = length(first(columns))
    index = Dict{Any,Int}()
    return [get!(index, Tuple(c[i] for c in columns), length(index) + 1) for i in 1:n],
           length(index)
end

# Consecutive blocks of (nearly) equal weight along the sorted parameter.
function _blocks_by_weight(x::AbstractVector, w::AbstractVector, k::Int)
    order = sortperm(x)
    total = sum(Float64, w)
    cell = zeros(Int, length(x))
    cumulative = 0.0
    for i in order
        mid = cumulative + Float64(w[i]) / 2
        cell[i] = clamp(floor(Int, mid / total * k) + 1, 1, k)
        cumulative += Float64(w[i])
    end
    return cell
end

function _fit(method::Grouping, columns::NamedTuple, Q, w)
    if method.bins !== nothing
        x = _single_numeric(columns, method)
        cell = _blocks_by_weight(x, w, method.bins)
        ncells = method.bins
    else
        cell, ncells = _cells(columns)
        n = length(cell)
        continuous = [k for (k, c) in pairs(columns) if eltype(c) <: AbstractFloat]
        !isempty(continuous) && n >= 2 && 2 * ncells > n &&
            throw(ArgumentError("Grouping() found $ncells cells in $n draws: " *
                                "$(join(continuous, ", ")) is continuous, so nearly " *
                                "every draw is a cell of its own and the result would " *
                                "be the EVPI whatever the parameter. Use " *
                                "Grouping(; bins=k) for one numeric parameter, " *
                                "StrongOakley(), Sadatsafavi() or a regression"))
    end
    sizes = zeros(Int, ncells)
    foreach(c -> sizes[c] += 1, cell)
    return _cell_means(Q, w, cell, ncells),
           (; cells=count(>(0), sizes), smallest_cell=minimum(filter(>(0), sizes)))
end

function _fit(method::StrongOakley, columns::NamedTuple, Q, w)
    x = _single_numeric(columns, method)
    n = length(x)
    k = method.blocks === nothing ? clamp(round(Int, sqrt(n) / 2), 2, 50) : method.blocks
    k = min(k, n)
    return _cell_means(Q, w, _blocks_by_weight(x, w, k), k), (; blocks=k)
end

function _fit(method::Sadatsafavi, columns::NamedTuple, Q, w)
    x = _single_numeric(columns, method)
    order = sortperm(x)
    n = length(order)
    C = cumsum(Float64.(Q[:, order]) .* Float64.(w[order])'; dims=2)
    total = C[:, n]
    # A cut after sorted position s is allowed only where the parameter changes.
    allowed = [s for s in 1:(n - 1) if x[order[s]] != x[order[s + 1]]]
    segment(from, to) = maximum(from == 0 ? C[:, to] : C[:, to] .- C[:, from])
    cuts = Int[]
    best = maximum(total)
    head = [segment(0, s) for s in allowed]
    tail = [maximum(total .- C[:, s]) for s in allowed]
    for (i, s) in enumerate(allowed)
        v = head[i] + tail[i]
        v > best && ((best, cuts) = (v, [s]))
    end
    if method.separators == 2
        # Two cuts s < t with action a in between are worth
        # (head[s] − C[a, s]) + (C[a, t] + tail[t]): for each a, one pass over t with
        # the running maximum of the first bracket over the cuts before it.
        for a in axes(C, 1)
            before, at = -Inf, 0
            for (j, t) in enumerate(allowed)
                if j > 1
                    v = before + C[a, t] + tail[j]
                    v > best && ((best, cuts) = (v, [at, t]))
                end
                lead = head[j] - C[a, t]
                lead > before && ((before, at) = (lead, t))
            end
        end
    end
    cell = ones(Int, n)
    for (position, i) in enumerate(order)
        cell[i] = 1 + count(<(position), cuts)
    end
    return _cell_means(Q, w, cell, length(cuts) + 1),
           (; cut_points=[(x[order[s]] + x[order[s + 1]]) / 2 for s in cuts])
end

# Design matrix of the additive regression spline.
function _spline_blocks(method::SplineRegression, columns::NamedTuple, w)
    blocks = Matrix{Float64}[]
    for x in columns
        levels = unique(x)
        size_of_basis = method.degree + method.knots
        if !_is_numeric(x) || length(levels) <= size_of_basis + 1
            # A factor: one indicator per level but the first.
            push!(blocks,
                  Float64[isequal(x[i], l) for i in eachindex(x), l in levels[2:end]])
            continue
        end
        lo, hi = extrema(x)
        z = (Float64.(x) .- lo) ./ (hi - lo)
        probabilities = [k / (method.knots + 1) for k in 1:(method.knots)]
        knots = unique(_weighted_quantiles(z, w, probabilities))
        B = zeros(length(z), method.degree + length(knots))
        for p in 1:(method.degree)
            B[:, p] .= z .^ p
        end
        for (k, κ) in enumerate(knots)
            B[:, method.degree + k] .= max.(z .- κ, 0.0) .^ method.degree
        end
        push!(blocks, B)
    end
    return blocks
end

function _design(method::SplineRegression, columns::NamedTuple, w)
    blocks = _spline_blocks(method, columns, w)
    n = length(first(columns))
    parts = Matrix{Float64}[ones(n, 1)]
    append!(parts, blocks)
    if method.interactions
        for i in eachindex(blocks), j in (i + 1):length(blocks)
            A, B = blocks[i], blocks[j]
            push!(parts,
                  reduce(hcat,
                         [A[:, p] .* B[:, q] for p in axes(A, 2) for q in axes(B, 2)];
                         init=zeros(n, 0)))
        end
    end
    return reduce(hcat, parts)
end

"""
    fitted_conditional_means(method::AbstractRegression, columns, Q, w) -> Matrix

The regression step that every regression method implements: the fitted conditional
means `G[a, i]` of each row of `Q` (actions × draws) given the `columns` (a named
tuple of vectors, one entry per draw), fitted with the draw weights `w`. The weighted
mean of each fitted row must equal that of the row itself (true of any weighted
least-squares fit with an intercept). A new regression backend subtypes
[`AbstractRegression`](@ref) and adds a method.
"""
function fitted_conditional_means(method::SplineRegression, columns::NamedTuple,
                                  Q::AbstractMatrix, w::AbstractVector)
    A = _design(method, columns, w)
    n, p = size(A)
    p <= n ||
        throw(ArgumentError("the regression has $p coefficients for $n draws; use " *
                            "fewer knots, fewer parameters or no interactions"))
    root = sqrt.(Float64.(w))
    F = qr(root .* A, ColumnNorm())
    coefficients = F \ (root .* permutedims(Float64.(Q)))
    return permutedims(A * coefficients)
end

function fitted_conditional_means(method::GAMRegression, columns, Q, w)
    throw(MissingExtensionError(:GAMRegression, :GAM))
end

function _fit(method::AbstractRegression, columns::NamedTuple, Q, w)
    G = fitted_conditional_means(method, columns, Q, w)
    residual = sum(Float64(w[i]) * (Float64(Q[a, i]) - G[a, i])^2
                   for a in axes(Q, 1), i in axes(Q, 2)) / size(Q, 1)
    return G, (; residual_variance=residual)
end

# One estimate from a fit, with its errors.
function _estimate(quantity::Symbol, method::AbstractVOIMethod, columns::NamedTuple,
                   Q::AbstractMatrix, w::AbstractVector; bootstrap::Integer, seed::Integer,
                   cells=nothing, extra=(;))
    G, info = _fit(method, columns, Q, w)
    value, v0, a = _voi_from_fit(G, Q, w)
    rng = Xoshiro(seed)
    bias = (; act_now=_act_now_bias(Q, w, rng), cells)
    gains = _policy_gains(method, G, Q, w, a)
    se = _weighted_mean_se(gains, w)
    if bootstrap > 0
        se = _bootstrap_se(length(w), bootstrap, rng) do idx
            wb = Float64.(w[idx])
            wb ./= sum(wb)
            Qb = Q[:, idx]
            Gb, _ = _fit(method, map(c -> c[idx], columns), Qb, wb)
            return first(_voi_from_fit(Gb, Qb, wb))
        end
    end
    sample_evpi = dot(w, [maximum(view(Q, :, i)) for i in axes(Q, 2)]) - v0
    above = _above_sample_evpi(value, sample_evpi, se, v0)
    above &&
        @warn "The $quantity estimate by $(_method_name(method)), $(_cell(value)), is " *
              "above the EVPI of the same sample, $(_cell(sample_evpi)), by more than " *
              "twice its standard error. No information is worth more than perfect " *
              "information, so the fitted values are not conditional means here (too " *
              "few draws for the fit, or a misspecified regression) and the estimate " *
              "should not be used. The standard error does not show this. Use fewer " *
              "knots or parameters, StrongOakley() or Grouping(; bins), or more draws."
    details = merge((; act_now_value=v0, act_now_index=a, draws=length(w),
                     effective_sample_size=Float64(effective_sample_size(w)),
                     bootstrap=Int(bootstrap), sample_evpi,
                     policy_gain=sum(Float64(w[i]) * gains[i] for i in eachindex(gains)),
                     above_sample_evpi=above), info, extra)
    return VOIEstimate(quantity, _method_name(method), value, se, bias, details)
end

# Whether an estimate is above the plain EVPI of the same sample by more than twice its
# standard error (and more than rounding). A set of cell means never is (Jensen: the
# maximum of a mean is at most the mean of the maxima); a regression can be, when its
# fitted values are not conditional means.
function _above_sample_evpi(value, sample_evpi, se::Real, v0)
    scale = max(1.0, abs(Float64(v0)), abs(Float64(sample_evpi)))
    return Float64(value) - Float64(sample_evpi) > 2 * se + sqrt(eps()) * scale
end

# EVPI
# ----

"""
    evpi(problem::DecisionProblem, method::AbstractVOIMethod; bootstrap=0, seed=1,
         noise_correlation=0) -> VOIEstimate

The expected value of perfect information estimated from a [`SampleBelief`](@ref),
with its standard error and biases kept apart ([`VOIEstimate`](@ref)).

- [`SampleAverage`](@ref): the definition on the sample, the number `evpi(problem)`
  gives. Right when the cells are exact conditional expectations. When the table has
  standard errors, `bias.cells` estimates how much the cell noise inflates its
  perfect-information term. That estimate takes the noise of different actions in a
  cell to have the correlation `noise_correlation` (`SampleAverage` only). The
  default, zero, is right for independent runs and **overstates the bias under
  common random numbers**, where the noise is shared between actions; pass the
  achieved correlation, [`crn_correlation`](@ref)`(outcomes, utility)`.
- A regression ([`SplineRegression`](@ref), [`GAMRegression`](@ref),
  [`GaussianProcessRegression`](@ref) when the draws carry many parameters): the
  **all-parameter regression estimate**. Each action's payoff is regressed on all the
  parameters of the draws and the fitted values replace the noisy cells, so the
  simulation noise is smoothed out instead of being maximised over. This is the
  estimator to use for a stochastic simulator; it reads "the value of learning every
  parameter", which is the EVPI when the parameters are all that is uncertain apart
  from the simulator's own randomness. It needs the draws to carry parameters.

`bootstrap = B` replaces the linearised standard error by the standard deviation
over `B` resamples of the draws; `seed` seeds the `Xoshiro` used for the resamples
and the bias simulations. See [`variance_components`](@ref) for the split between
parameter and simulation variance.
"""
function evpi(problem::DecisionProblem, method::SampleAverage; bootstrap::Integer=0,
              seed::Integer=1, noise_correlation::Real=0)
    -1 <= noise_correlation <= 1 ||
        throw(ArgumentError("noise_correlation must lie in [-1, 1], got " *
                            "$noise_correlation"))
    Q, w = _sample_problem(problem, :evpi)
    se = _feasible_se(problem)
    cells = se === nothing ? 0.0 :
            _cell_noise_bias(Q, se, w, Xoshiro(seed + 1), noise_correlation)
    return _estimate(:evpi, method, (;), Q, w; bootstrap, seed, cells,
                     extra=(; single_replicate=single_replicate(problem.payoff),
                            noise_correlation=Float64(noise_correlation)))
end
_fit(::SampleAverage, ::NamedTuple, Q, w) = (Q, (;))

function evpi(problem::DecisionProblem, method::AbstractRegression; bootstrap::Integer=0,
              seed::Integer=1)
    Q, w = _sample_problem(problem, :evpi)
    columns = problem.belief.draws
    isempty(columns) &&
        throw(ArgumentError("the draws carry no parameters to regress on; build the " *
                            "SampleBelief from a table of parameter draws"))
    return _estimate(:evpi, method, columns, Q, w; bootstrap, seed,
                     extra=(; parameters=collect(keys(columns))))
end

function evpi(problem::DecisionProblem, method::AbstractVOIMethod; kw...)
    throw(ArgumentError("$(_method_name(method)) estimates a partial value of " *
                        "information; for EVPI use SampleAverage() or a regression"))
end

"""
    variance_components(problem::DecisionProblem) -> Vector{NamedTuple}

The analysis-of-variance split of each feasible action's payoff between **parameter**
and **simulation** variance, for a sample problem whose table has standard errors.
One `(; action, total, simulation, parameter, parameter_share)` per action:

- `total`: the weighted variance of the cell means across the draws;
- `simulation`: the weighted mean of the squared standard errors of the cells, the
  part of `total` that is simulation noise left in the cell means;
- `parameter = total − simulation`: the variance of the conditional expected payoff
  across draws, which is what information about the parameters can act on. It is
  returned as computed, so it can be slightly negative when noise dominates;
- `parameter_share = parameter / total`.

A small parameter share says that the cells need more replicates before a
[`SampleAverage`](@ref) value of information means much. To split the variance of a
difference between two actions, apply this to a table of gains
(`payoff_table(outcomes, utility; baseline)`).
"""
function variance_components(problem::DecisionProblem)
    Q, w = _sample_problem(problem, :variance_components)
    se = _feasible_se(problem)
    se === nothing &&
        throw(ArgumentError("the payoff table has no standard errors, so the " *
                            "simulation variance is unknown; build it with " *
                            "payoff_table from an OutcomeTable with replicates"))
    wf = Float64.(w)
    return map(enumerate(feasible_actions(problem))) do (a, action)
        q = Float64.(view(Q, a, :))
        m = dot(wf, q)
        total = sum(wf .* (q .- m) .^ 2)
        simulation = sum(wf .* Float64.(view(se, a, :)) .^ 2)
        parameter = total - simulation
        return (; action, total, simulation, parameter,
                parameter_share=iszero(total) ? NaN : parameter / total)
    end
end

# EVPPI
# -----

# The default method by the number of focal parameters, as the R package `voi` chooses:
# a regression with smooths up to four, a Gaussian process from five on. A parameter
# that is not numeric is grouped; when it comes with a continuous one, grouping would
# make every draw a cell, and the spline regression takes it as a factor instead.
const _GP_FROM = 5
function _default_method(columns::NamedTuple)
    if !all(_is_numeric, columns)
        return any(c -> eltype(c) <: AbstractFloat, columns) ? SplineRegression() :
               Grouping()
    end
    return length(columns) < _GP_FROM ? SplineRegression() : GaussianProcessRegression()
end

"""
    evppi(problem::DecisionProblem, focal, method; bootstrap=0, seed=1) -> VOIEstimate
    evppi(problem::DecisionProblem, focal::Symbol; kw...) -> VOIEstimate

The expected value of partial perfect information about the named parameters of a
[`SampleBelief`](@ref): the value of learning `focal` (a parameter name, or a tuple or
vector of names, columns of the draws) exactly and nothing else. An estimate of the
EVPPI defined for [`evppi`](@ref)`(problem, partition)`, returned with its standard
error and biases ([`VOIEstimate`](@ref)). Methods:

| method | focal | idea |
|:--|:--|:--|
| [`Grouping`](@ref) | discrete, any number | conditional means by cell |
| [`StrongOakley`](@ref) | one numeric | means within blocks of the sorted parameter |
| [`Sadatsafavi`](@ref) | one numeric | best one or two cut points |
| [`SplineRegression`](@ref) | up to about four | additive regression spline, built in |
| [`GAMRegression`](@ref) | up to about four | penalised smooths, needs GAM.jl |
| [`GaussianProcessRegression`](@ref) | five or more | Gaussian process, built in |

**Without a method, the method is chosen by the focal parameters**, by the rule of
the R package `voi`: a regression with smooths for up to four parameters and a
Gaussian process from five on. Here that is `Grouping()` when some focal parameter is
not numeric and none is floating-point, `SplineRegression()` for one to four numeric
parameters (the built-in stand-in for the generalised additive model, which needs
GAM.jl: pass `GAMRegression()` to use it) and `GaussianProcessRegression()` for five
or more. A floating-point parameter named together with a non-numeric one gets
`SplineRegression()`, which takes the non-numeric one as a factor (grouping the two
would make every draw a cell of its own). The
method used is recorded in the `method` field of the result. The single-parameter
methods and `Grouping` need no regression and are useful cross-checks on it. All of them average over the
simulation noise in the cells, provided each block, segment or cell holds many
draws.

A regression estimate is only as good as its fit. Fitted values are not conditional
means when the regression is misspecified or has too few draws for its coefficients,
and the estimate is then not bounded by the EVPI: a value above the plain EVPI of
the same sample is a sign of a poor fit, not of valuable information, and **its
standard error does not show it**. Such an estimate is flagged
(`details.above_sample_evpi`, with `details.sample_evpi` and `details.policy_gain`
beside it: see [`VOIEstimate`](@ref)) and a warning is logged; the value is returned
as computed. `Grouping()` with no `bins` on a continuous parameter, where nearly
every draw is a cell of its own, would return the EVPI whatever the parameter is,
and throws an `ArgumentError` instead.

`bootstrap` and `seed` are as for [`evpi`](@ref).

```jldoctest
julia> belief = SampleBelief((; site=[:wet, :wet, :dry, :dry], x=[1, 2, 3, 4]));

julia> p = DecisionProblem(PayoffTable([0 0 0 0; 2 4 -4 -6]), belief);

julia> evppi(p, :site).value
3//2
```
"""
function evppi(problem::DecisionProblem, focal, method::AbstractVOIMethod;
               bootstrap::Integer=0, seed::Integer=1)
    Q, w = _sample_problem(problem, :evppi)
    method isa SampleAverage &&
        throw(ArgumentError("SampleAverage has no notion of a focal parameter; use " *
                            "Grouping(), StrongOakley(), Sadatsafavi() or a regression"))
    columns = _focal_columns(problem.belief, focal)
    return _estimate(:evppi, method, columns, Q, w; bootstrap, seed,
                     extra=(; focal=collect(keys(columns))))
end
function evppi(problem::DecisionProblem, focal::Union{Symbol,Tuple{Vararg{Symbol}}}; kw...)
    _sample_problem(problem, :evppi)
    return evppi(problem, focal, _default_method(_focal_columns(problem.belief, focal));
                 kw...)
end

# EVSI
# ----

"""
    SummaryStatistics(summaries; regression=SplineRegression())

A study described by **summary statistics of its simulated data**, for
[`evsi`](@ref) by the regression method of Strong, Oakley, Brennan and Breeze (2015,
*Medical Decision Making* 35:570). `summaries` is a Tables.jl table with one row per
draw: for each draw, one data set is simulated from the study under that draw's
parameters, and reduced to a few numbers (a sample mean, a count of positives). The
EVSI is then the value of learning the summaries, estimated by regressing the
payoffs on them with `regression`. The summaries must be (close to) sufficient for
what the study tells about the parameters, and few: the built-in regression suits up
to about four.
"""
struct SummaryStatistics{C<:NamedTuple,R<:AbstractRegression}
    summaries::C
    regression::R
end
function SummaryStatistics(summaries; regression::AbstractRegression=SplineRegression())
    columns = _column_table(summaries)
    isempty(columns) && throw(ArgumentError("no summary statistic was given"))
    return SummaryStatistics(columns, regression)
end

"""
    ImportanceSampling(loglikelihood; weights=nothing)

A study described by **the likelihood of its simulated data**, for [`evsi`](@ref) by
the importance-sampling method of Menzies (2016, *Medical Decision Making* 36:308).
`loglikelihood[k, j]` is the log-likelihood of simulated data set `k` under the
parameters of draw `j` (rows = data sets, columns = draws; any additive constant per
row is harmless). Data set `k` must have been simulated from the prior predictive
distribution: draw a parameter set from the sample, then data given it.

`weights` are the weights of the data sets. With `nothing`, there must be one data
set per draw, data set `k` simulated from draw `k`, and they take the weights of the
draws; if the data sets were instead simulated from draws picked with probability
proportional to their weights, pass `weights = :equal`. A vector gives them
explicitly.

For each data set the draws are reweighted by their likelihood, which gives the
posterior expected payoff of every action without refitting anything. See
[`importance_loglikelihoods`](@ref) to build the matrix from a data simulator and a
log-likelihood function.

**Effective sample size.** The estimate degrades when the data are so informative
that a few draws carry all the weight: each posterior mean is then an average over
few draws, the maximum over actions is taken over noisy means, and the estimate
drifts upwards towards the EVPI. The result therefore reports the posterior
effective sample size `1 / Σ_j π_kj²` of the data sets in `details` (`minimum_ess`,
`median_ess`, `mean_ess`, beside `prior_ess`, that of the draws themselves) and
prints the median and the smallest with the estimate. A median of a few tens or less
says that more draws are needed, or a regression on summary statistics
([`SummaryStatistics`](@ref)).

**Standard error.** The estimate has two sources of Monte Carlo error and
`standard_error` combines them, `√(datasets² + draws²)`, so that it is comparable
with the other estimators:

- `details.datasets_standard_error`: from the finite number of simulated data sets,
  the error of the weighted mean of the per-data-set gains;
- `details.draws_standard_error`: from the finite number of draws, by the delta
  method. Each posterior mean is a self-normalised weighted mean over the draws; its
  linearisation gives the first-order effect of draw `j` on the whole estimate,
  `Σ_k v_k π_kj (D_kj − g_k)`, with `π_kj` the posterior weight of draw `j` for data
  set `k`, `D_kj` the payoff of the action chosen for data set `k` minus that of the
  act-now action at draw `j`, and `g_k` its posterior mean; the sum of the squares of
  these effects over the draws is the variance.

The two are treated as independent, which ignores that a data set was simulated from
one of the draws, and the delta method is first-order: with a small effective sample
size it understates the error and says nothing of the upward drift above. On the
normal model of the R `voi` vignette (EVSI of 10 or 100 observations of `p1`; 400 to
2000 draws, 100 to 1000 data sets, 150 repetitions of each of five designs) the reported standard
error was between 0.91 and 1.13 times the standard deviation of the estimate over
the repetitions; the error from the data sets alone was between 0.27 and 0.79 times.
"""
struct ImportanceSampling{M<:AbstractMatrix{<:Real},W}
    loglikelihood::M
    weights::W
    function ImportanceSampling(loglikelihood::AbstractMatrix{<:Real}; weights=nothing)
        isempty(loglikelihood) && throw(ArgumentError("the log-likelihood matrix is empty"))
        any(isnan, loglikelihood) &&
            throw(ArgumentError("the log-likelihood matrix has a NaN entry"))
        any(==(Inf), loglikelihood) &&
            throw(ArgumentError("the log-likelihood matrix has a +Inf entry"))
        weights === nothing || weights === :equal ||
            (weights isa AbstractVector{<:Real} &&
             length(weights) == size(loglikelihood, 1)) ||
            throw(ArgumentError("weights must be nothing, :equal or one number per " *
                                "data set (row)"))
        return new{typeof(loglikelihood),typeof(weights)}(loglikelihood, weights)
    end
end

"""
    importance_loglikelihoods(simulate_data, loglikelihood, belief::SampleBelief;
                              datasets=nothing, seed) -> ImportanceSampling

Build the log-likelihood matrix of an [`ImportanceSampling`](@ref) study from two
functions of the user's model:

- `simulate_data(rng, θ)`: one data set of the study under the parameters `θ` (a
  named tuple, one row of the draws);
- `loglikelihood(data, θ)`: the log-likelihood of that data set under `θ`.

With `datasets = nothing`, one data set is simulated from every draw (and they carry
the draws' weights). With `datasets = K`, `K` draws are picked with probability
proportional to their weights and one data set is simulated from each (equal
weights). Every random number comes from `Xoshiro(seed)`; `seed` is required. The
cost is `K × n` likelihood evaluations.
"""
function importance_loglikelihoods(simulate_data, loglikelihood, belief::SampleBelief;
                                   datasets::Union{Nothing,Integer}=nothing,
                                   seed::Union{Nothing,Integer}=nothing)
    seed === nothing && throw(ArgumentError("give `seed`"))
    isempty(belief.draws) &&
        throw(ArgumentError("the draws carry no parameters to simulate data from"))
    rng = Xoshiro(seed)
    rows = Tables.rowtable(belief.draws)
    n = length(rows)
    if datasets === nothing
        sources = collect(1:n)
        weights = nothing
    else
        datasets >= 1 || throw(ArgumentError("datasets must be at least 1"))
        cumulative = cumsum(Float64.(belief.weights))
        sources = [min(searchsortedfirst(cumulative, rand(rng) * cumulative[end]), n)
                   for _ in 1:datasets]
        weights = :equal
    end
    L = zeros(length(sources), n)
    for (k, source) in enumerate(sources)
        data = simulate_data(rng, rows[source])
        for j in 1:n
            L[k, j] = loglikelihood(data, rows[j])
        end
    end
    return ImportanceSampling(L; weights)
end

"""
    GaussianApproximation(focal; n0, n, regression=nothing)

A study that measures the focal parameters with `n` observations, for [`evsi`](@ref)
by the Gaussian approximation of Jalal and Alarid-Escudero (2018, *Medical Decision
Making* 38:174). `n0` is the **prior effective sample size**: the number of
observations of the study that the prior on the focal parameters is worth (for a
normal prior with variance `τ²` and observations with variance `σ²`, `n0 = σ²/τ²`).
The conditional means of the payoffs given the focal parameters are fitted once (as
for [`evppi`](@ref), with `regression`, or the default method for the columns) and
shrunk towards their mean by the factor `√(n / (n0 + n))`, which approximates their
preposterior distribution. One fit therefore gives the EVSI at any sample size: `n`
may be a number or a vector, and with a vector [`evsi`](@ref) returns one estimate per
size, a curve that rises to the EVPPI of the focal parameters.

The approximation is exact when the conditional means are linear in a normal
parameter with normal data, and otherwise as good as the normal approximation to the
prior and the posterior.

**Standard error.** The default standard error is the linearised one of
[`VOIEstimate`](@ref) with the residuals of the fit scaled by the shrinkage factor.
It covers the sampling of the draws and, to first order, the fit; it does **not**
cover the error of the approximation itself (a wrong `n0`, a prior or posterior far
from normal), which is a bias and not a Monte Carlo error. On the normal model of the
R `voi` vignette (EVSI of 1, 10 and 100 observations of `p1`; 1000 and 4000 draws,
300 repetitions of each) it was between 1.02 and 1.05 times the standard deviation of
the estimate over the repetitions with the default regression, and 1.25 times
(conservative) with a linear fit, whose error is smaller than the linearisation
allows for; `bootstrap = 100` gave 0.94 to 0.99 times.
"""
struct GaussianApproximation{N,M}
    focal::Vector{Symbol}
    n0::Float64
    n::N
    regression::M
    function GaussianApproximation(focal; n0::Real, n::Union{Real,AbstractVector{<:Real}},
                                   regression::Union{Nothing,AbstractVOIMethod}=nothing)
        n0 > 0 || throw(ArgumentError("the prior effective sample size n0 must be " *
                                      "positive, got $n0"))
        all(>=(0), n) || throw(ArgumentError("sample sizes must be nonnegative"))
        regression isa SampleAverage &&
            throw(ArgumentError("SampleAverage cannot fit conditional means"))
        names = _focal_names(focal)
        return new{typeof(n),typeof(regression)}(names, Float64(n0), n, regression)
    end
end

"""
    evsi(problem::DecisionProblem, study; bootstrap=0, seed=1)

The expected value of sample information estimated from a [`SampleBelief`](@ref):
what the data of `study` are worth before they are seen, gross of any cost. An
estimate of the EVSI defined for [`evsi`](@ref)`(problem, experiment)`, with its
standard error and biases kept apart ([`VOIEstimate`](@ref)). `study` says what would
be observed and fixes the method:

- [`SummaryStatistics`](@ref): regression of the payoffs on summary statistics of
  one simulated data set per draw (Strong et al. 2015). Use it when the data reduce
  to a few numbers.
- [`ImportanceSampling`](@ref): reweighting of the draws by the likelihood of
  simulated data sets (Menzies 2016). Use it when a log-likelihood is available and
  no low-dimensional summary is; watch the effective sample sizes in `details`.
- [`GaussianApproximation`](@ref): shrinkage of the fitted conditional means (Jalal
  and Alarid-Escudero 2018). Use it for a curve over the sample size; with a vector
  of sizes the result is a vector of estimates.

The choice follows Kunst et al. (2020, *Value in Health* 23:734). The estimates
are bounded by the EVPPI of what the study informs and by the EVPI only up to Monte
Carlo error, since each comes from a different estimator. `bootstrap` (not available
for importance sampling, whose standard error already covers both the data sets and
the draws: see [`ImportanceSampling`](@ref)) and `seed` are as for [`evpi`](@ref).
"""
function evsi(problem::DecisionProblem, study::SummaryStatistics; bootstrap::Integer=0,
              seed::Integer=1)
    Q, w = _sample_problem(problem, :evsi)
    n = length(first(study.summaries))
    n == length(w) ||
        throw(ArgumentError("$n rows of summary statistics for $(length(w)) draws"))
    return _estimate(:evsi, study.regression, study.summaries, Q, w; bootstrap, seed,
                     extra=(; summaries=collect(keys(study.summaries))))
end

function evsi(problem::DecisionProblem, study::ImportanceSampling; seed::Integer=1)
    Q, w = _sample_problem(problem, :evsi)
    L = study.loglikelihood
    K, n = size(L)
    n == length(w) ||
        throw(ArgumentError("the log-likelihood matrix has $n columns for " *
                            "$(length(w)) draws"))
    study.weights === nothing && K != n &&
        throw(ArgumentError("with weights = nothing there must be one data set per " *
                            "draw ($n), got $K; pass weights = :equal or a vector"))
    v = study.weights === nothing ? Float64.(w) :
        study.weights === :equal ? fill(1 / K, K) :
        Float64.(_normalise_weights(collect(study.weights)))
    v0, a = _act_now(Q, w)
    wf = Float64.(w)
    Qf = Float64.(Q)
    gains = zeros(K)
    informed = zeros(K)
    ess = zeros(K)
    posterior = zeros(n)
    # influence[j]: the first-order effect of draw j on the estimate, summed over the
    # data sets, from the linearisation of each self-normalised posterior mean.
    influence = zeros(n)
    for k in 1:K
        top = maximum(view(L, k, :))
        isfinite(top) ||
            throw(ArgumentError("data set $k has zero likelihood under every draw"))
        posterior .= wf .* exp.(view(L, k, :) .- top)
        posterior ./= sum(posterior)
        m = Qf * posterior
        b = argmax(m)
        informed[k] = m[b]
        gains[k] = m[b] - m[a]
        ess[k] = 1 / sum(abs2, posterior)
        for j in 1:n
            influence[j] += v[k] * posterior[j] * (Qf[b, j] - Qf[a, j] - gains[k])
        end
    end
    value = dot(v, gains)
    sorted = sort(ess)
    datasets_se = _weighted_mean_se(gains, v)
    draws_se = sqrt(sum(abs2, influence))
    details = (; act_now_value=v0, act_now_index=a, draws=n, datasets=K,
               direct_value=dot(v, informed) - Float64(v0), minimum_ess=sorted[1],
               median_ess=sorted[(K + 1) ÷ 2], mean_ess=dot(v, ess),
               prior_ess=Float64(effective_sample_size(w)),
               datasets_standard_error=datasets_se, draws_standard_error=draws_se)
    bias = (; act_now=_act_now_bias(Q, w, Xoshiro(seed)), cells=nothing)
    return VOIEstimate(:evsi, :ImportanceSampling, value, hypot(datasets_se, draws_se),
                       bias, details)
end

function evsi(problem::DecisionProblem, study::GaussianApproximation; bootstrap::Integer=0,
              seed::Integer=1)
    Q, w = _sample_problem(problem, :evsi)
    columns = _focal_columns(problem.belief, study.focal)
    method = study.regression === nothing ? _default_method(columns) : study.regression
    function one_size(n)
        return _estimate(:evsi, _Shrunk(method, sqrt(n / (study.n0 + n))), columns, Q,
                         w; bootstrap, seed,
                         extra=(; focal=study.focal, n0=study.n0, n=n))
    end
    return study.n isa Real ? one_size(study.n) : [one_size(n) for n in study.n]
end

# A fit shrunk towards its weighted mean: the preposterior conditional means.
struct _Shrunk{M<:AbstractVOIMethod} <: AbstractVOIMethod
    method::M
    factor::Float64
end
_method_name(::_Shrunk) = :GaussianApproximation
function _fit(s::_Shrunk, columns::NamedTuple, Q, w)
    G, info = _fit(s.method, columns, Q, w)
    centre = G * Float64.(w)
    return centre .+ s.factor .* (G .- centre),
           merge(info, (; shrinkage=s.factor, fit=_method_name(s.method)))
end
# The shrunk fit is `centre + factor · (fit − centre)`, so the residuals of the fit
# enter the estimate scaled by the factor.
function _policy_gains(s::_Shrunk, G::AbstractMatrix, Q::AbstractMatrix, w::AbstractVector,
                       a::Int)
    iszero(s.factor) && return zeros(size(G, 2))
    centre = G * Float64.(w)
    R = Float64.(Q) .- (centre .+ (G .- centre) ./ s.factor)
    return map(axes(G, 2)) do i
        b = argmax(view(G, :, i))
        return G[b, i] - G[a, i] + s.factor * (R[b, i] - R[a, i])
    end
end

"""
    enbs(estimate::VOIEstimate; cost, population=1) -> NamedTuple

The expected net benefit of sampling from an estimated EVSI:
`population · EVSI − cost`, returned as `(; value, standard_error)` with the standard
error of the estimate scaled by `population`.

**Only for additive payoffs, in the payoff's own unit.** The study's cost must be stated
in the unit of the payoff (individuals removed or killed by the sampling for a count,
money where the payoff is a published valuation), and paying it must lower every payoff
by the same amount whatever the action and the state. Never invent a cost. When
the cost of a study is not additive in the payoff, it belongs in the outcomes of a
larger problem that has "run the study" among its actions.
"""
function enbs(estimate::VOIEstimate; cost::Real, population::Real=1)
    estimate.quantity === :evsi ||
        throw(ArgumentError("enbs is the net benefit of sampling and needs an EVSI " *
                            "estimate, got $(estimate.quantity)"))
    population >= 0 ||
        throw(ArgumentError("the population must be nonnegative, got $population"))
    return (; value=population * estimate.value - cost,
            standard_error=population * estimate.standard_error)
end

# Expected utility with its Monte Carlo error
# -------------------------------------------

"""
    expected_utility_estimates(problem::DecisionProblem) -> Vector{NamedTuple}

The expected utility of every feasible action of a sample problem, with its Monte
Carlo error, one `(; action, value, standard_error, simulation_error)` per action:

- `value`: `Σ_i w_i Q[a, i]`, as [`aggregate`](@ref)`(Expectation(), problem)`;
- `standard_error`: the error of that mean **when the draws are a random sample**
  of the belief, from the spread of the cell means across draws (which already
  contains the simulation noise of the cells);
- `simulation_error`: the error from the simulator alone, **holding the draws
  fixed**, from the standard errors of the cells (`nothing` when the table has none).
  This is the relevant error when the draws are a fixed design.

For the expected gain over the baseline with the right (paired) errors, build the
problem on a table of gains: `payoff_table(outcomes, utility; baseline)`.
"""
function expected_utility_estimates(problem::DecisionProblem)
    Q, w = _sample_problem(problem, :expected_utility_estimates)
    se = _feasible_se(problem)
    wf = Float64.(w)
    return map(enumerate(feasible_actions(problem))) do (a, action)
        q = view(Q, a, :)
        return (; action, value=dot(w, q), standard_error=_weighted_mean_se(q, w),
                simulation_error=se === nothing ? nothing :
                                 sqrt(sum(wf .^ 2 .* Float64.(view(se, a, :)) .^ 2)))
    end
end

"""
    expected_gain_estimates(problem::DecisionProblem) -> Vector{NamedTuple}

The expected gain of every feasible action over the baseline of a sample problem,
one `(; action, value, standard_error)` per action. `value` is
[`expected_gain_over_baseline`](@ref); `standard_error` is the error of the weighted
mean of the per-draw gains `Q[a, i] − Q[baseline, i]` when the draws are a random
sample, so the correlation between an action and the baseline across draws is
accounted for. Throws [`MissingBaselineError`](@ref) without a baseline.
"""
function expected_gain_estimates(problem::DecisionProblem)
    _, w = _sample_problem(problem, :expected_gain_estimates)
    row = _baseline_row(problem, :expected_gain_estimates)
    G = problem.payoff.Q[problem.feasible, :] .- problem.payoff.Q[row:row, :]
    return map(enumerate(feasible_actions(problem))) do (a, action)
        g = view(G, a, :)
        return (; action, value=dot(w, g), standard_error=_weighted_mean_se(g, w))
    end
end

"""
    conditional_problem(problem::DecisionProblem, focal) -> DecisionProblem

The sample problem collapsed onto a discrete parameter: a finite problem whose states
are the cells of `focal` (a parameter name, or several, of the draws of a
[`SampleBelief`](@ref); one cell per combination of values, in sorted order when the
values can be sorted and otherwise in order of first appearance), whose payoffs are the
weighted mean payoffs of the feasible actions within each cell, and whose belief is
the [`FiniteBelief`](@ref) of the cells' total weights. Feasibility and baseline
carry over.

It is the table a manager would look at when the cell is what could be learned: the
expected payoff of each action **by group**. Its [`evpi`](@ref) is the grouping
estimate of the EVPPI of `focal` (`evppi(problem, focal, Grouping())`), and with a
[`ScenarioSet`](@ref) over its states the cells become scenarios for maximin and
minimax regret, without weights. Each cell mean is an estimate from the draws in
the cell; cells with few draws are noisy.

```jldoctest
julia> belief = SampleBelief((; site=[:wet, :wet, :dry, :dry]));

julia> p = DecisionProblem(PayoffTable([0 0 0 0; 2 4 -4 -6]), belief);

julia> c = conditional_problem(p, :site); (state_labels(c), c.payoff.Q)
([:dry, :wet], Rational{Int64}[0 0; -5 3])

julia> evpi(c) == evppi(p, :site).value
true
```
"""
function conditional_problem(problem::DecisionProblem, focal)
    _sample_problem(problem, :conditional_problem)
    w = problem.belief.weights
    columns = _focal_columns(problem.belief, focal)
    cell, ncells = _cells(columns)
    representative = [findfirst(==(c), cell) for c in 1:ncells]
    # Cells in sorted order of their values when these can be sorted, otherwise in
    # order of first appearance.
    keys = [Tuple(column[i] for column in columns) for i in representative]
    order = try
        sortperm(keys)
    catch err
        err isa MethodError || rethrow()
        collect(1:ncells)
    end
    representative = representative[order]
    G = _cell_means(problem.payoff.Q, w, cell, ncells)
    mass = [sum(w[i] for i in eachindex(w) if cell[i] == cell[r]) for r in representative]
    labels = [Symbol(join((string(column[i]) for column in columns), "_"))
              for i in representative]
    payoff = PayoffTable(problem.payoff.actions, labels, G[:, representative])
    return DecisionProblem(payoff, FiniteBelief(labels, mass), problem.feasible,
                           problem.baseline)
end
