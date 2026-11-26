# Value of information for estimation, population value, net benefit of sampling over
# sample sizes, and checks of the regressions: the parts of the R package `voi`
# (`evsivar`, `pop_voi`, `enbs`, `enbs_opt`, `check_regression`) and of Jackson et al.
# (2019, 2021, 2022) that sit on top of the estimators.

_estimation_belief(b::SampleBelief) = b
function _estimation_belief(p::DecisionProblem)
    _sample_problem(p, :expected_variance_reduction)
    return p.belief
end

function _estimation_quantity(belief::SampleBelief, quantity)
    g = quantity isa Union{Symbol,AbstractString} ? only(_focal_columns(belief, quantity)) :
        quantity
    (g isa AbstractVector{<:Real} && length(g) == length(belief)) ||
        throw(ArgumentError("`quantity` must be the name of a numeric parameter of the " *
                            "draws or a vector with one number per draw"))
    return Float64.(g)
end

# Variance of the fitted conditional mean of g given `columns`, by `method`.
function _explained_variance(method::AbstractVOIMethod, columns::NamedTuple,
                             g::Vector{Float64}, w::AbstractVector)
    wf = Float64.(w)
    G, _ = _fit(method, columns, reshape(g, 1, :), w)
    centre = dot(wf, g)
    total = sum(wf .* (g .- centre) .^ 2)
    between = sum(wf[i] * (Float64(G[1, i]) - centre)^2 for i in eachindex(wf))
    return total, between
end

"""
    expected_variance_reduction(belief, focal; quantity, method=nothing)
    expected_variance_reduction(belief, study; quantity)

The value of information for estimation from a [`SampleBelief`](@ref) alone, with no
decision and no payoff table: a model that estimates a quantity (a growth rate, an
abundance, a prevalence) from uncertain inputs. `belief` is the sample (or a sample
problem, whose belief is used); `quantity` is a vector with one number per draw, the
model's output at that draw, or the name of a column.

With `focal` this is the **EVPPI for estimation**, the expected reduction in the
variance of the quantity from learning those parameters exactly,
`Var(E[quantity | focal])` (Jackson et al. 2019; `evppivar` of the R package `voi`).
With a study it is the **EVSI for estimation** (`evsivar`), the reduction expected from
the study's data, `Var(quantity) − E Var(quantity | data)`:

- [`SummaryStatistics`](@ref): the variance of the regression of the quantity on the
  summaries of one simulated data set per draw;
- [`ImportanceSampling`](@ref): the variance, over the simulated data sets, of the
  posterior mean of the quantity;
- [`GaussianApproximation`](@ref): the variance explained by its focal parameters
  times `n / (n0 + n)`; with a vector of sample sizes, a vector of results.

The result is as for the other methods, `(; prior_variance,
expected_posterior_variance, reduction, share)`. Under squared-error loss the
reduction *is* the value of the information, in the units of the quantity squared; a
weighted sum of several outputs is one quantity, `quantity = c₁ g₁ + c₂ g₂`.
"""
function expected_variance_reduction(belief::SampleBelief,
                                     focal::Union{Symbol,Tuple{Vararg{Symbol}}}; quantity,
                                     method::Union{Nothing,AbstractVOIMethod}=nothing)
    g = _estimation_quantity(belief, quantity)
    columns = _focal_columns(belief, focal)
    fit = _require_fit(_method_for(method, columns))
    return _variance_split(_explained_variance(fit, columns, g, belief.weights)...)
end

function expected_variance_reduction(source::Union{SampleBelief,DecisionProblem},
                                     study::SummaryStatistics; quantity)
    belief = _estimation_belief(source)
    g = _estimation_quantity(belief, quantity)
    length(first(study.summaries)) == length(g) ||
        throw(ArgumentError("$(length(first(study.summaries))) rows of summary " *
                            "statistics for $(length(g)) draws"))
    return _variance_split(_explained_variance(study.regression, study.summaries, g,
                                               belief.weights)...)
end

function expected_variance_reduction(source::Union{SampleBelief,DecisionProblem},
                                     study::GaussianApproximation; quantity)
    belief = _estimation_belief(source)
    g = _estimation_quantity(belief, quantity)
    columns = _focal_columns(belief, study.focal)
    method = _method_for(study.regression, columns)
    total, between = _explained_variance(method, columns, g, belief.weights)
    one(n) = _variance_split(total, between * n / (study.n0 + n))
    return study.n isa Real ? one(study.n) : [one(n) for n in study.n]
end

function expected_variance_reduction(source::Union{SampleBelief,DecisionProblem},
                                     study::ImportanceSampling; quantity)
    belief = _estimation_belief(source)
    g = _estimation_quantity(belief, quantity)
    L = study.loglikelihood
    K, n = size(L)
    n == length(g) ||
        throw(ArgumentError("the log-likelihood matrix has $n columns for $(length(g)) " *
                            "draws"))
    wf = Float64.(belief.weights)
    v = _dataset_weights(study, wf, K)
    centre = dot(wf, g)
    total = sum(wf .* (g .- centre) .^ 2)
    remaining = 0.0
    posterior = zeros(n)
    for k in 1:K
        _posterior!(posterior, L, k, wf)
        m = dot(posterior, g)
        remaining += v[k] * sum(posterior .* (g .- m) .^ 2)
    end
    return _variance_split(total, total - remaining)
end

"""
    variance_reduction_table(belief, inputs, outputs; method=nothing) -> NamedTuple

The EVPPI for estimation of every output against every input: which uncertain inputs
drive the uncertainty of which estimates. `inputs` is a list of focal parameters (a
name, or a tuple of names learned together, each as for
[`expected_variance_reduction`](@ref)); `outputs` is a named collection of quantities
(`name => vector` pairs, a named tuple of vectors, or names of columns of the draws).

The result is `(; inputs, outputs, share, reduction, prior_variance, remaining_sd)`:
`share[i, j]` is the share of the variance of output `j` that learning input `i`
would remove (its main-effect or Sobol index, here read as a value of information),
`reduction[i, j]` the reduction itself, `prior_variance[j]` the variance now, and
`remaining_sd[i, j] = √(prior_variance − reduction)`, the standard deviation of the
estimate expected to be left. The shares of the inputs need not add up to one: inputs
that interact share credit, and correlated inputs explain the same variance twice.
This is the table behind a heat map of inputs by outputs (Jackson et al. 2019).
"""
function variance_reduction_table(source::Union{SampleBelief,DecisionProblem}, inputs,
                                  outputs; method::Union{Nothing,AbstractVOIMethod}=nothing)
    belief = _estimation_belief(source)
    focal_sets = [f isa Union{Symbol,AbstractString} ? Symbol(f) : Tuple(_focal_names(f))
                  for f in
                      (inputs isa Union{Symbol,AbstractString,Tuple{Vararg{Symbol}}} ?
                       [inputs] : inputs)]
    isempty(focal_sets) && throw(ArgumentError("no input was given"))
    named = outputs isa NamedTuple ? [k => v for (k, v) in pairs(outputs)] :
            [o isa Pair ? Symbol(first(o)) => last(o) : Symbol(o) => Symbol(o)
             for o in (outputs isa Union{Symbol,AbstractString} ? [outputs] : outputs)]
    isempty(named) && throw(ArgumentError("no output was given"))
    results = [expected_variance_reduction(belief, f; quantity=q, method)
               for f in focal_sets, (_, q) in named]
    prior = [results[1, j].prior_variance for j in axes(results, 2)]
    reduction = [r.reduction for r in results]
    return (; inputs=[f isa Symbol ? String(f) : join(f, ", ") for f in focal_sets],
            outputs=first.(named), share=[r.share for r in results], reduction,
            prior_variance=prior,
            remaining_sd=sqrt.(max.(permutedims(prior) .- reduction, 0.0)))
end

"""
    equivalent_binomial_sample_size(mean, standard_error) -> Float64

The number of yes-or-no observations that an estimate of a proportion is worth:
`n = mean (1 − mean) / standard_error² − 1`, the sample size of the binomial data that
would give a Beta posterior with that mean and standard deviation from a flat
`Beta(0, 0)` prior (Jackson et al. 2021). It turns a published estimate with its
standard error into the prior effective sample size `n0` of a
[`GaussianApproximation`](@ref), and into the `a = n · mean`, `b = n · (1 − mean)` of
the Beta prior of a [`BinomialStudy`](@ref).

```jldoctest
julia> equivalent_binomial_sample_size(0.2, 0.1)
15.0
```
"""
function equivalent_binomial_sample_size(mean::Real, standard_error::Real)
    0 < mean < 1 || throw(ArgumentError("the mean must lie in (0, 1), got $mean"))
    standard_error > 0 ||
        throw(ArgumentError("the standard error must be positive, got $standard_error"))
    n = mean * (1 - mean) / standard_error^2 - 1
    n > 0 || throw(ArgumentError("a proportion with mean $mean cannot have a standard " *
                                 "error as large as $standard_error"))
    return round(Float64(n); digits=10)
end

# Population value and the net benefit of sampling
# ------------------------------------------------

function _population_factor(population::Real, horizon, discount::Real)
    population >= 0 ||
        throw(ArgumentError("the population must be nonnegative, got $population"))
    discount >= 0 ||
        throw(ArgumentError("the discount rate must be nonnegative, got $discount"))
    horizon === nothing && return population
    horizon >= 0 || throw(ArgumentError("the horizon must be nonnegative, got $horizon"))
    return iszero(discount) ? population * horizon :
           population * (1 - exp(-discount * horizon)) / discount
end

"""
    population_value(value; population, horizon=nothing, discount=0)

A value of information per unit (per pond, per site, per person) scaled to everything
the decision applies to. With `horizon = nothing` it is `value · population`: the
decision is taken once for `population` units. With a `horizon`, `population` is the
number of units that come under the decision **per unit of time**, the decision stands
for `horizon` units of time, and later units are discounted continuously at the rate
`discount`:

    value · population · (1 − exp(−discount · horizon)) / discount,

or `value · population · horizon` with no discounting (the formula of `pop_voi` of the
R package `voi`, which however returns `value · population` at a zero rate). `value` is
a number, or a [`VOIEstimate`](@ref), for which `(; value, standard_error)` is
returned. The population and the horizon are assumptions about the future, and the
result is proportional to both: show it for a range of them.
"""
function population_value(value::Real; population::Real, horizon=nothing, discount::Real=0)
    return value * _population_factor(population, horizon, discount)
end
function population_value(estimate::VOIEstimate; population::Real, horizon=nothing,
                          discount::Real=0)
    factor = _population_factor(population, horizon, discount)
    return (; value=estimate.value * factor,
            standard_error=estimate.standard_error * factor)
end

# A cost: a number, a (lower, upper) 95% interval, or (mean, lower, upper).
function _cost(c, what::AbstractString)
    c isa Real && return (Float64(c), 0.0)
    (c isa Union{Tuple,AbstractVector} && length(c) in (2, 3) && all(x -> x isa Real, c)) ||
        throw(ArgumentError("$what must be a number, a (lower, upper) 95% interval or " *
                            "(mean, lower, upper); got $(repr(c))"))
    lower, upper = c[end - 1], c[end]
    lower <= upper || throw(ArgumentError("$what has its interval the wrong way round"))
    centre = length(c) == 3 ? c[1] : (lower + upper) / 2
    return (Float64(centre), (upper - lower) / 4)
end

_normal_cdf(x::Real) = erfc(-x / sqrt(2)) / 2

"""
    enbs_curve(estimates; n, setup_cost=0, per_sample_cost=0, population=1,
               horizon=nothing, discount=0) -> Vector{NamedTuple}

The expected net benefit of sampling of one study design at several sample sizes
(`enbs` of the R package `voi`). `estimates` holds the EVSI at each size of `n`
(numbers, or [`VOIEstimate`](@ref)s as [`evsi`](@ref) returns for a
[`GaussianApproximation`](@ref) with a vector of sizes), per unit. For each size,

    ENBS = population value of the EVSI − (setup cost + per-sample cost · n),

with the population value of [`population_value`](@ref). A cost is a number, or a 95%
interval `(lower, upper)` (its midpoint is used), or `(mean, lower, upper)`; an
interval is read as that of a normal distribution, four standard deviations wide (as
`voi` reads it).

One row per size: `(; n, evsi, population_evsi, cost, enbs, standard_deviation,
probability_positive, standard_error)`. `standard_deviation` is the uncertainty of the
ENBS **from the costs**, and `probability_positive` the probability under it that the
study is worth its cost (`nothing` when the costs are certain). `standard_error` is the
Monte Carlo error of the population EVSI (`nothing` for plain numbers), kept apart.

As for [`enbs`](@ref): only for additive payoffs, with the costs in the payoff's own
unit (individuals removed by the sampling for a count; money only where the payoff is
a published valuation), and never an invented cost. See
[`optimal_sample_size`](@ref).
"""
function enbs_curve(estimates::AbstractVector; n::AbstractVector{<:Real}, setup_cost=0,
                    per_sample_cost=0, population::Real=1, horizon=nothing,
                    discount::Real=0)
    length(estimates) == length(n) ||
        throw(ArgumentError("$(length(estimates)) estimates for $(length(n)) sample sizes"))
    isempty(n) && throw(ArgumentError("no sample size was given"))
    for e in estimates
        e isa VOIEstimate && e.quantity !== :evsi &&
            throw(ArgumentError("enbs_curve needs EVSI estimates, got $(e.quantity)"))
    end
    factor = _population_factor(population, horizon, discount)
    setup, setup_sd = _cost(setup_cost, "setup_cost")
    each, each_sd = _cost(per_sample_cost, "per_sample_cost")
    return map(zip(estimates, n)) do (e, size)
        value = e isa VOIEstimate ? e.value : e
        cost = setup + each * size
        sd = hypot(setup_sd, each_sd * size)
        net = value * factor - cost
        return (; n=size, evsi=value, population_evsi=value * factor, cost, enbs=net,
                standard_deviation=sd,
                probability_positive=iszero(sd) ? nothing : _normal_cdf(net / sd),
                standard_error=e isa VOIEstimate ? e.standard_error * factor : nothing)
    end
end

# Least-squares cubic regression spline of y on x with `df` basis functions, at xnew.
function _smooth_curve(x::Vector{Float64}, y::Vector{Float64}, xnew::Vector{Float64},
                       df::Int)
    lo, hi = extrema(x)
    scale(v) = (v .- lo) ./ (hi - lo)
    interior = df - 4
    knots = [k / (interior + 1) for k in 1:interior]
    function basis(z)
        return hcat(ones(length(z)), z, z .^ 2, z .^ 3,
                    (max.(z .- κ, 0.0) .^ 3 for κ in knots)...)
    end
    coefficients = qr(basis(scale(x)), ColumnNorm()) \ y
    return basis(scale(xnew)) * coefficients
end

"""
    optimal_sample_size(curve; near=0.05, smooth=false, df=nothing) -> NamedTuple

The sample size with the largest expected net benefit of sampling, from the rows of
[`enbs_curve`](@ref) (`enbs_opt` of the R package `voi`). The result is `(; n, enbs,
lower, upper, smoothed)`: the best size, its ENBS, and the smallest and largest sizes
whose ENBS is within the share `near` of the maximum (`enbs ≥ max − near · |max|`),
the range of **near-optimal** designs. ENBS curves are flat near their top, so that
range is usually wide and the smaller end of it costs little.

With `smooth = false` the search is over the sizes given. With `smooth = true` the
ENBS is first smoothed and interpolated, by a least-squares cubic regression spline
in `√n` with `df` coefficients (default: half the number of sizes, between 4 and 6;
4 is a cubic), and the search is over every whole size in the range: this removes the
Monte Carlo jitter of the estimates, which otherwise picks a size by its noise. The
smooth is only as good as the sizes given: they should bracket the optimum and be
spread over the range, ten or more of them, not a few sizes decades apart. If the
best size is the largest or the smallest given, the range was too narrow to find the
optimum and the result says only that.
"""
function optimal_sample_size(curve::AbstractVector; near::Real=0.05, smooth::Bool=false,
                             df::Union{Nothing,Integer}=nothing)
    isempty(curve) && throw(ArgumentError("the curve is empty"))
    0 <= near < 1 || throw(ArgumentError("near must lie in [0, 1), got $near"))
    sizes = Float64[row.n for row in curve]
    values = Float64[row.enbs for row in curve]
    if smooth
        distinct = length(unique(sizes))
        distinct >= 5 ||
            throw(ArgumentError("smoothing needs at least 5 sample sizes, got $distinct"))
        k = df === nothing ? clamp(distinct ÷ 2, 4, 6) : Int(df)
        4 <= k < distinct ||
            throw(ArgumentError("df must be at least 4 and below the number of sample " *
                                "sizes ($distinct), got $k"))
        grid = collect(Float64, ceil(Int, minimum(sizes)):floor(Int, maximum(sizes)))
        values = _smooth_curve(sqrt.(sizes), values, sqrt.(grid), k)
        sizes = grid
    end
    top, at = findmax(values)
    close = findall(>=(top - near * abs(top)), values)
    as_size(x) = isinteger(x) ? Int(x) : x
    return (; n=as_size(sizes[at]), enbs=top, lower=as_size(minimum(sizes[close])),
            upper=as_size(maximum(sizes[close])), smoothed=smooth)
end

# Checks of a regression
# ----------------------

const _Checkable = Union{AbstractRegression,StrongOakley}

function _check_columns(problem::DecisionProblem, columns)
    Q, w = _sample_problem(problem, :regression_check)
    cols = columns isa SummaryStatistics ? columns.summaries :
           _focal_columns(problem.belief, columns)
    length(first(cols)) == length(w) ||
        throw(ArgumentError("$(length(first(cols))) rows of summary statistics for " *
                            "$(length(w)) draws"))
    return Q, w, cols
end

"""
    regression_check(problem, focal, method; bins=10) -> NamedTuple
    regression_check(problem, study::SummaryStatistics; bins=10) -> NamedTuple

Check the regression behind an [`evppi`](@ref) or an [`evsi`](@ref) by summary
statistics (`check_regression` of the R package `voi`). What matters for a value of
information is that the fitted values are conditional **means**: the residuals must
average to zero wherever the fitted value is, and it does not matter that their
spread varies or that they are far from normal.

For every action but the act-now one (the fit of an action enters the estimate only
through its difference from another), the draws are cut into `bins` groups of equal
weight by the fitted advantage over the act-now action, and the mean residual of that
difference is reported with its standard error. The result is `(; method,
residual_variance, r_squared, bins, worst)`: `bins` holds one `(; action, bin, fitted,
residual_mean, standard_error, z)` per action and group, and `worst` is the largest
`|z|`. Groups whose residuals average several standard errors from zero show a fit
that misses the shape of the conditional mean there; a regression that passes can
still overfit, which [`compare_regressions`](@ref) tests on draws left out.
"""
function regression_check(problem::DecisionProblem, focal, method::AbstractVOIMethod;
                          bins::Integer=10)
    return _regression_check(problem, focal, method, bins)
end
function regression_check(problem::DecisionProblem, study::SummaryStatistics;
                          bins::Integer=10)
    return _regression_check(problem, study, study.regression, bins)
end

function _regression_check(problem, columns, method, bins)
    bins >= 2 || throw(ArgumentError("bins must be at least 2, got $bins"))
    method isa SampleAverage &&
        throw(ArgumentError("SampleAverage fits nothing to check"))
    Q, w, cols = _check_columns(problem, columns)
    G, _ = _fit(method, cols, Q, w)
    wf = Float64.(w)
    Qf, Gf = Float64.(Q), Float64.(G)
    _, a = _act_now(Q, w)
    labels = feasible_actions(problem)
    rows = NamedTuple[]
    explained, total = 0.0, 0.0
    for b in axes(Q, 1)
        b == a && continue
        observed = Qf[b, :] .- Qf[a, :]
        fitted = Gf[b, :] .- Gf[a, :]
        residual = observed .- fitted
        centre = dot(wf, observed)
        total += sum(wf .* (observed .- centre) .^ 2)
        explained += sum(wf .* residual .^ 2)
        k = min(Int(bins), length(wf))
        cell = _blocks_by_weight(fitted, wf, k)
        for c in 1:k
            members = findall(==(c), cell)
            isempty(members) && continue
            mass = sum(wf[members])
            iszero(mass) && continue
            share = wf[members] ./ mass
            m = dot(share, residual[members])
            se = sqrt(sum(share .^ 2 .* (residual[members] .- m) .^ 2))
            push!(rows,
                  (; action=labels[b], bin=c, fitted=dot(share, fitted[members]),
                   residual_mean=m, standard_error=se,
                   z=iszero(se) ? (iszero(m) ? 0.0 : Inf * sign(m)) : m / se))
        end
    end
    return (; method=_method_name(method),
            residual_variance=explained / max(size(Q, 1) - 1, 1),
            r_squared=iszero(total) ? 1.0 : 1 - explained / total, bins=rows,
            worst=maximum(r -> abs(r.z), rows; init=0.0))
end

"""
    compare_regressions(problem, focal, methods; folds=5, seed=1) -> NamedTuple
    compare_regressions(problem, study::SummaryStatistics, methods; folds=5, seed=1)

Compare regression methods for an [`evppi`](@ref) (or an [`evsi`](@ref) by summary
statistics) by how well each predicts **draws it was not fitted to**. No method is
reliable everywhere, and the estimate's own standard error does not show a bad fit
(Jackson et al. 2022), so the comparison is by `folds`-fold cross-validation: the draws
are split at random (`Xoshiro(seed)`), each part in turn is given zero weight in the
fit, and its payoffs' differences from the act-now action are predicted from the rest.

`methods` is a named collection or a list of regressions ([`SplineRegression`](@ref),
[`MARSRegression`](@ref), [`GaussianProcessRegression`](@ref), [`GAMRegression`](@ref))
or [`StrongOakley`](@ref). The result is `(; methods, best)`: one row per method,
`(; name, method, cv_error, cv_r_squared, value, standard_error, above_sample_evpi)`,
where `cv_error` is the weighted mean squared prediction error, `cv_r_squared` its
share of the variance explained (negative when the method predicts worse than a
constant), and the last three are the estimate on all the draws; `best` is the name
with the smallest `cv_error`. A method that fits noise has a good in-sample fit, a
poor `cv_error` and a large `value`: prefer the method that predicts best, not the one
that reports the most value.
"""
function compare_regressions(problem::DecisionProblem, columns, methods; folds::Integer=5,
                             seed::Integer=1)
    Q, w, cols = _check_columns(problem, columns)
    named = methods isa AbstractVOIMethod ? [_method_name(methods) => methods] :
            methods isa NamedTuple ? [k => v for (k, v) in pairs(methods)] :
            [m isa Pair ? Symbol(first(m)) => last(m) : _method_name(m) => m
             for m in methods]
    isempty(named) && throw(ArgumentError("no method was given"))
    allunique(first.(named)) ||
        throw(ArgumentError("two methods have the same name; give them as " *
                            "`name => method` pairs"))
    for (name, m) in named
        m isa _Checkable ||
            throw(ArgumentError("$name is a $(typeof(m)); compare_regressions takes " *
                                "regressions and StrongOakley"))
    end
    n = length(w)
    2 <= folds <= n || throw(ArgumentError("folds must lie between 2 and $n, got $folds"))
    fold = [mod1(i, Int(folds)) for i in 1:n][shuffle!(Xoshiro(seed), collect(1:n))]
    wf = Float64.(w)
    Qf = Float64.(Q)
    _, a = _act_now(Q, w)
    observed = Qf .- permutedims(Qf[a, :])
    centre = observed * wf
    total = sum(wf[i] * (observed[b, i] - centre[b])^2 for b in axes(Q, 1), i in 1:n)
    quantity = columns isa SummaryStatistics ? :evsi : :evppi
    rows = map(named) do (name, method)
        error = 0.0
        for k in 1:Int(folds)
            held = fold .== k
            training = ifelse.(held, 0.0, wf)
            training ./= sum(training)
            G, _ = _fit(method, cols, Q, training)
            predicted = Float64.(G) .- permutedims(Float64.(G[a, :]))
            error += sum(wf[i] * (observed[b, i] - predicted[b, i])^2
                         for b in axes(Q, 1), i in findall(held))
        end
        e = with_logger(NullLogger()) do
            return _estimate(quantity, method, cols, Q, w; bootstrap=0, seed)
        end
        return (; name, method=_method_name(method), cv_error=error,
                cv_r_squared=iszero(total) ? 1.0 : 1 - error / total, value=e.value,
                standard_error=e.standard_error,
                above_sample_evpi=e.details.above_sample_evpi)
    end
    return (; methods=rows, best=rows[argmin([r.cv_error for r in rows])].name)
end
