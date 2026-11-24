# What a study is worth outcome by outcome, what it does to the uncertainty of a
# quantity, and which of several candidate studies to run. Nothing here defines a new
# value of information: the profile sums to the EVSI of `voi.jl`, and `optimal_design`
# compares EVSIs.

"""
    voi_profile(problem::DecisionProblem, experiment::Experiment; atol=0, rtol) -> NamedTuple

The value of an experiment **outcome by outcome** (Akinlotan et al. 2024): the
expected value of sample information hides how the realised value varies over what
the study might show. For each signal `z` of positive probability the profile gives

- `probability`: `P(z) = Σ_θ p(θ) L[z, θ]`;
- `best`: the actions that are optimal after `z` (ties as in [`decide`](@ref));
- `posterior_value`: the expected utility of the best action after `z`;
- `value`: the **value of the information in this outcome**, the posterior expected
  utility of the best action minus that of the act-now action `a*` (the first Bayes
  action under the prior). It is never negative, and zero exactly when `a*` is still
  optimal after `z`;
- `changes_decision`: whether `a*` is no longer optimal after `z`.

The result is `(; act_now_action, outcomes, evsi, variance, standard_deviation,
probability_zero)`: `evsi = Σ_z P(z) value_z` is [`evsi`](@ref)`(problem, experiment)`
exactly; `variance` and `standard_deviation` are those of the per-outcome value over
the signals; `probability_zero` is the probability that the study changes nothing. A
study with a large EVSI that comes from a rare outcome has a large standard deviation
and a `probability_zero` near one, which matters to a risk-averse manager. A signal of
probability zero is listed with `value` zero and no posterior. Exact in, exact out,
except the standard deviation. Needs a single prior.

```jldoctest
julia> p = DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5]);

julia> profile = voi_profile(p, Experiment([4//5 1//5; 1//5 4//5]));

julia> [(o.signal, o.probability, o.value) for o in profile.outcomes]
2-element Vector{Tuple{Symbol, Rational{Int64}, Rational{Int64}}}:
 (:z1, 14//25, 110//7)
 (:z2, 11//25, 0)

julia> profile.evsi, profile.probability_zero
(44//5, 11//25)
```
"""
function voi_profile(problem::DecisionProblem, experiment::Experiment; atol::Real=0,
                     rtol::Real=_default_rtol(promote_type(eltype(problem.payoff),
                                                           eltype(experiment))))
    _check_states(experiment, problem.payoff.states, "payoff table")
    p = _prior(problem, :voi_profile)
    Q = _feasible_Q(problem)
    L = experiment.likelihood
    actions = feasible_actions(problem)
    a0 = argmax(Q * p)
    T = promote_type(eltype(Q), eltype(p), eltype(L))
    V = Base.promote_op(/, T, T)
    total = zero(V)
    second = zero(V)
    unchanged = zero(promote_type(eltype(p), eltype(L)))
    outcomes = map(axes(L, 1)) do z
        joint = p .* view(L, z, :)
        pz = sum(joint)
        signal = experiment.signals[z]
        iszero(pz) &&
            return (; signal, probability=pz, best=Symbol[], posterior_value=nothing,
                    value=zero(V), changes_decision=false)
        scores = Q * joint
        best = _select(Optimize(), scores, :max, _tolerance(scores, atol, rtol))
        stays = a0 in best
        value = stays ? zero(V) : (maximum(scores) - scores[a0]) / pz
        total += pz * value
        second += pz * value^2
        stays && (unchanged += pz)
        return (; signal, probability=pz, best=actions[best],
                posterior_value=maximum(scores) / pz, value, changes_decision=!stays)
    end
    variance = second - total^2
    return (; act_now_action=actions[a0], outcomes, evsi=total, variance,
            standard_deviation=sqrt(max(Float64(variance), 0.0)),
            probability_zero=unchanged)
end

function _variance_split(total::Real, between::Real)
    return (; prior_variance=total, expected_posterior_variance=total - between,
            reduction=between, share=iszero(total) ? zero(between) : between / total)
end

function _state_quantity(problem::DecisionProblem, quantity::AbstractVector{<:Real})
    n = size(problem.payoff, 2)
    length(quantity) == n ||
        throw(ArgumentError("the quantity has $(length(quantity)) entries for the $n " *
                            "states of the payoff table"))
    return quantity
end

"""
    expected_variance_reduction(problem, experiment::Experiment, quantity)
    expected_variance_reduction(problem, partition::AbstractVector, quantity)
    expected_variance_reduction(problem, focal; quantity, method=nothing)

The **value of information for estimation**: by how much, on average, a study
reduces the variance of a quantity of interest (Jackson et al. 2019). `quantity` has
one number per state (the abundance in that state, a parameter, the payoff of one
action). With `g` the quantity,

    reduction = Var(g) − E_z Var(g | z) = Var(E[g | z]),

which is never negative. The result is `(; prior_variance,
expected_posterior_variance, reduction, share)` with `share = reduction /
prior_variance`. It measures learning, not deciding: a study can remove most of the
variance of a quantity and still have an [`evsi`](@ref) of zero, when no outcome would
change the action, so read the two side by side.

- With an [`Experiment`](@ref): the reduction from observing its signal. Exact.
- With a partition of the states (as for [`evppi`](@ref)): the reduction from
  learning the cell. Exact.
- With a [`SampleBelief`](@ref) and `focal` (a parameter name or a tuple of names):
  the reduction from learning those parameters, estimated as the weighted variance of
  the fitted conditional mean of the quantity given them, with `method` as for
  [`evppi`](@ref) (default: chosen by the focal parameters). `quantity` is a vector
  with one number per draw, or the name of a parameter of the draws.

Units are those of the quantity squared (individuals squared for a count). Needs a
single prior.

```jldoctest
julia> p = DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5]);

julia> r = expected_variance_reduction(p, Experiment([4//5 1//5; 1//5 4//5]), [10, 60]);

julia> r.prior_variance, r.reduction, r.share
(600//1, 16200//77, 27//77)
```
"""
function expected_variance_reduction(problem::DecisionProblem, experiment::Experiment,
                                     quantity::AbstractVector{<:Real})
    _check_states(experiment, problem.payoff.states, "payoff table")
    p = _prior(problem, :expected_variance_reduction)
    g = _state_quantity(problem, quantity)
    L = experiment.likelihood
    centre = dot(p, g)
    total = sum(p[j] * g[j]^2 for j in eachindex(g)) - centre^2
    explained = zero(Base.promote_op(/, typeof(total), eltype(L)))
    for z in axes(L, 1)
        pz = sum(p[j] * L[z, j] for j in eachindex(g))
        iszero(pz) && continue
        explained += sum(p[j] * L[z, j] * g[j] for j in eachindex(g))^2 / pz
    end
    return _variance_split(total, explained - centre^2)
end

function expected_variance_reduction(problem::DecisionProblem, partition::AbstractVector,
                                     quantity::AbstractVector{<:Real})
    p = _prior(problem, :expected_variance_reduction)
    g = _state_quantity(problem, quantity)
    length(partition) == length(g) ||
        throw(ArgumentError("the partition has $(length(partition)) entries for the " *
                            "$(length(g)) states of the payoff table"))
    centre = dot(p, g)
    total = sum(p[j] * g[j]^2 for j in eachindex(g)) - centre^2
    explained = zero(Base.promote_op(/, typeof(total), eltype(p)))
    for cell in unique(partition)
        cols = findall(isequal(cell), partition)
        mass = sum(p[cols])
        iszero(mass) && continue
        explained += sum(p[j] * g[j] for j in cols)^2 / mass
    end
    return _variance_split(total, explained - centre^2)
end

function expected_variance_reduction(problem::DecisionProblem,
                                     focal::Union{Symbol,Tuple{Vararg{Symbol}}}; quantity,
                                     method::Union{Nothing,AbstractVOIMethod}=nothing)
    _, w = _sample_problem(problem, :expected_variance_reduction)
    belief = problem.belief
    g = quantity isa Union{Symbol,AbstractString} ? only(_focal_columns(belief, quantity)) :
        quantity
    (g isa AbstractVector{<:Real} && length(g) == length(w)) ||
        throw(ArgumentError("`quantity` must be the name of a numeric parameter of the " *
                            "draws or a vector with one number per draw"))
    columns = _focal_columns(belief, focal)
    fit = method === nothing ? _default_method(columns) : method
    fit isa SampleAverage &&
        throw(ArgumentError("SampleAverage cannot fit conditional means"))
    wf = Float64.(w)
    gf = Float64.(g)
    G, _ = _fit(fit, columns, reshape(gf, 1, :), w)
    centre = dot(wf, gf)
    total = sum(wf .* (gf .- centre) .^ 2)
    between = sum(wf[i] * (Float64(G[1, i]) - centre)^2 for i in eachindex(wf))
    return _variance_split(total, between)
end

# Candidate studies
# -----------------

const _Study = Union{Experiment,SummaryStatistics,ImportanceSampling,GaussianApproximation}

function _design_list(designs)
    if designs isa GaussianApproximation && !(designs.n isa Real)
        return [Symbol("n = ", n) => GaussianApproximation(designs.focal; n0=designs.n0, n,
                                                           designs.regression)
                for n in designs.n]
    end
    designs isa _Study && return [:design => designs]
    named = _named_items(designs, _Study, "designs")
    isempty(named) && throw(ArgumentError("no candidate design was given"))
    for (name, d) in named
        d isa GaussianApproximation && !(d.n isa Real) &&
            throw(ArgumentError("the design $name has several sample sizes; pass it " *
                                "alone to compare them, or give one size per design"))
    end
    return named
end

function _design_costs(costs, named)
    costs === nothing &&
        return [d isa Experiment ? d.cost : 0 for (_, d) in named]
    if costs isa Function
        return [costs(d isa GaussianApproximation ? d.n : name) for (name, d) in named]
    end
    if costs isa AbstractVector{<:Real}
        length(costs) == length(named) ||
            throw(ArgumentError("$(length(costs)) costs for $(length(named)) designs"))
        return collect(costs)
    end
    given = Dict(_named_items(costs, Real, "costs"))
    return map(named) do (name, _)
        haskey(given, name) ||
            throw(ArgumentError("no cost was given for the design $name"))
        return given[name]
    end
end

"""
    optimal_design(problem::DecisionProblem, designs; costs=nothing, population=1,
                   seed=1) -> NamedTuple

Compare a finite list of candidate studies and name the best. `designs` is a named
collection (a named tuple, a dictionary or `name => design` pairs) of

- [`Experiment`](@ref)s, valued exactly by [`evsi`](@ref) on a finite problem, or
- studies of a sample problem ([`SummaryStatistics`](@ref),
  [`ImportanceSampling`](@ref), [`GaussianApproximation`](@ref)), valued by the
  sample-based [`evsi`](@ref), each with its standard error;

or a single [`GaussianApproximation`](@ref) with a vector of sample sizes, which is
compared across its sizes (designs named `n = 10`, ...).

Each design is scored by its **expected net benefit of sampling**,
`population · EVSI − cost`. `costs` gives the cost of each design **in individuals**
(animals removed or killed by the sampling, say): `nothing` uses the `cost` of each
experiment and zero for the studies; otherwise a vector in the order of the designs,
a named collection, or a function of the sample size (of the name, for designs
without one). There are no monetary costs in this package, and a net benefit is
meaningful **only for additive payoffs**, as for [`enbs`](@ref). When every cost is
zero the comparison is by EVSI alone, and the most informative design wins.

The result is `(; designs, best, tied, criterion, population)`: `designs` holds one
`(; name, evsi, standard_error, cost, net, net_standard_error)` per design (standard
errors `nothing` for exact values); `best` is the name of the design with the largest
net benefit, the first in the order given when several tie (`tied` lists them all);
`criterion` is `:net_benefit` or `:evsi`.

With estimated EVSIs the best design is picked from noisy numbers, so its reported
net benefit overstates on average what it is worth (the optimiser's curse, as for the
act-now value): read the standard errors, and prefer the smaller study when two are
within them.

```jldoctest
julia> p = DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5]);

julia> coarse = Experiment([4//5 1//5; 1//5 4//5]; cost=2);

julia> fine = Experiment([19//20 1//20; 1//20 19//20]; cost=8);

julia> d = optimal_design(p, (; coarse, fine)); (d.best, [x.net for x in d.designs])
(:fine, Rational{Int64}[34//5, 77//10])
```
"""
function optimal_design(problem::DecisionProblem, designs; costs=nothing,
                        population::Real=1, seed::Integer=1)
    population >= 0 ||
        throw(ArgumentError("the population must be nonnegative, got $population"))
    named = _design_list(designs)
    cost = _design_costs(costs, named)
    rows = map(zip(named, cost)) do ((name, design), c)
        (c isa Real && isfinite(c)) ||
            throw(ArgumentError("the cost of the design $name is $c, not a finite number"))
        if design isa Experiment
            value = evsi(problem, design)
            return (; name, evsi=value, standard_error=nothing, cost=c,
                    net=population * value - c, net_standard_error=nothing)
        end
        e = evsi(problem, design; seed)
        return (; name, evsi=e.value, standard_error=e.standard_error, cost=c,
                net=population * e.value - c,
                net_standard_error=population * e.standard_error)
    end
    nets = [r.net for r in rows]
    tied = [r.name for r in rows if r.net == maximum(nets)]
    return (; designs=rows, best=first(tied), tied,
            criterion=all(iszero, cost) ? :evsi : :net_benefit, population)
end
