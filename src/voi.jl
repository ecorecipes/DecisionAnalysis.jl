# Exact value of information on finite problems. The definitions below are a contract
# with the Lean development (theorem tier T); change neither side alone.
#
#   act-now value   V0        = max_a Σ_θ p(θ) Q[a, θ]
#   value(e)                  = Σ_z max_a Σ_θ p(θ) L[z, θ] Q[a, θ]      (joint form)
#   EVSI(e)                   = value(e) − V0
#   EVPI                      = Σ_θ p(θ) max_a Q[a, θ] − V0
#   EVPPI(partition)          = Σ_cells max_a Σ_{θ ∈ cell} p(θ) Q[a, θ] − V0
#
# The joint form has no division, so a signal of probability zero contributes zero and
# needs no special case.

function _feasible_Q(problem::DecisionProblem)
    return problem.payoff.Q[problem.feasible, :]
end

"""
    act_now_value(problem::DecisionProblem)

The value of deciding now, with no further information: the largest expected utility,
`V0 = max_a Σ_θ p(θ) Q[a, θ]`, over the feasible actions. Needs a single prior
([`FiniteBelief`](@ref) or [`PointBelief`](@ref)).
"""
function act_now_value(problem::DecisionProblem)
    return maximum(_feasible_Q(problem) * _prior(problem, :act_now_value))
end

"""
    value_with_information(problem::DecisionProblem, experiment::Experiment)

The value of observing the signal of `experiment` and then choosing the best action
for it:

    value(e) = Σ_z max_a Σ_θ p(θ) L[z, θ] Q[a, θ].

This is the joint form: the posterior is never formed, nothing is divided by `P(z)`,
and a signal of probability zero contributes exactly zero. The cost of the experiment
is not subtracted (see [`enbs`](@ref)). Needs a single prior.
"""
function value_with_information(problem::DecisionProblem, experiment::Experiment)
    _check_states(experiment, problem.payoff.states, "payoff table")
    p = _prior(problem, :value_with_information)
    Q = _feasible_Q(problem)
    L = experiment.likelihood
    return sum(maximum(Q * (p .* view(L, z, :))) for z in axes(L, 1))
end

"""
    evsi(problem::DecisionProblem, experiment::Experiment)

The expected value of sample information: what observing the signal of `experiment`
before choosing is worth, gross of its cost,

    EVSI = value(e) − V0,

with `value(e)` from [`value_with_information`](@ref) and `V0` from
[`act_now_value`](@ref). What is learned is the signal of this experiment about the
states of this problem.

Under a single prior, `0 ≤ EVSI ≤ EVPI`, and a garbling of an experiment
([`garble`](@ref)) has an EVSI no larger than the original (Blackwell). Both
statements are for a single prior; under a credal set neither holds in general, and
[`robust_evsi`](@ref) can be negative.

```jldoctest
julia> p = DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5]);

julia> evsi(p, Experiment([4//5 1//5; 1//5 4//5]))
44//5
```
"""
function evsi(problem::DecisionProblem, experiment::Experiment)
    return value_with_information(problem, experiment) - act_now_value(problem)
end

"""
    information_can_change_decision(problem::DecisionProblem, experiment::Experiment;
                                    atol=0, rtol) -> Bool

Whether observing the signal of `experiment` can change what is done: `true` exactly
when **no single action is optimal after every signal of positive probability**. When
it is `false`, one action is best whatever is observed, the experiment cannot change
the decision, and it is worth nothing.

This is equivalent to a positive value of sample information,

    information_can_change_decision(problem, e)  ⟺  evsi(problem, e) > 0,

for a single prior (theorem `evsi_eq_zero_iff` of the Lean development: EVSI is zero
exactly when one action maximises `Σ_θ p(θ) L[z, θ] Q[a, θ]` for every signal `z`).
The check does not compute the EVSI: it intersects, over the signals of positive
probability, the sets of actions that are optimal after each. Ties count: an action
that is only tied for best after some signal is still optimal after it.

Optimality after a signal is judged with the tie tolerance of [`decide`](@ref)
(`max(atol, rtol · max|score|)`; zero by default for exact problems, where the
equivalence is exact, and `rtol = sqrt(eps)` in floating point, where it holds up to
that tolerance). Needs a single prior.

```jldoctest
julia> p = DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5]);

julia> information_can_change_decision(p, Experiment([4//5 1//5; 1//5 4//5]))
true

julia> information_can_change_decision(p, Experiment([1//2 1//2; 1//2 1//2]))
false
```
"""
function information_can_change_decision(problem::DecisionProblem, experiment::Experiment;
                                         atol::Real=0,
                                         rtol::Real=_default_rtol(promote_type(eltype(problem.payoff),
                                                                               eltype(experiment))))
    _check_states(experiment, problem.payoff.states, "payoff table")
    p = _prior(problem, :information_can_change_decision)
    Q = _feasible_Q(problem)
    L = experiment.likelihood
    always_best = trues(size(Q, 1))
    for z in axes(L, 1)
        joint = p .* view(L, z, :)
        iszero(sum(joint)) && continue
        scores = Q * joint
        best = _select(Optimize(), scores, :max, _tolerance(scores, atol, rtol))
        always_best .&= [a in best for a in eachindex(scores)]
    end
    return !any(always_best)
end

"""
    evpi(problem::DecisionProblem)

The expected value of perfect information about the state:

    EVPI = Σ_θ p(θ) max_a Q[a, θ] − V0.

It is nonnegative, equals the smallest [`expected_regret`](@ref), and bounds the
[`evsi`](@ref) of every experiment on the same states. Needs a single prior; for a
[`CredalSet`](@ref) use [`robust_evpi`](@ref).

If the payoff table was built from a stochastic simulator with few replicates per
cell, every maximum is taken over noisy cells: the act-now value and the value with
perfect information are each biased upwards, and the chosen action is on average
worth less than its reported score. EVPI is the difference of the two, so the sign
of its bias is not determined in general. With exact payoffs and a sampled belief,
EVPI is biased downwards. This is a property of the table, not corrected here; see
[`VOIEstimate`](@ref) for the estimates of both biases.

```jldoctest
julia> evpi(DecisionProblem(PayoffTable([0 -100; -30 -30]), [3 // 5, 2 // 5]))
18//1
```
"""
function evpi(problem::DecisionProblem)
    p = _prior(problem, :evpi)
    Q = _feasible_Q(problem)
    best = [maximum(view(Q, :, j)) for j in axes(Q, 2)]
    return dot(p, best) - maximum(Q * p)
end

"""
    evppi(problem::DecisionProblem, partition::AbstractVector)

The expected value of partial perfect information: the value of learning **which
cell of a partition** the state lies in, and nothing more. `partition` has one entry
per state of the payoff table, in order; states with equal entries are in the same
cell (for example the value of one focal parameter when the state is a tuple of
parameters).

    EVPPI = Σ_cells max_a Σ_{θ ∈ cell} p(θ) Q[a, θ] − V0.

The one-cell partition gives zero and the partition into single states gives
[`evpi`](@ref); refining a partition never lowers the value. Needs a single prior.

```jldoctest
julia> p = DecisionProblem(PayoffTable([0 -10 -60 -150; -20 -20 -25 -90]),
                           [1 // 2, 3 // 10, 3 // 20, 1 // 20]);

julia> evppi(p, [:mild, :mild, :severe, :severe])
33//4
```
"""
function evppi(problem::DecisionProblem, partition::AbstractVector)
    n = size(problem.payoff, 2)
    length(partition) == n ||
        throw(ArgumentError("the partition has $(length(partition)) entries for the " *
                            "$n states of the payoff table"))
    p = _prior(problem, :evppi)
    Q = _feasible_Q(problem)
    informed = sum(unique(partition)) do cell
        cols = findall(isequal(cell), partition)
        return maximum(view(Q, :, cols) * view(p, cols))
    end
    return informed - maximum(Q * p)
end

"""
    enbs(problem::DecisionProblem, experiment::Experiment; population=1)

The expected net benefit of sampling: `population · EVSI − cost`, where `cost` is the
cost of the experiment and `population` scales the per-decision value to the number
of decisions the information serves.

This is meaningful **only for additive payoffs**: the payoff must be in the same
units as the cost, and paying the cost must shift every payoff by the same amount
whatever the action and state. When utility is not additive in the cost (a
risk-averse utility of wealth, for example) the cost belongs in the outcome, as part
of each payoff of a larger problem that has "run the experiment" among its actions,
and this function does not apply.
"""
function enbs(problem::DecisionProblem, experiment::Experiment; population::Real=1)
    population >= 0 ||
        throw(ArgumentError("the population must be nonnegative, got $population"))
    return population * evsi(problem, experiment) - experiment.cost
end

# Robust value of information
# ---------------------------

"""
    RobustVOI

The value of information under a [`CredalSet`](@ref), as returned by
[`robust_evpi`](@ref) and [`robust_evsi`](@ref). Two different objects are kept
apart. Fields:

- `quantity`: `:evpi` or `:evsi`.
- `rule`: the rule under which the rule-specific value is computed.
- `act_now`: the rule's value of deciding now.
- `informed`: the rule's value of the strategy that waits for the information.
- `value`: `informed − act_now`, the **rule-specific** value. It may be negative and
  is returned as computed, never clamped.
- `members`: the single-prior value ([`evpi`](@ref) or [`evsi`](@ref)) at each
  generating point of the credal set, in order.
- `lower`, `upper`: the **envelope** of the single-prior EVPI over the whole credal
  set. For `:evsi` both are `nothing`: EVSI is neither convex nor concave in the
  prior, so neither end of its envelope is found at a generating point, and
  version 0.1 does not compute it.
- `upper_prior`: a prior in the credal set **at which the upper EVPI is attained**:
  the probabilities of the states, in the order of the payoff table, of the mixture
  of the generating points found by the programme of
  [`evpi_envelope_model`](@ref), so that `evpi` under it equals `upper`. It is the
  belief in the set under which learning the state is worth most, and it is often an
  interior prior, not a generating point. It need not be unique. `nothing` for
  `:evsi`.
"""
struct RobustVOI{R<:AbstractRule,T,E,P}
    quantity::Symbol
    rule::R
    act_now::T
    informed::T
    value::T
    members::Vector{T}
    lower::E
    upper::E
    upper_prior::P
end

# The parts of a rule that a rule-specific robust value can use.
function _prior_rule(rule::AbstractRule, operation::Symbol)
    ok = rule isa DecisionRule && rule.selector isa Optimize &&
         rule.metric.transform isa IdentityTransform &&
         rule.metric.scenarios === nothing &&
         rule.metric.summary isa
         Union{Expectation,LowerExpectation,UpperExpectation,AlphaMaxmin}
    ok ||
        throw(ArgumentError("$operation needs a rule that maximises an aggregate of " *
                            "utilities over priors, such as GammaMaximin() or " *
                            "DecisionRule(AlphaMaxmin(α)); got $(_rule_name(rule))"))
    return rule.metric.summary
end

function _credal_view(problem::DecisionProblem, operation::Symbol)
    v = _view(problem)
    v.priors === nothing &&
        throw(BeliefFormError(operation, v.form,
                              "a scenario set carries no probabilities; supply a " *
                              "CredalSet"))
    v.form === :PointBelief && return _View(_feasible_Q(problem),
                                            [_prior(problem, operation)], v.form)
    return v
end

# Largest single-prior EVPI over the convex hull of the priors, by the programme of
# `evpi_envelope_model`, with the weights on the generating points that attain it.
function _evpi_upper(E::AbstractMatrix, perfect::AbstractVector; optimizer=nothing,
                     rounding_tol=_ROUNDING_TOLS)
    T = promote_type(eltype(E), eltype(perfect))
    sol = solve_program(_evpi_envelope_program(E, perfect); optimizer, rounding_tol)
    sol.status === :optimal || throw(LinearProgramError(:evpi_envelope, sol.status))
    _is_exact(T) && !sol.exact &&
        throw(ExactVerificationError(:evpi_envelope,
                                     "the rounded solution of the floating-point " *
                                     "optimiser is not certified optimal"))
    return _from_big(T, sol.objective), sol.values.λ
end

"""
    robust_evpi(problem::DecisionProblem; rule=GammaMaximin(), optimizer=nothing,
                rounding_tol=(1e-8, 1e-10, 1e-12)) -> RobustVOI

The value of perfect information when the belief is a [`CredalSet`](@ref). Two
objects are returned in one [`RobustVOI`](@ref), because a single "robust EVPI"
conflates them.

**The rule-specific value.** `rule` must maximise an aggregate over priors
([`GammaMaximin`](@ref), `DecisionRule(UpperExpectation())` or
`DecisionRule(AlphaMaxmin(α))`). Acting now is worth the rule's best score. With
perfect information the best payoff `max_a Q[a, θ]` is obtained in each state, and
that state-contingent payoff is aggregated over the priors in the same way. `value`
is the difference.

**The envelope.** `lower` and `upper` are the smallest and largest single-prior
[`evpi`](@ref) over the whole set. EVPI is concave in the prior, so the smallest is
attained at a generating point (`lower == minimum(members)`), while the largest may
lie at an interior prior and is found by the linear programme of
[`evpi_envelope_model`](@ref) (`upper ≥ maximum(members)`); `upper_prior` is a prior
of the set at which it is attained. `optimizer` and
`rounding_tol` are passed to [`solve_program`](@ref); with a floating-point
optimiser on an exact problem the optimum must be certified exactly, or
[`ExactVerificationError`](@ref) is thrown.

A [`FiniteBelief`](@ref) is treated as a one-point credal set; every number then
equals [`evpi`](@ref).
"""
function robust_evpi(problem::DecisionProblem; rule::AbstractRule=GammaMaximin(),
                     optimizer=nothing, rounding_tol=_ROUNDING_TOLS)
    agg = _prior_rule(rule, :robust_evpi)
    v = _credal_view(problem, :robust_evpi)
    Q = v.Q
    best = [maximum(view(Q, :, j)) for j in axes(Q, 2)]
    act_now = maximum(_aggregate(agg, Q, v))
    informed = only(_aggregate(agg, permutedims(best), v))
    E = _expectations(Q, v.priors)
    perfect = [dot(q, best) for q in v.priors]
    members = [perfect[k] - maximum(view(E, :, k)) for k in eachindex(perfect)]
    act_now, informed, least = promote(act_now, informed, minimum(members))
    most, λ = _evpi_upper(E, perfect; optimizer, rounding_tol)
    lower, upper = promote(least, most)
    P = eltype(first(v.priors))
    upper_prior = [_from_big(P, sum(λ[k] * v.priors[k][j] for k in eachindex(λ)))
                   for j in axes(Q, 2)]
    return RobustVOI(:evpi, rule, act_now, informed, informed - act_now,
                     collect(typeof(least), members), lower, upper, upper_prior)
end

"""
    robust_evsi(problem::DecisionProblem, experiment::Experiment;
                rule=GammaMaximin()) -> RobustVOI

The value of an experiment when the belief is a [`CredalSet`](@ref), under a rule
that maximises an aggregate over priors (as for [`robust_evpi`](@ref)).

The decision-maker updates every prior in the set by Bayes' rule on the signal
(priors that give the signal probability zero drop out), applies `rule` to the
posterior set, and takes its [`first_choice`](@ref). That fixes a strategy, one
action per signal. `informed` is the rule's value of that strategy evaluated before
the signal is seen, by aggregating its expected payoff over the prior set; `act_now`
is the rule's best score without the signal; `value` is the difference.

**`value` can be negative**, and is returned as computed. Updating a set of priors
can widen it (dilation), the posterior choices need not form the strategy the rule
would have picked in advance, and so the signal can be worth less than nothing to
this decision-maker. Blackwell monotonicity is a statement about single priors.

`members` holds the single-prior [`evsi`](@ref) at each generating point. `lower` and
`upper` are `nothing`; see [`RobustVOI`](@ref).
"""
function robust_evsi(problem::DecisionProblem, experiment::Experiment;
                     rule::AbstractRule=GammaMaximin())
    agg = _prior_rule(rule, :robust_evsi)
    _check_states(experiment, problem.payoff.states, "payoff table")
    v = _credal_view(problem, :robust_evsi)
    Q, L = v.Q, experiment.likelihood
    scores = _aggregate(agg, Q, v)
    act_now = maximum(scores)
    # The state-contingent expected payoff of the strategy: w[θ] = Σ_z L[z, θ] Q[a_z, θ].
    w = zeros(promote_type(eltype(Q), eltype(L)), size(Q, 2))
    tol_type = promote_type(eltype(Q), eltype(L), eltype(first(v.priors)))
    for z in axes(L, 1)
        posteriors = [q .* view(L, z, :) for q in v.priors]
        posteriors = [q ./ sum(q) for q in posteriors if !iszero(sum(q))]
        # A signal that no prior allows contributes nothing whatever is chosen.
        isempty(posteriors) && continue
        s = _aggregate(agg, Q, _View(Q, posteriors, :CredalSet))
        a = first(_select(Optimize(), s, :max,
                          _tolerance(s, 0, _default_rtol(tol_type))))
        w .+= view(L, z, :) .* view(Q, a, :)
    end
    informed = only(_aggregate(agg, permutedims(w), v))
    E = _expectations(Q, v.priors)
    members = [sum(maximum(Q * (q .* view(L, z, :))) for z in axes(L, 1)) -
               maximum(view(E, :, k)) for (k, q) in enumerate(v.priors)]
    act_now, informed, m = promote(act_now, informed, first(members))
    return RobustVOI(:evsi, rule, act_now, informed, informed - act_now,
                     collect(typeof(m), members), nothing, nothing, nothing)
end

# The bridge to influence diagrams
# --------------------------------

"""
    normal_form(model; states=nothing, baseline=nothing, atol=1e-9) -> NamedTuple

The **normal form** of a single-decision influence diagram: the payoff table and
belief that the diagram defines, as a [`DecisionProblem`](@ref), and the decision's
information as an [`Experiment`](@ref). `model` is an `InfluenceDiagramModel` of
InfluenceDiagrams.jl (`github.com/ecorecipes/InfluenceDiagrams.jl`); the method lives
in a weak-dependency extension, so load that package first
([`MissingExtensionError`](@ref) otherwise). InfluenceDiagrams.jl is not a dependency
of this package.

- The **actions** are the states of the decision's action variable.
- The **state** `θ` is the joint value of the chance variables named in `states`; by
  default, every chance variable that could be observed before the decision (those
  not downstream of it). The belief is their joint distribution, `p(θ)`.
- The **payoff** is the conditional expected total utility,
  `Q[a, θ] = E[U | do(a), θ]`, summing the additive utility nodes of the diagram.
- The **experiment** describes what the decision already observes: its information
  variables that are not part of the state, with likelihood `L[z, θ] = P(z | θ)`;
  `nothing` when there are none.

The result is `(; problem, experiment, decision, action, states, information,
observed)`. With it, the structural quantities of the diagram are values of
information on a payoff table:

- `decide(problem, MaxExpectedUtility())` is the best action with no information;
- `evsi(problem, experiment)` is the value of what the decision observes (the
  diagram's `expected_value_of_information` for those variables, from none);
- `evpi(problem)` is the value of observing the whole state (its
  `expected_value_of_perfect_information` from no information), and
  `evpi(problem) − evsi(problem, experiment)` what is left to learn.

`states` must carry everything the information tells about the utility
(`E[U | a, θ, z] = E[U | a, θ]`), otherwise `ArgumentError` is thrown: add the
variables the utility depends on. The diagram is evaluated by enumerating its joint
distribution once per action, so this is for small diagrams, and the numbers are
floating point, as in InfluenceDiagrams.jl. Utilities are whatever the diagram's
utility nodes hold; this package itself counts utility in individuals.
"""
function normal_form(model; kw...)
    throw(MissingExtensionError(:normal_form, :InfluenceDiagrams))
end
