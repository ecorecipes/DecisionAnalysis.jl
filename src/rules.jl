# Rules. A scalar rule is a robustness metric (transform, scenario subset, summary)
# followed by a selector; the named rules are built from those combinators. The
# set-valued rules over a credal set, and the Pareto front, are separate types.

# Transforms
# ----------

"""
    AbstractTransform

Supertype of the transforms applied to a payoff table before it is aggregated:
[`IdentityTransform`](@ref), [`RegretTransform`](@ref) and
[`RelativeRegretTransform`](@ref).
"""
abstract type AbstractTransform end

"""
    IdentityTransform()

Leave the payoff table as it is. Scores are utilities: higher is better.
"""
struct IdentityTransform <: AbstractTransform end

"""
    RegretTransform()

Replace the payoff table by its [`regret_table`](@ref). Scores are losses: lower is
better, and an aggregator's "worst case" is the largest regret.
"""
struct RegretTransform <: AbstractTransform end

"""
    RelativeRegretTransform()

Replace the payoff table by its [`relative_regret_table`](@ref). Scores are losses.
"""
struct RelativeRegretTransform <: AbstractTransform end

_sense(::IdentityTransform) = :max
_sense(::AbstractTransform) = :min
_apply(::IdentityTransform, M) = M
_apply(::RegretTransform, M) = _regret_matrix(M)
_apply(::RelativeRegretTransform, M) = _relative_regret_matrix(M)

# Robustness metrics
# ------------------

"""
    RobustnessMetric(summary; transform=IdentityTransform(), scenarios=nothing)

A robustness metric in three steps: restrict the problem to a subset of `scenarios`
(state labels; `nothing` keeps all, see [`restrict_scenarios`](@ref)), apply
`transform` to the restricted payoff table, and collapse each action's row with the
aggregator `summary`. The subset is taken first, so regret is measured against the
best action within the scenarios kept.

[`robustness`](@ref) evaluates a metric; [`DecisionRule`](@ref) adds a selector to
one. Named instances: [`maximin_metric`](@ref), [`maximax_metric`](@ref),
[`hurwicz_metric`](@ref), [`laplace_metric`](@ref) and
[`minimax_regret_metric`](@ref).
"""
struct RobustnessMetric{A<:AbstractAggregator,X<:AbstractTransform,
                        S<:Union{Nothing,Vector{Symbol}}}
    summary::A
    transform::X
    scenarios::S
end

function RobustnessMetric(summary::AbstractAggregator;
                          transform::AbstractTransform=IdentityTransform(),
                          scenarios=nothing)
    subset = scenarios === nothing ? nothing : _labels(scenarios)
    return RobustnessMetric(summary, transform, subset)
end

_sense(m::RobustnessMetric) = _sense(m.transform)

"""
    robustness(problem::DecisionProblem, metric::RobustnessMetric) -> Vector

The value of `metric` for every feasible action, in [`feasible_actions`](@ref) order.
With the identity transform a higher value is better; with a regret transform a
lower one is.

```jldoctest
julia> p = DecisionProblem(PayoffTable([0 -1000; -200 -200]), ScenarioSet([:s1, :s2]));

julia> robustness(p, minimax_regret_metric())
2-element Vector{Int64}:
 800
 200
```
"""
function robustness(problem::DecisionProblem, metric::RobustnessMetric)
    p = metric.scenarios === nothing ? problem :
        restrict_scenarios(problem, metric.scenarios)
    v = _view(p)
    return _scores(metric.summary, _apply(metric.transform, v.Q), v, _sense(metric))
end

"""
    maximin_metric(; scenarios=nothing)

The worst payoff over the scenarios: `RobustnessMetric(WorstCase())`.
"""
maximin_metric(; scenarios=nothing) = RobustnessMetric(WorstCase(); scenarios)

"""
    maximax_metric(; scenarios=nothing)

The best payoff over the scenarios: `RobustnessMetric(BestCase())`.
"""
maximax_metric(; scenarios=nothing) = RobustnessMetric(BestCase(); scenarios)

"""
    hurwicz_metric(α; scenarios=nothing)

The optimism-pessimism mix `α · best + (1 − α) · worst` over the scenarios:
`RobustnessMetric(Hurwicz(α))`.
"""
hurwicz_metric(α::Real; scenarios=nothing) = RobustnessMetric(Hurwicz(α); scenarios)

"""
    laplace_metric(; scenarios=nothing)

The unweighted mean payoff over the scenarios: `RobustnessMetric(Laplace())`.
"""
laplace_metric(; scenarios=nothing) = RobustnessMetric(Laplace(); scenarios)

"""
    minimax_regret_metric(; scenarios=nothing)

The largest regret over the scenarios:
`RobustnessMetric(WorstCase(); transform=RegretTransform())`. Lower is better.
"""
function minimax_regret_metric(; scenarios=nothing)
    return RobustnessMetric(WorstCase(); transform=RegretTransform(), scenarios)
end

# Selectors
# ---------

"""
    AbstractSelector

Supertype of the selectors that turn scores into a chosen set: [`Optimize`](@ref) and
[`Threshold`](@ref).
"""
abstract type AbstractSelector end

"""
    Optimize()

Choose every action whose score is best (largest for utilities, smallest for
losses). Scores within the tie tolerance of [`decide`](@ref) of the best are tied and
all are returned.
"""
struct Optimize <: AbstractSelector end

"""
    Threshold(level)

Choose every action whose score is at least `level` (for utilities) or at most
`level` (for losses). The chosen set may be empty.
"""
struct Threshold{T<:Real} <: AbstractSelector
    level::T
end

# Rules
# -----

"""
    AbstractRule

Supertype of everything [`decide`](@ref) accepts: [`DecisionRule`](@ref) (which the
named scalar rules construct) and the set-valued rules [`EAdmissible`](@ref),
[`Maximal`](@ref), [`IntervalDominance`](@ref) and [`ParetoFront`](@ref).
"""
abstract type AbstractRule end

"""
    DecisionRule(summary; transform=IdentityTransform(), scenarios=nothing,
                 selector=Optimize(), name=:DecisionRule)
    DecisionRule(metric::RobustnessMetric; selector=Optimize(), name=:DecisionRule)

A scalar decision rule: a [`RobustnessMetric`](@ref) (transform, scenario subset,
aggregator) that scores each action, and a selector that picks from the scores.
`name` labels the rule in results and reports.

The named rules are instances: [`MaxExpectedUtility`](@ref), [`Maximin`](@ref),
[`Maximax`](@ref), [`MinimaxRegret`](@ref), [`GammaMaximin`](@ref) and
[`Satisfice`](@ref). Others are one line, for example
`DecisionRule(Hurwicz(0.3))`, `DecisionRule(AlphaMaxmin(0.7))`,
`DecisionRule(CVaR(0.1))` or
`DecisionRule(Expectation(); transform=RegretTransform())`.
"""
struct DecisionRule{M<:RobustnessMetric,S<:AbstractSelector} <: AbstractRule
    name::Symbol
    metric::M
    selector::S
end

function DecisionRule(metric::RobustnessMetric; selector::AbstractSelector=Optimize(),
                      name::Symbol=:DecisionRule)
    return DecisionRule(name, metric, selector)
end

function DecisionRule(summary::AbstractAggregator;
                      transform::AbstractTransform=IdentityTransform(), scenarios=nothing,
                      selector::AbstractSelector=Optimize(), name::Symbol=:DecisionRule)
    return DecisionRule(name, RobustnessMetric(summary; transform, scenarios), selector)
end

"""
    MaxExpectedUtility()

Maximise expected utility under a single prior: `DecisionRule(Expectation())`. The
chosen actions are the Bayes actions.
"""
MaxExpectedUtility() = DecisionRule(Expectation(); name=:MaxExpectedUtility)

"""
    Maximin()

Maximise the worst payoff over states (Wald): `DecisionRule(WorstCase())`.
Probabilities are ignored.
"""
Maximin() = DecisionRule(WorstCase(); name=:Maximin)

"""
    Maximax()

Maximise the best payoff over states: `DecisionRule(BestCase())`. Probabilities are
ignored.
"""
Maximax() = DecisionRule(BestCase(); name=:Maximax)

"""
    MinimaxRegret()

Minimise the largest regret over states (Savage):
`DecisionRule(WorstCase(); transform=RegretTransform())`. The scores are the
[`max_regret`](@ref) values, and lower is better.

**Minimax regret depends on the menu of actions.** It violates independence of
irrelevant alternatives: adding or removing an action, even one that is never
chosen, can change which of the others is chosen, because it changes the regret
table (proved as `minimaxRegret_violates_iia` in the Lean development). A
feasibility filter changes the menu in the same way. The result of
[`decide`](@ref) carries this warning as `details.note`, with
`details.menu_dependent == true`, and a [`DecisionReport`](@ref) repeats it.
"""
function MinimaxRegret()
    return DecisionRule(WorstCase(); transform=RegretTransform(), name=:MinimaxRegret)
end

"""
    GammaMaximin()

Maximise the lower expectation over a [`CredalSet`](@ref) (Gilboa and Schmeidler's
maxmin expected utility): `DecisionRule(LowerExpectation())`. Only the generating
points of the set are needed.
"""
GammaMaximin() = DecisionRule(LowerExpectation(); name=:GammaMaximin)

"""
    Satisfice(threshold; aggregator=Expectation(), transform=IdentityTransform())

Accept every action whose score reaches `threshold`:
`DecisionRule(aggregator; transform, selector=Threshold(threshold))`. For utilities
an action is accepted when its score is at least `threshold`; with a regret transform,
when it is at most `threshold`. The chosen set lists all accepted actions in table
order, the first choice is the first of them, and the set is empty (first choice
`nothing`) when no action is good enough.
"""
function Satisfice(threshold::Real; aggregator::AbstractAggregator=Expectation(),
                   transform::AbstractTransform=IdentityTransform())
    return DecisionRule(aggregator; transform, selector=Threshold(threshold),
                        name=:Satisfice)
end

"""
    EAdmissible()

Set-valued rule over a [`CredalSet`](@ref): an action is E-admissible when it
maximises expected utility under **some** prior in the set (Levi).

The set is the convex hull of its generating points and an action can be optimal
only at an interior prior, so checking the generators alone is wrong. For each action
`a`, [`decide`](@ref) solves the linear programme

    maximise t  subject to  Σ_k λ_k (E[a, k] − E[b, k]) ≥ t for every action b,
                            Σ_k λ_k = 1,  λ ≥ 0,

where `E[a, k]` is the expected utility of `a` at generator `k`. The optimum is at
most zero and `a` is E-admissible exactly when it is zero. The result's
`details.witnesses` holds, for each chosen action, a prior in the set under which it
is optimal, and `details.weights` the convex weights on the generating points that
give that prior (`witness = Σ_k weights[k] · point_k`). For a `Rational` problem both
are exact, and [`export_eadmissible_witness`](@ref) writes them for an independent
checker.

The programme is [`eadmissibility_model`](@ref), and `decide` takes the keywords
`optimizer` and `rounding_tol` of [`solve_program`](@ref): by default the built-in
exact simplex, otherwise any JuMP optimiser. Whatever solved it, on an exact problem
with zero tolerances an action is reported E-admissible only with weights that have
been checked exactly (nonnegative, summing to one, the action optimal under their
mixture). `details.exact` maps every feasible action to whether its yes or no is an
exact statement: always `true` by the default route and with an exact optimiser on
an exact problem; with a floating-point optimiser a yes is exact (its witness was
rounded and verified) and a no is numerical, to the tolerance of the optimiser,
unless its optimum was certified by duality; on a floating-point problem every answer
is numerical. `details.route` names the route (see [`ProgramSolution`](@ref)).

An action that is not [`Maximal`](@ref) is strictly beaten by another action at every
generating point, hence under every prior of the set, so it is **not E-admissible and
no programme is solved for it** (theorem `isMixedDominated_of_not_isMaximal` of the
Lean development). `details.dominated_by` maps each such action to the action that
beats it; its "no" is exact on an exact problem with zero tolerances. If a
floating-point optimiser reports an action E-admissible and the rounded witness fails
the exact check, [`ExactVerificationError`](@ref) is thrown.
"""
struct EAdmissible <: AbstractRule end

"""
    Maximal()

Set-valued rule over a [`CredalSet`](@ref): an action is maximal when no other action
has a strictly higher expected utility under **every** prior in the set (Walley).
Only the generating points are needed. Every E-admissible action is maximal.
"""
struct Maximal <: AbstractRule end

"""
    IntervalDominance()

Set-valued rule over a [`CredalSet`](@ref): an action is kept when its upper
expectation is at least the largest lower expectation of any action. Only the
generating points are needed. Every maximal action is kept, so this is the largest of
the three sets. The result's `details` hold the `lower` and `upper` expectations.
"""
struct IntervalDominance <: AbstractRule end

"""
    ParetoFront(aggregator=Expectation())

Set-valued rule over several objectives. `decide(problems, ParetoFront())` takes a
vector of [`DecisionProblem`](@ref)s with the same feasible actions, one per
objective, scores each action on each objective with `aggregator`, and keeps the
actions that are not dominated: no other action is at least as good on every
objective and strictly better on one. All objectives are utilities (higher is
better). The result's `scores` is the matrix of scores, rows = actions and columns =
objectives.
"""
struct ParetoFront{A<:AbstractAggregator} <: AbstractRule
    aggregator::A
end
ParetoFront() = ParetoFront(Expectation())

_rule_name(r::DecisionRule) = r.name
_rule_name(r::AbstractRule) = nameof(typeof(r))

# Results
# -------

"""
    DecisionResult

What [`decide`](@ref) returns. Fields:

- `rule`: the rule that was applied.
- `actions`: the feasible actions, in table order.
- `scores`: one score per action for a scalar rule; `nothing` for the set-valued
  rules over a credal set; the action-by-objective matrix for [`ParetoFront`](@ref).
- `chosen`: the labels of all chosen actions, in table order. Ties are all listed.
- `sense`: `:max` when a higher score is better, `:min` when a lower one is.
- `details`: a named tuple of rule-specific extras (see each set-valued rule, and
  [`MinimaxRegret`](@ref) for the note every regret-based rule attaches).

A result prints as a table of the actions with their scores and whether each was
chosen, and renders as a Markdown or HTML table in a notebook or a Quarto document.

[`first_choice`](@ref) gives the deterministic single choice.
"""
struct DecisionResult{R<:AbstractRule,S,D<:NamedTuple}
    rule::R
    actions::Vector{Symbol}
    scores::S
    chosen::Vector{Symbol}
    sense::Symbol
    details::D
end

"""
    first_choice(result::DecisionResult) -> Union{Symbol,Nothing}

The deterministic single choice of a result: the chosen action that comes first in
the order of the payoff table, or `nothing` when the chosen set is empty (possible
only for [`Satisfice`](@ref)). Tied actions are never broken at random.
"""
first_choice(r::DecisionResult) = isempty(r.chosen) ? nothing : first(r.chosen)

_tolerance(x, atol::Real, rtol::Real) = max(atol, rtol * maximum(abs, x; init=zero(rtol)))

function _select(::Optimize, scores::AbstractVector, sense::Symbol, tol::Real)
    if sense === :max
        best = maximum(scores)
        return findall(s -> s >= best - tol, scores)
    end
    best = minimum(scores)
    return findall(s -> s <= best + tol, scores)
end
function _select(sel::Threshold, scores::AbstractVector, sense::Symbol, ::Real)
    return sense === :max ? findall(>=(sel.level), scores) :
           findall(<=(sel.level), scores)
end

"""
    decide(problem::DecisionProblem, rule; atol=0, rtol) -> DecisionResult
    decide(problems::AbstractVector{<:DecisionProblem}, rule::ParetoFront; atol=0, rtol)
    decide(tables, belief, rule::ParetoFront; feasible=nothing, baseline=nothing, atol=0, rtol)

Apply `rule` to `problem` and return a [`DecisionResult`](@ref) holding the score of
every feasible action, the chosen actions and the rule.

**Ties.** Every action whose score is within `max(atol, rtol · max|score|)` of the
best is chosen. `rtol` defaults to zero for `Rational` and integer problems, where
ties are exact, and to `sqrt(eps)` in floating point. The chosen actions are listed
in the order of the payoff table and [`first_choice`](@ref) is the first of them;
nothing is random.

A rule that is not defined for the form of the belief throws
[`BeliefFormError`](@ref): for example [`MaxExpectedUtility`](@ref) on a
[`ScenarioSet`](@ref). The set-valued rules [`EAdmissible`](@ref), [`Maximal`](@ref)
and [`IntervalDominance`](@ref) take a [`CredalSet`](@ref); under a single prior all
three return the Bayes actions.

```jldoctest
julia> t = PayoffTable([:nothing, :levee, :relocate], [:low, :moderate, :high, :extreme],
                       [0 -10 -60 -150; -20 -20 -25 -90; -48 -48 -48 -48]);

julia> p = DecisionProblem(t, [1 // 2, 3 // 10, 3 // 20, 1 // 20]);

julia> decide(p, MaxExpectedUtility()).chosen
1-element Vector{Symbol}:
 :nothing

julia> decide(p, Maximin()).chosen
1-element Vector{Symbol}:
 :relocate

julia> r = decide(p, MinimaxRegret()); (r.scores, r.chosen)
([102, 42, 48], [:levee])
```
"""
function decide(problem::DecisionProblem, rule::DecisionRule; atol::Real=0,
                rtol::Real=_default_rtol(eltype(problem.payoff)))
    scores = robustness(problem, rule.metric)
    sense = _sense(rule.metric)
    idx = _select(rule.selector, scores, sense, _tolerance(scores, atol, rtol))
    actions = feasible_actions(problem)
    details = rule.metric.transform isa IdentityTransform ? (;) :
              (; menu_dependent=true, note=_MENU_NOTE)
    if problem.belief isa SampleBelief &&
       rule.metric.summary isa Union{WorstCase,BestCase,Hurwicz}
        details = merge(details, (; sample_dependent=true, sample_note=_SAMPLE_NOTE))
    end
    return DecisionResult(rule, actions, scores, actions[idx], sense, details)
end

# Attached to the result of a state-wise rule applied to a sample.
const _SAMPLE_NOTE = "The worst and best cases are taken over the draws in the " *
                     "sample, whatever their weights, so this score depends on which " *
                     "draws were sampled: a larger sample can only make the worst " *
                     "case worse and the best case better."

# Attached to the result of every regret-based rule.
const _MENU_NOTE = "Regret is measured against the best feasible action in each " *
                   "state, so this rule depends on the menu of actions: adding or " *
                   "removing an action, even one that is never chosen, can change " *
                   "the choice (independence of irrelevant alternatives fails)."

# Expected utility of each feasible action at each generating point of the belief.
function _expectation_matrix(problem::DecisionProblem, rule::AbstractRule)
    v = _view(problem)
    v.priors === nothing &&
        throw(BeliefFormError(_rule_name(rule), v.form,
                              "a scenario set carries no probabilities; supply a " *
                              "CredalSet"))
    return _expectations(v.Q, v.priors), v
end

function decide(problem::DecisionProblem, rule::IntervalDominance; atol::Real=0,
                rtol::Real=_default_rtol(eltype(problem.payoff)))
    E, _ = _expectation_matrix(problem, rule)
    lower = vec(minimum(E; dims=2))
    upper = vec(maximum(E; dims=2))
    tol = _tolerance(E, atol, rtol)
    idx = findall(u -> u >= maximum(lower) - tol, upper)
    actions = feasible_actions(problem)
    return DecisionResult(rule, actions, nothing, actions[idx], :max, (; lower, upper))
end

function decide(problem::DecisionProblem, rule::Maximal; atol::Real=0,
                rtol::Real=_default_rtol(eltype(problem.payoff)))
    E, _ = _expectation_matrix(problem, rule)
    tol = _tolerance(E, atol, rtol)
    n = size(E, 1)
    dominated(a) = any(b -> all(k -> E[b, k] - E[a, k] > tol, axes(E, 2)), 1:n)
    idx = findall(!dominated, 1:n)
    actions = feasible_actions(problem)
    return DecisionResult(rule, actions, nothing, actions[idx], :max, (;))
end

# Whether action `a` maximises expected utility under the mixture of the generating
# points with weights `w`, exactly. `E[b, k]` is the expected utility of `b` at point k.
function _optimal_under(E::AbstractMatrix, w::AbstractVector, a::Int)
    eu = E * w
    return all(>=(0), w) && isone(sum(w)) && eu[a] == maximum(eu)
end

# Decide E-admissibility of action `a` by the programme of `eadmissibility_model`.
# Returns `(admitted, weights, exact, route)`: `weights` are the convex weights on the
# generating points (`nothing` when not admitted) and `exact` says whether the yes or no
# is an exact statement. For an exact problem and a zero tolerance, a yes is returned
# only with weights that pass `_optimal_under`, whatever solved the programme.
function _eadmissible_witness(E::AbstractMatrix, a::Int; tol::Real=0, optimizer=nothing,
                              rounding_tol=_ROUNDING_TOLS)
    sol = solve_program(_eadmissibility_program(E, a); optimizer, rounding_tol)
    sol.status === :optimal || throw(LinearProgramError(:e_admissibility, sol.status))
    slack, λ = sol.objective, sol.values.λ
    strict = _is_exact(eltype(E)) && iszero(tol)
    if !strict
        # Floating point, or an exact problem with a tolerance: admitted to tolerance.
        return slack >= -tol, slack >= -tol ? λ : nothing, false, sol.route
    end
    Eb = _BigRat.(E)
    if sol.exact
        slack >= 0 || return false, nothing, true, sol.route
        _optimal_under(Eb, λ, a) ||
            throw(ExactVerificationError(:e_admissibility,
                                         "the weights found for action $a do not make " *
                                         "it optimal under their mixture"))
        return true, λ, true, sol.route
    end
    # A rounded floating-point solution that was not certified optimal. A yes needs
    # only a witness, so clean the weights up and check them exactly.
    w = _clean_weights(λ)
    w !== nothing && _optimal_under(Eb, w, a) && return true, w, true, sol.route
    if slack < -1e-6 * max(1, maximum(abs, Eb))
        # A "no" from a floating-point optimiser is exact once a mixed action that
        # strictly beats `a` at every generating point passes the exact check.
        certified = _mixed_dominator(E, a; optimizer, rounding_tol) !== nothing
        return false, nothing, certified, sol.route
    end
    throw(ExactVerificationError(:e_admissibility,
                                 "the floating-point optimiser reports action $a " *
                                 "E-admissible to its tolerance, but the rounded " *
                                 "weights do not make it exactly optimal"))
end

function decide(problem::DecisionProblem, rule::EAdmissible; atol::Real=0,
                rtol::Real=_default_rtol(eltype(problem.payoff)), optimizer=nothing,
                rounding_tol=_ROUNDING_TOLS)
    E, v = _expectation_matrix(problem, rule)
    tol = _tolerance(E, atol, rtol)
    T = eltype(E)
    actions = feasible_actions(problem)
    idx = Int[]
    witnesses = Dict{Symbol,Vector}()
    weights = Dict{Symbol,Vector}()
    exact = Dict{Symbol,Bool}()
    route = :internal
    # An action that is not maximal is strictly beaten at every generating point by
    # another action, hence under every prior of the set, so it is not E-admissible
    # (Lean `isMixedDominated_of_not_isMaximal`) and no programme is solved for it.
    dominated_by = Dict{Symbol,Symbol}()
    strict = _is_exact(T) && iszero(tol)
    for a in axes(E, 1)
        b = findfirst(b -> all(k -> E[b, k] - E[a, k] > tol, axes(E, 2)), axes(E, 1))
        if b !== nothing
            dominated_by[actions[a]] = actions[b]
            exact[actions[a]] = strict
            continue
        end
        admitted, λ, sure, route = _eadmissible_witness(E, a; tol, optimizer,
                                                        rounding_tol)
        exact[actions[a]] = sure
        admitted || continue
        push!(idx, a)
        witnesses[actions[a]] = [_from_big(T,
                                           sum(λ[k] * v.priors[k][j] for k in eachindex(λ)))
                                 for j in axes(v.Q, 2)]
        weights[actions[a]] = _from_big.(T, λ)
    end
    return DecisionResult(rule, actions, nothing, actions[idx], :max,
                          (; witnesses, weights, exact, route, dominated_by))
end

# Rows of `S` not dominated by another row (all >=, one > by more than `tol`).
function _nondominated(S::AbstractMatrix, tol::Real)
    function dominates(b, a)
        return all(j -> S[b, j] >= S[a, j] - tol, axes(S, 2)) &&
               any(j -> S[b, j] > S[a, j] + tol, axes(S, 2))
    end
    return findall(a -> !any(b -> dominates(b, a), axes(S, 1)), axes(S, 1))
end

function decide(problems::AbstractVector{<:DecisionProblem}, rule::ParetoFront;
                objectives=nothing, atol::Real=0,
                rtol::Real=mapreduce(p -> _default_rtol(eltype(p.payoff)), max, problems;
                                     init=0))
    isempty(problems) && throw(ArgumentError("ParetoFront needs at least one objective"))
    actions = feasible_actions(first(problems))
    for p in problems
        feasible_actions(p) == actions ||
            throw(LabelMismatchError("feasible actions of the objectives of a Pareto " *
                                     "front", actions, feasible_actions(p)))
    end
    S = reduce(hcat, [aggregate(rule.aggregator, p) for p in problems])
    idx = _nondominated(S, _tolerance(S, atol, rtol))
    names = objectives === nothing ? _default_labels("objective", length(problems)) :
            _labels(objectives)
    length(names) == length(problems) ||
        throw(ArgumentError("$(length(names)) objective names for " *
                            "$(length(problems)) objectives"))
    return DecisionResult(rule, actions, S, actions[idx], :max, (; objectives=names))
end

function decide(tables, belief::Union{AbstractBelief,AbstractVector{<:Real}},
                rule::ParetoFront; feasible=nothing, baseline=nothing, kw...)
    named = _named_tables(tables)
    problems = [DecisionProblem(t, belief; feasible, baseline) for (_, t) in named]
    return decide(problems, rule; objectives=first.(named), kw...)
end
