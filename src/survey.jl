# Survey first or act now, unit by unit, under a budget (after Raymond et al. 2020,
# "Combining species distribution models and value of information analysis for spatial
# allocation of conservation resources", J Appl Ecol 57:819-830). The values are the
# expected gains over the baseline of `problem.jl`; what is new here is that actions and
# the survey have costs, and that options are ranked by value per unit of expected cost.
# Written from the equations and the description of the method in the paper; nothing is
# translated from its R listing.

function _action_costs(problem::DecisionProblem, costs::AbstractVector{<:Real})
    n = length(problem.payoff.actions)
    length(costs) == n ||
        throw(ArgumentError("action_costs has $(length(costs)) entries for the $n " *
                            "actions of the payoff table"))
    return collect(costs)
end
function _action_costs(problem::DecisionProblem, costs)
    actions = problem.payoff.actions
    given = Dict(_named_items(costs, Real, "action costs"))
    for a in keys(given)
        a in actions || throw(ArgumentError("action_costs names $a, which is not an " *
                                            "action of the payoff table"))
    end
    for a in actions
        haskey(given, a) || throw(ArgumentError("action_costs has no cost for $a"))
    end
    return [given[a] for a in actions]
end

function _imposed_policy(policy, signals::Vector{Symbol}, actions::Vector{Symbol})
    given = Dict(name => Symbol(action)
                 for (name, action) in
                     _named_items(policy, Union{Symbol,AbstractString},
                                  "actions of a policy"))
    return map(signals) do z
        haskey(given, z) || throw(ArgumentError("policy has no action for the signal $z"))
        a = findfirst(==(given[z]), actions)
        a === nothing &&
            throw(ArgumentError("policy sends $z to $(given[z]), which is not a " *
                                "feasible action"))
        return a
    end
end

# The policy (one action per signal) with the largest expected gain per unit of
# expected cost, by Dinkelbach's iteration: at the ratio λ of the current policy,
# choose after each signal the action with the largest `gain − λ cost`; if no signal
# changes its action the current policy is optimal (Lean: `ratio_le_of_parametric`).
# A new policy is kept only if its ratio is strictly larger, so the loop ends in
# floating point as well.
function _cost_effective_policy(G, K, survey_cost, base::Int)
    nz = size(G, 2)
    current = fill(base, nz)
    value(policy) = sum(G[policy[z], z] for z in 1:nz)
    cost(policy) = survey_cost + sum(K[policy[z], z] for z in 1:nz)
    ratio = _div(value(current), cost(current))
    while true
        candidate = copy(current)
        for z in 1:nz
            best = current[z]
            score = G[best, z] - ratio * K[best, z]
            for a in axes(G, 1)
                s = G[a, z] - ratio * K[a, z]
                s > score && ((best, score) = (a, s))
            end
            candidate[z] = best
        end
        candidate == current && return current
        r = _div(value(candidate), cost(candidate))
        r > ratio || return current
        current, ratio = candidate, r
    end
end

"""
    survey_or_act(problem::DecisionProblem, experiment::Experiment;
                  action_costs, survey_cost, policy=nothing) -> NamedTuple

For one management unit: **act now, or survey first and act on what the survey
shows**, when the actions and the survey both cost money and the question is what each
unit of money buys (Raymond et al. 2020). The problem needs a baseline (no
intervention) and a single prior; values are expected gains over the baseline, as in
[`expected_gain_over_baseline`](@ref).

- `action_costs`: the cost of each action over and above the baseline, as a vector
  in the order of the actions of the payoff table or as `action => cost` pairs. The
  baseline costs zero and every other feasible action costs more than zero.
- `survey_cost`: the cost of the survey, more than zero, paid whatever it shows.

Two options are compared.

- **Act now** with action `a`: value `Σ_θ p(θ) (Q[a, θ] − Q[baseline, θ])` at cost
  `action_costs[a]`. The action reported is the one with the largest value per unit of
  cost (the first of them on a tie).
- **Survey first** with a policy `z ↦ a_z`: value
  `Σ_z Σ_θ p(θ) L[z, θ] (Q[a_z, θ] − Q[baseline, θ])` at the *expected* cost
  `survey_cost + Σ_z P(z) action_costs[a_z]`. With `policy=nothing` the policy is the
  one with the largest value per unit of expected cost over all policies; `policy`
  given as `signal => action` pairs imposes one (Raymond et al. buy a unit only where
  the species is found).

The unit is surveyed first when that option has **strictly** more value per unit of
cost; on a tie it is acted on now. The result is `(; act_now, survey, survey_first,
choice, value, cost, largest_cost, cost_effectiveness)`: `act_now` is `(; action,
value, cost, cost_effectiveness)`; `survey` is `(; policy, value, cost, largest_cost,
probability_of_acting, cost_effectiveness)`, where `cost` is the expected cost,
`largest_cost` the cost if the survey returns its most expensive outcome, and
`probability_of_acting` the probability that the survey is followed by an action other
than the baseline; `choice` is `:act_now` or `:survey`, and the last four fields are
those of the chosen option. Exact in, exact out.

Surveying and then taking the same action whatever is seen has the value of acting
now at a higher cost, so a survey is chosen only when some outcome leads to a cheaper
action: its worth here is the money it saves on units where acting would gain
little, which a budget can then spend elsewhere ([`select_under_budget`](@ref)). This
is not [`evsi`](@ref), which values a study when the costs are already in the payoffs.

For one action of value 1 where a species is present (prior `p`), cost `C`, and a
survey of cost `M` that finds a present species with probability `d` and never finds
an absent one, the unit is surveyed first exactly when `d C (1 − p) > M` (Lean:
`survey_first_iff`).

```jldoctest
julia> t = PayoffTable([:none, :protect], [:absent, :present], [0 0; 0 1]);

julia> p = DecisionProblem(t, [9 // 10, 1 // 10]; baseline=:none);

julia> e = Experiment([1 3//10; 0 7//10]; signals=[:not_found, :found]);

julia> u = survey_or_act(p, e; action_costs=[0, 5000], survey_cost=750);

julia> u.choice, u.survey.policy
(:survey, [:not_found => :none, :found => :protect])

julia> u.act_now.value, u.act_now.cost, u.survey.value, u.survey.cost
(1//10, 5000, 7//100, 1100//1)
```
"""
function survey_or_act(problem::DecisionProblem, experiment::Experiment; action_costs,
                       survey_cost::Real, policy=nothing)
    _check_states(experiment, problem.payoff.states, "payoff table")
    p = _prior(problem, :survey_or_act)
    row = _baseline_row(problem, :survey_or_act)
    problem.feasible[row] ||
        throw(ArgumentError("the baseline $(problem.baseline) is not a feasible action"))
    (isfinite(survey_cost) && survey_cost > 0) ||
        throw(ArgumentError("survey_cost must be positive and finite, got $survey_cost"))
    all_costs = _action_costs(problem, action_costs)
    iszero(all_costs[row]) ||
        throw(ArgumentError("the baseline $(problem.baseline) must cost zero: " *
                            "action_costs are costs over and above no intervention"))
    rows = findall(problem.feasible)
    actions = problem.payoff.actions[rows]
    costs = all_costs[rows]
    base = findfirst(==(row), rows)
    others = setdiff(eachindex(rows), base)
    isempty(others) &&
        throw(ArgumentError("no feasible action other than the baseline"))
    all(c -> isfinite(c) && c > 0, costs[others]) ||
        throw(ArgumentError("every feasible action other than the baseline must have a " *
                            "positive finite cost"))
    L = experiment.likelihood
    gains = problem.payoff.Q[rows, :] .- problem.payoff.Q[row:row, :]
    # G[a, z]: the gain of action a jointly with signal z; K[a, z]: its expected cost.
    G = gains * (p .* transpose(L))
    pz = L * p
    K = costs * transpose(pz)

    now_values = gains * p
    a0 = others[argmax([_div(now_values[a], costs[a]) for a in others])]
    act_now = (; action=actions[a0], value=now_values[a0], cost=costs[a0],
               cost_effectiveness=_div(now_values[a0], costs[a0]))

    chosen = policy === nothing ? _cost_effective_policy(G, K, survey_cost, base) :
             _imposed_policy(policy, experiment.signals, actions)
    nz = length(pz)
    value = sum(G[chosen[z], z] for z in 1:nz)
    cost = survey_cost + sum(K[chosen[z], z] for z in 1:nz)
    possible = [z for z in 1:nz if !iszero(pz[z])]
    largest = survey_cost + maximum(costs[chosen[z]] for z in possible)
    acting = sum(pz[z] for z in possible if chosen[z] != base; init=zero(eltype(pz)))
    survey = (; policy=[experiment.signals[z] => actions[chosen[z]] for z in 1:nz], value,
              cost, largest_cost=largest, probability_of_acting=acting,
              cost_effectiveness=_div(value, cost))

    survey_first = survey.cost_effectiveness > act_now.cost_effectiveness
    return (; act_now, survey, survey_first,
            choice=survey_first ? :survey : :act_now,
            value=survey_first ? survey.value : act_now.value,
            cost=survey_first ? survey.cost : act_now.cost,
            largest_cost=survey_first ? survey.largest_cost : act_now.cost,
            cost_effectiveness=survey_first ? survey.cost_effectiveness :
                               act_now.cost_effectiveness)
end

function _unit_option(unit, option::Symbol)
    option === :chosen &&
        return (; choice=unit.choice, unit.value, unit.cost, unit.largest_cost)
    option === :act_now &&
        return (; choice=:act_now, unit.act_now.value, unit.act_now.cost,
                largest_cost=unit.act_now.cost)
    return (; choice=:survey, unit.survey.value, unit.survey.cost,
            unit.survey.largest_cost)
end

# Exact 0-1 knapsack branch-and-bound in exact arithmetic using the Dantzig fractional
# upper bound (sorted by decreasing value/cost ratio). Works with no external solver.
function _knapsack_upper_bound(idx::Int, cur_val, rem_budget, vals, costs)
    ub = float(cur_val)
    rem = float(rem_budget)
    n = length(vals)
    @inbounds for j in idx:n
        cj = float(costs[j])
        if cj <= rem
            rem -= cj
            ub += float(vals[j])
        else
            rem > 0.0 && (ub += float(vals[j]) * (rem / cj))
            break
        end
    end
    return ub
end

function _exact_knapsack(picks, budget)
    cand = [i for i in eachindex(picks) if picks[i].value > 0 && picks[i].cost <= budget]
    isempty(cand) && return Int[]
    ratio(i) = _div(picks[i].value, picks[i].cost)
    sort!(cand; by=ratio, rev=true, alg=MergeSort)
    n = length(cand)
    vals = [picks[i].value for i in cand]
    costs = [picks[i].cost for i in cand]
    # Initial incumbent from greedy pass.
    best_val = zero(promote_type(typeof(budget), eltype(vals)))
    best_mask = falses(n)
    rem0 = budget
    for j in 1:n
        if costs[j] <= rem0
            rem0 -= costs[j]
            best_val += vals[j]
            best_mask[j] = true
        end
    end
    tol = _is_exact(typeof(best_val)) ? 0.0 : 1e-12 * max(1.0, abs(float(best_val)))
    cur_mask = falses(n)
    # Iterative DFS stack: (idx, phase, cur_val, rem_budget)
    stack = Tuple{Int,Int,typeof(best_val),typeof(budget)}[(1, 0, zero(best_val), budget)]
    while !isempty(stack)
        idx, phase, cval, rem = pop!(stack)
        if phase == 2
            cur_mask[idx] = false
            continue
        end
        if idx > n
            if cval > best_val
                best_val = cval
                copyto!(best_mask, cur_mask)
            end
            continue
        end
        if _knapsack_upper_bound(idx, cval, rem, vals, costs) <= float(best_val) + tol
            continue
        end
        if phase == 0
            # Explore skipping `idx` after exploring taking `idx`.
            push!(stack, (idx, 1, cval, rem))
            if costs[idx] <= rem
                cur_mask[idx] = true
                push!(stack, (idx, 2, cval, rem))
                push!(stack, (idx + 1, 0, cval + vals[idx], rem - costs[idx]))
            end
        else
            cur_mask[idx] = false
            push!(stack, (idx + 1, 0, cval, rem))
        end
    end
    return sort!([cand[j] for j in 1:n if best_mask[j]])
end

"""
    select_under_budget(units, budget; option=:chosen, passes=nothing,
                        method=:greedy) -> NamedTuple

Choose which management units to work on with a fixed budget, from the results of
[`survey_or_act`](@ref) (Raymond et al. 2020). Each unit enters with one option: the
one it chose (`option=:chosen`), acting now (`:act_now`) or surveying first
(`:survey`); the last two give the "never survey" and "always survey" comparisons.

- `method = :greedy` (default): Raymond et al. (2020)'s ranking heuristic by value per
  unit of cost, with optional `passes` cap (`passes = nothing` runs until no unit fits;
  `passes = 2` matches the printed supplement listing).
- `method = :exact`: exact 0–1 knapsack optimisation via branch-and-bound in exact
  arithmetic (no external solver required), returning the provably value-maximising subset
  within `budget` (in index order; `passes = 1`). For a JuMP formulation or LP
  relaxation, see [`budget_selection_model`](@ref).

The result is `(; selected, choices, value, cost, largest_cost, remaining, passes)`:
the indices of the units taken; the option of each; the total value; the total cost;
the total if every survey returned its most expensive outcome; the budget left; and the
number of passes made.

```jldoctest
julia> units = [(; choice=:act_now, value=v, cost=c, largest_cost=c)
                for (v, c) in [(6, 60), (5, 55), (1, 20), (1, 30)]];

julia> s = select_under_budget(units, 100);

julia> s.selected, s.value, s.cost, s.remaining
([1, 3], 7, 80, 20)
```
"""
function select_under_budget(units::AbstractVector, budget::Real; option::Symbol=:chosen,
                             passes::Union{Nothing,Integer}=nothing,
                             method::Symbol=:greedy)
    option in (:chosen, :act_now, :survey) ||
        throw(ArgumentError("option is :chosen, :act_now or :survey, not :$option"))
    method in (:greedy, :exact) ||
        throw(ArgumentError("method must be :greedy or :exact, not :$method"))
    passes === nothing || passes >= 1 ||
        throw(ArgumentError("passes must be at least 1"))
    (isfinite(budget) && budget >= 0) ||
        throw(ArgumentError("the budget must be nonnegative and finite, got $budget"))
    picks = [_unit_option(u, option) for u in units]
    if method === :exact
        selected = _exact_knapsack(picks, budget)
        cost = sum(picks[i].cost for i in selected; init=zero(budget))
        return (; selected, choices=[picks[i].choice for i in selected],
                value=sum(picks[i].value for i in selected; init=zero(budget)),
                cost,
                largest_cost=sum(picks[i].largest_cost for i in selected;
                                 init=zero(budget)), remaining=budget - cost, passes=1)
    end
    ratio(i) = _div(picks[i].value, picks[i].cost)
    left = [i for i in eachindex(picks) if picks[i].value > 0]
    sort!(left; by=ratio, rev=true, alg=MergeSort)
    selected = Int[]
    remaining = budget
    made = 0
    while !isempty(left) && (passes === nothing || made < passes)
        if made >= 1
            let limit = remaining
                filter!(i -> picks[i].cost <= limit, left)
            end
            isempty(left) && break
            sort!(left; by=i -> (-ratio(i), picks[i].cost), alg=MergeSort)
        end
        made += 1
        taken = 0
        for i in left
            picks[i].cost <= remaining || break
            remaining -= picks[i].cost
            push!(selected, i)
            taken += 1
        end
        deleteat!(left, 1:taken)
        made > 1 && taken == 0 && break
    end
    return (; selected, choices=[picks[i].choice for i in selected],
            value=sum(picks[i].value for i in selected; init=zero(budget)),
            cost=budget - remaining,
            largest_cost=sum(picks[i].largest_cost for i in selected; init=zero(budget)),
            remaining, passes=made)
end
