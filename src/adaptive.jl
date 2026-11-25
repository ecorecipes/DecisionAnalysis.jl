# Sequential (adaptive) management: the same decision taken period after period, with
# what each action reveals used in the next. The recursion is the one of
# `proofs/DecisionAnalysisProofs/Finite/Adaptive.lean`, on unnormalised weights (the
# joint form of `voi.jl`): nothing is divided by the probability of a signal.

# The experiment that goes with each feasible action; `nothing` where the action
# reveals nothing.
function _action_experiments(problem::DecisionProblem, experiment::Experiment)
    _check_states(experiment, problem.payoff.states, "payoff table")
    return Union{Nothing,Experiment}[experiment for _ in feasible_actions(problem)]
end
function _action_experiments(problem::DecisionProblem, experiments)
    named = Dict{Symbol,Any}(Symbol(k) => v
                             for (k, v) in (experiments isa AbstractVector{<:Pair} ?
                                            experiments : pairs(experiments)))
    for (a, e) in named
        a in problem.payoff.actions ||
            throw(ArgumentError("experiments names $a, which is not an action of the " *
                                "payoff table"))
        e isa Experiment ||
            throw(ArgumentError("the entry of experiments for $a is not an Experiment"))
        _check_states(e, problem.payoff.states, "payoff table")
    end
    return Union{Nothing,Experiment}[get(named, a, nothing)
                                     for a in feasible_actions(problem)]
end

_weights_key(w::AbstractVector{<:Union{Integer,Rational}}) = w
_weights_key(w::AbstractVector) = round.(w; sigdigits=12)

function _horizon_weight(discount, horizon::Integer)
    return sum(discount^k for k in 0:(horizon - 1))
end

"""
    adaptive_management(problem::DecisionProblem, experiments;
                        horizon, discount=1, depth=min(horizon, 3)) -> NamedTuple

**Sequential (adaptive) management**: the decision of `problem` is taken in each of
`horizon` periods. The state is fixed and unknown; in each period the action `a` pays
`Q[a, θ]` and reveals a signal, the belief is updated, and the next action is chosen
on the new belief. What is learned depends on what is done:

- `experiments` is one [`Experiment`](@ref), observed whatever the action, or a
  collection of `action => Experiment` pairs (a vector of pairs, a named tuple or a
  dictionary). An action that is not named reveals nothing.
- `discount` multiplies the payoffs of each later period (`δ ≥ 0`; period `t` has
  weight `δ^(t−1)`).

Four ways of managing are valued, each as a total over the horizon:

- `no_learning`: the best single action on the prior, repeated, ignoring what is seen;
- `passive`: in each period the best action for the current belief, which is updated
  as results arrive, but no action is ever taken *in order to* learn (passive adaptive
  management);
- `active`: the best policy, which may take an action that pays less now because of
  what it reveals (active adaptive management). It is found by the recursion
  `V_n(w) = max_a [Σ_θ w(θ) Q[a, θ] + δ Σ_z V_{n−1}(w · L_a[z, ·])]`;
- `perfect`: the state known from the start.

Always `no_learning ≤ passive ≤ active ≤ perfect` (proved in the Lean development,
`Finite/Adaptive.lean`). The result is `(; horizon, discount, no_learning, passive,
active, perfect, value_of_learning, value_of_probing, evpi, baseline, act_now_action,
first_action, policy, beliefs_evaluated)`: `value_of_learning = active − no_learning`;
`value_of_probing = active − passive`, what deliberate experimentation adds;
`evpi = perfect − no_learning`, which bounds both; `baseline` is the value of taking
the baseline action in every period (`nothing` without a baseline); `act_now_action`
is the first action of the passive and no-learning managers and `first_action` that of
the active one, so the two differ exactly when it pays to probe at once.

`policy` is the active policy as a tree, to `depth` periods: each node is
`(; period, probability, belief, action, value, branches)`, with the probability of
reaching it, the belief there, the action taken, the expected value from there on
given that the node is reached, and `branches`, a vector of `signal => node` for the
signals of positive probability (empty at the last period shown).

Ties go to the first action. Exact in, exact out. The recursion visits every distinct
belief once (`beliefs_evaluated`); beliefs reached by the same observations in a
different order coincide, so repeated trials of one kind stay cheap, but the number
grows quickly with the number of signals, actions and periods. With floating-point
inputs, beliefs that agree to twelve significant digits are treated as one. The exact
weights are products of one likelihood per period, so over many periods rational
inputs should be `Rational{BigInt}` (machine rationals overflow with an error). Needs a
single prior; infeasible actions are left out.

```jldoctest
julia> t = PayoffTable([:safe, :risky], [:bad, :good], [0 0; -3//2 1]);

julia> p = DecisionProblem(t, [1 // 2, 1 // 2]; baseline=:safe);

julia> shows = Experiment([1 0; 0 1]; signals=[:failed, :worked]);

julia> r = adaptive_management(p, (; risky=shows); horizon=3);

julia> r.no_learning, r.passive, r.active, r.perfect
(0//1, 0//1, 3//4, 3//2)

julia> r.act_now_action, r.first_action, r.value_of_probing
(:safe, :risky, 3//4)

julia> [signal => node.action for (signal, node) in r.policy.branches]
2-element Vector{Pair{Symbol, Symbol}}:
 :failed => :safe
 :worked => :risky
```
"""
function adaptive_management(problem::DecisionProblem, experiments; horizon::Integer,
                             discount::Real=1, depth::Integer=min(horizon, 3))
    horizon >= 1 || throw(ArgumentError("the horizon must be at least 1, got $horizon"))
    discount >= 0 ||
        throw(ArgumentError("the discount cannot be negative, got $discount"))
    depth >= 0 || throw(ArgumentError("the depth cannot be negative, got $depth"))
    p = _prior(problem, :adaptive_management)
    Q = _feasible_Q(problem)
    actions = feasible_actions(problem)
    studies = _action_experiments(problem, experiments)
    likelihoods = [e === nothing ? nothing : e.likelihood for e in studies]
    T = promote_type(eltype(Q), eltype(p), typeof(discount),
                     (eltype(L) for L in likelihoods if L !== nothing)...)
    w0 = collect(T, p)
    # The weights after each signal of positive probability under action a.
    function updates(w, a)
        local L = likelihoods[a]
        L === nothing && return [(0, w)]
        local out = Tuple{Int,Vector{T}}[]
        for z in axes(L, 1)
            local next = w .* view(L, z, :)
            iszero(sum(next)) || push!(out, (z, next))
        end
        return out
    end
    # The closures below recurse, so every variable they assign is declared local: a
    # name shared with the enclosing function would be overwritten by the inner calls.
    active = Dict{Tuple{Int,Vector{T}},Tuple{T,Int}}()
    function best(n, w)
        n == 0 && return (zero(T), 0)
        local key = (n, _weights_key(w))
        haskey(active, key) && return active[key]
        local scores = Q * w
        local value, choice = zero(T), 0
        for a in eachindex(actions)
            local v = scores[a]
            if n > 1
                for (_, next) in updates(w, a)
                    v += discount * first(best(n - 1, next))
                end
            end
            (choice == 0 || v > value) && ((value, choice) = (v, a))
        end
        active[key] = (value, choice)
        return (value, choice)
    end
    myopic = Dict{Tuple{Int,Vector{T}},T}()
    function passive(n, w)
        n == 0 && return zero(T)
        local key = (n, _weights_key(w))
        haskey(myopic, key) && return myopic[key]
        local scores = Q * w
        local a = argmax(scores)
        local v = scores[a]
        if n > 1
            for (_, next) in updates(w, a)
                v += discount * passive(n - 1, next)
            end
        end
        myopic[key] = v
        return v
    end
    function node(n, w, levels)
        local value, a = best(n, w)
        local total = sum(w)
        local branches = levels <= 1 || n <= 1 || likelihoods[a] === nothing ? [] :
                         [studies[a].signals[z] => node(n - 1, next, levels - 1)
                          for (z, next) in updates(w, a)]
        return (; period=horizon - n + 1, probability=total, belief=w ./ total,
                action=actions[a], value=value / total, branches)
    end

    weight = _horizon_weight(discount, horizon)
    now = Q * w0
    a0 = argmax(now)
    no_learning = weight * now[a0]
    perfect = weight * sum(w0[t] * maximum(view(Q, :, t)) for t in eachindex(w0))
    active_value, first_choice_index = best(horizon, w0)
    passive_value = passive(horizon, w0)
    base = problem.baseline === nothing ? nothing :
           findfirst(==(problem.baseline), actions)
    return (; horizon, discount, no_learning, passive=passive_value, active=active_value,
            perfect, value_of_learning=active_value - no_learning,
            value_of_probing=active_value - passive_value, evpi=perfect - no_learning,
            baseline=base === nothing ? nothing : weight * now[base],
            act_now_action=actions[a0], first_action=actions[first_choice_index],
            policy=depth == 0 ? nothing : node(horizon, w0, depth),
            beliefs_evaluated=length(active))
end
