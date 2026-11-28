# Certainty-equivalent value of information (`ce_evpi`, `ce_evppi`, `ce_evsi`):
# willingness-to-pay / willingness-to-trade in outcome units (individuals) under a
# risk-attitude utility. For linear utilities this equals plain EVPI/EVPPI/EVSI; for
# CARA (`ExponentialUtility`) it has a closed form `σ * (CE_info - CE_0)` computed in
# log-space; for CRRA (`PowerUtility`) it solves Howard's (1966) buying-price equation
# `V_info(v*) = V_0` by monotone bisection alongside reporting `ce_gain`.

const _CESingleUtility = Union{IndividualsUtility,LinearUtility,ExponentialUtility,
                               PowerUtility}

function _check_ce_utility(u::AbstractUtility)
    u isa _CESingleUtility ||
        throw(ArgumentError("certainty-equivalent VOI requires a single-attribute utility " *
                            "(IndividualsUtility, LinearUtility, ExponentialUtility or " *
                            "PowerUtility), got $(nameof(typeof(u)))"))
    return u
end

# Utility of an outcome level `x`, checking domain (e.g., nonnegative counts for
# IndividualsUtility or PowerUtility).
_ce_apply(u::_AttributeUtility, x::Real) = _value(u, x)
function _ce_apply(::IndividualsUtility, x::Real)
    x >= 0 ||
        throw(InvalidOutcomeError("the individuals count is $x; a count cannot be negative"))
    return x
end

# Shift-invariant log-sum-exp certainty equivalent for a list of (outcome, weight) pairs.
function _ce_exp_informed_finite(u::ExponentialUtility, X::AbstractMatrix,
                                 p::AbstractVector,
                                 group_weights::Vector{Vector{Float64}})
    r = u.risk_aversion
    σ = _direction(u)
    na, ns = size(X)
    if iszero(r)
        v0 = maximum(sum(Float64(p[j]) * Float64(σ * X[a, j]) for j in 1:ns) for a in 1:na)
        v_inf = sum(maximum(sum(gw[j] * Float64(σ * X[a, j]) for j in 1:ns) for a in 1:na)
                    for gw in group_weights)
        ce0 = σ * v0
        ce_inf = σ * v_inf
        return (; value=v_inf - v0, ce_gain=v_inf - v0, act_now_ce=ce0,
                informed_ce=ce_inf, act_now_utility=Float64(_value(u, ce0)),
                informed_utility=Float64(_value(u, ce_inf)), domain_limited=false)
    end
    α = -r * σ
    pf = Float64.(p)
    # Act-now CE per action (using shift-invariant log-sum-exp); higher σ*CE is better.
    act_ces = [_exp_certainty_equivalent(u, view(X, a, :), pf) for a in 1:na]
    a0 = argmax([σ * c for c in act_ces])
    ce0 = act_ces[a0]
    # Informed lottery: for each signal/cell group g with joint weights gw (Σ_g gw = pf),
    # choose the action maximising posterior expected utility (equivalently σ * CE_g).
    xs_inf = Float64[]
    ws_inf = Float64[]
    for gw in group_weights
        mass = sum(gw)
        mass > 0 || continue
        best_a = 1
        best_score = -Inf
        for a in 1:na
            c = _exp_certainty_equivalent(u, view(X, a, :), gw ./ mass)
            sc = σ * c
            if sc > best_score
                best_score = sc
                best_a = a
            end
        end
        for j in 1:ns
            gw[j] > 0 || continue
            push!(xs_inf, Float64(X[best_a, j]))
            push!(ws_inf, gw[j])
        end
    end
    ce_inf = _exp_certainty_equivalent(u, xs_inf, ws_inf)
    gain = max(0.0, σ * (ce_inf - ce0))
    return (; value=gain, ce_gain=gain, act_now_ce=ce0, informed_ce=ce_inf,
            act_now_utility=Float64(_value(u, ce0)),
            informed_utility=Float64(_value(u, ce_inf)), domain_limited=false)
end

# Evaluate informed expected utility under `PowerUtility` when deducting `v` from every
# outcome, returning `-Inf` if no action remains in domain at some positive-weight group.
function _power_informed_at(u::PowerUtility, X::AbstractMatrix,
                            group_weights::Vector{Vector{Float64}}, v::Float64)
    ρ = u.risk_aversion
    na, ns = size(X)
    total = 0.0
    for gw in group_weights
        best = -Inf
        for a in 1:na
            ok = true
            acc = 0.0
            for j in 1:ns
                w = gw[j]
                w > 0 || continue
                x = Float64(X[a, j]) - v
                if !(x > 0.0 || (x == 0.0 && ρ < 1.0))
                    ok = false
                    break
                end
                acc += w * _value(u, x)
            end
            ok && acc > best && (best = acc)
        end
        isfinite(best) || return -Inf
        total += best
    end
    return total
end

# Largest deduction `v >= 0` for which at least one action stays in `PowerUtility`'s domain
# at every positive-probability group.
function _power_max_deduction(X::AbstractMatrix, group_weights::Vector{Vector{Float64}})
    na, ns = size(X)
    vmax = Inf
    for gw in group_weights
        any(>(0), gw) || continue
        gmax = -Inf
        for a in 1:na
            amin = Inf
            for j in 1:ns
                gw[j] > 0 || continue
                x = Float64(X[a, j])
                x < amin && (amin = x)
            end
            amin > gmax && (gmax = amin)
        end
        gmax < vmax && (vmax = gmax)
    end
    return max(0.0, vmax)
end

function _solve_power_buying_price(eval_v, v0::Float64, v_info0::Float64, vmax::Float64,
                                   ρ::Float64)
    if v_info0 <= v0 + 1e-14 * max(1.0, abs(v0))
        return 0.0, false
    end
    if ρ < 1.0
        v_at_max = eval_v(vmax)
        if v_at_max > v0 + 1e-13 * max(1.0, abs(v0))
            return nothing, true
        end
        lo, hi = 0.0, vmax
        for _ in 1:80
            mid = 0.5 * (lo + hi)
            if eval_v(mid) > v0
                lo = mid
            else
                hi = mid
            end
            hi - lo <= 1e-13 * max(1.0, hi) && break
        end
        return 0.5 * (lo + hi), false
    end
    # ρ >= 1: V_info(v) -> -Inf as v -> vmax⁻, so a unique root exists in [0, vmax).
    lo = 0.0
    hi = vmax
    # Step slightly inside vmax until finite and <= v0.
    gap = max(eps(vmax), 1e-15 * max(1.0, vmax))
    while hi > 0.0 && eval_v(max(0.0, vmax - gap)) > v0
        gap *= 0.5
        gap == 0.0 && break
    end
    hi = max(0.0, vmax - gap)
    if eval_v(hi) > v0
        return nothing, true
    end
    for _ in 1:80
        mid = 0.5 * (lo + hi)
        if eval_v(mid) > v0
            lo = mid
        else
            hi = mid
        end
        hi - lo <= 1e-13 * max(1.0, hi) && break
    end
    return 0.5 * (lo + hi), false
end

function _ce_voi_power_finite(X::AbstractMatrix, p::AbstractVector, u::PowerUtility,
                              group_weights)
    na, ns = size(X)
    pf = Float64.(p)
    gwf = [Float64.(gw) for gw in group_weights]
    U = [Float64(_value(u, X[a, j])) for a in 1:na, j in 1:ns]
    v0 = maximum(U * pf)
    v_inf0 = sum(maximum(U * gw) for gw in gwf)
    ce0 = inverse_utility(u, v0)
    ce_inf = inverse_utility(u, v_inf0)
    ce_gain = max(0.0, ce_inf - ce0)
    vmax = _power_max_deduction(X, gwf)
    eval_v(v) = _power_informed_at(u, X, gwf, v)
    bp, limited = _solve_power_buying_price(eval_v, v0, v_inf0, vmax, u.risk_aversion)
    return (; value=bp, ce_gain, act_now_ce=ce0, informed_ce=ce_inf, act_now_utility=v0,
            informed_utility=v_inf0, domain_limited=limited)
end

function _ce_voi_finite(X::AbstractMatrix, p::AbstractVector, u::_CESingleUtility,
                        group_weights)
    na, ns = size(X)
    if u isa Union{IndividualsUtility,LinearUtility}
        # Validate domain and compute exact expected utilities and certainty equivalents.
        U = [_ce_apply(u, X[a, j]) for a in 1:na, j in 1:ns]
        v0 = maximum(U * p)
        v_inf = sum(maximum(U * gw) for gw in group_weights)
        ce0 = inverse_utility(u, v0)
        ce_inf = inverse_utility(u, v_inf)
        σ = _direction(u)
        gain = σ * (ce_inf - ce0)
        return (; value=gain, ce_gain=gain, act_now_ce=ce0, informed_ce=ce_inf,
                act_now_utility=v0, informed_utility=v_inf, domain_limited=false)
    elseif u isa ExponentialUtility
        gwf = [Float64.(gw) for gw in group_weights]
        return _ce_exp_informed_finite(u, X, p, gwf)
    end
    return _ce_voi_power_finite(X, p, u, group_weights)
end

# Group joint-weight vectors for perfect info, partition, and experiment on finite beliefs.
function _perfect_groups(p::AbstractVector)
    T = eltype(p)
    ns = length(p)
    return [begin
                v = zeros(T, ns)
                v[j] = p[j]
                v
            end
            for j in 1:ns]
end

function _partition_groups(p::AbstractVector, partition::AbstractVector)
    T = eltype(p)
    ns = length(p)
    length(partition) == ns ||
        throw(ArgumentError("the partition has $(length(partition)) entries for the " *
                            "$ns states of the payoff table"))
    return [begin
                v = zeros(T, ns)
                for j in cols
                    v[j] = p[j]
                end
                v
            end
            for cols in _cell_columns(partition)]
end

function _experiment_groups(p::AbstractVector, experiment::Experiment)
    L = experiment.likelihood
    return [p .* view(L, z, :) for z in axes(L, 1)]
end

# OutcomeTable helper: evaluate replicate-averaged cell utilities at deduction `v`, always
# applying `u` to each usable run before averaging across replicates (Jensen's inequality).
function _outcome_table_ce_slices(t::OutcomeTable, u::_CESingleUtility)
    k = findfirst(==(u.name), t.outcomes)
    k === nothing &&
        throw(InvalidOutcomeError("there is no outcome named $(u.name) among $(t.outcomes)"))
    ok = t.status .== _OK
    na, nd, nr = size(ok)
    cells = Matrix{Vector{Float64}}(undef, na, nd)
    for d in 1:nd, a in 1:na
        xs = Float64[]
        for r in 1:nr
            ok[a, d, r] && push!(xs, Float64(t.values[a, d, r, k]))
        end
        isempty(xs) &&
            throw(InvalidOutcomeError("action $(t.actions[a]) at draw $(t.draws[d]) has " *
                                      "no usable run"))
        cells[a, d] = xs
    end
    return cells, SampleBelief(t).weights
end

function _ce_exp_informed_outcomes(u::ExponentialUtility, cells::Matrix{Vector{Float64}},
                                   p::AbstractVector,
                                   group_weights::Vector{Vector{Float64}})
    r = u.risk_aversion
    σ = _direction(u)
    na, nd = size(cells)
    pf = Float64.(p)
    act_ces = Float64[]
    for a in 1:na
        xs = Float64[]
        ws = Float64[]
        for d in 1:nd
            pf[d] > 0 || continue
            c = cells[a, d]
            wr = pf[d] / length(c)
            for x in c
                push!(xs, x)
                push!(ws, wr)
            end
        end
        push!(act_ces, _exp_certainty_equivalent(u, xs, ws))
    end
    a0 = argmax([σ * c for c in act_ces])
    ce0 = act_ces[a0]
    xs_inf = Float64[]
    ws_inf = Float64[]
    for gw in group_weights
        mass = sum(gw)
        mass > 0 || continue
        best_a = 1
        best_score = -Inf
        for a in 1:na
            xs_g = Float64[]
            ws_g = Float64[]
            for d in 1:nd
                gw[d] > 0 || continue
                c = cells[a, d]
                wr = gw[d] / (mass * length(c))
                for x in c
                    push!(xs_g, x)
                    push!(ws_g, wr)
                end
            end
            sc = σ * _exp_certainty_equivalent(u, xs_g, ws_g)
            if sc > best_score
                best_score = sc
                best_a = a
            end
        end
        for d in 1:nd
            gw[d] > 0 || continue
            c = cells[best_a, d]
            wr = gw[d] / length(c)
            for x in c
                push!(xs_inf, x)
                push!(ws_inf, wr)
            end
        end
    end
    ce_inf = _exp_certainty_equivalent(u, xs_inf, ws_inf)
    gain = max(0.0, σ * (ce_inf - ce0))
    return (; value=gain, ce_gain=gain, act_now_ce=ce0, informed_ce=ce_inf,
            act_now_utility=Float64(_value(u, ce0)),
            informed_utility=Float64(_value(u, ce_inf)), domain_limited=false)
end

function _power_outcomes_at(u::PowerUtility, cells::Matrix{Vector{Float64}},
                            group_weights::Vector{Vector{Float64}}, v::Float64)
    ρ = u.risk_aversion
    na, nd = size(cells)
    total = 0.0
    for gw in group_weights
        best = -Inf
        for a in 1:na
            ok = true
            acc = 0.0
            for d in 1:nd
                w = gw[d]
                w > 0 || continue
                c = cells[a, d]
                cell_u = 0.0
                for x0 in c
                    x = x0 - v
                    if !(x > 0.0 || (x == 0.0 && ρ < 1.0))
                        ok = false
                        break
                    end
                    cell_u += _value(u, x)
                end
                ok || break
                acc += w * (cell_u / length(c))
            end
            ok && acc > best && (best = acc)
        end
        isfinite(best) || return -Inf
        total += best
    end
    return total
end

function _ce_voi_outcomes(t::OutcomeTable, u::_CESingleUtility, group_weights)
    if u isa Union{IndividualsUtility,LinearUtility}
        # Exact linear utility commutes with replicate averaging.
        pt = payoff_table(t, IndividualsUtility(u.name))
        p = SampleBelief(t).weights
        return _ce_voi_finite(pt.Q, p, u, group_weights)
    end
    cells, p = _outcome_table_ce_slices(t, u)
    gwf = [Float64.(gw) for gw in group_weights]
    if u isa ExponentialUtility
        return _ce_exp_informed_outcomes(u, cells, p, gwf)
    end
    # PowerUtility: apply u to every run before averaging over replicates.
    na, nd = size(cells)
    pf = Float64.(p)
    U = [sum(_value(u, x) for x in cells[a, d]) / length(cells[a, d])
         for a in 1:na,
             d in 1:nd]
    v0 = maximum(U * pf)
    v_inf0 = sum(maximum(U * gw) for gw in gwf)
    ce0 = inverse_utility(u, v0)
    ce_inf = inverse_utility(u, v_inf0)
    ce_gain = max(0.0, ce_inf - ce0)
    # Domain upper bound across replicates.
    vmax = Inf
    for gw in gwf
        any(>(0), gw) || continue
        gmax = -Inf
        for a in 1:na
            amin = Inf
            for d in 1:nd
                gw[d] > 0 || continue
                m = minimum(cells[a, d])
                m < amin && (amin = m)
            end
            amin > gmax && (gmax = amin)
        end
        gmax < vmax && (vmax = gmax)
    end
    vmax = max(0.0, vmax)
    eval_v(v) = _power_outcomes_at(u, cells, gwf, v)
    bp, limited = _solve_power_buying_price(eval_v, v0, v_inf0, vmax, u.risk_aversion)
    return (; value=bp, ce_gain, act_now_ce=ce0, informed_ce=ce_inf,
            act_now_utility=v0, informed_utility=v_inf0, domain_limited=limited)
end

"""
    ce_evpi(problem::DecisionProblem, utility::AbstractUtility) -> NamedTuple
    ce_evpi(outcomes::OutcomeTable, utility::AbstractUtility) -> NamedTuple

The **certainty-equivalent expected value of perfect information** in outcome units
(individuals): the maximum state-independent deduction `v* ≥ 0` along the improvement
direction that a decision-maker with single-attribute risk utility `utility`
([`IndividualsUtility`](@ref), [`LinearUtility`](@ref), [`ExponentialUtility`](@ref),
[`PowerUtility`](@ref)) would trade for perfect information about the state (Howard's
buying price, solving `V_info(v*) = V_0`).

On a [`DecisionProblem`](@ref), `problem.payoff.Q` must hold **raw outcome levels**
(before `utility` is applied), with a single prior. On an [`OutcomeTable`](@ref),
`utility` is applied to every usable replicate before averaging, preserving Jensen's
inequality.

- For linear utilities (`IndividualsUtility`, `LinearUtility`), `value` equals the
  outcome-scale [`evpi`](@ref) (exact for rational inputs).
- For [`ExponentialUtility`](@ref) (CARA), translation invariance yields the exact
  closed form `σ · (informed_ce − act_now_ce)`, computed in log-space without saturation.
- For [`PowerUtility`](@ref) (CRRA), wealth effects separate the buying price `value`
  from `ce_gain = informed_ce − act_now_ce`; `value` is found by monotone bisection. If
  `ρ < 1` and even deducting the full domain bound still leaves expected utility above
  `V_0`, `domain_limited` is `true` and `value` is `nothing` while `ce_gain` is still
  reported.

Returns `(; value, ce_gain, act_now_ce, informed_ce, act_now_utility, informed_utility,
domain_limited)`.
"""
function ce_evpi(problem::DecisionProblem, u::AbstractUtility)
    _check_ce_utility(u)
    p = _prior(problem, :ce_evpi)
    X = _feasible_Q(problem)
    return _ce_voi_finite(X, p, u, _perfect_groups(p))
end
function ce_evpi(t::OutcomeTable, u::AbstractUtility)
    _check_ce_utility(u)
    p = SampleBelief(t).weights
    return _ce_voi_outcomes(t, u, _perfect_groups(p))
end

"""
    ce_evppi(problem::DecisionProblem, utility::AbstractUtility, partition::AbstractVector)
        -> NamedTuple
    ce_evppi(outcomes::OutcomeTable, utility::AbstractUtility, partition::AbstractVector)
        -> NamedTuple

The **certainty-equivalent expected value of partial perfect information** in outcome
units (individuals) for learning which cell of `partition` holds under single-attribute
risk utility `utility`. See [`ce_evpi`](@ref) for the definition of the buying price
`value` and returned fields, and [`evppi`](@ref) for `partition`.
"""
function ce_evppi(problem::DecisionProblem, u::AbstractUtility, partition::AbstractVector)
    _check_ce_utility(u)
    p = _prior(problem, :ce_evppi)
    X = _feasible_Q(problem)
    return _ce_voi_finite(X, p, u, _partition_groups(p, partition))
end
function ce_evppi(t::OutcomeTable, u::AbstractUtility, partition::AbstractVector)
    _check_ce_utility(u)
    p = SampleBelief(t).weights
    return _ce_voi_outcomes(t, u, _partition_groups(p, partition))
end

"""
    ce_evsi(problem::DecisionProblem, utility::AbstractUtility, experiment::Experiment)
        -> NamedTuple
    ce_evsi(outcomes::OutcomeTable, utility::AbstractUtility, experiment::Experiment)
        -> NamedTuple

The **certainty-equivalent expected value of sample information** in outcome units
(individuals) for observing `experiment` before acting under single-attribute risk
utility `utility`. See [`ce_evpi`](@ref) for the definition of the buying price `value`
and returned fields, and [`evsi`](@ref) for `experiment`.
"""
function ce_evsi(problem::DecisionProblem, u::AbstractUtility, experiment::Experiment)
    _check_ce_utility(u)
    _check_states(experiment, problem.payoff.states, "payoff table")
    p = _prior(problem, :ce_evsi)
    X = _feasible_Q(problem)
    return _ce_voi_finite(X, p, u, _experiment_groups(p, experiment))
end
function ce_evsi(t::OutcomeTable, u::AbstractUtility, experiment::Experiment)
    _check_ce_utility(u)
    _check_states(experiment, t.draws, "outcome table")
    p = SampleBelief(t).weights
    return _ce_voi_outcomes(t, u, _experiment_groups(p, experiment))
end
