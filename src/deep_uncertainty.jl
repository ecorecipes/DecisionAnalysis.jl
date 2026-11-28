# Non-parametric and deep-uncertainty rules:
#   - First- and second-order stochastic dominance (`fsd_dominates`, `ssd_dominates`,
#     `StochasticDominance`)
#   - Info-gap robustness and opportuneness (`info_gap_robustness`,
#     `info_gap_opportuneness`), exact in rational arithmetic on total-variation belief
#     balls and by bisection for general nested horizon models
#   - Scenario discovery via Patient Rule Induction Method (`prim_boxes`)

# Stochastic dominance
# --------------------

function _check_lottery(x::AbstractVector{<:Real}, p::AbstractVector{<:Real}, name)
    isempty(x) && throw(ArgumentError("$name lottery cannot be empty"))
    length(x) == length(p) ||
        throw(ArgumentError("$name outcomes ($(length(x))) and probabilities " *
                            "($(length(p))) must have the same length"))
    all(isfinite, x) || throw(ArgumentError("$name outcomes must be finite"))
    all(v -> isfinite(v) && v >= 0, p) ||
        throw(ArgumentError("$name probabilities must be finite and nonnegative"))
    s = sum(p)
    s > 0 || throw(ArgumentError("$name probabilities must have positive sum"))
    return x, [_div(v, s) for v in p]
end

# Sorted distinct support `t` and CDF values `Fx, Fy` of two discrete lotteries.
function _joint_cdfs(x::AbstractVector{<:Real}, px::AbstractVector{<:Real},
                     y::AbstractVector{<:Real}, py::AbstractVector{<:Real})
    TX = promote_type(eltype(x), eltype(y))
    TP = promote_type(eltype(px), eltype(py))
    mx = Dict{TX,TP}()
    my = Dict{TX,TP}()
    for i in eachindex(x)
        v = TX(x[i])
        mx[v] = get(mx, v, zero(TP)) + TP(px[i])
    end
    for i in eachindex(y)
        v = TX(y[i])
        my[v] = get(my, v, zero(TP)) + TP(py[i])
    end
    support = sort!(collect(union(keys(mx), keys(my))))
    M = length(support)
    Fx = Vector{TP}(undef, M)
    Fy = Vector{TP}(undef, M)
    cx = zero(TP)
    cy = zero(TP)
    @inbounds for k in 1:M
        t = support[k]
        cx += get(mx, t, zero(TP))
        cy += get(my, t, zero(TP))
        Fx[k] = cx
        Fy[k] = cy
    end
    return support, Fx, Fy
end

function _fsd_check(x, px, y, py; strict::Bool=true, atol::Real=0)
    _, Fx, Fy = _joint_cdfs(x, px, y, py)
    tol = _is_exact(eltype(Fx)) ? zero(eltype(Fx)) :
          max(float(atol), sqrt(eps(float(eltype(Fx)))))
    all_le = true
    any_lt = false
    for k in eachindex(Fx)
        diff = Fx[k] - Fy[k]
        if diff > tol
            all_le = false
            break
        elseif diff < -tol
            any_lt = true
        end
    end
    return all_le && (!strict || any_lt)
end

function _ssd_check(x, px, y, py; strict::Bool=true, atol::Real=0)
    support, Fx, Fy = _joint_cdfs(x, px, y, py)
    T = promote_type(eltype(support), eltype(Fx))
    M = length(support)
    Ix = Vector{T}(undef, M)
    Iy = Vector{T}(undef, M)
    Ix[1] = zero(T)
    Iy[1] = zero(T)
    @inbounds for k in 1:(M - 1)
        dt = T(support[k + 1] - support[k])
        Ix[k + 1] = Ix[k] + T(Fx[k]) * dt
        Iy[k + 1] = Iy[k] + T(Fy[k]) * dt
    end
    scale = max(one(T), maximum(abs, Ix; init=zero(T)), maximum(abs, Iy; init=zero(T)))
    tol = _is_exact(T) ? zero(T) : max(float(atol), sqrt(eps(float(T))) * float(scale))
    all_le = true
    any_lt = false
    for k in 1:M
        diff = Ix[k] - Iy[k]
        if diff > tol
            all_le = false
            break
        elseif diff < -tol
            any_lt = true
        end
    end
    return all_le && (!strict || any_lt)
end

function _action_row(problem::DecisionProblem, action::Union{Symbol,AbstractString})
    a = Symbol(action)
    idx = findfirst(==(a), problem.payoff.actions)
    idx === nothing &&
        throw(ArgumentError("$action is not among the actions $(problem.payoff.actions)"))
    return idx
end

"""
    fsd_dominates(x, px, y, py; strict=true, atol=0) -> Bool
    fsd_dominates(problem::DecisionProblem, a, b; strict=true, atol=0) -> Bool

Whether lottery `(x, px)` **first-order stochastically dominates** `(y, py)`:
`F_x(t) ≤ F_y(t)` for every outcome level `t` (with strict inequality somewhere when
`strict = true`). Equivalently, every decision-maker with a non-decreasing utility
function prefers `x` to `y`. Exact for rational inputs.
"""
function fsd_dominates(x::AbstractVector{<:Real}, px::AbstractVector{<:Real},
                       y::AbstractVector{<:Real}, py::AbstractVector{<:Real};
                       strict::Bool=true, atol::Real=0)
    xv, pxv = _check_lottery(x, px, "first")
    yv, pyv = _check_lottery(y, py, "second")
    return _fsd_check(xv, pxv, yv, pyv; strict, atol)
end
function fsd_dominates(problem::DecisionProblem, a::Union{Symbol,AbstractString},
                       b::Union{Symbol,AbstractString}; strict::Bool=true, atol::Real=0)
    p = _prior(problem, :fsd_dominates)
    ia = _action_row(problem, a)
    ib = _action_row(problem, b)
    return _fsd_check(view(problem.payoff.Q, ia, :), p, view(problem.payoff.Q, ib, :), p;
                      strict, atol)
end

"""
    ssd_dominates(x, px, y, py; strict=true, atol=0) -> Bool
    ssd_dominates(problem::DecisionProblem, a, b; strict=true, atol=0) -> Bool

Whether lottery `(x, px)` **second-order stochastically dominates** `(y, py)`:
`∫_{-∞}^z F_x(t) dt ≤ ∫_{-∞}^z F_y(t) dt` for every `z` (strictly somewhere when
`strict = true`). Equivalently, every risk-averse decision-maker (every non-decreasing
concave utility, or equivalently every `CVaR(α)` for `α ∈ (0, 1]`) prefers `x` to `y`.
Exact for rational inputs.
"""
function ssd_dominates(x::AbstractVector{<:Real}, px::AbstractVector{<:Real},
                       y::AbstractVector{<:Real}, py::AbstractVector{<:Real};
                       strict::Bool=true, atol::Real=0)
    xv, pxv = _check_lottery(x, px, "first")
    yv, pyv = _check_lottery(y, py, "second")
    return _ssd_check(xv, pxv, yv, pyv; strict, atol)
end
function ssd_dominates(problem::DecisionProblem, a::Union{Symbol,AbstractString},
                       b::Union{Symbol,AbstractString}; strict::Bool=true, atol::Real=0)
    p = _prior(problem, :ssd_dominates)
    ia = _action_row(problem, a)
    ib = _action_row(problem, b)
    return _ssd_check(view(problem.payoff.Q, ia, :), p, view(problem.payoff.Q, ib, :), p;
                      strict, atol)
end

"""
    StochasticDominance(; order=1)

Set-valued decision rule keeping every feasible action that is **not strictly dominated**
under first-order (`order = 1`, [`fsd_dominates`](@ref)) or second-order (`order = 2`,
[`ssd_dominates`](@ref)) stochastic dominance under a single prior.

`StochasticDominance(; order=1)` retains every action optimal for at least one
non-decreasing utility function; `StochasticDominance(; order=2)` retains the SSD
efficient set (non-dominated for risk-averse utilities).
"""
struct StochasticDominance <: AbstractRule
    order::Int
    function StochasticDominance(; order::Integer=1)
        order in (1, 2) ||
            throw(ArgumentError("StochasticDominance order must be 1 (FSD) or 2 (SSD), " *
                                "got $order"))
        return new(Int(order))
    end
end

function decide(problem::DecisionProblem, rule::StochasticDominance; atol::Real=0,
                rtol::Union{Nothing,Real}=nothing)
    p = _prior(problem, :StochasticDominance)
    Q = _feasible_Q(problem)
    actions = feasible_actions(problem)
    na = size(Q, 1)
    checker = rule.order == 1 ? _fsd_check : _ssd_check
    dominated_by = Dict{Symbol,Symbol}()
    kept = Int[]
    for a in 1:na
        dom = nothing
        for b in 1:na
            b == a && continue
            if checker(view(Q, b, :), p, view(Q, a, :), p; strict=true, atol)
                dom = actions[b]
                break
            end
        end
        if dom === nothing
            push!(kept, a)
        else
            dominated_by[actions[a]] = dom
        end
    end
    return DecisionResult(rule, actions, nothing, actions[kept], :max,
                          (; order=rule.order, dominated_by))
end

# Info-gap decision theory (Ben-Haim 2006)
# ----------------------------------------

# Exact worst-case expected utility over total-variation ball TV(p, p0) <= α, inverted to
# find α̂ = sup { α ∈ [0, 1] : min_{TV(p, p0) <= α} q' p >= requirement }.
function _tv_ball_robustness(q::AbstractVector, p0::AbstractVector, requirement::Real)
    T = _div_type(promote_type(eltype(q), eltype(p0), typeof(requirement)),
                  promote_type(eltype(q), Int))
    nom = dot(q, p0)
    nom < requirement && return zero(T)
    qmin = minimum(q)
    qmin >= requirement && return one(T)
    # Moving mass δ from highest-payoff states to qmin lowers expectation by Σ δ_j (q_j - qmin).
    order = sortperm(collect(q); rev=true)
    need = nom - requirement
    used = zero(T)
    for j in order
        drop = q[j] - qmin
        drop > 0 || break
        mass = p0[j]
        iszero(mass) && continue
        step = mass * drop
        if step >= need
            used += _div(need, drop)
            return min(one(T), used)
        end
        used += T(mass)
        need -= step
    end
    return min(one(T), used)
end

# Exact best-case expected utility over total-variation ball TV(p, p0) <= α, inverted to
# find β̂ = inf { α ∈ [0, 1] : max_{TV(p, p0) <= α} q' p >= windfall }.
function _tv_ball_opportuneness(q::AbstractVector, p0::AbstractVector, windfall::Real)
    T = _div_type(promote_type(eltype(q), eltype(p0), typeof(windfall)),
                  promote_type(eltype(q), Int))
    nom = dot(q, p0)
    nom >= windfall && return zero(T)
    qmax = maximum(q)
    qmax < windfall && return nothing
    order = sortperm(collect(q))
    need = windfall - nom
    used = zero(T)
    for j in order
        gain = qmax - q[j]
        gain > 0 || break
        mass = p0[j]
        iszero(mass) && continue
        step = mass * gain
        if step >= need
            used += _div(need, gain)
            return min(one(T), used)
        end
        used += T(mass)
        need -= step
    end
    return min(one(T), used)
end

"""
    info_gap_robustness(problem::DecisionProblem, requirement::Real) -> NamedTuple
    info_gap_robustness(worst_at, actions, requirement::Real; max_horizon=1.0,
                        tol=1e-9) -> NamedTuple

**Info-gap robustness** (Ben-Haim 2006): the largest horizon of uncertainty `α̂(a) ≥ 0`
at which action `a` is guaranteed to meet the critical performance `requirement`. Larger
`α̂` means greater immunity to uncertainty.

- On a [`DecisionProblem`](@ref) with a single prior `p₀`, uncertainty at horizon
  `α ∈ [0, 1]` is the total-variation ball `U(α, p₀) = {p ∈ Δ(S) : TV(p, p₀) ≤ α}`. The
  worst-case expected utility is piecewise-linear and concave in `α`, so `α̂(a)` is solved
  **exactly** (exact in, exact out on rationals): `0` when the nominal expectation already
  falls below `requirement`, and `1` when even the worst state meets `requirement`.
- In the callable form, `worst_at(action, α)` returns the worst-case performance of
  `action` over a user-defined nested uncertainty set `U(α)` (non-increasing in `α` on
  `[0, max_horizon]`), and `α̂(a)` is found by monotone bisection.

Returns `(; actions, robustness, chosen, requirement)`.
"""
function info_gap_robustness(problem::DecisionProblem, requirement::Real)
    isfinite(requirement) ||
        throw(ArgumentError("requirement must be finite, got $requirement"))
    p0 = _prior(problem, :info_gap_robustness)
    Q = _feasible_Q(problem)
    actions = feasible_actions(problem)
    vals = [_tv_ball_robustness(view(Q, a, :), p0, requirement) for a in axes(Q, 1)]
    tol = _tolerance(vals, 0, _default_rtol(eltype(vals)))
    top = maximum(vals)
    chosen = actions[findall(>=(top - tol), vals)]
    return (; actions, robustness=vals, chosen, requirement)
end
function info_gap_robustness(worst_at, actions, requirement::Real;
                             max_horizon::Real=1.0, tol::Real=1e-9)
    isfinite(requirement) ||
        throw(ArgumentError("requirement must be finite, got $requirement"))
    (isfinite(max_horizon) && max_horizon > 0) ||
        throw(ArgumentError("max_horizon must be positive and finite, got $max_horizon"))
    (isfinite(tol) && tol > 0) ||
        throw(ArgumentError("tol must be positive and finite, got $tol"))
    acts = _labels(actions)
    isempty(acts) && throw(ArgumentError("actions cannot be empty"))
    H = Float64(max_horizon)
    vals = map(acts) do a
        Float64(worst_at(a, 0.0)) < requirement && return 0.0
        Float64(worst_at(a, H)) >= requirement && return H
        lo, hi = 0.0, H
        for _ in 1:80
            mid = 0.5 * (lo + hi)
            if Float64(worst_at(a, mid)) >= requirement
                lo = mid
            else
                hi = mid
            end
            hi - lo <= tol && break
        end
        return lo
    end
    t = _tolerance(vals, 0, _default_rtol(Float64))
    top = maximum(vals)
    chosen = acts[findall(>=(top - t), vals)]
    return (; actions=acts, robustness=vals, chosen, requirement)
end

"""
    info_gap_opportuneness(problem::DecisionProblem, windfall::Real) -> NamedTuple
    info_gap_opportuneness(best_at, actions, windfall::Real; max_horizon=1.0,
                           tol=1e-9) -> NamedTuple

**Info-gap opportuneness** (Ben-Haim 2006): the smallest horizon of uncertainty
`β̂(a) ≥ 0` at which a favourable windfall performance `windfall` becomes achievable under
action `a` (`nothing` when unattainable within the horizon). Smaller `β̂` means greater
opportunity for windfall gain.

- On a [`DecisionProblem`](@ref) with a single prior `p₀`, uses the total-variation ball
  `U(α, p₀)` and solves `β̂(a)` **exactly** (exact in, exact out on rationals).
- In the callable form, `best_at(action, α)` returns the best-case performance over a
  user-defined nested uncertainty set `U(α)` (non-decreasing in `α` on `[0, max_horizon]`).

Returns `(; actions, opportuneness, chosen, windfall)`.
"""
function info_gap_opportuneness(problem::DecisionProblem, windfall::Real)
    isfinite(windfall) || throw(ArgumentError("windfall must be finite, got $windfall"))
    p0 = _prior(problem, :info_gap_opportuneness)
    Q = _feasible_Q(problem)
    actions = feasible_actions(problem)
    vals = [_tv_ball_opportuneness(view(Q, a, :), p0, windfall) for a in axes(Q, 1)]
    attainable = [v for v in vals if v !== nothing]
    chosen = if isempty(attainable)
        Symbol[]
    else
        tol = _tolerance(attainable, 0, _default_rtol(eltype(attainable)))
        bot = minimum(attainable)
        actions[[i for i in eachindex(vals) if vals[i] !== nothing && vals[i] <= bot + tol]]
    end
    return (; actions, opportuneness=vals, chosen, windfall)
end
function info_gap_opportuneness(best_at, actions, windfall::Real;
                                max_horizon::Real=1.0, tol::Real=1e-9)
    isfinite(windfall) || throw(ArgumentError("windfall must be finite, got $windfall"))
    (isfinite(max_horizon) && max_horizon > 0) ||
        throw(ArgumentError("max_horizon must be positive and finite, got $max_horizon"))
    (isfinite(tol) && tol > 0) ||
        throw(ArgumentError("tol must be positive and finite, got $tol"))
    acts = _labels(actions)
    isempty(acts) && throw(ArgumentError("actions cannot be empty"))
    H = Float64(max_horizon)
    vals = Vector{Union{Nothing,Float64}}(undef, length(acts))
    for (idx, a) in enumerate(acts)
        if Float64(best_at(a, 0.0)) >= windfall
            vals[idx] = 0.0
        elseif Float64(best_at(a, H)) < windfall
            vals[idx] = nothing
        else
            lo, hi = 0.0, H
            for _ in 1:80
                mid = 0.5 * (lo + hi)
                if Float64(best_at(a, mid)) >= windfall
                    hi = mid
                else
                    lo = mid
                end
                hi - lo <= tol && break
            end
            vals[idx] = hi
        end
    end
    attainable = Float64[v for v in vals if v !== nothing]
    chosen = if isempty(attainable)
        Symbol[]
    else
        t = _tolerance(attainable, 0, _default_rtol(Float64))
        bot = minimum(attainable)
        acts[[i for i in eachindex(vals) if vals[i] !== nothing && vals[i] <= bot + t]]
    end
    return (; actions=acts, opportuneness=vals, chosen, windfall)
end

# Scenario discovery: Patient Rule Induction Method (PRIM)
# --------------------------------------------------------

function _prim_box_record(cols::Vector{Vector{Float64}}, names::Vector{Symbol},
                          full_lo::Vector{Float64}, full_hi::Vector{Float64},
                          box_lo::Vector{Float64}, box_hi::Vector{Float64},
                          in_box::BitVector, cases::BitVector, wf::Vector{Float64},
                          total_case_mass::Float64)
    mass = 0.0
    case_mass = 0.0
    cnt = 0
    @inbounds for i in eachindex(in_box)
        in_box[i] || continue
        w = wf[i]
        mass += w
        cnt += 1
        cases[i] && (case_mass += w)
    end
    density = mass > 0.0 ? case_mass / mass : 0.0
    coverage = total_case_mass > 0.0 ? case_mass / total_case_mass : 0.0
    restricted = Symbol[]
    bounds = Dict{Symbol,Tuple{Float64,Float64}}()
    for j in eachindex(names)
        bounds[names[j]] = (box_lo[j], box_hi[j])
        if box_lo[j] > full_lo[j] || box_hi[j] < full_hi[j]
            push!(restricted, names[j])
        end
    end
    return (; bounds, coverage, density, mass, count=cnt,
            restricted_dimensions=restricted)
end

function _prim_weighted_quantile(vals::Vector{Float64}, ws::Vector{Float64}, q::Float64)
    ord = sortperm(vals)
    tot = sum(ws)
    target = q * tot
    acc = 0.0
    @inbounds for i in ord
        acc += ws[i]
        acc >= target && return vals[i]
    end
    return vals[ord[end]]
end

"""
    prim_boxes(belief::SampleBelief, cases; peel_alpha=0.05, min_mass=0.05,
               paste=true) -> NamedTuple
    prim_boxes(problem::DecisionProblem, cases; kw...) -> NamedTuple

**Scenario discovery** by the Patient Rule Induction Method (PRIM; Friedman and Fisher
1999, *Stat. Comput.* 9:123; Bryant and Lempert 2010, *Technol. Forecast. Soc. Change*
77:34). Given a [`SampleBelief`](@ref) with numeric parameter draws and a boolean vector
`cases` (or predicate `cases(row) -> Bool`) marking draws of interest — such as policy
failures, high-regret draws, or rank reversals — finds hyper-rectangular parameter boxes
with high **density** (weighted share of `cases` inside the box) and **coverage**
(weighted share of all `cases` captured by the box).

Returns `(; trajectory, frontier, base_rate, dimensions)`:
- `trajectory`: every box along the peeling (and optional `paste`) sequence, each a
  `(; bounds, coverage, density, mass, count, restricted_dimensions)` named tuple;
- `frontier`: the Pareto-optimal subset of `trajectory` trading off `coverage` and
  `density`;
- `base_rate`: overall weighted prevalence of `cases`;
- `dimensions`: numeric parameter columns searched.
"""
function prim_boxes(belief::SampleBelief, cases; peel_alpha::Real=0.05,
                    min_mass::Real=0.05, paste::Bool=true)
    (isfinite(peel_alpha) && 0 < peel_alpha < 0.5) ||
        throw(ArgumentError("peel_alpha must lie in (0, 0.5), got $peel_alpha"))
    (isfinite(min_mass) && 0 < min_mass < 1) ||
        throw(ArgumentError("min_mass must lie in (0, 1), got $min_mass"))
    isempty(belief.draws) &&
        throw(ArgumentError("the SampleBelief carries no parameter draws"))
    wf = Float64.(belief.weights)
    n = length(wf)
    y = if cases isa AbstractVector{Bool}
        length(cases) == n ||
            throw(ArgumentError("cases has length $(length(cases)) for $n draws"))
        BitVector(cases)
    elseif cases isa AbstractVector
        length(cases) == n ||
            throw(ArgumentError("cases has length $(length(cases)) for $n draws"))
        BitVector(Bool.(cases))
    else
        rows = Tables.rowtable(belief.draws)
        BitVector(Bool[cases(r) for r in rows])
    end
    # Keep numeric columns with non-trivial spread.
    names = Symbol[]
    cols = Vector{Float64}[]
    for (k, v) in pairs(belief.draws)
        _is_numeric(v) || continue
        vf = Float64.(v)
        all(isfinite, vf) || continue
        minimum(vf) < maximum(vf) || continue
        push!(names, k)
        push!(cols, vf)
    end
    isempty(names) &&
        throw(ArgumentError("prim_boxes needs at least one varying numeric parameter"))
    d = length(names)
    full_lo = [minimum(c) for c in cols]
    full_hi = [maximum(c) for c in cols]
    box_lo = copy(full_lo)
    box_hi = copy(full_hi)
    in_box = trues(n)
    total_case_mass = sum(wf[i] for i in 1:n if y[i]; init=0.0)
    base_rate = total_case_mass

    trajectory = NamedTuple[]
    push!(trajectory,
          _prim_box_record(cols, names, full_lo, full_hi, box_lo, box_hi, in_box, y, wf,
                           total_case_mass))

    cur_mass = 1.0
    # Peeling loop
    while cur_mass > min_mass
        best_dens = -1.0
        best_mass = -1.0
        best_j = 0
        best_side = 0 # 1 = lower peel, 2 = upper peel
        best_cut = 0.0
        idx_in = findall(in_box)
        length(idx_in) <= 2 && break
        w_in = wf[idx_in]
        for j in 1:d
            v_in = cols[j][idx_in]
            # Candidate 1: raise lower bound by peeling bottom `peel_alpha`.
            q_lo = _prim_weighted_quantile(v_in, w_in, Float64(peel_alpha))
            m1, c1 = 0.0, 0.0
            @inbounds for k in idx_in
                if cols[j][k] > q_lo
                    w = wf[k]
                    m1 += w
                    y[k] && (c1 += w)
                end
            end
            if m1 > 0.0 && m1 < cur_mass - 1e-12 && m1 >= min_mass * 0.5
                dens1 = c1 / m1
                if dens1 > best_dens + 1e-12 ||
                   (abs(dens1 - best_dens) <= 1e-12 && m1 > best_mass)
                    best_dens, best_mass, best_j, best_side, best_cut = dens1, m1, j, 1,
                                                                        q_lo
                end
            end
            # Candidate 2: lower upper bound by peeling top `peel_alpha`.
            q_hi = _prim_weighted_quantile(v_in, w_in, 1.0 - Float64(peel_alpha))
            m2, c2 = 0.0, 0.0
            @inbounds for k in idx_in
                if cols[j][k] < q_hi
                    w = wf[k]
                    m2 += w
                    y[k] && (c2 += w)
                end
            end
            if m2 > 0.0 && m2 < cur_mass - 1e-12 && m2 >= min_mass * 0.5
                dens2 = c2 / m2
                if dens2 > best_dens + 1e-12 ||
                   (abs(dens2 - best_dens) <= 1e-12 && m2 > best_mass)
                    best_dens, best_mass, best_j, best_side, best_cut = dens2, m2, j, 2,
                                                                        q_hi
                end
            end
        end
        best_j == 0 && break
        if best_side == 1
            box_lo[best_j] = best_cut
            @inbounds for k in idx_in
                cols[best_j][k] <= best_cut && (in_box[k] = false)
            end
        else
            box_hi[best_j] = best_cut
            @inbounds for k in idx_in
                cols[best_j][k] >= best_cut && (in_box[k] = false)
            end
        end
        rec = _prim_box_record(cols, names, full_lo, full_hi, box_lo, box_hi, in_box, y, wf,
                               total_case_mass)
        cur_mass = rec.mass
        cur_mass < min_mass && break
        push!(trajectory, rec)
    end

    # Optional pasting phase on the final box: greedily expand a restricted face if it
    # strictly improves density.
    if paste && length(trajectory) > 1
        cur_rec = trajectory[end]
        for _ in 1:20
            best_dens = cur_rec.density
            best_j = 0
            best_side = 0
            best_cut = 0.0
            for j in 1:d
                # Check pasting lower face downward.
                if box_lo[j] > full_lo[j]
                    outside = Int[]
                    for k in 1:n
                        in_box[k] && continue
                        cols[j][k] <= box_lo[j] || continue
                        ok = true
                        for r in 1:d
                            r == j && continue
                            if cols[r][k] < box_lo[r] || cols[r][k] > box_hi[r]
                                ok = false
                                break
                            end
                        end
                        ok && push!(outside, k)
                    end
                    if !isempty(outside)
                        q_cut = _prim_weighted_quantile(cols[j][outside], wf[outside],
                                                        1.0 - Float64(peel_alpha))
                        m_add, c_add = 0.0, 0.0
                        for k in outside
                            if cols[j][k] >= q_cut
                                m_add += wf[k]
                                y[k] && (c_add += wf[k])
                            end
                        end
                        if m_add > 0.0
                            new_dens = (cur_rec.density * cur_rec.mass + c_add) /
                                       (cur_rec.mass + m_add)
                            if new_dens > best_dens + 1e-12
                                best_dens, best_j, best_side, best_cut = new_dens, j, 1,
                                                                         q_cut
                            end
                        end
                    end
                end
                # Check pasting upper face upward.
                if box_hi[j] < full_hi[j]
                    outside = Int[]
                    for k in 1:n
                        in_box[k] && continue
                        cols[j][k] >= box_hi[j] || continue
                        ok = true
                        for r in 1:d
                            r == j && continue
                            if cols[r][k] < box_lo[r] || cols[r][k] > box_hi[r]
                                ok = false
                                break
                            end
                        end
                        ok && push!(outside, k)
                    end
                    if !isempty(outside)
                        q_cut = _prim_weighted_quantile(cols[j][outside], wf[outside],
                                                        Float64(peel_alpha))
                        m_add, c_add = 0.0, 0.0
                        for k in outside
                            if cols[j][k] <= q_cut
                                m_add += wf[k]
                                y[k] && (c_add += wf[k])
                            end
                        end
                        if m_add > 0.0
                            new_dens = (cur_rec.density * cur_rec.mass + c_add) /
                                       (cur_rec.mass + m_add)
                            if new_dens > best_dens + 1e-12
                                best_dens, best_j, best_side, best_cut = new_dens, j, 2,
                                                                         q_cut
                            end
                        end
                    end
                end
            end
            best_j == 0 && break
            if best_side == 1
                box_lo[best_j] = best_cut
            else
                box_hi[best_j] = best_cut
            end
            for k in 1:n
                in_box[k] = all(r -> box_lo[r] <= cols[r][k] <= box_hi[r], 1:d)
            end
            cur_rec = _prim_box_record(cols, names, full_lo, full_hi, box_lo, box_hi,
                                       in_box, y, wf, total_case_mass)
            push!(trajectory, cur_rec)
        end
    end

    # Extract coverage–density Pareto frontier.
    frontier = NamedTuple[]
    for (i, b) in enumerate(trajectory)
        dominated = false
        for (j, o) in enumerate(trajectory)
            j == i && continue
            if o.coverage >= b.coverage - 1e-12 && o.density >= b.density - 1e-12 &&
               (o.coverage > b.coverage + 1e-12 || o.density > b.density + 1e-12)
                dominated = true
                break
            end
        end
        dominated || push!(frontier, b)
    end
    sort!(frontier; by=b -> (-b.coverage, -b.density))
    return (; trajectory, frontier, base_rate, dimensions=names)
end

function prim_boxes(problem::DecisionProblem, cases; kw...)
    problem.belief isa SampleBelief ||
        throw(BeliefFormError(:prim_boxes, nameof(typeof(problem.belief)),
                              "prim_boxes requires a SampleBelief"))
    return prim_boxes(problem.belief, cases; kw...)
end
