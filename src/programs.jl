# The linear programmes of the package, each stated once as a JuMP model. A builder
# returns a `DecisionProgram`: the model and the references to its variables and
# constraints. `solve_program` (src/solve.jl) solves one by any of three routes; the
# user-facing functions (`decide` with `EAdmissible`, `is_garbling_of`, `deficiency`,
# `robust_evpi`) build, solve and then verify.

"""
    DecisionProgram{T}

A linear programme of this package as a JuMP model, returned by the builders
[`eadmissibility_model`](@ref), [`garbling_model`](@ref), [`deficiency_model`](@ref),
[`evpi_envelope_model`](@ref), [`mixed_dominance_model`](@ref) and
[`separating_payoff_model`](@ref). Fields:

- `name`: `:e_admissibility`, `:garbling`, `:deficiency`, `:evpi_envelope`,
  `:mixed_dominance` or `:separating_payoff`.
- `model`: the `JuMP.GenericModel{T}`. `T` is `Rational{BigInt}` when the problem is
  exact (`Rational` or integer entries) and the floating-point type of the problem
  otherwise. No optimiser is attached.
- `variables`: a named tuple of the model's variable references, with the names used
  in the builder's docstring.
- `constraints`: a named tuple of its constraint references.

The model is an ordinary JuMP model: it can be printed, extended, given an optimiser
and solved directly, or passed to [`solve_program`](@ref), which also reads the
solution back in the shape of `variables`.
"""
struct DecisionProgram{T,V<:NamedTuple,C<:NamedTuple}
    name::Symbol
    model::JuMP.GenericModel{T}
    variables::V
    constraints::C
end

# The number type of the model of a problem with entries of type `S`.
function _program_type(::Type{S}) where {S}
    return _is_exact(S) ? _BigRat :
           S <: AbstractFloat ? S : Float64
end

# E-admissibility
# ---------------

# `E[b, k]`: expected utility of action `b` at generating point `k`.
function _eadmissibility_program(E::AbstractMatrix, a::Integer)
    n, K = size(E)
    T = _program_type(eltype(E))
    D = T.(E[a:a, :]) .- T.(E)
    model = JuMP.GenericModel{T}()
    λ = JuMP.@variable(model, [1:K], lower_bound=zero(T), base_name="λ")
    t = JuMP.@variable(model, base_name="t")
    simplex = JuMP.@constraint(model, sum(λ) == one(T))
    optimal = JuMP.@constraint(model, [b = 1:n],
                               sum(D[b, k] * λ[k] for k in 1:K) - t >= zero(T))
    JuMP.@objective(model, Max, t)
    return DecisionProgram(:e_admissibility, model, (; λ, t), (; simplex, optimal))
end

"""
    eadmissibility_model(problem::DecisionProblem, action) -> DecisionProgram

The linear programme that decides whether `action` (a label among the feasible
actions) is E-admissible in `problem`, whose belief is a [`CredalSet`](@ref) or a
single prior. With `E[b, k] = Σ_θ p_k(θ) Q[b, θ]` the expected utility of the feasible
action `b` at the generating point `p_k`, `n` feasible actions and `K` generating
points, it is

    maximise    t
    subject to  Σ_k λ_k = 1,                                 (simplex)
                Σ_k λ_k (E[a, k] − E[b, k]) − t ≥ 0   for every action b,   (optimal[b])
                λ_k ≥ 0,  t free.

The row `b = a` reads `t ≤ 0`, so the optimum is at most zero, and `a` is
E-admissible exactly when it is zero; `Σ_k λ_k p_k` is then a prior in the set under
which `a` maximises expected utility.

Shape: `K + 1` variables (`variables.λ`, a vector of `K`, and `variables.t`) and
`n + 1` constraints (`constraints.simplex` and `constraints.optimal`, a vector of
`n`), plus the `K` lower bounds on `λ`.

This is the programme that `decide(problem, EAdmissible())` solves for each action.
"""
function eadmissibility_model(problem::DecisionProblem,
                              action::Union{Symbol,AbstractString})
    actions = feasible_actions(problem)
    a = findfirst(==(Symbol(action)), actions)
    a === nothing &&
        throw(ArgumentError("$action is not among the feasible actions $actions"))
    E, _ = _expectation_matrix(problem, EAdmissible())
    return _eadmissibility_program(E, a)
end

# Garbling and deficiency
# -----------------------

"""
    garbling_model(e′::Experiment, e::Experiment) -> DecisionProgram

The feasibility programme that decides whether `e′` is a garbling of `e`. With `L`
the likelihood of `e` (`nz` signals by `S` states) and `L′` that of `e′` (`nz′`
signals), find a matrix `G` (`nz′` by `nz`, `G[z′, z] = P(z′ | z)`) with

    Σ_z′ G[z′, z] = 1                      for every signal z of e,       (columns[z])
    Σ_z G[z′, z] L[z, θ] = L′[z′, θ]       for every z′ and state θ,      (product[z′, θ])
    G[z′, z] ≥ 0.

There is no objective. The programme is feasible exactly when `e′` is a garbling of
`e`, and any feasible `G` is a witness (`G * L == L′`).

Shape: `nz′ · nz` variables (`variables.G`, a matrix) and `nz + nz′ · S` equality
constraints (`constraints.columns`, `constraints.product`), plus the lower bounds.

This is the programme behind [`is_garbling_of`](@ref) for exact likelihoods and a
zero tolerance; a floating-point pair is tested through [`deficiency_model`](@ref).
"""
function garbling_model(e′::Experiment, e::Experiment)
    _check_same_states(e, e′)
    T = _program_type(promote_type(eltype(e), eltype(e′)))
    L, Lt = T.(e.likelihood), T.(e′.likelihood)
    nz, S = size(L)
    nt = size(Lt, 1)
    model = JuMP.GenericModel{T}()
    G = JuMP.@variable(model, [1:nt, 1:nz], lower_bound=zero(T), base_name="G")
    columns = JuMP.@constraint(model, [z = 1:nz], sum(G[zt, z] for zt in 1:nt) == one(T))
    product = JuMP.@constraint(model, [zt = 1:nt, θ = 1:S],
                               sum(L[z, θ] * G[zt, z] for z in 1:nz) == Lt[zt, θ])
    return DecisionProgram(:garbling, model, (; G), (; columns, product))
end

"""
    deficiency_model(e::Experiment, e′::Experiment) -> DecisionProgram

The linear programme for Le Cam's deficiency of `e` with respect to `e′`. With `L`
the likelihood of `e` (`nz` signals by `S` states) and `L′` that of `e′` (`nz′`
signals), over a garbling matrix `G` (`nz′` by `nz`), absolute deviations `u`
(`nz′` by `S`) and a bound `d`:

    minimise    d
    subject to  Σ_z′ G[z′, z] = 1                              for every z,        (columns[z])
                Σ_z G[z′, z] L[z, θ] − u[z′, θ] ≤ L′[z′, θ]    for every z′, θ,    (above[z′, θ])
                Σ_z G[z′, z] L[z, θ] + u[z′, θ] ≥ L′[z′, θ]    for every z′, θ,    (below[z′, θ])
                Σ_z′ u[z′, θ] − d ≤ 0                          for every θ,        (total[θ])
                G ≥ 0,  u ≥ 0,  d ≥ 0.

At the optimum `d = min_G max_θ ‖G L[·, θ] − L′[·, θ]‖₁`, twice the deficiency
(total variation is half the ℓ¹ distance): `deficiency(e, e′) = d / 2`. The programme
is always feasible and bounded.

Shape: `nz′ · nz + nz′ · S + 1` variables (`variables.G`, `variables.u`,
`variables.d`) and `nz + 2 · nz′ · S + S` constraints (`constraints.columns`,
`constraints.above`, `constraints.below`, `constraints.total`), plus the lower bounds.

This is the programme that [`deficiency`](@ref) solves.
"""
function deficiency_model(e::Experiment, e′::Experiment)
    _check_same_states(e, e′)
    T = _program_type(promote_type(eltype(e), eltype(e′)))
    L, Lt = T.(e.likelihood), T.(e′.likelihood)
    nz, S = size(L)
    nt = size(Lt, 1)
    model = JuMP.GenericModel{T}()
    G = JuMP.@variable(model, [1:nt, 1:nz], lower_bound=zero(T), base_name="G")
    u = JuMP.@variable(model, [1:nt, 1:S], lower_bound=zero(T), base_name="u")
    d = JuMP.@variable(model, lower_bound=zero(T), base_name="d")
    columns = JuMP.@constraint(model, [z = 1:nz], sum(G[zt, z] for zt in 1:nt) == one(T))
    above = JuMP.@constraint(model, [zt = 1:nt, θ = 1:S],
                             sum(L[z, θ] * G[zt, z] for z in 1:nz) - u[zt, θ] <=
                             Lt[zt, θ])
    below = JuMP.@constraint(model, [zt = 1:nt, θ = 1:S],
                             sum(L[z, θ] * G[zt, z] for z in 1:nz) + u[zt, θ] >=
                             Lt[zt, θ])
    total = JuMP.@constraint(model, [θ = 1:S],
                             sum(u[zt, θ] for zt in 1:nt) - d <= zero(T))
    JuMP.@objective(model, Min, d)
    return DecisionProgram(:deficiency, model, (; G, u, d),
                           (; columns, above, below, total))
end

# The upper EVPI envelope
# -----------------------

# `E[a, k]` as above; `perfect[k] = Σ_θ p_k(θ) max_a Q[a, θ]`.
function _evpi_envelope_program(E::AbstractMatrix, perfect::AbstractVector)
    n, K = size(E)
    T = _program_type(promote_type(eltype(E), eltype(perfect)))
    Et, m = T.(E), T.(perfect)
    model = JuMP.GenericModel{T}()
    λ = JuMP.@variable(model, [1:K], lower_bound=zero(T), base_name="λ")
    w = JuMP.@variable(model, base_name="w")
    simplex = JuMP.@constraint(model, sum(λ) == one(T))
    act_now = JuMP.@constraint(model, [a = 1:n],
                               sum(Et[a, k] * λ[k] for k in 1:K) - w <= zero(T))
    JuMP.@objective(model, Max, sum(m[k] * λ[k] for k in 1:K) - w)
    return DecisionProgram(:evpi_envelope, model, (; λ, w), (; simplex, act_now))
end

"""
    evpi_envelope_model(problem::DecisionProblem) -> DecisionProgram

The linear programme for the largest single-prior [`evpi`](@ref) over the credal set
of `problem` (the upper end of the envelope in [`robust_evpi`](@ref)). With
`E[a, k]` the expected utility of the feasible action `a` at the generating point
`p_k`, `m_k = Σ_θ p_k(θ) max_a Q[a, θ]` the value with perfect information at `p_k`,
`n` feasible actions and `K` generating points:

    maximise    Σ_k λ_k m_k − w
    subject to  Σ_k λ_k = 1,                                (simplex)
                Σ_k λ_k E[a, k] − w ≤ 0   for every action a,   (act_now[a])
                λ_k ≥ 0,  w free.

At the optimum `w` is the act-now value under the prior `Σ_k λ_k p_k`, so the
objective is the EVPI at that prior. EVPI is concave in the prior, which is why its
maximum over the hull is a linear programme while its minimum is at a generating
point. The programme is always feasible and bounded.

Shape: `K + 1` variables (`variables.λ`, `variables.w`) and `n + 1` constraints
(`constraints.simplex`, `constraints.act_now`), plus the `K` lower bounds on `λ`.
"""
function evpi_envelope_model(problem::DecisionProblem)
    v = _credal_view(problem, :evpi_envelope_model)
    Q = v.Q
    best = [maximum(view(Q, :, j)) for j in axes(Q, 2)]
    return _evpi_envelope_program(_expectations(Q, v.priors),
                                  [dot(q, best) for q in v.priors])
end

# Certified "no" answers
# ----------------------

# `E[b, k]`: expected utility of action `b` at generating point `k`.
function _mixed_dominance_program(E::AbstractMatrix, a::Integer)
    n, K = size(E)
    T = _program_type(eltype(E))
    D = T.(E) .- T.(E[a:a, :])
    model = JuMP.GenericModel{T}()
    μ = JuMP.@variable(model, [1:n], lower_bound=zero(T), base_name="μ")
    s = JuMP.@variable(model, base_name="s")
    simplex = JuMP.@constraint(model, sum(μ) == one(T))
    dominates = JuMP.@constraint(model, [k = 1:K],
                                 sum(D[b, k] * μ[b] for b in 1:n) - s >= zero(T))
    JuMP.@objective(model, Max, s)
    return DecisionProgram(:mixed_dominance, model, (; μ, s), (; simplex, dominates))
end

"""
    mixed_dominance_model(problem::DecisionProblem, action) -> DecisionProgram

The linear programme that looks for a **mixed action that strictly beats** `action`
at every generating point of the credal set of `problem`: the dual of
[`eadmissibility_model`](@ref), stated as a programme of its own so that a "not
E-admissible" answer has a witness that can be checked. With `E[b, k]` the expected
utility of the feasible action `b` at the generating point `p_k`, `n` feasible
actions and `K` generating points, it is

    maximise    s
    subject to  Σ_b μ_b = 1,                                         (simplex)
                Σ_b μ_b (E[b, k] − E[a, k]) − s ≥ 0   for every k,    (dominates[k])
                μ_b ≥ 0,  s free.

Taking `μ` to be all on `a` gives `s = 0`, so the optimum is at least zero, and it is
positive exactly when `a` is **not** E-admissible (linear-programming duality; Lean
theorem `not_isEAdmissible_iff_isMixedDominated`). The mixture `μ` is then better
than `a` by at least `s` under every prior of the set.

Shape: `n + 1` variables (`variables.μ`, a vector of `n`, and `variables.s`) and
`K + 1` constraints (`constraints.simplex` and `constraints.dominates`, a vector of
`K`), plus the `n` lower bounds on `μ`.

This is the programme behind [`export_not_eadmissible_witness`](@ref).
"""
function mixed_dominance_model(problem::DecisionProblem,
                               action::Union{Symbol,AbstractString})
    actions = feasible_actions(problem)
    a = findfirst(==(Symbol(action)), actions)
    a === nothing &&
        throw(ArgumentError("$action is not among the feasible actions $actions"))
    E, _ = _expectation_matrix(problem, EAdmissible())
    return _mixed_dominance_program(E, a)
end

"""
    separating_payoff_model(e′::Experiment, e::Experiment) -> DecisionProgram

The linear programme that looks for a **decision problem in which `e′` is worth
strictly more than `e`**, which exists exactly when `e′` is not a garbling of `e`
(Blackwell's theorem; Lean theorem `not_isGarblingOf_iff_exists_separating`). The
actions of the problem are the signals of `e′`, its prior is uniform, and its payoff
table is the unknown `V[z′, θ]`. With `L` the likelihood of `e` (`nz` signals by `S`
states) and `L′` that of `e′` (`nz′` signals),

    maximise    Σ_{z′, θ} L′[z′, θ] V[z′, θ] − Σ_z u_z
    subject to  u_z ≥ Σ_θ L[z, θ] V[z′, θ]     for every z and z′,   (best[z, z′])
                −1 ≤ V[z′, θ] ≤ 1,  u_z free.

At the optimum `u_z` is the best that can be done after signal `z` of `e`, so
`Σ_z u_z` is `S` times the value of `e`, and the first sum is `S` times the value of
taking action `z′` after signal `z′` of `e′`, a lower bound on the value of `e′`.
`V = 0` gives zero, so the optimum is at least zero, and it is positive exactly when
`e′` is not a garbling of `e`. The bounds on `V` only fix the scale.

Shape: `nz′ · S + nz` variables (`variables.V`, a matrix, and `variables.u`, a
vector) and `nz · nz′` constraints (`constraints.best`), plus the bounds on `V`.

This is the programme behind [`export_not_garbling_witness`](@ref).
"""
function separating_payoff_model(e′::Experiment, e::Experiment)
    _check_same_states(e, e′)
    T = _program_type(promote_type(eltype(e), eltype(e′)))
    L, Lt = T.(e.likelihood), T.(e′.likelihood)
    nz, S = size(L)
    nt = size(Lt, 1)
    model = JuMP.GenericModel{T}()
    V = JuMP.@variable(model, [1:nt, 1:S], lower_bound=-one(T), upper_bound=one(T),
                       base_name="V")
    u = JuMP.@variable(model, [1:nz], base_name="u")
    best = JuMP.@constraint(model, [z = 1:nz, zt = 1:nt],
                            u[z] - sum(L[z, θ] * V[zt, θ] for θ in 1:S) >= zero(T))
    JuMP.@objective(model, Max,
                    sum(Lt[zt, θ] * V[zt, θ] for zt in 1:nt, θ in 1:S) - sum(u))
    return DecisionProgram(:separating_payoff, model, (; V, u), (; best))
end

# Exact budgeted unit selection, constrained policies and credal-set pruning
# --------------------------------------------------------------------------

"""
    budget_selection_model(units, budget; option=:chosen, integer=false,
                           worst_case=false) -> DecisionProgram

The linear (or 0–1 integer) programme for selecting management units from [`survey_or_act`](@ref)
under a `budget` (the exact optimisation counterpart to the heuristic in
[`select_under_budget`](@ref)). With `N` units of value `v_i` and cost `c_i` (using
`largest_cost` when `worst_case = true`):

    maximise    Σ_i v_i x_i
    subject to  Σ_i c_i x_i ≤ budget,       (budget)
                0 ≤ x_i ≤ 1                 (binary when `integer = true`).

When `integer = false` (default), returns the exact LP relaxation solvable by
[`solve_program`](@ref) with the built-in exact rational simplex or any LP optimiser.
When `integer = true`, marks `x` binary for a MILP solver such as `HiGHS.Optimizer`.

Shape: `N` variables (`variables.x`) and `1` constraint (`constraints.budget`), plus bounds.
"""
function budget_selection_model(units::AbstractVector, budget::Real;
                                option::Symbol=:chosen, integer::Bool=false,
                                worst_case::Bool=false)
    option in (:chosen, :act_now, :survey) ||
        throw(ArgumentError("option is :chosen, :act_now or :survey, not :$option"))
    (isfinite(budget) && budget >= 0) ||
        throw(ArgumentError("the budget must be nonnegative and finite, got $budget"))
    isempty(units) && throw(ArgumentError("units cannot be empty"))
    picks = [_unit_option(u, option) for u in units]
    vs = [p.value for p in picks]
    cs = [worst_case ? p.largest_cost : p.cost for p in picks]
    S = promote_type(typeof(budget), eltype(vs), eltype(cs))
    T = integer ? Float64 : _program_type(S)
    N = length(picks)
    model = JuMP.GenericModel{T}()
    x = JuMP.@variable(model, [1:N], lower_bound=zero(T), upper_bound=one(T), base_name="x")
    if integer
        for i in 1:N
            JuMP.set_binary(x[i])
        end
    end
    bud = JuMP.@constraint(model, sum(T(cs[i]) * x[i] for i in 1:N) <= T(budget))
    JuMP.@objective(model, Max, sum(T(vs[i]) * x[i] for i in 1:N))
    return DecisionProgram(:budget_selection, model, (; x), (; budget=bud))
end

"""
    constrained_policy_model(problem::DecisionProblem; constraints=(), chance=())
        -> DecisionProgram

The linear programme for an optimal **randomised (mixed) policy** `μ ∈ Δ(A)` over the
`n` feasible actions of `problem` (which needs a single prior `p`), maximising expected
utility subject to linear expectation constraints and chance constraints:

    maximise    Σ_a μ_a E_p[Q_a]
    subject to  Σ_a μ_a = 1,                                          (simplex)
                Σ_a μ_a E_p[C_k,a] ⋚ rhs_k          for k in constraints, (expectation[k])
                Σ_a μ_a P_p(H_r,a ≥ τ_r) ≥ α_r      for r in chance,      (chance[r])
                μ_a ≥ 0.

- `constraints`: collection of `(table::PayoffTable, sense::Symbol, rhs::Real)` tuples
  with `sense ∈ (:<=, :>=, :(==))` and `table` sharing the action and state labels of
  `problem`.
- `chance`: collection of `(table::PayoffTable, threshold::Real, min_prob::Real)` or
  `(threshold::Real, min_prob::Real)` tuples (the two-argument form applies to
  `problem.payoff`), requiring probability at least `min_prob ∈ [0, 1]` of attaining
  payoff `≥ threshold`.

Shape: `n` variables (`variables.μ`) and `1 + length(constraints) + length(chance)`
constraints (`constraints.simplex`, `constraints.expectation`, `constraints.chance`),
plus lower bounds.
"""
function constrained_policy_model(problem::DecisionProblem; constraints=(), chance=())
    p = _prior(problem, :constrained_policy_model)
    Q = _feasible_Q(problem)
    na = size(Q, 1)
    eu = Q * p
    c_list = collect(constraints)
    ch_list = collect(chance)
    S = promote_type(eltype(eu), eltype(p))
    exp_coeffs = Vector{Vector{Any}}()
    exp_senses = Symbol[]
    exp_rhs = Any[]
    for item in c_list
        length(item) == 3 ||
            throw(ArgumentError("each expectation constraint must be (table, sense, rhs)"))
        tbl, sense, rhs = item
        tbl isa PayoffTable ||
            throw(ArgumentError("expected a PayoffTable in constraint, got $(typeof(tbl))"))
        _check_same_labels(problem.payoff, tbl)
        sense in (:<=, :>=, :(==)) ||
            throw(ArgumentError("constraint sense must be :<=, :>= or :(==), got :$sense"))
        (rhs isa Real && isfinite(rhs)) ||
            throw(ArgumentError("constraint rhs must be finite, got $rhs"))
        ce = _feasible_rows(problem, tbl.Q) * p
        S = promote_type(S, eltype(ce), typeof(rhs))
        push!(exp_coeffs, collect(ce))
        push!(exp_senses, sense)
        push!(exp_rhs, rhs)
    end
    ch_coeffs = Vector{Vector{Any}}()
    ch_rhs = Any[]
    for item in ch_list
        tbl, thresh, min_prob = if length(item) == 2
            (problem.payoff, item[1], item[2])
        elseif length(item) == 3
            (item[1], item[2], item[3])
        else
            throw(ArgumentError("each chance constraint must be (threshold, min_prob) or " *
                                "(table, threshold, min_prob)"))
        end
        tbl isa PayoffTable ||
            throw(ArgumentError("expected a PayoffTable in chance constraint, got " *
                                "$(typeof(tbl))"))
        _check_same_labels(problem.payoff, tbl)
        (thresh isa Real && isfinite(thresh)) ||
            throw(ArgumentError("chance threshold must be finite, got $thresh"))
        (min_prob isa Real && isfinite(min_prob) && 0 <= min_prob <= 1) ||
            throw(ArgumentError("chance min_prob must lie in [0, 1], got $min_prob"))
        M = _feasible_rows(problem, tbl.Q)
        probs = [sum(p[j] for j in axes(M, 2) if M[a, j] >= thresh; init=zero(eltype(p)))
                 for a in 1:na]
        S = promote_type(S, eltype(probs), typeof(min_prob))
        push!(ch_coeffs, collect(probs))
        push!(ch_rhs, min_prob)
    end
    T = _program_type(S)
    model = JuMP.GenericModel{T}()
    μ = JuMP.@variable(model, [1:na], lower_bound=zero(T), base_name="μ")
    simplex = JuMP.@constraint(model, sum(μ) == one(T))
    expectation_c = Any[]
    for k in eachindex(exp_coeffs)
        lhs = sum(T(exp_coeffs[k][a]) * μ[a] for a in 1:na)
        rhs = T(exp_rhs[k])
        if exp_senses[k] === :<=
            push!(expectation_c, JuMP.@constraint(model, lhs <= rhs))
        elseif exp_senses[k] === :>=
            push!(expectation_c, JuMP.@constraint(model, lhs >= rhs))
        else
            push!(expectation_c, JuMP.@constraint(model, lhs == rhs))
        end
    end
    chance_c = [JuMP.@constraint(model,
                                 sum(T(ch_coeffs[r][a]) * μ[a] for a in 1:na) >=
                                 T(ch_rhs[r])) for r in eachindex(ch_coeffs)]
    JuMP.@objective(model, Max, sum(T(eu[a]) * μ[a] for a in 1:na))
    return DecisionProgram(:constrained_policy, model, (; μ),
                           (; simplex, expectation=expectation_c, chance=chance_c))
end

"""
    generator_redundancy_model(credal::CredalSet, index::Integer) -> DecisionProgram

The feasibility linear programme deciding whether generating point `index` of `credal`
lies in the convex hull of the other generating points `{p_k : k ≠ index}`:

    find        λ_k ≥ 0   (k ≠ index)
    subject to  Σ_{k ≠ index} λ_k = 1,             (simplex)
                Σ_{k ≠ index} λ_k p_k = p_index.   (match[θ])

Feasible if and only if point `index` is redundant (not a strict vertex of the credal set).

Shape: `K - 1` variables (`variables.λ`) and `1 + S` equality constraints
(`constraints.simplex`, `constraints.match`), plus lower bounds.
"""
function generator_redundancy_model(credal::CredalSet, index::Integer)
    K = length(credal.points)
    1 <= index <= K ||
        throw(ArgumentError("index $index out of range 1:$K"))
    K >= 2 ||
        throw(ArgumentError("generator_redundancy_model needs at least 2 generating points"))
    S = length(credal.states)
    T = _program_type(eltype(first(credal.points)))
    others = [credal.points[k] for k in 1:K if k != index]
    target = credal.points[index]
    m = length(others)
    model = JuMP.GenericModel{T}()
    λ = JuMP.@variable(model, [1:m], lower_bound=zero(T), base_name="λ")
    simplex = JuMP.@constraint(model, sum(λ) == one(T))
    match = JuMP.@constraint(model, [s = 1:S],
                             sum(T(others[k][s]) * λ[k] for k in 1:m) == T(target[s]))
    return DecisionProgram(:generator_redundancy, model, (; λ), (; simplex, match))
end

function _is_redundant_generator(states::Vector{Symbol}, kept::Vector{Vector{T}}, idx::Int,
                                 rational::Bool, optimizer, rounding_tol) where {T}
    cur = CredalSet(states, kept)
    prog = generator_redundancy_model(cur, idx)
    sol = solve_program(prog; optimizer, rounding_tol)
    sol.status === :optimal || return false
    λ = sol.values.λ
    others = [kept[k] for k in eachindex(kept) if k != idx]
    target = kept[idx]
    if rational
        sol.exact || return false
        for s in eachindex(states)
            recon = sum(λ[k] * _BigRat(others[k][s]) for k in eachindex(others))
            recon == _BigRat(target[s]) || return false
        end
        return true
    end
    tol = 100 * _default_atol(T)
    for s in eachindex(states)
        recon = sum(λ[k] * others[k][s] for k in eachindex(others))
        abs(recon - target[s]) <= tol || return false
    end
    return true
end

"""
    prune_generators(credal::CredalSet; optimizer=nothing,
                     rounding_tol=(1e-8, 1e-10, 1e-12)) -> CredalSet

Remove duplicate and redundant (non-extreme) generating points from `credal`, returning
a minimal [`CredalSet`](@ref) with the exact same convex hull. Each candidate point is
tested against the remaining points via [`generator_redundancy_model`](@ref) and
[`solve_program`](@ref); a point is dropped only when a convex-combination witness is
verified (`exact` for rational credal sets).
"""
function prune_generators(credal::CredalSet; optimizer=nothing,
                          rounding_tol=_ROUNDING_TOLS)
    pts = copy(credal.points)
    T = eltype(first(pts))
    rational = _is_exact(T)
    # First deduplicate identical points in order.
    kept = Vector{T}[]
    for p in pts
        any(q -> q == p, kept) || push!(kept, p)
    end
    length(kept) <= 2 && return CredalSet(credal.states, kept)
    i = 1
    while i <= length(kept) && length(kept) > 1
        if _is_redundant_generator(credal.states, kept, i, rational, optimizer,
                                   rounding_tol)
            deleteat!(kept, i)
        else
            i += 1
        end
    end
    return CredalSet(credal.states, kept)
end
