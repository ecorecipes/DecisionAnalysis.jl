# The linear programmes of the package, each stated once as a JuMP model. A builder
# returns a `DecisionProgram`: the model and the references to its variables and
# constraints. `solve_program` (src/solve.jl) solves one by any of three routes; the
# user-facing functions (`decide` with `EAdmissible`, `is_garbling_of`, `deficiency`,
# `robust_evpi`) build, solve and then verify.

"""
    DecisionProgram{T}

A linear programme of this package as a JuMP model, returned by the builders
[`eadmissibility_model`](@ref), [`garbling_model`](@ref), [`deficiency_model`](@ref)
and [`evpi_envelope_model`](@ref). Fields:

- `name`: `:e_admissibility`, `:garbling`, `:deficiency` or `:evpi_envelope`.
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
    λ = JuMP.@variable(model, [1:K], lower_bound = zero(T), base_name = "λ")
    t = JuMP.@variable(model, base_name = "t")
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
    G = JuMP.@variable(model, [1:nt, 1:nz], lower_bound = zero(T), base_name = "G")
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
    G = JuMP.@variable(model, [1:nt, 1:nz], lower_bound = zero(T), base_name = "G")
    u = JuMP.@variable(model, [1:nt, 1:S], lower_bound = zero(T), base_name = "u")
    d = JuMP.@variable(model, lower_bound = zero(T), base_name = "d")
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
    λ = JuMP.@variable(model, [1:K], lower_bound = zero(T), base_name = "λ")
    w = JuMP.@variable(model, base_name = "w")
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
