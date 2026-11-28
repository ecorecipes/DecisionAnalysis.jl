"""
The bridge to BayesianNetworks.jl. An influence diagram becomes a Bayesian network once
its decision is made, so a decision problem is also a set of networks, one for each
action. `normal_form` reads the payoff table, the belief and the experiments off such a
set: from one network with a management node, from a named collection of networks, from
the networks of several hypotheses (with optional model-discrimination observations), or
from a lag-1 `DynamicBayesModel` for `adaptive_management`. Loaded when BayesianNetworks
is; automatically uses variable elimination when BayesianNetworkInference is also loaded.
"""
module DecisionAnalysisBayesianNetworksExt

using DecisionAnalysis: DecisionAnalysis, DecisionProblem, Experiment, FiniteBelief,
                        PayoffTable, ScenarioSet, _is_exact
using BayesianNetworks: BayesianNetworks

const BN = BayesianNetworks

_names(x::Union{Symbol,AbstractString}) = Symbol[Symbol(x)]
_names(x) = Symbol[Symbol(v) for v in x]
function _unique_joined(tuples)
    labels = Symbol[Symbol(join(t, "_")) for t in tuples]
    allunique(labels) && return labels
    return Symbol[Symbol(join(t, "|")) for t in tuples]
end

function _to_named_tuple(x::NamedTuple)
    isempty(x) && throw(ArgumentError("expected a non-empty named collection of networks"))
    return x
end
function _to_named_tuple(x::Union{AbstractDict,AbstractVector{<:Pair}})
    iter = x isa AbstractDict ? pairs(x) : x
    items = [Symbol(first(p)) => last(p) for p in iter]
    isempty(items) &&
        throw(ArgumentError("expected a non-empty named collection of networks"))
    ks = first.(items)
    allunique(ks) || throw(ArgumentError("duplicate names in collection: $ks"))
    return NamedTuple{Tuple(ks)}(Tuple(last.(items)))
end

function _variable_states(model::BN.BayesModel, x::Symbol)
    BN.has_variable(BN.syntax(model), x) ||
        throw(ArgumentError("the network has no variable $x; its variables are " *
                            "$(BN.variable_names(BN.syntax(model)))"))
    return Symbol.(BN.states(BN.syntax(model), x))
end

# Marginal distribution over `query` variables in `query` order: uses variable elimination
# when BayesianNetworkInference is loaded, otherwise falls back to BN.marginal.
function _marginal_table(model::BN.BayesModel, query::Vector{Symbol}; max_states, atol)
    isempty(query) && return fill(1.0)
    ext = Base.get_extension(DecisionAnalysis,
                             :DecisionAnalysisBayesianNetworkInferenceExt)
    if ext !== nothing
        return ext._infer_marginal(model, query; atol)
    end
    return BN.marginal(model, query; max_states, atol).table
end

# One outcome term `variables => values` as (variables, table): `values` has one number
# for each state of the variables, as an array with one axis per variable, or, for one
# variable, a vector or a named tuple or dictionary by state.
function _outcome_term(model::BN.BayesModel, term::Pair)
    variables = _names(first(term))
    isempty(variables) && throw(ArgumentError("an outcome term needs a variable"))
    values = last(term)
    sizes = Tuple(length(_variable_states(model, x)) for x in variables)
    if values isa Union{NamedTuple,AbstractDict}
        length(variables) == 1 ||
            throw(ArgumentError("values by state name are for one variable; give an " *
                                "array for $(variables)"))
        labels = _variable_states(model, only(variables))
        given = Dict{Symbol,Any}(Symbol(k) => v for (k, v) in pairs(values))
        Set(keys(given)) == Set(labels) ||
            throw(ArgumentError("the outcome of $(only(variables)) needs one value for " *
                                "each of its states $labels, got $(collect(keys(given)))"))
        raw = [given[s] for s in labels]
        T = all(v -> _is_exact(typeof(v)), raw) ? Rational{BigInt} : Float64
        return variables, T[T(v) for v in raw]
    end
    values isa AbstractArray{<:Real} ||
        throw(ArgumentError("the values of an outcome term are an array of numbers, or a " *
                            "named tuple by state for one variable; got $(typeof(values))"))
    size(values) == sizes ||
        throw(ArgumentError("the outcome of $variables needs an array of size $sizes " *
                            "(one axis per variable, in the order of its states), got " *
                            "$(size(values))"))
    T = _is_exact(eltype(values)) ? Rational{BigInt} : Float64
    return variables, T.(values)
end

_outcome_terms(model, outcome::Pair) = [_outcome_term(model, outcome)]
_outcome_terms(model, outcome) = [_outcome_term(model, term) for term in outcome]

function _retime_term(term::Pair, t::Integer)
    vars = _names(first(term))
    timed = length(vars) == 1 ? BN.variable_at(only(vars), t) :
            Tuple(BN.variable_at(v, t) for v in vars)
    return timed => last(term)
end
_retime_outcome(outcome::Pair, t::Integer) = _retime_term(outcome, t)
_retime_outcome(outcome, t::Integer) = [_retime_term(term, t) for term in outcome]

function _axis_strides(dims::Tuple, axes::Vector{Int})
    st = Vector{Int}(undef, length(axes))
    acc = 1
    @inbounds for k in eachindex(axes)
        st[k] = acc
        acc *= dims[axes[k]]
    end
    return st
end

function _lin_idx(ci::CartesianIndex, axes::Vector{Int}, strides::Vector{Int})
    idx = 1
    @inbounds for k in eachindex(axes)
        idx += (ci[axes[k]] - 1) * strides[k]
    end
    return idx
end

function _joint_labels(model::BN.BayesModel, variables::Vector{Symbol})
    isempty(variables) && return Symbol[]
    state_vecs = [_variable_states(model, x) for x in variables]
    dims = Tuple(length(s) for s in state_vecs)
    return _unique_joined(Tuple(state_vecs[k][c[k]] for k in eachindex(variables))
                          for c in vec(collect(CartesianIndices(dims))))
end

# From one network: mass[θ, z] = P(state = θ, observed = z) and
# value[θ, z] = Σ P(state = θ, observed = z, x) × outcome(x), with the state and signal
# labels. θ and z run over the joint states of `states` and `observed`, the first
# variable fastest.
function _tables(model::BN.BayesModel, states::Vector{Symbol}, observed::Vector{Symbol},
                 terms; max_states, atol)
    query = unique!(vcat(states, observed, (first(t) for t in terms)...))
    labels = [_variable_states(model, x) for x in query]
    joint = _marginal_table(model, query; max_states, atol)
    position = Dict{Symbol,Int}(x => i for (i, x) in enumerate(query))
    dims = size(joint)
    s_axes = Int[position[x] for x in states]
    z_axes = Int[position[x] for x in observed]
    s_strides = _axis_strides(dims, s_axes)
    z_strides = _axis_strides(dims, z_axes)
    term_specs = [(Int[position[x] for x in vars],
                   _axis_strides(dims, Int[position[x] for x in vars]), table)
                  for (vars, table) in terms]
    nθ = prod(dims[ax] for ax in s_axes; init=1)
    nz = prod(dims[ax] for ax in z_axes; init=1)
    exact = _is_exact(eltype(joint)) && all(t -> _is_exact(eltype(last(t))), terms)
    T = exact ? Rational{BigInt} : Float64
    mass = zeros(T, nθ, nz)
    value = zeros(T, nθ, nz)
    for ci in CartesianIndices(dims)
        p = T(joint[ci])
        iszero(p) && continue
        θ = _lin_idx(ci, s_axes, s_strides)
        z = _lin_idx(ci, z_axes, z_strides)
        mass[θ, z] += p
        acc = zero(T)
        for (t_axes, t_strides, table) in term_specs
            acc += T(table[_lin_idx(ci, t_axes, t_strides)])
        end
        value[θ, z] += p * acc
    end
    label(variables) = _unique_joined(Tuple(labels[position[x]][c[k]]
                                            for (k, x) in enumerate(variables))
                                      for c in
                                          vec(collect(CartesianIndices(Tuple(dims[position[x]]
                                                                             for x in
                                                                                 variables)))))
    return mass, value, label(states), label(observed),
           Tuple(dims[ax] for ax in z_axes)
end

# The experiment with likelihood P(z | θ) from mass[θ, z].
function _experiment(mass::AbstractMatrix{T}, prior, signals, states) where {T<:Real}
    L = zeros(T, size(mass, 2), size(mass, 1))
    for θ in eachindex(prior)
        if iszero(prior[θ])
            L[1, θ] = one(T)          # a state of probability zero: any column will do
        else
            L[:, θ] .= view(mass, θ, :) ./ prior[θ]
        end
    end
    return Experiment(L; signals, states, atol=T <: Rational ? 0 : 1e-9)
end

function DecisionAnalysis.normal_form(models::NamedTuple{<:Any,
                                                         <:Tuple{BN.BayesModel,
                                                                 Vararg{BN.BayesModel}}};
                                      states, outcome, observed=(), baseline=nothing,
                                      atol::Real=1e-9, max_states::Integer=1_000_000)
    actions = collect(keys(models))
    chosen, seen = _names(states), _names(observed)
    isempty(chosen) && throw(ArgumentError("`states` names at least one variable"))
    allunique(vcat(chosen, seen)) ||
        throw(ArgumentError("a variable is named twice in `states` and `observed`"))
    masses = Matrix[]
    raw_values = Matrix[]
    state_labels, signal_labels, signal_sizes = Symbol[], Symbol[], ()
    for (action, model) in pairs(models)
        m, v, sl, zl, sizes = _tables(model, chosen, seen, _outcome_terms(model, outcome);
                                      max_states, atol)
        if isempty(masses)
            state_labels, signal_labels, signal_sizes = sl, zl, sizes
        else
            (sl == state_labels && zl == signal_labels) ||
                throw(ArgumentError("the networks do not agree on the states of " *
                                    "$(vcat(chosen, seen))"))
            diff_ok = (eltype(m) <: Rational && eltype(masses[1]) <: Rational) ?
                      (m == masses[1]) : (maximum(abs, m .- masses[1]) <= atol)
            diff_ok ||
                throw(ArgumentError("the distribution of $(vcat(chosen, seen)) under " *
                                    "$action differs from that under $(first(actions)): " *
                                    "a state, or something observed before the decision, " *
                                    "cannot depend on the action"))
        end
        push!(masses, m)
        push!(raw_values, v)
    end
    exact = all(x -> eltype(x) <: Rational, masses) &&
            all(x -> eltype(x) <: Rational, raw_values)
    T = exact ? Rational{BigInt} : Float64
    mass = Matrix{T}(masses[1])
    values = Matrix{T}[Matrix{T}(v) for v in raw_values]
    prior = vec(sum(mass; dims=2))
    Q = zeros(T, length(actions), length(prior))
    for (a, v) in enumerate(values), θ in eachindex(prior)
        iszero(prior[θ]) && continue
        Q[a, θ] = sum(view(v, θ, :)) / prior[θ]
    end
    # The state must carry everything the observation says about the outcome.
    tol_q = exact ? zero(T) : 1e-7 * max(1.0, maximum(abs, Q; init=0.0))
    mass_cut = exact ? zero(T) : atol
    for (a, v) in enumerate(values), θ in eachindex(prior)
        iszero(prior[θ]) && continue
        for z in axes(v, 2)
            mass[θ, z] > mass_cut || continue
            abs(v[θ, z] / mass[θ, z] - Q[a, θ]) <= tol_q ||
                throw(ArgumentError("given the state $chosen, the expected outcome still " *
                                    "depends on the observed $seen; add to `states` the " *
                                    "variables the outcome depends on"))
        end
    end
    problem = DecisionProblem(PayoffTable(actions, state_labels, Q),
                              FiniteBelief(state_labels, prior ./ sum(prior);
                                           atol=exact ? 0 : 1e-9); baseline)
    experiment = isempty(seen) ? nothing :
                 _experiment(mass, prior, signal_labels, state_labels)
    # One experiment for each observed variable alone: the joint mass summed over the
    # others.
    cube = reshape(mass, length(prior), signal_sizes...)
    alone = map(enumerate(seen)) do (k, x)
        others = Tuple(j + 1 for j in eachindex(seen) if j != k)
        part = reshape(sum(cube; dims=others), length(prior), signal_sizes[k])
        return _experiment(part, prior, _variable_states(first(models), x), state_labels)
    end
    experiments = NamedTuple{Tuple(seen)}(Tuple(alone))
    return (; problem, experiment, experiments, actions, states=chosen, observed=seen)
end

function DecisionAnalysis.normal_form(model::BN.BayesModel; decision, states, outcome,
                                      observed=(), kw...)
    management = Symbol(decision)
    (management in _names(states) || management in _names(observed)) &&
        throw(ArgumentError("the decision $management cannot be part of `states` or " *
                            "`observed`"))
    actions = _variable_states(model, management)
    models = NamedTuple{Tuple(actions)}(Tuple(BN.do_intervention(model, management => a)
                                              for a in actions))
    return merge(DecisionAnalysis.normal_form(models; states, outcome, observed, kw...),
                 (; decision=management))
end

function DecisionAnalysis.normal_form(hypotheses::NamedTuple{<:Any,
                                                             <:Tuple{NamedTuple,
                                                                     Vararg{NamedTuple}}};
                                      outcome, observed=(), prior=nothing, baseline=nothing,
                                      atol::Real=1e-9, max_states::Integer=1_000_000)
    names = collect(keys(hypotheses))
    actions = collect(keys(first(hypotheses)))
    seen = _names(observed)
    allunique(seen) || throw(ArgumentError("duplicate variable in `observed`: $seen"))
    Q = zeros(Float64, length(actions), length(names))
    obs_cols = Vector{Vector{Float64}}(undef, length(names))
    signal_labels = Symbol[]
    signal_sizes = ()
    first_obs_states = Vector{Symbol}[]
    for (h, models) in enumerate(values(hypotheses))
        collect(keys(models)) == actions ||
            throw(ArgumentError("hypothesis $(names[h]) has the actions " *
                                "$(collect(keys(models))), not $actions"))
        obs_dist = nothing
        for (a, model) in enumerate(values(models))
            model isa BN.BayesModel ||
                throw(ArgumentError("hypothesis $(names[h]), action $(actions[a]): " *
                                    "expected a BayesModel, got $(typeof(model))"))
            _, value, _, _, _ = _tables(model, Symbol[], Symbol[],
                                        _outcome_terms(model, outcome); max_states, atol)
            Q[a, h] = only(value)
            if !isempty(seen)
                obs_states = [_variable_states(model, x) for x in seen]
                if isempty(first_obs_states)
                    first_obs_states = obs_states
                    signal_sizes = Tuple(length(s) for s in obs_states)
                    signal_labels = _joint_labels(model, seen)
                else
                    obs_states == first_obs_states ||
                        throw(ArgumentError("the hypothesis networks do not agree on the " *
                                            "states of observed variables $seen"))
                end
                p_obs = vec(Float64.(_marginal_table(model, seen; max_states, atol)))
                if obs_dist === nothing
                    obs_dist = p_obs
                else
                    maximum(abs, p_obs .- obs_dist) <= atol ||
                        throw(ArgumentError("under hypothesis $(names[h]), the " *
                                            "distribution of observed $seen under action " *
                                            "$(actions[a]) differs from $(first(actions)): " *
                                            "observed variables cannot depend on the action"))
                end
            end
        end
        if !isempty(seen)
            obs_cols[h] = obs_dist ./ sum(obs_dist)
        end
    end
    belief = prior === nothing ? ScenarioSet(names) : FiniteBelief(names, collect(prior))
    problem = DecisionProblem(PayoffTable(actions, names, Q), belief; baseline)
    if isempty(seen)
        return (; problem, experiment=nothing, experiments=(;), actions,
                states=[:hypothesis], observed=Symbol[])
    end
    L = reduce(hcat, obs_cols)
    experiment = Experiment(L; signals=signal_labels, states=names)
    cube = reshape(L, signal_sizes..., length(names))
    alone = map(enumerate(seen)) do (k, x)
        others = Tuple(j for j in eachindex(seen) if j != k)
        part = isempty(others) ? L :
               reshape(sum(cube; dims=others), signal_sizes[k], length(names))
        return Experiment(part; signals=first_obs_states[k], states=names)
    end
    experiments = NamedTuple{Tuple(seen)}(Tuple(alone))
    return (; problem, experiment, experiments, actions, states=[:hypothesis],
            observed=seen)
end

function _state_assignments(vars::Vector{Symbol}, state_vecs::Vector{Vector{Symbol}})
    dims = Tuple(length(s) for s in state_vecs)
    out = Vector{Vector{Pair{Symbol,Symbol}}}(undef, prod(dims))
    for (idx, ci) in enumerate(CartesianIndices(dims))
        out[idx] = Pair{Symbol,Symbol}[vars[k] => state_vecs[k][ci[k]]
                                       for k in eachindex(vars)]
    end
    return out
end

function _extract_dbn_transition(m_a::BN.BayesModel, s0::Vector{Symbol},
                                 s1::Vector{Symbol}, state_vecs::Vector{Vector{Symbol}},
                                 prior::Vector{Float64}; max_states, atol)
    nθ = length(prior)
    joint = reshape(Float64.(_marginal_table(m_a, vcat(s0, s1); max_states, atol)), nθ, nθ)
    T_a = zeros(Float64, nθ, nθ)
    assignments = _state_assignments(s0, state_vecs)
    for θ in 1:nθ
        if prior[θ] > atol
            col = vec(joint[θ, :]) ./ prior[θ]
            T_a[:, θ] .= col ./ sum(col)
        else
            m_fixed = BN.do_intervention(m_a, assignments[θ])
            col = vec(Float64.(_marginal_table(m_fixed, s1; max_states, atol)))
            T_a[:, θ] .= col ./ sum(col)
        end
    end
    return T_a
end

function DecisionAnalysis.normal_form(dm::BN.DynamicBayesModel; decision, states, outcome,
                                      observed=(), signal_of::Symbol=:current,
                                      baseline=nothing, atol::Real=1e-9,
                                      max_states::Integer=1_000_000)
    BN.lags(dm) == 1 ||
        throw(ArgumentError("normal_form supports lag-1 DynamicBayesModel, got lags = " *
                            "$(BN.lags(dm))"))
    signal_of in (:current, :next) ||
        throw(ArgumentError("signal_of must be :current or :next, got :$signal_of"))
    management = Symbol(decision)
    chosen, seen = _names(states), _names(observed)
    isempty(chosen) && throw(ArgumentError("`states` names at least one variable"))
    (management in chosen || management in seen) &&
        throw(ArgumentError("the decision $management cannot be part of `states` or " *
                            "`observed`"))
    allunique(vcat(chosen, seen)) ||
        throw(ArgumentError("a variable is named twice in `states` and `observed`"))
    cur = BN.current_variables(dm)
    for x in vcat([management], chosen, seen)
        x in cur ||
            throw(ArgumentError("$x is not a current variable of the DynamicBayesModel " *
                                "(variables: $cur)"))
    end
    actions = _variable_states(dm.initial, management)
    u1 = BN.unroll(dm, 1)
    d0 = BN.variable_at(management, 0)
    d1 = BN.variable_at(management, 1)
    s0 = Symbol[BN.variable_at(x, 0) for x in chosen]
    s1 = Symbol[BN.variable_at(x, 1) for x in chosen]
    state_vecs = [_variable_states(dm.initial, x) for x in chosen]
    state_labels = _joint_labels(dm.initial, chosen)
    nθ = length(state_labels)
    assignments0 = _state_assignments(s0, state_vecs)
    assignments1 = _state_assignments(s1, state_vecs)
    terms0 = _outcome_terms(u1, _retime_outcome(outcome, 0))
    obs_t = signal_of === :current ? 0 : 1
    obs_vars = Symbol[BN.variable_at(x, obs_t) for x in seen]
    signal_labels = isempty(seen) ? Symbol[] : _joint_labels(dm.initial, seen)
    nz = length(signal_labels)

    prior = nothing
    Q = zeros(Float64, length(actions), nθ)
    trans_list = Matrix{Float64}[]
    exp_list = Experiment[]
    for (ai, a) in enumerate(actions)
        m_a = BN.do_intervention(u1, [d0 => a, d1 => first(actions)])
        # Slice-0 prior and expected outcome per state.
        mass0, val0, _, _, _ = _tables(m_a, s0,
                                       signal_of === :current ? obs_vars : Symbol[],
                                       terms0; max_states, atol)
        prior_a = vec(sum(mass0; dims=2))
        if prior === nothing
            prior = prior_a
        else
            maximum(abs, prior_a .- prior) <= atol ||
                throw(ArgumentError("the initial distribution of $chosen depends on the " *
                                    "action $a; states cannot be downstream of the decision"))
        end
        for θ in 1:nθ
            if prior[θ] > atol
                Q[ai, θ] = sum(view(val0, θ, :)) / prior[θ]
            else
                m_fixed = BN.do_intervention(m_a, assignments0[θ])
                _, vf, _, _, _ = _tables(m_fixed, Symbol[], Symbol[], terms0; max_states,
                                         atol)
                Q[ai, θ] = only(vf)
            end
        end
        # Check outcome conditional independence on current observation.
        if signal_of === :current && !isempty(seen)
            scale = max(1.0, maximum(abs, view(Q, ai, :); init=0.0))
            for θ in 1:nθ
                prior[θ] > atol || continue
                for z in 1:nz
                    mass0[θ, z] > atol || continue
                    abs(val0[θ, z] / mass0[θ, z] - Q[ai, θ]) <= 1e-7 * scale ||
                        throw(ArgumentError("given state $chosen and action $a, the " *
                                            "expected outcome still depends on observed " *
                                            "$seen; include those variables in `states`"))
                end
            end
        end
        # Transition matrix T_a[θ', θ] = P(S_1 = θ' | S_0 = θ, do(d_0 = a)).
        T_a = _extract_dbn_transition(m_a, s0, s1, state_vecs, prior; max_states, atol)
        if length(actions) >= 2
            m_alt = BN.do_intervention(u1, [d0 => a, d1 => actions[2]])
            T_alt = _extract_dbn_transition(m_alt, s0, s1, state_vecs, prior; max_states,
                                            atol)
            maximum(abs, T_a .- T_alt) <= atol ||
                throw(ArgumentError("the next-period state $chosen depends on the " *
                                    "next-period decision $management; transitions must " *
                                    "depend only on the current state and current action"))
        end
        push!(trans_list, T_a)
        if !isempty(seen)
            if signal_of === :current
                # Verify Markov sufficiency: P(S_1 | S_0, obs_0, do(a)) == T_a[:, θ].
                j_szo = reshape(Float64.(_marginal_table(m_a, vcat(s0, obs_vars, s1);
                                                         max_states, atol)), nθ, nz, nθ)
                for θ in 1:nθ, z in 1:nz
                    p_sz = mass0[θ, z]
                    p_sz > atol || continue
                    cond_s1 = vec(j_szo[θ, z, :]) ./ p_sz
                    maximum(abs, cond_s1 .- view(T_a, :, θ)) <= 1e-6 ||
                        throw(ArgumentError("under action $a, observed $seen carries " *
                                            "transition information about the next state " *
                                            "not captured by `states` $chosen; add those " *
                                            "variables to `states`"))
                end
                L_a = zeros(Float64, nz, nθ)
                for θ in 1:nθ
                    if prior[θ] > atol
                        col = vec(view(mass0, θ, :)) ./ prior[θ]
                        L_a[:, θ] .= col ./ sum(col)
                    else
                        m_fixed = BN.do_intervention(m_a, assignments0[θ])
                        col = vec(Float64.(_marginal_table(m_fixed, obs_vars; max_states,
                                                           atol)))
                        L_a[:, θ] .= col ./ sum(col)
                    end
                end
                push!(exp_list, Experiment(L_a; signals=signal_labels, states=state_labels))
            else
                # signal_of === :next: P(obs_1 = z | S_1 = θ', do(d_0 = a))
                j_so = reshape(Float64.(_marginal_table(m_a, vcat(s1, obs_vars); max_states,
                                                        atol)), nθ, nz)
                p_s1 = vec(sum(j_so; dims=2))
                L_a = zeros(Float64, nz, nθ)
                for θ in 1:nθ
                    if p_s1[θ] > atol
                        col = vec(view(j_so, θ, :)) ./ p_s1[θ]
                        L_a[:, θ] .= col ./ sum(col)
                    else
                        m_fixed = BN.do_intervention(m_a, assignments1[θ])
                        col = vec(Float64.(_marginal_table(m_fixed, obs_vars; max_states,
                                                           atol)))
                        L_a[:, θ] .= col ./ sum(col)
                    end
                end
                if length(actions) >= 2
                    m_alt = BN.do_intervention(u1, [d0 => a, d1 => actions[2]])
                    j_alt = reshape(Float64.(_marginal_table(m_alt, vcat(s1, obs_vars);
                                                             max_states, atol)), nθ, nz)
                    maximum(abs, j_so .- j_alt) <= atol ||
                        throw(ArgumentError("next-period observation $seen depends on the " *
                                            "next-period decision $management; cannot form " *
                                            "signal_of = :next experiment"))
                end
                push!(exp_list, Experiment(L_a; signals=signal_labels, states=state_labels))
            end
        end
    end
    problem = DecisionProblem(PayoffTable(actions, state_labels, Q),
                              FiniteBelief(state_labels, prior ./ sum(prior)); baseline)
    transitions = NamedTuple{Tuple(actions)}(Tuple(trans_list))
    experiments = isempty(seen) ? nothing : NamedTuple{Tuple(actions)}(Tuple(exp_list))
    return (; problem, experiments, transitions, signal_of, decision=management, actions,
            states=chosen, observed=seen)
end

function DecisionAnalysis.normal_form(models::Union{AbstractDict,AbstractVector{<:Pair}};
                                      kw...)
    nt = _to_named_tuple(models)
    first_val = first(values(nt))
    if first_val isa BN.BayesModel
        all(v -> v isa BN.BayesModel, values(nt)) ||
            throw(ArgumentError("expected a collection of BayesModel values"))
        return DecisionAnalysis.normal_form(nt; kw...)
    end
    nested = NamedTuple{keys(nt)}(Tuple(_to_named_tuple(v) for v in values(nt)))
    return DecisionAnalysis.normal_form(nested; kw...)
end

end # module
