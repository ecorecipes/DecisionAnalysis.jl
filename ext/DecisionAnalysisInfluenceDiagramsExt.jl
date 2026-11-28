"""
The bridge to InfluenceDiagrams.jl: `normal_form` turns a single-decision influence
diagram into a `DecisionProblem` (and the decision's information into an `Experiment`),
using variable elimination (`BayesianNetworkInference.VariableElimination`) rather than
enumerating the full joint table. Preserves exact `Rational{BigInt}` arithmetic when all
bound chance kernels and utilities are exact. Loaded when InfluenceDiagrams is.
"""
module DecisionAnalysisInfluenceDiagramsExt

using DecisionAnalysis: DecisionAnalysis, DecisionProblem, Experiment, FiniteBelief,
                        PayoffTable, _is_exact
using InfluenceDiagrams: InfluenceDiagrams

const ID = InfluenceDiagrams
const BNI = InfluenceDiagrams.BayesianNetworkInference

function _unique_joined(tuples)
    labels = Symbol[Symbol(join(t, "_")) for t in tuples]
    allunique(labels) && return labels
    return Symbol[Symbol(join(t, "|")) for t in tuples]
end

function _strides(sizes::Vector{Int})
    st = Vector{Int}(undef, length(sizes))
    acc = 1
    @inbounds for k in eachindex(sizes)
        st[k] = acc
        acc *= sizes[k]
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

function _variable_names(id, ids)
    return Symbol[ID.variable_name(id, v) for v in ids]
end

function _extract_mass(::Type{T}, fac, chosen::Vector{Symbol},
                       observed::Vector{Symbol}) where {T<:Real}
    pos = Dict{Symbol,Int}(x => i for (i, x) in enumerate(fac.vars))
    s_axes = Int[pos[x] for x in chosen]
    z_axes = Int[pos[x] for x in observed]
    s_sizes = Int[length(fac.axes[i].labels) for i in s_axes]
    z_sizes = Int[length(fac.axes[i].labels) for i in z_axes]
    s_strides = _strides(s_sizes)
    z_strides = _strides(z_sizes)
    Z = zeros(T, prod(s_sizes; init=1), prod(z_sizes; init=1))
    @inbounds for ci in CartesianIndices(fac.table)
        p = T(fac.table[ci])
        iszero(p) && continue
        θ = _lin_idx(ci, s_axes, s_strides)
        z = _lin_idx(ci, z_axes, z_strides)
        Z[θ, z] += p
    end
    state_labels = _unique_joined(Tuple(fac.axes[i].labels[c[k]]
                                        for (k, i) in enumerate(s_axes))
                                  for c in vec(collect(CartesianIndices(Tuple(s_sizes)))))
    signal_labels = _unique_joined(Tuple(fac.axes[i].labels[c[k]]
                                         for (k, i) in enumerate(z_axes))
                                   for c in vec(collect(CartesianIndices(Tuple(z_sizes)))))
    return Z, state_labels, signal_labels
end

function _accumulate_utility!(N_u::Matrix{T}, N_total::Matrix{T}, fac,
                              chosen::Vector{Symbol}, observed::Vector{Symbol},
                              sc::Vector{Symbol},
                              u_tab::AbstractArray{<:Real}) where {T<:Real}
    pos = Dict{Symbol,Int}(x => i for (i, x) in enumerate(fac.vars))
    s_axes = Int[pos[x] for x in chosen]
    z_axes = Int[pos[x] for x in observed]
    u_axes = Int[pos[x] for x in sc]
    s_sizes = Int[length(fac.axes[i].labels) for i in s_axes]
    z_sizes = Int[length(fac.axes[i].labels) for i in z_axes]
    u_sizes = Int[size(u_tab, k) for k in eachindex(sc)]
    s_strides = _strides(s_sizes)
    z_strides = _strides(z_sizes)
    u_strides = _strides(u_sizes)
    @inbounds for ci in CartesianIndices(fac.table)
        p = T(fac.table[ci])
        iszero(p) && continue
        θ = _lin_idx(ci, s_axes, s_strides)
        z = _lin_idx(ci, z_axes, z_strides)
        u_val = T(u_tab[_lin_idx(ci, u_axes, u_strides)])
        contrib = p * u_val
        N_u[θ, z] += contrib
        N_total[θ, z] += contrib
    end
    return nothing
end

function _check_state_sufficiency(weighted::Vector{Matrix{T}}, Q::Matrix{T},
                                  mass::Matrix{T}, prior::Vector{T},
                                  chosen::Vector{Symbol}, observed::Vector{Symbol},
                                  atol::Real) where {T<:Real}
    exact = T <: Rational
    tol_q = exact ? zero(T) : 1e-7 * max(1.0, maximum(abs, Q; init=0.0))
    mass_cut = exact ? zero(T) : atol
    for (a, N) in enumerate(weighted), θ in eachindex(prior)
        iszero(prior[θ]) && continue
        for z in axes(N, 2)
            mass[θ, z] > mass_cut || continue
            abs(N[θ, z] / mass[θ, z] - Q[a, θ]) <= tol_q ||
                throw(ArgumentError("given the state $chosen, the expected utility " *
                                    "still depends on the information $observed; add " *
                                    "the variables the utility depends on to `states`"))
        end
    end
    return nothing
end

function _all_exact_diagram(m::ID.InfluenceDiagramModel, u_names::Vector{Symbol})
    hasmethod(BNI.compile, Tuple{ID.BayesModel}, (:atol, :exact)) || return false
    ks = ID.kernels(m)
    !isempty(ks) && all(k -> _is_exact(eltype(k)), values(ks)) || return false
    for uname in u_names
        u = ID.utility(m, uname)
        u isa ID.TabularUtility && _is_exact(eltype(u.table)) || return false
    end
    return true
end

function DecisionAnalysis.normal_form(m::ID.InfluenceDiagramModel; states=nothing,
                                      baseline=nothing, multi_objective::Bool=false,
                                      atol::Real=1e-9)
    id = ID.syntax(m)
    decisions = ID.decision_names(id)
    length(decisions) == 1 ||
        throw(ArgumentError("normal_form needs an influence diagram with exactly one " *
                            "decision; this one has $(length(decisions)): $decisions"))
    isempty(ID.evidence(m)) ||
        throw(ArgumentError("the model carries evidence; condition the diagram first " *
                            "or remove the evidence"))
    decision = only(decisions)
    action = only(ID.action_names(id))
    actions = ID.states(id, action)
    information = ID.information_names(id, decision)
    available = vcat(information,
                     _variable_names(id, ID.admissible_information(id, decision)))
    # The state: by default every chance variable that could be observed before the
    # decision, in the order of the diagram.
    chosen = states === nothing ? [x for x in ID.chance_names(id) if x in available] :
             Symbol[Symbol(x)
                    for x in (states isa Union{Symbol,AbstractString} ?
                              (states,) : states)]
    isempty(chosen) &&
        throw(ArgumentError("the diagram has no chance variable upstream of the " *
                            "decision to serve as the state"))
    for x in chosen
        x in available ||
            throw(ArgumentError("$x cannot be part of the state: it is not a chance " *
                                "variable that could be observed before the decision " *
                                "(available: $available)"))
    end
    # Signals: the information variables that are not themselves part of the state.
    observed = [x for x in information if !(x in chosen)]
    base_vars = vcat(chosen, observed)
    u_names = Symbol[ID.utility_names(id)...]
    nu = length(u_names)
    exact = _all_exact_diagram(m, u_names)
    T = exact ? Rational{BigInt} : Float64

    mass = nothing      # mass[θ, z] = P(state = θ, signal = z), the same for every action
    weighted = Matrix{T}[]
    weighted_per_u = [Matrix{T}[] for _ in 1:nu]
    state_labels = Symbol[]
    signal_labels = Symbol[]
    backend = BNI.VariableElimination()
    for a in actions
        bm = ID.instantiate(ID.fix_decision(m, decision => a); atol)
        fg = exact ? BNI.compile(bm; atol, exact=true) : BNI.compile(bm; atol)
        base_fac, _ = BNI.infer(fg, base_vars; backend)
        Z, sl, zl = _extract_mass(T, base_fac, chosen, observed)
        if mass === nothing
            mass = Z
            state_labels = sl
            signal_labels = zl
        else
            diff_ok = exact ? (Z == mass) : (maximum(abs, Z .- mass) <= atol)
            diff_ok ||
                throw(ArgumentError("the distribution of the state and the " *
                                    "information changes with the action, so they are " *
                                    "not upstream of the decision"))
        end
        N_total = zeros(T, size(Z))
        for (ui, uname) in enumerate(u_names)
            u = ID.utility(m, uname)
            sc = ID.scope(u)
            u_axes = ID.FiniteAxis[only(ID.factors(ID.space(bm, x))) for x in sc]
            u_tab = ID.utility_table(u, u_axes)
            all(isfinite, u_tab) ||
                throw(ArgumentError("utility $uname has non-finite entries"))
            N_u = zeros(T, size(Z))
            if isempty(sc)
                val = T(u_tab[])
                N_u .= val .* Z
                N_total .+= N_u
            else
                qvars = unique(vcat(base_vars, sc))
                fac = length(qvars) == length(base_vars) ? base_fac :
                      first(BNI.infer(fg, qvars; backend))
                _accumulate_utility!(N_u, N_total, fac, chosen, observed, sc, u_tab)
            end
            push!(weighted_per_u[ui], N_u)
        end
        push!(weighted, N_total)
    end
    prior = vec(sum(mass; dims=2))
    Q = zeros(T, length(actions), length(prior))
    for (a, N) in enumerate(weighted), θ in eachindex(prior)
        iszero(prior[θ]) && continue
        Q[a, θ] = sum(view(N, θ, :)) / prior[θ]
    end
    _check_state_sufficiency(weighted, Q, mass, prior, chosen, observed, atol)
    payoff = PayoffTable(actions, state_labels, Q)
    belief = FiniteBelief(state_labels, prior ./ sum(prior);
                          atol=exact ? 0 : 1e-9)
    problem = DecisionProblem(payoff, belief; baseline)
    experiment = nothing
    if !isempty(observed)
        L = zeros(T, size(mass, 2), size(mass, 1))
        for θ in eachindex(prior)
            if iszero(prior[θ])
                L[1, θ] = one(T)             # a state of probability zero: any column will do
            else
                L[:, θ] .= view(mass, θ, :) ./ prior[θ]
            end
        end
        experiment = Experiment(L; signals=signal_labels, states=state_labels,
                                atol=exact ? 0 : 1e-9)
    end
    if multi_objective
        obj_tables = map(1:nu) do ui
            Qu = zeros(T, length(actions), length(prior))
            for (a, Nu) in enumerate(weighted_per_u[ui]), θ in eachindex(prior)
                iszero(prior[θ]) && continue
                Qu[a, θ] = sum(view(Nu, θ, :)) / prior[θ]
            end
            _check_state_sufficiency(weighted_per_u[ui], Qu, mass, prior, chosen, observed,
                                     atol)
            return PayoffTable(actions, state_labels, Qu)
        end
        objectives = NamedTuple{Tuple(u_names)}(Tuple(obj_tables))
        return (; problem, objectives, experiment, decision, action, states=chosen,
                information, observed)
    end
    return (; problem, experiment, decision, action, states=chosen, information,
            observed)
end

end # module
