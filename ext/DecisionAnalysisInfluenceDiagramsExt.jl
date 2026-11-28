"""
The bridge to InfluenceDiagrams.jl: `normal_form` turns a single-decision influence
diagram into a `DecisionProblem` (and the decision's information into an `Experiment`),
so that the structural values of information of the diagram can be read, and checked,
as `evpi` and `evsi` on a payoff table. Loaded when InfluenceDiagrams is.
"""
module DecisionAnalysisInfluenceDiagramsExt

using DecisionAnalysis: DecisionAnalysis, DecisionProblem, Experiment, FiniteBelief,
                        PayoffTable
using InfluenceDiagrams: InfluenceDiagrams

const ID = InfluenceDiagrams

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

function DecisionAnalysis.normal_form(m::ID.InfluenceDiagramModel; states=nothing,
                                      baseline=nothing, atol::Real=1e-9)
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

    mass = nothing      # mass[θ, z] = P(state = θ, signal = z), the same for every action
    weighted = Matrix{Float64}[]          # per action: Σ P(x) U(x) over the cells
    state_labels = Symbol[]
    signal_labels = Symbol[]
    for a in actions
        J = ID.joint_table(ID.instantiate(ID.fix_decision(m, decision => a)))
        position = Dict(x => i for (i, x) in enumerate(J.variables))
        s_axes = Int[position[x] for x in chosen]
        z_axes = Int[position[x] for x in observed]
        s_sizes = Int[length(J.states[i]) for i in s_axes]
        z_sizes = Int[length(J.states[i]) for i in z_axes]
        s_strides = _strides(s_sizes)
        z_strides = _strides(z_sizes)
        Z = zeros(prod(s_sizes), prod(z_sizes))
        N = zeros(size(Z))
        assignment = Dict{Symbol,Symbol}()
        sizehint!(assignment, length(J.variables))
        for ci in CartesianIndices(J.table)
            p = J.table[ci]
            iszero(p) && continue
            @inbounds for i in eachindex(J.variables)
                assignment[J.variables[i]] = J.states[i][ci[i]]
            end
            θ = _lin_idx(ci, s_axes, s_strides)
            z = _lin_idx(ci, z_axes, z_strides)
            Z[θ, z] += p
            N[θ, z] += p * ID.total_utility(m, assignment)
        end
        if mass === nothing
            mass = Z
            state_labels = _unique_joined(Tuple(J.states[i][c[k]]
                                                for (k, i) in enumerate(s_axes))
                                          for c in
                                              vec(collect(CartesianIndices(Tuple(s_sizes)))))
            signal_labels = _unique_joined(Tuple(J.states[i][c[k]]
                                                 for (k, i) in enumerate(z_axes))
                                           for c in
                                               vec(collect(CartesianIndices(Tuple(z_sizes)))))
        else
            maximum(abs, Z .- mass) <= atol ||
                throw(ArgumentError("the distribution of the state and the " *
                                    "information changes with the action, so they are " *
                                    "not upstream of the decision"))
        end
        push!(weighted, N)
    end
    prior = vec(sum(mass; dims=2))
    Q = zeros(length(actions), length(prior))
    for (a, N) in enumerate(weighted), θ in eachindex(prior)
        iszero(prior[θ]) && continue
        Q[a, θ] = sum(view(N, θ, :)) / prior[θ]
    end
    # The state must carry everything the information says about the utility:
    # E[U | action, state, signal] = E[U | action, state] wherever it is defined.
    scale = max(1.0, maximum(abs, Q))
    for (a, N) in enumerate(weighted), θ in eachindex(prior)
        iszero(prior[θ]) && continue
        for z in axes(N, 2)
            mass[θ, z] > atol || continue
            abs(N[θ, z] / mass[θ, z] - Q[a, θ]) <= 1e-7 * scale ||
                throw(ArgumentError("given the state $chosen, the expected utility " *
                                    "still depends on the information $observed; add " *
                                    "the variables the utility depends on to `states`"))
        end
    end
    payoff = PayoffTable(actions, state_labels, Q)
    problem = DecisionProblem(payoff, FiniteBelief(state_labels, prior ./ sum(prior));
                              baseline)
    experiment = nothing
    if !isempty(observed)
        L = zeros(size(mass, 2), size(mass, 1))
        for θ in eachindex(prior)
            if iszero(prior[θ])
                L[1, θ] = 1.0             # a state of probability zero: any column will do
            else
                L[:, θ] .= view(mass, θ, :) ./ prior[θ]
            end
        end
        experiment = Experiment(L; signals=signal_labels, states=state_labels)
    end
    return (; problem, experiment, decision, action, states=chosen, information,
            observed)
end

end # module
