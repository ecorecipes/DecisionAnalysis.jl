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

_joined(labels) = Symbol(join(labels, "_"))

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
        s_axes = [position[x] for x in chosen]
        z_axes = [position[x] for x in observed]
        s_sizes = [length(J.states[i]) for i in s_axes]
        z_sizes = [length(J.states[i]) for i in z_axes]
        s_index = LinearIndices(Tuple(s_sizes))
        z_index = LinearIndices(Tuple(z_sizes))
        Z = zeros(prod(s_sizes), prod(z_sizes))
        N = zeros(size(Z))
        for ci in CartesianIndices(J.table)
            p = J.table[ci]
            iszero(p) && continue
            assignment = Dict{Symbol,Symbol}(x => J.states[i][ci[i]]
                                             for (i, x) in enumerate(J.variables))
            θ = s_index[(ci[i] for i in s_axes)...]
            z = z_index[(ci[i] for i in z_axes)...]
            Z[θ, z] += p
            N[θ, z] += p * ID.total_utility(m, assignment)
        end
        if mass === nothing
            mass = Z
            state_labels = [_joined(J.states[i][c[k]] for (k, i) in enumerate(s_axes))
                            for c in vec(collect(CartesianIndices(Tuple(s_sizes))))]
            signal_labels = [_joined(J.states[i][c[k]] for (k, i) in enumerate(z_axes))
                             for c in vec(collect(CartesianIndices(Tuple(z_sizes))))]
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
        # The state must carry everything the information says about the utility:
        # E[U | action, state, signal] = E[U | action, state] wherever it is defined.
        scale = max(1.0, maximum(abs, Q))
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
