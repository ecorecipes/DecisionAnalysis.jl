"""
The bridge to BayesianNetworks.jl. An influence diagram becomes a Bayesian network once
its decision is made, so a decision problem is also a set of networks, one for each
action. `normal_form` reads the payoff table, the belief and the experiments off such a
set: from one network with a management node, from a named collection of networks, or
from the networks of several hypotheses. Loaded when BayesianNetworks is.
"""
module DecisionAnalysisBayesianNetworksExt

using DecisionAnalysis: DecisionAnalysis, DecisionProblem, Experiment, FiniteBelief,
                        PayoffTable, ScenarioSet
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
        return variables, Float64[given[s] for s in labels]
    end
    values isa AbstractArray{<:Real} ||
        throw(ArgumentError("the values of an outcome term are an array of numbers, or a " *
                            "named tuple by state for one variable; got $(typeof(values))"))
    size(values) == sizes ||
        throw(ArgumentError("the outcome of $variables needs an array of size $sizes " *
                            "(one axis per variable, in the order of its states), got " *
                            "$(size(values))"))
    return variables, Float64.(values)
end

_outcome_terms(model, outcome::Pair) = [_outcome_term(model, outcome)]
_outcome_terms(model, outcome) = [_outcome_term(model, term) for term in outcome]

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

# From one network: mass[θ, z] = P(state = θ, observed = z) and
# value[θ, z] = Σ P(state = θ, observed = z, x) × outcome(x), with the state and signal
# labels. θ and z run over the joint states of `states` and `observed`, the first
# variable fastest.
function _tables(model::BN.BayesModel, states::Vector{Symbol}, observed::Vector{Symbol},
                 terms; max_states, atol)
    query = unique!(vcat(states, observed, (first(t) for t in terms)...))
    labels = [_variable_states(model, x) for x in query]
    joint = BN.marginal(model, query; max_states, atol).table
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
    mass = zeros(nθ, nz)
    value = zeros(nθ, nz)
    for ci in CartesianIndices(dims)
        p = joint[ci]
        iszero(p) && continue
        θ = _lin_idx(ci, s_axes, s_strides)
        z = _lin_idx(ci, z_axes, z_strides)
        mass[θ, z] += p
        acc = 0.0
        for (t_axes, t_strides, table) in term_specs
            acc += table[_lin_idx(ci, t_axes, t_strides)]
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
function _experiment(mass::AbstractMatrix, prior, signals, states)
    L = zeros(size(mass, 2), size(mass, 1))
    for θ in eachindex(prior)
        if iszero(prior[θ])
            L[1, θ] = 1.0             # a state of probability zero: any column will do
        else
            L[:, θ] .= view(mass, θ, :) ./ prior[θ]
        end
    end
    return Experiment(L; signals, states)
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
    mass = nothing
    values = Matrix{Float64}[]
    state_labels, signal_labels, signal_sizes = Symbol[], Symbol[], ()
    for (action, model) in pairs(models)
        m, v, sl, zl, sizes = _tables(model, chosen, seen, _outcome_terms(model, outcome);
                                      max_states, atol)
        if mass === nothing
            mass, state_labels, signal_labels, signal_sizes = m, sl, zl, sizes
        else
            (sl == state_labels && zl == signal_labels) ||
                throw(ArgumentError("the networks do not agree on the states of " *
                                    "$(vcat(chosen, seen))"))
            maximum(abs, m .- mass) <= atol ||
                throw(ArgumentError("the distribution of $(vcat(chosen, seen)) under " *
                                    "$action differs from that under $(first(actions)): " *
                                    "a state, or something observed before the decision, " *
                                    "cannot depend on the action"))
        end
        push!(values, v)
    end
    prior = vec(sum(mass; dims=2))
    Q = zeros(length(actions), length(prior))
    for (a, v) in enumerate(values), θ in eachindex(prior)
        iszero(prior[θ]) && continue
        Q[a, θ] = sum(view(v, θ, :)) / prior[θ]
    end
    # The state must carry everything the observation says about the outcome.
    scale = max(1.0, maximum(abs, Q))
    for (a, v) in enumerate(values), θ in eachindex(prior)
        iszero(prior[θ]) && continue
        for z in axes(v, 2)
            mass[θ, z] > atol || continue
            abs(v[θ, z] / mass[θ, z] - Q[a, θ]) <= 1e-7 * scale ||
                throw(ArgumentError("given the state $chosen, the expected outcome still " *
                                    "depends on the observed $seen; add to `states` the " *
                                    "variables the outcome depends on"))
        end
    end
    problem = DecisionProblem(PayoffTable(actions, state_labels, Q),
                              FiniteBelief(state_labels, prior ./ sum(prior)); baseline)
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
                                      outcome, prior=nothing, baseline=nothing,
                                      atol::Real=1e-9, max_states::Integer=1_000_000)
    names = collect(keys(hypotheses))
    actions = collect(keys(first(hypotheses)))
    Q = zeros(length(actions), length(names))
    for (h, models) in enumerate(values(hypotheses))
        collect(keys(models)) == actions ||
            throw(ArgumentError("hypothesis $(names[h]) has the actions " *
                                "$(collect(keys(models))), not $actions"))
        for (a, model) in enumerate(values(models))
            model isa BN.BayesModel ||
                throw(ArgumentError("hypothesis $(names[h]), action $(actions[a]): " *
                                    "expected a BayesModel, got $(typeof(model))"))
            _, value, _, _, _ = _tables(model, Symbol[], Symbol[],
                                        _outcome_terms(model, outcome); max_states, atol)
            Q[a, h] = only(value)
        end
    end
    belief = prior === nothing ? ScenarioSet(names) : FiniteBelief(names, collect(prior))
    problem = DecisionProblem(PayoffTable(actions, names, Q), belief; baseline)
    return (; problem, experiment=nothing, experiments=(;), actions, states=[:hypothesis],
            observed=Symbol[])
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
