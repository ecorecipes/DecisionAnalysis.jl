"""
    DecisionAnalysisFiniteKernelsExt

Bidirectional bridge to FiniteKernels.jl (and `ResilienceCore.MarkovKernel`):

- `FiniteBelief(k::FiniteKernel)` and `Experiment(k::FiniteKernel)`;
- `FiniteKernel(b::FiniteBelief)` and `FiniteKernel(e::Experiment)` (exact-preserving);
- `FiniteKernel` and `ResilienceCore.MarkovKernel` transition kernels accepted directly
  as `transitions` in [`adaptive_management`](@ref).

Loaded when both DecisionAnalysis and FiniteKernels are.
"""
module DecisionAnalysisFiniteKernelsExt

using DecisionAnalysis: DecisionAnalysis, DecisionProblem, Experiment, FiniteBelief,
                        LabelMismatchError, _default_atol
using FiniteKernels: FiniteKernels, FiniteAxis, FiniteKernel, FiniteSpace, ResilienceCore,
                     axis_names, factors, joint_states, kernel_matrix, state

# A one-axis space keeps its labels; the joint states of a several-axis space are
# labelled by joining the axis labels with `_`, in `joint_states` order.
function _space_labels(X::FiniteSpace)
    ndims(X) == 1 && return copy(only(factors(X)).labels)
    labels = [Symbol(join(s, "_")) for s in joint_states(X)]
    allunique(labels) && return labels
    return [Symbol(join(s, "|")) for s in joint_states(X)]
end

function DecisionAnalysis.FiniteBelief(k::FiniteKernel; kw...)
    ndims(k.dom) == 0 ||
        throw(ArgumentError("a FiniteBelief needs a state (a kernel with an empty " *
                            "domain), got a kernel with domain axes $(axis_names(k.dom))"))
    return FiniteBelief(_space_labels(k.codom), vec(k.table); kw...)
end

function DecisionAnalysis.Experiment(k::FiniteKernel; cost::Real=0, kw...)
    (ndims(k.dom) >= 1 && ndims(k.codom) >= 1) ||
        throw(ArgumentError("an experiment needs a kernel from a state space to a " *
                            "signal space, both non-empty"))
    return Experiment(kernel_matrix(k); signals=_space_labels(k.codom),
                      states=_space_labels(k.dom), cost, kw...)
end

function FiniteKernels.FiniteKernel(b::FiniteBelief; name::Symbol=:state,
                                    atol::Real=_default_atol(eltype(b.probabilities)))
    X = FiniteSpace(FiniteAxis(name, b.states))
    return state(X, b.probabilities; atol)
end

function FiniteKernels.FiniteKernel(e::Experiment; state_name::Symbol=:state,
                                    signal_name::Symbol=:signal,
                                    atol::Real=_default_atol(eltype(e.likelihood)))
    st = e.states === nothing ? Symbol[Symbol(:s, i) for i in axes(e.likelihood, 2)] :
         e.states
    dom = FiniteSpace(FiniteAxis(state_name, st))
    codom = FiniteSpace(FiniteAxis(signal_name, e.signals))
    return FiniteKernel(dom, codom, e.likelihood; atol)
end

DecisionAnalysis._is_single_transition(::FiniteKernel) = true
DecisionAnalysis._is_single_transition(::ResilienceCore.MarkovKernel) = true

function DecisionAnalysis._transition_matrix(problem::DecisionProblem, k::FiniteKernel,
                                             who::AbstractString)
    (ndims(k.dom) >= 1 && ndims(k.codom) >= 1) ||
        throw(ArgumentError("$who must be a kernel from states to states, both non-empty"))
    dom_labels = _space_labels(k.dom)
    codom_labels = _space_labels(k.codom)
    expected = problem.payoff.states
    dom_labels == expected ||
        throw(LabelMismatchError("domain states of $who", expected, dom_labels))
    codom_labels == expected ||
        throw(LabelMismatchError("codomain states of $who", expected, codom_labels))
    return kernel_matrix(k)
end

function DecisionAnalysis._transition_matrix(::DecisionProblem,
                                             mk::ResilienceCore.MarkovKernel,
                                             ::AbstractString)
    # ResilienceCore.MarkovKernel stores row-stochastic P[present, next];
    # adaptive_management expects column-stochastic T[next, present].
    return permutedims(mk.P)
end

end # module
