"""
    DecisionAnalysisFiniteKernelsExt

Beliefs and experiments from the objects of FiniteKernels.jl: a `FiniteBelief` from a
state and an `Experiment` from a kernel. Loaded when both DecisionAnalysis and
FiniteKernels are.
"""
module DecisionAnalysisFiniteKernelsExt

using DecisionAnalysis: DecisionAnalysis, Experiment, FiniteBelief
using FiniteKernels: FiniteKernel, FiniteSpace, axis_names, factors, joint_states,
                     kernel_matrix

# A one-axis space keeps its labels; the joint states of a several-axis space are
# labelled by joining the axis labels with `_`, in `joint_states` order.
function _space_labels(X::FiniteSpace)
    return ndims(X) == 1 ? copy(only(factors(X)).labels) :
           [Symbol(join(s, "_")) for s in joint_states(X)]
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

end # module
