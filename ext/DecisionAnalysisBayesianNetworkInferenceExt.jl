"""
    DecisionAnalysisBayesianNetworkInferenceExt

Variable-elimination backend for `DecisionAnalysisBayesianNetworksExt`: when
`BayesianNetworkInference` is loaded alongside `BayesianNetworks`, `normal_form`
marginalises via `BayesianNetworkInference.VariableElimination` instead of enumerating
the full joint table.
"""
module DecisionAnalysisBayesianNetworkInferenceExt

using DecisionAnalysis: DecisionAnalysis
using BayesianNetworks: BayesModel
using BayesianNetworkInference: BayesianNetworkInference

const BNI = BayesianNetworkInference

function _infer_marginal(model::BayesModel, query::Vector{Symbol}; atol::Real)
    isempty(query) && return fill(1.0)
    fac, _ = BNI.infer(model, query; backend=BNI.VariableElimination(), atol)
    if fac.vars == query
        return fac.table
    end
    pos = Dict{Symbol,Int}(x => i for (i, x) in enumerate(fac.vars))
    perm = Tuple(pos[x] for x in query)
    return permutedims(fac.table, perm)
end

end # module
