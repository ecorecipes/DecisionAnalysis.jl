"""
    DecisionAnalysisBayesianNetworkInferenceExt

Variable-elimination backend for `DecisionAnalysisBayesianNetworksExt`: when
`BayesianNetworkInference` is loaded alongside `BayesianNetworks`, `normal_form`
marginalises via `BayesianNetworkInference.VariableElimination` instead of enumerating
the full joint table.
"""
module DecisionAnalysisBayesianNetworkInferenceExt

using DecisionAnalysis: DecisionAnalysis, _is_exact
using BayesianNetworks: BayesModel, kernels
using BayesianNetworkInference: BayesianNetworkInference

const BNI = BayesianNetworkInference

function _all_exact_kernels(model::BayesModel)
    ks = kernels(model)
    return !isempty(ks) && all(k -> _is_exact(eltype(k)), values(ks))
end

function _infer_marginal(model::BayesModel, query::Vector{Symbol}; atol::Real)
    exact = _all_exact_kernels(model) &&
            hasmethod(BNI.compile, Tuple{BayesModel}, (:atol, :exact))
    if isempty(query)
        return exact ? fill(one(Rational{BigInt})) : fill(1.0)
    end
    fac, _ = exact ?
             BNI.infer(model, query; backend=BNI.VariableElimination(), atol, exact=true) :
             BNI.infer(model, query; backend=BNI.VariableElimination(), atol)
    if fac.vars == query
        return fac.table
    end
    pos = Dict{Symbol,Int}(x => i for (i, x) in enumerate(fac.vars))
    perm = Tuple(pos[x] for x in query)
    return permutedims(fac.table, perm)
end

end # module
