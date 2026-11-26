"""
    DecisionAnalysisGAMExt

The regression backend of `GAMRegression`: fitted conditional means from a
generalised additive model of GAM.jl, one model per action. Loaded when both
DecisionAnalysis and GAM are.
"""
module DecisionAnalysisGAMExt

using DecisionAnalysis: DecisionAnalysis, GAMRegression, _is_numeric
using GAM: GAM

# The formula `y ~ ...` and the data columns, with safe names. The formula is built
# from terms, not parsed from text: nothing is evaluated at run time, so the same
# formula is not compiled again for every action and every bootstrap resample.
function _formula(method::GAMRegression, columns::NamedTuple)
    data = Pair{Symbol,Any}[]
    smooth = Symbol[]
    basis = Int[]
    terms = GAM.StatsModels.AbstractTerm[]
    for (j, column) in enumerate(columns)
        name = Symbol(:x, j)
        distinct = length(unique(column))
        if _is_numeric(column) && distinct > 3
            push!(data, name => Float64.(column))
            push!(smooth, name)
            push!(basis, min(method.k, distinct - 1))
        else
            push!(data, name => string.(column))
            distinct > 1 && push!(terms, GAM.term(name))
        end
    end
    if method.tensor && length(smooth) > 1
        k = max(3, min(minimum(basis), 5))
        push!(terms, GAM.SmoothTerm(GAM.te(smooth...; k)))
    else
        for (name, k) in zip(smooth, basis)
            push!(terms, GAM.SmoothTerm(GAM.s(name; k)))
        end
    end
    isempty(terms) && push!(terms, GAM.term(1))
    return (GAM.term(:y) ~ reduce(+, terms)), data
end

function DecisionAnalysis.fitted_conditional_means(method::GAMRegression,
                                                   columns::NamedTuple, Q::AbstractMatrix,
                                                   w::AbstractVector)
    formula, data = _formula(method, columns)
    weights = Float64.(w) .* length(w)
    G = zeros(Float64, size(Q))
    # The first action is fitted as it is and every other one through its difference
    # from the first, as voi and SAVI do. Something that does not vary is its own
    # conditional mean: there is nothing to fit, and a fit would not converge.
    function fitted(y)
        allequal(y) && return y
        return GAM.gam(formula, (; y, data...); weights, method=method.smoothing).fitted_values
    end
    first_row = Float64.(view(Q, 1, :))
    G[1, :] .= fitted(first_row)
    for a in 2:size(Q, 1)
        G[a, :] .= view(G, 1, :) .+ fitted(Float64.(view(Q, a, :)) .- first_row)
    end
    return G
end

end # module
