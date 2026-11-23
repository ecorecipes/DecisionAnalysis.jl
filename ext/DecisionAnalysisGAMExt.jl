"""
    DecisionAnalysisGAMExt

The regression backend of `GAMRegression`: fitted conditional means from a
generalised additive model of GAM.jl, one model per action. Loaded when both
DecisionAnalysis and GAM are.
"""
module DecisionAnalysisGAMExt

using DecisionAnalysis: DecisionAnalysis, GAMRegression
using GAM: GAM

_numeric(column::AbstractVector) = eltype(column) <: Real && !(eltype(column) <: Bool)

# The right-hand side of the formula and the data columns, with safe names.
function _terms(method::GAMRegression, columns::NamedTuple)
    data = Pair{Symbol,Any}[]
    smooth = Symbol[]
    basis = Int[]
    terms = String[]
    for (j, column) in enumerate(columns)
        name = Symbol(:x, j)
        distinct = length(unique(column))
        if _numeric(column) && distinct > 3
            push!(data, name => Float64.(column))
            push!(smooth, name)
            push!(basis, min(method.k, distinct - 1))
        else
            push!(data, name => string.(column))
            distinct > 1 && push!(terms, String(name))
        end
    end
    if method.tensor && length(smooth) > 1
        k = max(3, min(minimum(basis), 5))
        push!(terms, "te(" * join(smooth, ", ") * ", k = $k)")
    else
        for (name, k) in zip(smooth, basis)
            push!(terms, "s($name, k = $k)")
        end
    end
    return isempty(terms) ? "1" : join(terms, " + "), data
end

function DecisionAnalysis.fitted_conditional_means(method::GAMRegression,
                                                   columns::NamedTuple, Q::AbstractMatrix,
                                                   w::AbstractVector)
    rhs, data = _terms(method, columns)
    formula = Core.eval(@__MODULE__, :(GAM.@formula($(Meta.parse("y ~ " * rhs)))))
    weights = Float64.(w) .* length(w)
    G = zeros(Float64, size(Q))
    for a in axes(Q, 1)
        y = Float64.(view(Q, a, :))
        table = (; y, data...)
        model = Base.invokelatest(GAM.gam, formula, table; weights)
        G[a, :] .= model.fitted_values
    end
    return G
end

end # module
