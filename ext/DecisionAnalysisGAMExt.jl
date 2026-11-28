"""
    DecisionAnalysisGAMExt

The regression backend of `GAMRegression`: fitted conditional means from a
generalised additive model of GAM.jl, one model per action. Loaded when both
DecisionAnalysis and GAM are.
"""
module DecisionAnalysisGAMExt

using DecisionAnalysis: DecisionAnalysis, GAMRegression, _is_numeric
using GAM: GAM

# The REML score is very flat in the smoothing parameters, and the value of
# information is not: at GAM.jl's default tolerance (1e-7, on the relative change of
# the score) the search stops where an estimate can still be 6% from the one at the
# optimum. The tolerance here is where the estimates stop moving; the fits are of a
# linear model, so the extra iterations cost little. Still needed with the GAM.jl that
# no longer stops at the first flat step of the score: at its defaults all but one of
# the REML comparisons in test/test_savi.jl pass, and that one gives 0.94 where mgcv
# gives 0.14.
const _CONTROL = GAM.gam_control(; epsilon=1e-11, outer_maxit=2000)

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

function _fit_gam_row(method::GAMRegression, formula, fit_data, full_data,
                      weights::Vector{Float64}, y::Vector{Float64}, usable::Vector{Int},
                      held::Vector{Int})
    y_fit = isempty(held) ? y : y[usable]
    allequal(y_fit) && return fill(first(y_fit), length(y))
    m = GAM.gam(formula, (; y=y_fit, fit_data...); weights, method=method.smoothing,
                control=_CONTROL)
    isempty(held) && return m.fitted_values
    out = Vector{Float64}(undef, length(y))
    out[usable] .= m.fitted_values
    held_data = Pair{Symbol,Any}[name => col[held] for (name, col) in full_data]
    try
        out[held] .= GAM.predict(m, (; held_data...); type=:response)
    catch
        out[held] .= sum(weights .* y_fit) / sum(weights)
    end
    return out
end

function DecisionAnalysis.fitted_conditional_means(method::GAMRegression,
                                                   columns::NamedTuple, Q::AbstractMatrix,
                                                   w::AbstractVector)
    wf = Float64.(w)
    usable = findall(>(0), wf)
    held = findall(<=(0), wf)
    if isempty(held)
        formula, full_data = _formula(method, columns)
        fit_data = full_data
        weights = wf .* length(wf)
    else
        sub_columns = map(col -> col[usable], columns)
        formula, fit_data = _formula(method, sub_columns)
        full_data = Pair{Symbol,Any}[]
        for (j, column) in enumerate(columns)
            name = Symbol(:x, j)
            is_num = fit_data[j].second isa Vector{Float64}
            push!(full_data, name => (is_num ? Float64.(column) : string.(column)))
        end
        sub_w = wf[usable]
        weights = sub_w .* (length(usable) / sum(sub_w))
    end
    G = zeros(Float64, size(Q))
    # The first action is fitted as it is and every other one through its difference
    # from the first, as voi and SAVI do. Something that does not vary is its own
    # conditional mean: there is nothing to fit, and a fit would not converge.
    first_row = Float64.(view(Q, 1, :))
    G[1, :] .= _fit_gam_row(method, formula, fit_data, full_data, weights, first_row,
                            usable, held)
    for a in 2:size(Q, 1)
        diff_row = Float64.(view(Q, a, :)) .- first_row
        G[a, :] .= view(G, 1, :) .+
                   _fit_gam_row(method, formula, fit_data, full_data, weights, diff_row,
                                usable, held)
    end
    return G
end

end # module
