"""
    DecisionAnalysisGAMExt

The regression backend of `GAMRegression`: fitted conditional means from a
generalised additive model of GAM.jl (`gam`, `bam`, or `scam`), one model per action,
plus `parameter_draws` and `SampleBelief` adapters for `GAM.GamModel` posterior
samples. Loaded when both DecisionAnalysis and GAM are.
"""
module DecisionAnalysisGAMExt

using DecisionAnalysis: DecisionAnalysis, GAMRegression, SampleBelief, _is_numeric
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
    parametric = Symbol[]
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
            if distinct > 1
                push!(parametric, name)
                push!(terms, GAM.term(name))
            end
        end
    end
    if method.backend === :scam && !isempty(smooth)
        bs = something(method.shape, :mpi)
        specs = GAM.SmoothSpec[GAM.s(name; k=max(k, 4), bs)
                               for (name, k) in zip(smooth, basis)]
        return GAM.GamFormula(:y, parametric, true, specs), data, true
    elseif method.tensor && length(smooth) > 1
        k = max(3, min(minimum(basis), 5))
        if method.interaction === :ti
            for (name, kj) in zip(smooth, basis)
                push!(terms, GAM.SmoothTerm(GAM.s(name; k=kj)))
            end
            push!(terms, GAM.SmoothTerm(GAM.ti(smooth...; k)))
        else
            push!(terms, GAM.SmoothTerm(GAM.te(smooth...; k)))
        end
    else
        for (name, k) in zip(smooth, basis)
            push!(terms, GAM.SmoothTerm(GAM.s(name; k)))
        end
    end
    isempty(terms) && push!(terms, GAM.term(1))
    return (GAM.term(:y) ~ reduce(+, terms)), data, !isempty(smooth)
end

function _fit_model(method::GAMRegression, formula, fit_tbl, weights::Vector{Float64},
                    has_smooth::Bool)
    if method.backend === :bam
        return GAM.bam(formula, fit_tbl; weights, method=method.smoothing,
                       discrete=method.discrete, control=_CONTROL)
    elseif method.backend === :scam && has_smooth
        return GAM.scam(formula, fit_tbl; weights, method=method.smoothing)
    end
    return GAM.gam(formula, fit_tbl; weights, method=method.smoothing, control=_CONTROL)
end

function _fit_gam_row(method::GAMRegression, formula, fit_data, full_data,
                      weights::Vector{Float64}, y::Vector{Float64}, usable::Vector{Int},
                      held::Vector{Int}, has_smooth::Bool)
    y_fit = isempty(held) ? y : y[usable]
    allequal(y_fit) && return fill(first(y_fit), length(y)), nothing
    m = _fit_model(method, formula, (; y=y_fit, fit_data...), weights, has_smooth)
    isempty(held) && return m.fitted_values, m
    out = Vector{Float64}(undef, length(y))
    out[usable] .= m.fitted_values
    held_data = Pair{Symbol,Any}[name => col[held] for (name, col) in full_data]
    try
        out[held] .= GAM.predict(m, (; held_data...); type=:response)
    catch
        out[held] .= sum(weights .* y_fit) / sum(weights)
    end
    return out, m
end

function _gam_fit_all(method::GAMRegression, columns::NamedTuple, Q::AbstractMatrix,
                      w::AbstractVector)
    wf = Float64.(w)
    usable = findall(>(0), wf)
    held = findall(<=(0), wf)
    if isempty(held)
        formula, full_data, has_smooth = _formula(method, columns)
        fit_data = full_data
        weights = wf .* length(wf)
    else
        sub_columns = map(col -> col[usable], columns)
        formula, fit_data, has_smooth = _formula(method, sub_columns)
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
    models = Vector{Union{Nothing,GAM.GamModel}}(undef, size(Q, 1))
    # The first action is fitted as it is and every other one through its difference
    # from the first, as voi and SAVI do. Something that does not vary is its own
    # conditional mean: there is nothing to fit, and a fit would not converge.
    first_row = Float64.(view(Q, 1, :))
    fit1, m1 = _fit_gam_row(method, formula, fit_data, full_data, weights, first_row,
                            usable, held, has_smooth)
    G[1, :] .= fit1
    models[1] = m1
    for a in 2:size(Q, 1)
        diff_row = Float64.(view(Q, a, :)) .- first_row
        fita, ma = _fit_gam_row(method, formula, fit_data, full_data, weights, diff_row,
                                usable, held, has_smooth)
        G[a, :] .= view(G, 1, :) .+ fita
        models[a] = ma
    end
    return G, models
end

function DecisionAnalysis.fitted_conditional_means(method::GAMRegression,
                                                   columns::NamedTuple, Q::AbstractMatrix,
                                                   w::AbstractVector)
    G, _ = _gam_fit_all(method, columns, Q, w)
    return G
end

function DecisionAnalysis._fit(method::GAMRegression, columns::NamedTuple,
                               Q::AbstractMatrix, w::AbstractVector)
    G, models = _gam_fit_all(method, columns, Q, w)
    residual = sum(Float64(w[i]) * (Float64(Q[a, i]) - G[a, i])^2
                   for a in axes(Q, 1), i in axes(Q, 2)) / size(Q, 1)
    edfs = Float64[m === nothing ? 1.0 : Float64(GAM.model_edf(m)) for m in models]
    checks = [m === nothing ? NamedTuple[] : GAM.k_check(m; n_rep=50, seed=11)
              for m in models]
    return G,
           (; residual_variance=residual, backend=method.backend,
            smoothing=method.smoothing, edf=edfs, k_check=checks)
end

function _clean_coef_symbols(names::AbstractVector{<:AbstractString})
    seen = Dict{Symbol,Int}()
    out = Vector{Symbol}(undef, length(names))
    for (i, raw) in enumerate(names)
        s = lowercase(replace(strip(raw), r"[^A-Za-z0-9_]+" => "_"))
        s = strip(s, '_')
        isempty(s) && (s = "coef_$i")
        isdigit(first(s)) && (s = "c_" * s)
        sym = Symbol(s)
        cnt = get(seen, sym, 0) + 1
        seen[sym] = cnt
        out[i] = cnt == 1 ? sym : Symbol(sym, :_, cnt)
    end
    return out
end

function DecisionAnalysis.parameter_draws(m::GAM.GamModel; n::Integer=1000,
                                          seed::Integer=1, unconditional::Bool=false)
    n >= 1 || throw(ArgumentError("n must be positive, got $n"))
    draws = GAM.posterior_samples(m; n=Int(n), seed=Int(seed), unconditional)
    syms = _clean_coef_symbols(GAM.coefnames(m))
    return NamedTuple{Tuple(syms)}(Tuple(draws[:, j] for j in eachindex(syms)))
end

function DecisionAnalysis.SampleBelief(m::GAM.GamModel; n::Integer=1000,
                                       seed::Integer=1, unconditional::Bool=false)
    return SampleBelief(DecisionAnalysis.parameter_draws(m; n, seed, unconditional))
end

end # module
