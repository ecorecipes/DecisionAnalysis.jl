# Multivariate adaptive regression splines (Friedman 1991, *Annals of Statistics* 19:1),
# a regression backend for the sample-based value of information: the method that the R
# package `voi` reaches through `earth`. Written from the algorithm as published; it
# shares no code with `earth`, and differs from it in where knots may be placed.

"""
    MARSRegression(; degree=1, max_terms=21, knots=20, penalty=nothing, threshold=1e-3)

Regression by **multivariate adaptive regression splines** (Friedman 1991): the
conditional means are sums of products of hinge functions `max(x − t, 0)` and
`max(t − x, 0)`, with the variables and knots `t` chosen from the draws. The basis is
grown greedily, a pair of hinges at a time, each pair the one that most reduces the
residual sum of squares, and then pruned term by term to the subset with the smallest
generalised cross-validation score

    GCV = (RSS / n) / (1 − C / n)²,   C = M + penalty · (M − 1) / 2,

with `M` the number of terms. One basis is chosen for all actions together (their
payoffs standardised), and each action then has its own coefficients.

- `degree`: the largest number of variables in a term. 1 is additive; 2 allows
  products of hinges in two variables.
- `max_terms`: the size the basis may grow to before pruning, the intercept included.
- `knots`: the candidate knots per variable, at weighted quantiles of the draws (all
  the distinct values when there are no more than that). `earth` tries every draw;
  a grid is the difference from it, and makes the estimate depend little on `knots`
  once it is a few tens.
- `penalty`: the cost of a knot in the score; `nothing` is 2 for `degree = 1` and 3
  otherwise, the defaults of Friedman and of `earth`.
- `threshold`: growth stops when a pair improves the fit by less than this share of
  the total sum of squares.

Unlike [`SplineRegression`](@ref) it chooses its own flexibility, and spends it where
the payoff bends, so it suits conditional means that are flat over most of the range
and change sharply somewhere, and several focal parameters of which few matter. A
non-numeric column enters through an indicator for each of its levels but the first.
The pruning is by a score, not a test: with few draws the fit can still follow noise,
and the check of the estimate against the EVPI of the sample applies as for any
regression (see [`VOIEstimate`](@ref), [`compare_regressions`](@ref)).
"""
struct MARSRegression <: AbstractRegression
    degree::Int
    max_terms::Int
    knots::Int
    penalty::Union{Nothing,Float64}
    threshold::Float64
    function MARSRegression(; degree::Integer=1, max_terms::Integer=21, knots::Integer=20,
                            penalty::Union{Nothing,Real}=nothing, threshold::Real=1e-3)
        degree >= 1 || throw(ArgumentError("degree must be at least 1, got $degree"))
        max_terms >= 1 ||
            throw(ArgumentError("max_terms must be at least 1, got $max_terms"))
        knots >= 1 || throw(ArgumentError("knots must be at least 1, got $knots"))
        penalty === nothing || penalty >= 0 ||
            throw(ArgumentError("penalty must be nonnegative, got $penalty"))
        threshold >= 0 ||
            throw(ArgumentError("threshold must be nonnegative, got $threshold"))
        return new(degree, max_terms, knots,
                   penalty === nothing ? nothing : Float64(penalty), Float64(threshold))
    end
end

# A term: a product of hinges, one per variable in `variables`.
struct _MARSTerm
    variables::Vector{Int}
    knots::Vector{Float64}
    signs::Vector{Int}
end

# The numeric predictors: numeric columns as they are, the others as indicators.
function _mars_predictors(columns::NamedTuple)
    predictors = Vector{Float64}[]
    names = String[]
    for (name, x) in pairs(columns)
        if _is_numeric(x)
            push!(predictors, Float64.(x))
            push!(names, String(name))
        else
            for level in unique(x)[2:end]
                push!(predictors, Float64[isequal(v, level) for v in x])
                push!(names, string(name, " = ", level))
            end
        end
    end
    return predictors, names
end

function _mars_knots(x::Vector{Float64}, w::AbstractVector, k::Int)
    values = sort(unique(x))
    length(values) <= 1 && return Float64[]
    interior = values[1:(end - 1)]
    length(interior) <= k && return interior
    grid = unique(_weighted_quantiles(x, w, [(i - 0.5) / k for i in 1:k]))
    return filter(<(values[end]), grid)
end

function _mars_column(parent::AbstractVector, x::AbstractVector, t::Real, sign::Int)
    return parent .* max.(sign .* (x .- t), 0.0)
end

# Orthogonalise `c` against the orthonormal columns of `basis`; `nothing` when what is
# left is negligible beside the column itself.
function _mars_orthogonal(c::Vector{Float64}, basis::Vector{Vector{Float64}})
    size0 = dot(c, c)
    size0 > 0 || return nothing
    u = copy(c)
    for _ in 1:2, q in basis
        u .-= dot(q, u) .* q
    end
    left = dot(u, u)
    left > 1e-10 * size0 || return nothing
    return u ./ sqrt(left)
end

function _mars(method::MARSRegression, columns::NamedTuple, Q::AbstractMatrix,
               w::AbstractVector)
    predictors, names = _mars_predictors(columns)
    n = length(w)
    wf = Float64.(w) ./ sum(Float64, w)
    root = sqrt.(wf .* n)
    Y = permutedims(Float64.(Q))
    centre = permutedims(wf) * Y
    spread = sqrt.(permutedims(wf) * (Y .- centre) .^ 2)
    varying = findall(>(0), vec(spread))
    info(terms, gcv) = (; terms, gcv)
    isempty(varying) && return repeat(permutedims(centre), 1, n), info(String[], 0.0)
    # Standardised, weighted responses that vary: what the basis is chosen on.
    Z = root .* ((Y[:, varying] .- centre[:, varying]) ./ spread[:, varying])
    total = sum(abs2, Z)
    knots = [_mars_knots(x, wf, method.knots) for x in predictors]

    terms = [_MARSTerm(Int[], Float64[], Int[])]
    raw = [ones(n)]                       # the basis functions at the draws
    ortho = [root ./ sqrt(sum(abs2, root))]
    R = Z .- ortho[1] .* (permutedims(ortho[1]) * Z)
    while length(terms) + 1 <= method.max_terms && sum(abs2, R) > 1e-12 * total
        best = (0.0, 0, 0, 0.0)
        for (b, term) in enumerate(terms)
            length(term.variables) < method.degree || continue
            for v in eachindex(predictors)
                v in term.variables && continue
                for t in knots[v]
                    gain = 0.0
                    added = Vector{Float64}[]
                    for sign in (1, -1)
                        c = root .* _mars_column(raw[b], predictors[v], t, sign)
                        u = _mars_orthogonal(c, vcat(ortho, added))
                        u === nothing && continue
                        gain += sum(abs2, permutedims(u) * R)
                        push!(added, u)
                    end
                    gain > best[1] && (best = (gain, b, v, t))
                end
            end
        end
        gain, b, v, t = best
        gain > method.threshold * total || break
        for sign in (1, -1)
            length(terms) < method.max_terms || break
            column = _mars_column(raw[b], predictors[v], t, sign)
            u = _mars_orthogonal(root .* column, ortho)
            u === nothing && continue
            push!(terms,
                  _MARSTerm(vcat(terms[b].variables, v), vcat(terms[b].knots, t),
                            vcat(terms[b].signs, sign)))
            push!(raw, column)
            push!(ortho, u)
            R .-= u .* (permutedims(u) * R)
        end
    end

    # Backward pass on the normal equations: drop the term whose loss is least, and
    # keep the subset with the smallest generalised cross-validation score.
    B = reduce(hcat, raw)
    Bw = root .* B
    gram = Bw' * Bw
    cross = Bw' * Z
    penalty = method.penalty === nothing ? (method.degree == 1 ? 2.0 : 3.0) :
              method.penalty
    function rss(S)
        fit = qr(gram[S, S], ColumnNorm()) \ cross[S, :]
        return max(total - sum(cross[S, :] .* fit), 0.0)
    end
    function score(S)
        cost = length(S) + penalty * (length(S) - 1) / 2
        cost < n || return Inf
        return rss(S) / n / (1 - cost / n)^2
    end
    kept = collect(eachindex(terms))
    chosen, lowest = copy(kept), score(kept)
    while length(kept) > 1
        losses = [rss(filter(!=(j), kept)) for j in kept[2:end]]
        deleteat!(kept, 1 + argmin(losses))
        s = score(kept)
        s <= lowest && ((chosen, lowest) = (copy(kept), s))
    end
    coefficients = qr(Bw[:, chosen], ColumnNorm()) \ (root .* Y)
    function describe(term)
        return isempty(term.variables) ? "1" :
               join(("h($(s > 0 ? "" : "−")($(names[v]) − $(round(t; sigdigits=4))))"
                     for (v, t, s) in zip(term.variables, term.knots, term.signs)),
                    " · ")
    end
    return permutedims(B[:, chosen] * coefficients),
           info([describe(terms[j]) for j in chosen], lowest)
end

function fitted_conditional_means(method::MARSRegression, columns::NamedTuple,
                                  Q::AbstractMatrix, w::AbstractVector)
    return first(_mars(method, columns, Q, w))
end

function _fit(method::MARSRegression, columns::NamedTuple, Q, w)
    G, info = _mars(method, columns, Q, w)
    residual = sum(Float64(w[i]) * (Float64(Q[a, i]) - G[a, i])^2
                   for a in axes(Q, 1), i in axes(Q, 2)) / size(Q, 1)
    return G, (; residual_variance=residual, terms=info.terms, gcv=info.gcv)
end
