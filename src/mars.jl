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

function _mars_weighted_column!(dst::Vector{Float64}, root::Vector{Float64},
                                parent::Vector{Float64}, x::Vector{Float64}, t::Float64,
                                sign::Int)
    @inbounds for i in eachindex(dst)
        d = sign * (x[i] - t)
        dst[i] = d > 0.0 ? root[i] * parent[i] * d : 0.0
    end
    return dst
end

# Orthogonalise `c` against the orthonormal columns of `basis` (and optional `extra`)
# in-place into `u`; returns `true` when the residual norm is non-negligible.
function _mars_orthogonal!(u::Vector{Float64}, c::Vector{Float64},
                           basis::Vector{Vector{Float64}},
                           extra::Union{Nothing,Vector{Float64}}=nothing)
    size0 = dot(c, c)
    size0 > 0 || return false
    copyto!(u, c)
    for _ in 1:2
        for q in basis
            d = dot(q, u)
            @inbounds for i in eachindex(u)
                u[i] -= d * q[i]
            end
        end
        if extra !== nothing
            d = dot(extra, u)
            @inbounds for i in eachindex(u)
                u[i] -= d * extra[i]
            end
        end
    end
    left = dot(u, u)
    left > 1e-10 * size0 || return false
    inv_n = inv(sqrt(left))
    @inbounds for i in eachindex(u)
        u[i] *= inv_n
    end
    return true
end

function _mars_orthogonal(c::Vector{Float64}, basis::Vector{Vector{Float64}})
    u = similar(c)
    return _mars_orthogonal!(u, c, basis) ? u : nothing
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
    c_buf = Vector{Float64}(undef, n)
    u1_buf = Vector{Float64}(undef, n)
    u2_buf = Vector{Float64}(undef, n)
    r_buf = Vector{Float64}(undef, size(R, 2))
    while length(terms) + 1 <= method.max_terms && sum(abs2, R) > 1e-12 * total
        best = (0.0, 0, 0, 0.0)
        for (b, term) in enumerate(terms)
            length(term.variables) < method.degree || continue
            parent_b = raw[b]
            for v in eachindex(predictors)
                v in term.variables && continue
                xv = predictors[v]
                for t in knots[v]
                    gain = 0.0
                    _mars_weighted_column!(c_buf, root, parent_b, xv, t, 1)
                    has_first = _mars_orthogonal!(u1_buf, c_buf, ortho)
                    if has_first
                        mul!(r_buf, transpose(R), u1_buf)
                        gain += dot(r_buf, r_buf)
                    end
                    _mars_weighted_column!(c_buf, root, parent_b, xv, t, -1)
                    if _mars_orthogonal!(u2_buf, c_buf, ortho,
                                         has_first ? u1_buf : nothing)
                        mul!(r_buf, transpose(R), u2_buf)
                        gain += dot(r_buf, r_buf)
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
            mul!(r_buf, transpose(R), u)
            @inbounds for j in axes(R, 2), i in axes(R, 1)
                R[i, j] -= u[i] * r_buf[j]
            end
        end
    end

    # Backward pass via thin QR of Bw: for any subset S, columns of Bw[:, S] lie in
    # span(Q), so ||Z - Bw[:, S] β||² = base_rss + ||QtZ - R[:, S] β||² on an M × |S|
    # matrix (exact QR conditioning without squaring κ(Bw) or re-factoring n rows).
    B = reduce(hcat, raw)
    Bw = root .* B
    r_bw = min(size(Bw, 1), size(Bw, 2))
    F_bw = qr(Bw)
    R_bw = Matrix(F_bw.R)
    QtZ_full = F_bw.Q' * Z
    QtZ = QtZ_full[1:r_bw, :]
    base_rss = r_bw < n ? sum(abs2, view(QtZ_full, (r_bw + 1):n, :)) : 0.0
    n_eff = Float64(effective_sample_size(wf))
    penalty = method.penalty === nothing ? (method.degree == 1 ? 2.0 : 3.0) :
              method.penalty
    function rss(S)
        Rs = view(R_bw, :, S)
        fit = qr(Rs, ColumnNorm()) \ QtZ
        return base_rss + sum(abs2, QtZ .- Rs * fit)
    end
    function score(S)
        cost = length(S) + penalty * (length(S) - 1) / 2
        cost < n_eff || return Inf
        return (rss(S) / n) / (1 - cost / n_eff)^2
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
