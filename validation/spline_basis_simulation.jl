# The simulation behind the choice of basis of `SplineRegression()` (finding 10 of the
# README). Run with `julia --project validation/spline_basis_simulation.jl`; a few minutes.
# Columns: the cubic with four knots, limits of 10 and 20 draws per coefficient,
# generalised cross-validation without and with the constant, and leave-one-out.
using DecisionAnalysis, Random, Statistics, Printf, LinearAlgebra
const DA = DecisionAnalysis
ladder = [(0, 0), (1, 0), (2, 0), (3, 0), (3, 1), (3, 2), (3, 3), (3, 4)]
ncoef(p, c) = 1 + p * (c[1] + c[2])
function fmeans(prob, θ)
    n, p = size(θ)
    F = zeros(prob == :linear ? 2 : 3, n)
    for i in 1:n
        if prob == :linear
            F[2, i] = 0.3 + sum(θ[i, :]) / sqrt(p)
        elseif prob == :nonlinear
            F[2, i] = 0.2 + sin(2θ[i, 1]) +
                      sum(0.5 * (θ[i, j]^2 - 1) for j in 2:p; init=0.0)
            F[3, i] = -0.2 + 1.2tanh(2(θ[i, 1] - 0.5)) +
                      sum(0.4θ[i, j] for j in 2:p; init=0.0)
        elseif prob == :weak
            F[2, i] = 0.1 + 0.3θ[i, 1]
            F[3, i] = -0.1
        else
            F[2, i] = 0.1
            F[3, i] = -0.1
        end
    end
    return F
end
value(G) = mean(maximum(G; dims=1)) - maximum(mean(G; dims=2))
truth(prob, p) = value(fmeans(prob, randn(Xoshiro(7), 2_000_000, p)))
# returns value, gcv, loo for each ladder entry
function fits(θ, Q)
    n, p = size(θ)
    cols = NamedTuple{ntuple(j -> Symbol(:x, j), p)}(ntuple(j -> θ[:, j], p))
    D = permutedims(Q[2:end, :])               # differences from the reference action
    out = []
    for cfg in ladder
        if ncoef(p, cfg) > n ÷ 2
            push!(out, (NaN, Inf, Inf))
            continue
        end
        A = cfg == (0, 0) ? ones(n, 1) :
            DA._design(SplineRegression(; degree=cfg[1], knots=cfg[2]), cols,
                       fill(1 / n, n))
        F = qr(A)
        Qt = Matrix(F.Q)[:, 1:size(A, 2)]
        fitted = Qt * (Qt' * D)
        R = D .- fitted
        h = vec(sum(abs2, Qt; dims=2))
        k = size(A, 2)
        G = vcat(zeros(1, n), permutedims(fitted))
        push!(out, (value(G), n * sum(abs2, R) / (n - k)^2, sum(abs2, R ./ (1 .- h))))
    end
    return out
end
pick(n, p, c) = something(findlast(cfg -> ncoef(p, cfg) <= n / c, ladder), 2)
ns = [30, 50, 100, 200, 400, 1000];
reps = 300;
names = ["default(3,4)", "limit c=10", "limit c=20", "GCV", "GCV+none", "LOO+none"]
println("bias / rmse of each selector; truth in brackets")
tot = Dict()
for prob in (:linear, :nonlinear, :weak, :null), s in (0.5, 2.0), p in (1, 2, 4)
    t = truth(prob, p)
    for n in ns
        rng = Xoshiro(hash((prob, s, p, n)))
        v = zeros(reps, 6)
        chosen = zeros(Int, reps)
        for r in 1:reps
            θ = randn(rng, n, p)
            F = fmeans(prob, θ)
            Q = F .+ s .* randn(rng, size(F)...)
            Q[1, :] .= 0
            o = fits(θ, Q)
            last = findlast(x -> !isnan(x[1]), o)
            g = [x[2] for x in o]
            l = [x[3] for x in o]
            v[r, 1] = o[last][1]
            v[r, 2] = o[min(pick(n, p, 10), last)][1]
            v[r, 3] = o[min(pick(n, p, 20), last)][1]
            v[r, 4] = o[1 + argmin(g[2:end])][1]
            v[r, 5] = o[argmin(g)][1]
            chosen[r] = argmin(g)
            v[r, 6] = o[argmin(l)][1]
        end
        b = [mean(v[:, k]) - t for k in 1:6]
        e = [sqrt(mean((v[:, k] .- t) .^ 2)) for k in 1:6]
        tot[(prob, s, p, n)] = (b, e)
        @printf("%-9s s=%.1f p=%d n=%4d [%.3f] %s | none chosen %.2f\n", prob, s, p, n, t,
                join((@sprintf("%+.3f/%.3f", b[k], e[k]) for k in 1:6), "  "),
                mean(chosen .== 1))
    end
end
println("\nmean rmse over noise levels and p, by problem and n: ", names)
for prob in (:linear, :nonlinear, :weak, :null), n in ns
    m = [mean(tot[(prob, s, p, n)][2][k] for s in (0.5, 2.0), p in (1, 2, 4)) for k in 1:6]
    bb = [mean(tot[(prob, s, p, n)][1][k] for s in (0.5, 2.0), p in (1, 2, 4)) for k in 1:6]
    @printf("%-9s n=%4d rmse %s | bias %s\n", prob, n,
            join((@sprintf("%.3f", x) for x in m), " "),
            join((@sprintf("%+.3f", x) for x in bb), " "))
end
