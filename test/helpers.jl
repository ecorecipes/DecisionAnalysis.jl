# Shared test helpers: the fixture reader and the slow oracles that every fast path is
# checked against. The oracles are written from the definitions with explicit loops and
# share no code with the package.

const BigRat = Rational{BigInt}
const FIXTURES = joinpath(@__DIR__, "fixtures", "worked_examples.json")

# "n/d" or "n" -> Rational{BigInt}
function parse_rational(s::AbstractString)
    parts = split(s, '/')
    num = parse(BigInt, parts[1])
    return length(parts) == 2 ? num // parse(BigInt, parts[2]) : num // 1
end
rational_vector(v) = BigRat[parse_rational(x) for x in v]
function rational_matrix(rows)
    return permutedims(reduce(hcat, [rational_vector(r) for r in rows]))
end
symbols(v) = Symbol[Symbol(x) for x in v]
one_based(v) = Int[i + 1 for i in v]

load_examples() = JSON.parse(read(FIXTURES, String))["examples"]

# Oracles on plain matrices: Q[a, θ], prior p[θ], likelihood L[z, θ].
function oracle_expected_utility(Q, p)
    return [sum(p[t] * Q[a, t] for t in axes(Q, 2)) for a in axes(Q, 1)]
end

function oracle_regret(Q)
    R = similar(Q)
    for t in axes(Q, 2), a in axes(Q, 1)
        R[a, t] = maximum(Q[b, t] for b in axes(Q, 1)) - Q[a, t]
    end
    return R
end

function oracle_evpi(Q, p)
    return sum(p[t] * maximum(Q[a, t] for a in axes(Q, 1)) for t in axes(Q, 2)) -
           maximum(oracle_expected_utility(Q, p))
end

# Posterior form: Σ_z P(z) max_a E[Q[a, ·] | z], skipping signals of probability zero.
# The package uses the joint form; this is the textbook one.
function oracle_value_with_information(Q, p, L)
    total = zero(promote_type(eltype(Q), eltype(p), eltype(L)))
    for z in axes(L, 1)
        pz = sum(p[t] * L[z, t] for t in axes(L, 2))
        iszero(pz) && continue
        posterior = [p[t] * L[z, t] / pz for t in axes(L, 2)]
        total += pz * maximum(oracle_expected_utility(Q, posterior))
    end
    return total
end

function oracle_evsi(Q, p, L)
    return oracle_value_with_information(Q, p, L) - maximum(oracle_expected_utility(Q, p))
end

# The deterministic experiment that reveals the cell of a partition.
function partition_likelihood(partition, ::Type{T}) where {T}
    cells = unique(partition)
    return T[partition[t] == c ? one(T) : zero(T) for c in cells, t in eachindex(partition)]
end

argmax_set(v) = findall(==(maximum(v)), v)
argmin_set(v) = findall(==(minimum(v)), v)

# Indices of the chosen actions of a result among its feasible actions.
chosen_indices(result) = findall(in(result.chosen), result.actions)

# A random column-stochastic matrix with exact entries.
function random_stochastic(rng, ::Type{T}, nrow, ncol) where {T}
    cols = map(1:ncol) do _
        w = rand(rng, 0:6, nrow)
        all(iszero, w) && (w[1] = 1)
        return T[x // sum(w) for x in w]
    end
    return reduce(hcat, cols)
end

function exact_problem(rng, m, n)
    return DecisionProblem(random_payoff_table(rng, m, n; T=BigRat),
                           random_belief(rng, n; T=BigRat))
end
