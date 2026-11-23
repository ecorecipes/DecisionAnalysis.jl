# A small dense two-phase simplex method in exact rational arithmetic, used for
# E-admissibility, the garbling test, deficiency and the upper EVPI envelope. It is
# internal: nothing here is exported.
#
# Why built in rather than an extension on an external solver: the programmes of this
# package are tiny (tens of variables), and the answers that depend on them ("is this
# action E-admissible", "is the deficiency zero") are exact statements that a
# floating-point solver can only answer up to a tolerance. Every number is converted to
# `Rational{BigInt}` (a `Float64` is a dyadic rational, so the conversion is exact) and
# Bland's rule guarantees termination. The cost is speed on large problems, which
# version 0.1 does not target.

const _BigRat = Rational{BigInt}

struct _LPSolution
    status::Symbol              # :optimal, :infeasible or :unbounded
    x::Vector{_BigRat}
    objective::_BigRat
end

# One pivot on row `r`, column `s` of the tableau `T` with reduced-cost row `z`.
function _pivot!(T::Matrix{_BigRat}, z::Vector{_BigRat}, basis::Vector{Int}, r::Int,
                 s::Int)
    ncol = size(T, 2)
    piv = T[r, s]
    for j in 1:ncol
        T[r, j] /= piv
    end
    for i in axes(T, 1)
        i == r && continue
        f = T[i, s]
        iszero(f) && continue
        for j in 1:ncol
            T[i, j] -= f * T[r, j]
        end
    end
    f = z[s]
    if !iszero(f)
        for j in 1:ncol
            z[j] -= f * T[r, j]
        end
    end
    basis[r] = s
    return nothing
end

# Simplex iterations with Bland's rule; entering columns are restricted to `1:n`.
function _iterate!(T::Matrix{_BigRat}, z::Vector{_BigRat}, basis::Vector{Int}, n::Int)
    while true
        s = findfirst(j -> z[j] < 0, 1:n)
        s === nothing && return :optimal
        r = 0
        best = zero(_BigRat)
        for i in axes(T, 1)
            T[i, s] > 0 || continue
            ratio = T[i, end] / T[i, s]
            if r == 0 || ratio < best || (ratio == best && basis[i] < basis[r])
                r = i
                best = ratio
            end
        end
        r == 0 && return :unbounded
        _pivot!(T, z, basis, r, s)
    end
end

# minimise c'x subject to A x = b, x >= 0.
function _simplex(c::Vector{_BigRat}, A::Matrix{_BigRat}, b::Vector{_BigRat})
    m, n = size(A)
    T = zeros(_BigRat, m, n + m + 1)
    for i in 1:m
        s = b[i] < 0 ? -1 : 1
        for j in 1:n
            T[i, j] = s * A[i, j]
        end
        T[i, n + i] = 1
        T[i, end] = s * b[i]
    end
    basis = collect((n + 1):(n + m))
    # Phase 1: minimise the sum of the artificial variables.
    z = zeros(_BigRat, n + m + 1)
    for j in 1:n
        z[j] = -sum(@view T[:, j]; init=zero(_BigRat))
    end
    z[end] = -sum(@view T[:, end]; init=zero(_BigRat))
    _iterate!(T, z, basis, n)
    iszero(z[end]) || return _LPSolution(:infeasible, _BigRat[], zero(_BigRat))
    # Drive the remaining (degenerate) artificial variables out of the basis; a row
    # with no structural entry is redundant and is dropped.
    keep = trues(m)
    for i in 1:m
        basis[i] > n || continue
        j = findfirst(j -> !iszero(T[i, j]), 1:n)
        if j === nothing
            keep[i] = false
        else
            _pivot!(T, z, basis, i, j)
        end
    end
    T = T[keep, vcat(1:n, n + m + 1)]
    basis = basis[keep]
    # Phase 2.
    z = zeros(_BigRat, n + 1)
    for j in 1:n
        z[j] = c[j]
    end
    for (i, bi) in enumerate(basis)
        cb = c[bi]
        iszero(cb) && continue
        for j in 1:(n + 1)
            z[j] -= cb * T[i, j]
        end
    end
    status = _iterate!(T, z, basis, n)
    status === :optimal || return _LPSolution(status, _BigRat[], zero(_BigRat))
    x = zeros(_BigRat, n)
    for (i, bi) in enumerate(basis)
        x[bi] = T[i, end]
    end
    return _LPSolution(:optimal, x, -z[end])
end

_bigrat(A::AbstractArray) = map(_BigRat, A)

# minimise c'x subject to Aeq x = beq, Aub x <= bub, and x[j] >= 0 for j not in `free`.
function _solve_lp(c::AbstractVector, Aeq::AbstractMatrix, beq::AbstractVector,
                   Aub::AbstractMatrix, bub::AbstractVector; free::Vector{Int}=Int[])
    n = length(c)
    me, mu = size(Aeq, 1), size(Aub, 1)
    nf = length(free)
    A = zeros(_BigRat, me + mu, n + nf + mu)
    A[1:me, 1:n] = _bigrat(Aeq)
    A[(me + 1):(me + mu), 1:n] = _bigrat(Aub)
    cc = zeros(_BigRat, n + nf + mu)
    cc[1:n] = _bigrat(c)
    for (k, j) in enumerate(free)
        A[:, n + k] = -A[:, j]
        cc[n + k] = -cc[j]
    end
    for i in 1:mu
        A[me + i, n + nf + i] = 1
    end
    b = vcat(_bigrat(beq), _bigrat(bub))
    sol = _simplex(cc, A, b)
    sol.status === :optimal || return sol
    x = sol.x[1:n]
    for (k, j) in enumerate(free)
        x[j] -= sol.x[n + k]
    end
    return _LPSolution(:optimal, x, sol.objective)
end

# Results go back to the caller's arithmetic: floating point stays floating point,
# anything exact is returned as `Rational{BigInt}`.
_from_big(::Type{T}, x) where {T<:AbstractFloat} = T(x)
_from_big(::Type, x) = x
