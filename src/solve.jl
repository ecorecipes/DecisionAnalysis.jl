# Solving a `DecisionProgram` by one of three routes, and the exact checks that stand
# between a solver and anything the package returns.
#
#   optimizer = nothing        the built-in exact simplex (src/lp.jl), fed from
#                              `JuMP.lp_matrix_data` of the model;
#   an exact optimiser         one that accepts the model's number type, for example
#                              `CDDLib.Optimizer{Rational{BigInt}}` on a rational model;
#   a floating-point optimiser on a rational model: a `Float64` copy of the matrix form
#                              is solved, the primal and dual solutions are rounded to
#                              rationals, and the pair is checked exactly against the
#                              rational data.
#
# No route is trusted: the callers verify every witness exactly before returning it.

const _MOI = JuMP.MOI

"""
    ProgramSolution{T}

What [`solve_program`](@ref) returns. Fields:

- `status`: `:optimal`, `:infeasible`, `:unbounded`, or the lower-case name of another
  termination status of an external optimiser.
- `objective`: the objective value at `values` (zero for a feasibility programme), or
  `nothing` when the status is not `:optimal`.
- `values`: the solution, a named tuple with the names and shapes of the programme's
  `variables`; empty when the status is not `:optimal`.
- `route`: `:internal` (the built-in exact simplex), `:exact_optimizer` (an optimiser
  working in the rational number type of the model) or `:float_optimizer` (a
  floating-point optimiser: with rounding on a rational model, as it is on a
  floating-point one).
- `exact`: whether `status`, `objective` and `values` are exact statements about the
  programme. It is `true` for a rational programme solved by the internal simplex or
  by an exact optimiser, and for a rounded floating-point solution whose optimality
  was certified exactly (see [`solve_program`](@ref)). It is `false` for a
  floating-point programme, for a rounded solution that was not certified (then
  `values` and `objective` are the rounded numbers, unverified), and for an
  `:infeasible` or `:unbounded` verdict of a floating-point optimiser.
"""
struct ProgramSolution{T,V<:NamedTuple}
    status::Symbol
    objective::Union{Nothing,T}
    values::V
    route::Symbol
    exact::Bool
end

# Whether the optimiser accepts linear constraints with coefficients of type `T`.
function _supports_number_type(optimizer, ::Type{T}) where {T}
    opt = _MOI.instantiate(optimizer)
    F = _MOI.ScalarAffineFunction{T}
    return _MOI.supports_constraint(opt, F, _MOI.LessThan{T}) ||
           _MOI.supports_constraint(opt, F, _MOI.EqualTo{T}) ||
           _MOI.supports_constraint(opt, F, _MOI.GreaterThan{T})
end

function _status_symbol(status)
    status == _MOI.OPTIMAL && return :optimal
    status == _MOI.INFEASIBLE && return :infeasible
    status == _MOI.DUAL_INFEASIBLE && return :unbounded
    return Symbol(lowercase(string(status)))
end

function _run!(model, optimizer)
    JuMP.set_optimizer(model, optimizer)
    _MOI.supports(JuMP.backend(model), _MOI.Silent()) && JuMP.set_silent(model)
    JuMP.optimize!(model)
    return _status_symbol(JuMP.termination_status(model))
end

_minimising_sign(data) = data.sense == _MOI.MAX_SENSE ? -1 : 1
_objective_at(data, x) = dot(data.c, x) + data.c_offset

# The matrix form  bl ≤ A x ≤ bu, xl ≤ x ≤ xu  as the form of the built-in simplex:
# minimise c'x subject to Aeq x = beq, Aub x ≤ bub, x ≥ 0 except for the free variables.
# Variables with a finite lower bound are shifted (`x' = x - xl >= 0`) rather than split.
function _solve_internal(data)
    A = Matrix(data.A)
    m, n = size(A)
    R = _BigRat
    shift = zeros(R, n)
    free = Int[]
    ub_rows, ub_rhs = Vector{R}[], R[]
    unit(j, s) = R[k == j ? s : 0 for k in 1:n]
    for j in 1:n
        lo, hi = data.x_lower[j], data.x_upper[j]
        if isfinite(lo) && isfinite(hi) && lo > hi
            return :infeasible, R[]
        end
        if isfinite(lo)
            shift[j] = R(lo)
            if isfinite(hi)
                push!(ub_rows, unit(j, 1))
                push!(ub_rhs, R(hi) - shift[j])
            end
        else
            push!(free, j)
            if isfinite(hi)
                push!(ub_rows, unit(j, 1))
                push!(ub_rhs, R(hi))
            end
        end
    end
    has_shift = any(!iszero, shift)
    eq_rows, eq_rhs = Vector{R}[], R[]
    for i in 1:m
        row = R.(A[i, :])
        offset = has_shift ? dot(row, shift) : zero(R)
        lo, hi = data.b_lower[i], data.b_upper[i]
        if lo == hi
            push!(eq_rows, row)
            push!(eq_rhs, R(lo) - offset)
        else
            if isfinite(hi)
                push!(ub_rows, row)
                push!(ub_rhs, R(hi) - offset)
            end
            if isfinite(lo)
                push!(ub_rows, -row)
                push!(ub_rhs, offset - R(lo))
            end
        end
    end
    stack(rows) = isempty(rows) ? zeros(R, 0, n) : permutedims(reduce(hcat, rows))
    c = _minimising_sign(data) .* R.(data.c)
    sol = _solve_lp(c, stack(eq_rows), eq_rhs, stack(ub_rows), ub_rhs; free)
    sol.status === :optimal || return sol.status, sol.x
    return sol.status, has_shift ? (sol.x .+ shift) : sol.x
end

# A typed copy of the matrix form, with its variables and rows in the same order
# (a row with no finite bound has no constraint and the entry `nothing`). Building a fresh
# twin leaves `prog.model` untouched and gives row handles in the order of `data.A`.
function _typed_twin(data, ::Type{S}) where {S<:Real}
    m, n = size(data.A)
    twin = JuMP.GenericModel{S}()
    x = JuMP.@variable(twin, [1:n])
    for j in 1:n
        isfinite(data.x_lower[j]) && JuMP.set_lower_bound(x[j], S(data.x_lower[j]))
        isfinite(data.x_upper[j]) && JuMP.set_upper_bound(x[j], S(data.x_upper[j]))
    end
    A = Matrix(data.A)
    rows = Any[]
    for i in 1:m
        expr = sum(S(A[i, j]) * x[j] for j in 1:n if !iszero(A[i, j]); init=zero(S))
        expr = expr + JuMP.GenericAffExpr{S,JuMP.GenericVariableRef{S}}(zero(S))
        lo, hi = data.b_lower[i], data.b_upper[i]
        if lo == hi
            push!(rows, JuMP.@constraint(twin, expr == S(lo)))
        elseif isfinite(lo) && isfinite(hi)
            push!(rows, JuMP.@constraint(twin, S(lo) <= expr <= S(hi)))
        elseif isfinite(lo)
            push!(rows, JuMP.@constraint(twin, expr >= S(lo)))
        elseif isfinite(hi)
            push!(rows, JuMP.@constraint(twin, expr <= S(hi)))
        else
            push!(rows, nothing)
        end
    end
    objective = sum(S(data.c[j]) * x[j] for j in 1:n; init=zero(S)) + S(data.c_offset)
    if data.sense == _MOI.MAX_SENSE
        JuMP.@objective(twin, Max, objective)
    elseif data.sense == _MOI.MIN_SENSE
        JuMP.@objective(twin, Min, objective)
    end
    return twin, x, rows
end
_float_twin(data) = _typed_twin(data, Float64)

# `nothing` if `x` satisfies every row and bound of the matrix form exactly.
function _primal_defect(data, x::AbstractVector)
    for j in eachindex(x)
        data.x_lower[j] <= x[j] <= data.x_upper[j] ||
            return "variable $j = $(x[j]) is outside its bounds"
    end
    Ax = data.A * x
    for i in eachindex(Ax)
        data.b_lower[i] <= Ax[i] <= data.b_upper[i] ||
            return "constraint $i is violated"
    end
    return nothing
end

# Whether the row multipliers `y` are dual feasible for the minimising form and give a
# dual objective equal to the primal objective at `x`. With an exactly feasible `x`
# this is a proof, by weak duality, that `x` is optimal.
function _dual_certifies(data, x::AbstractVector, y::AbstractVector)
    c = _minimising_sign(data) .* data.c
    r = c .- data.A' * y
    bound = zero(eltype(c))
    for i in eachindex(y)
        if y[i] > 0
            isfinite(data.b_lower[i]) || return false
            bound += data.b_lower[i] * y[i]
        elseif y[i] < 0
            isfinite(data.b_upper[i]) || return false
            bound += data.b_upper[i] * y[i]
        end
    end
    for j in eachindex(r)
        if r[j] > 0
            isfinite(data.x_lower[j]) || return false
            bound += data.x_lower[j] * r[j]
        elseif r[j] < 0
            isfinite(data.x_upper[j]) || return false
            bound += data.x_upper[j] * r[j]
        end
    end
    return dot(c, x) == bound
end

_round_rational(x::Real, tol::Real) = rationalize(BigInt, Float64(x); tol)

# The rounding tolerances tried, in order, on a floating-point solution of a rational
# model. Trying several is sound because each rounding is verified exactly.
const _ROUNDING_TOLS = (1e-8, 1e-10, 1e-12)
function _rounding_tols(tol)
    tols = tol isa Real ? (tol,) : Tuple(tol)
    (!isempty(tols) && all(t -> t isa Real && t >= 0, tols)) ||
        throw(ArgumentError("rounding_tol must be a nonnegative number or a non-empty " *
                            "collection of them, got $tol"))
    return tols
end

# The values of the model's variables in the shape of the programme's `variables`.
function _shape_values(prog::DecisionProgram, data, x::AbstractVector)
    lookup = Dict(zip(data.variables, x))
    return map(v -> _value_of(lookup, v), prog.variables)
end
_value_of(lookup::AbstractDict, v::JuMP.AbstractVariableRef) = lookup[v]
_value_of(lookup::AbstractDict, v::AbstractArray) = map(x -> _value_of(lookup, x), v)

function _failed(prog::DecisionProgram{T}, status::Symbol, route::Symbol,
                 exact::Bool) where {T}
    return ProgramSolution{T,@NamedTuple{}}(status, nothing, (;), route, exact)
end

function _solved(prog::DecisionProgram{T}, data, x::AbstractVector, route::Symbol,
                 exact::Bool) where {T}
    xt = T <: AbstractFloat ? T.(x) : x
    values = _shape_values(prog, data, xt)
    return ProgramSolution{T,typeof(values)}(:optimal, T(_objective_at(data, xt)), values,
                                             route, exact)
end

"""
    solve_program(program::DecisionProgram; optimizer=nothing, rounding_tol=(1e-8, 1e-10, 1e-12))
        -> ProgramSolution

Solve a programme built by [`eadmissibility_model`](@ref), [`garbling_model`](@ref),
[`deficiency_model`](@ref), [`evpi_envelope_model`](@ref), [`mixed_dominance_model`](@ref)
or [`separating_payoff_model`](@ref). There are three routes to the same model.

- **`optimizer = nothing` (the default): the built-in exact simplex.** The matrix
  form of the model is read with `JuMP.lp_matrix_data` and solved in
  `Rational{BigInt}` by a dense two-phase simplex method with Bland's rule. No solver
  needs to be installed. It is meant for small problems (tens of variables and
  constraints). A floating-point model is solved as the exact rationals its numbers
  are and the result converted back.
- **An exact optimiser**, one that accepts the model's number type, for example
  `CDDLib.Optimizer{Rational{BigInt}}` for a rational model (or any floating-point
  optimiser for a floating-point model). It is attached to `program.model` and run.
  For a rational model the point it returns is checked to satisfy every constraint
  exactly ([`ExactVerificationError`](@ref) otherwise).
- **A floating-point optimiser on a rational model**, for example
  `HiGHS.Optimizer`. A `Float64` copy of the matrix form is solved; the primal and
  dual solutions are rounded with `rationalize(BigInt, x; tol)`; and the rounded pair
  is checked against the rational data: the primal point satisfies every constraint
  exactly, the dual point is dual feasible, and the two objectives are equal (for a
  feasibility programme, the first of these alone). When they hold the rounded point
  is optimal by weak duality and the solution is marked `exact`. `rounding_tol` is
  the tolerance `tol`, or several to be tried in order until one passes (a coarse
  tolerance recovers simple fractions from a loose solution, a fine one recovers
  large denominators from a tight solution; trying several is sound because each
  rounding is verified). When none passes, the point rounded at the first tolerance
  is returned with `exact = false`, and it is for the caller to verify what it needs
  or to refuse: the
  functions of this package that call `solve_program` throw
  [`ExactVerificationError`](@ref) before they would return an unverified number.

Which of the last two routes applies is decided by asking the optimiser whether it
supports linear constraints in the model's number type. An optimiser that supports
neither that type nor `Float64` throws an `ArgumentError`.

`optimizer` is anything `JuMP.set_optimizer` accepts: an optimiser type or
`JuMP.optimizer_with_attributes(...)`.
"""
function solve_program(prog::DecisionProgram{T}; optimizer=nothing,
                       rounding_tol=_ROUNDING_TOLS) where {T}
    tols = _rounding_tols(rounding_tol)
    data = JuMP.lp_matrix_data(prog.model)
    rational = _is_exact(T)
    if optimizer === nothing
        status, x = _solve_internal(data)
        status === :optimal || return _failed(prog, status, :internal, rational)
        return _solved(prog, data, x, :internal, rational)
    elseif _supports_number_type(optimizer, T)
        twin, xs, rows = _typed_twin(data, T)
        status = _run!(twin, optimizer)
        route = rational ? :exact_optimizer : :float_optimizer
        status === :optimal || return _failed(prog, status, route, rational)
        x = T[JuMP.value(v) for v in xs]
        if rational
            defect = _primal_defect(data, x)
            defect === nothing ||
                throw(ExactVerificationError(prog.name,
                                             "the point returned by the exact " *
                                             "optimiser is not feasible: $defect"))
            optimising = data.sense != _MOI.FEASIBILITY_SENSE
            if optimising && JuMP.has_duals(twin)
                y = T[r === nothing ? zero(T) : JuMP.dual(r) for r in rows]
                (_dual_certifies(data, x, y) || _dual_certifies(data, x, -y)) ||
                    throw(ExactVerificationError(prog.name,
                                                 "the dual multipliers returned by the " *
                                                 "exact optimiser do not certify " *
                                                 "optimality"))
            end
        end
        return _solved(prog, data, x, route, rational)
    elseif rational && _supports_number_type(optimizer, Float64)
        twin, xs, rows = _float_twin(data)
        status = _run!(twin, optimizer)
        status === :optimal || return _failed(prog, status, :float_optimizer, false)
        xf = Float64[JuMP.value(v) for v in xs]
        optimising = data.sense != _MOI.FEASIBILITY_SENSE
        yf = optimising && JuMP.has_duals(twin) ?
             Float64[r === nothing ? 0.0 : JuMP.dual(r) for r in rows] : nothing
        for tol in tols
            x = T[_round_rational(v, tol) for v in xf]
            _primal_defect(data, x) === nothing || continue
            if optimising
                yf === nothing && break
                y = T[_round_rational(v, tol) for v in yf]
                (_dual_certifies(data, x, y) || _dual_certifies(data, x, -y)) || continue
            end
            return _solved(prog, data, x, :float_optimizer, true)
        end
        x = T[_round_rational(v, first(tols)) for v in xf]
        return _solved(prog, data, x, :float_optimizer, false)
    end
    return throw(ArgumentError("the optimiser supports linear constraints neither in the " *
                               "model's number type $T nor in Float64"))
end

# Clean-up of a rounded point that should lie in the probability simplex: negative
# entries to zero, then normalise. `nothing` if nothing is left.
function _clean_weights(w::AbstractVector)
    v = max.(w, zero(eltype(w)))
    s = sum(v)
    return iszero(s) ? nothing : v ./ s
end

# The same for every column of a matrix that should be column-stochastic.
function _clean_columns(G::AbstractMatrix)
    H = max.(G, zero(eltype(G)))
    s = sum(H; dims=1)
    any(iszero, s) && return nothing
    return H ./ s
end
