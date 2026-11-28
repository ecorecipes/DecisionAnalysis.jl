"""
    DecisionAnalysisSciMLBaseExt

Generic adapter for SciML problems (`SciMLBase.AbstractSciMLProblem` — `ODEProblem`,
`SDEProblem`, `DiscreteProblem`, `JumpProblem`, `EnsembleProblem`, etc.): implements
`with_parameters`, `apply_action`, `simulate`, and `outcome` using `SciMLBase.remake` and
`SciMLBase.solve` so any SciML ecological or epidemiological simulator works directly with
`DecisionAnalysis.outcomes`.

When using a lightweight solver package without a default algorithm (such as
`OrdinaryDiffEqTsit5` alone), pass `alg = Tsit5()` as a keyword argument on the problem
(or via `action = (; kwargs = (; alg = Tsit5()))`); when `OrdinaryDiffEqDefault` or
`DifferentialEquations` is loaded, `SciMLBase.solve(prob)` selects a solver automatically.

Loaded when both `DecisionAnalysis` and `SciMLBase` are loaded.
"""
module DecisionAnalysisSciMLBaseExt

using DecisionAnalysis: DecisionAnalysis
using Random: AbstractRNG
using SciMLBase: SciMLBase

function _merge_params(p::NamedTuple, nt::NamedTuple)
    fields = keys(p)
    used = filter(in(fields), keys(nt))
    isempty(used) &&
        throw(ArgumentError("the draw names none of the problem parameters $fields: $(keys(nt))"))
    vals = ntuple(length(fields)) do i
        k = fields[i]
        return k in used ? getfield(nt, k) : getfield(p, k)
    end
    return NamedTuple{fields}(vals)
end

function _merge_params(p::AbstractDict, nt::NamedTuple)
    q = copy(p)
    matched = false
    K = keytype(q)
    for k in keys(nt)
        if K === Symbol && haskey(q, k)
            q[k] = getfield(nt, k)
            matched = true
        elseif K === String && haskey(q, string(k))
            q[string(k)] = getfield(nt, k)
            matched = true
        elseif haskey(q, k)
            q[k] = getfield(nt, k)
            matched = true
        end
    end
    matched ||
        throw(ArgumentError("the draw names none of the dictionary parameter keys $(collect(keys(p))): $(keys(nt))"))
    return q
end

function _merge_params(p::AbstractVector, nt::NamedTuple)
    haskey(nt, :p) && return collect(nt.p)
    return throw(ArgumentError("vector-parameterised SciML problems require a `:p` column in the draw " *
                               "(or a NamedTuple/Dict parameter container)"))
end

function _merge_params(p, nt::NamedTuple)
    haskey(nt, :p) && return nt.p
    return throw(ArgumentError("cannot update parameters of type $(typeof(p)) from draw keys $(keys(nt)); " *
                               "pass a NamedTuple or Dict as `prob.p`, or supply a `:p` field in the draw"))
end

function DecisionAnalysis.with_parameters(prob::SciMLBase.AbstractSciMLProblem, θ)
    nt = NamedTuple(θ)
    has_u0 = haskey(nt, :u0)
    has_tspan = haskey(nt, :tspan)
    has_p = if prob.p isa NamedTuple
        any(k -> k in keys(prob.p), keys(nt))
    elseif prob.p isa AbstractDict
        K = keytype(prob.p)
        any(k -> (K === String ? haskey(prob.p, string(k)) : haskey(prob.p, k)), keys(nt))
    else
        haskey(nt, :p)
    end
    (has_p || has_u0 || has_tspan) ||
        throw(ArgumentError("the draw $(keys(nt)) matches none of `p`, `u0`, or `tspan` of the SciML problem"))
    new_p = has_p ? _merge_params(prob.p, nt) : prob.p
    new_u0 = has_u0 ? nt.u0 : prob.u0
    new_tspan = has_tspan ? nt.tspan : prob.tspan
    return SciMLBase.remake(prob; u0=new_u0, tspan=new_tspan, p=new_p)
end

function DecisionAnalysis.apply_action(prob::SciMLBase.AbstractSciMLProblem, action)
    (action === nothing || action == (;)) && return prob
    action isa Function && return action(prob)
    action isa NamedTuple ||
        throw(ArgumentError("an action for a SciML problem is `nothing`, `(;)`, a function " *
                            "`prob -> prob′`, or a NamedTuple with any of (:p, :u0, :tspan, :kwargs); " *
                            "got $(typeof(action))"))
    allowed = (:p, :u0, :tspan, :kwargs)
    for k in keys(action)
        k in allowed ||
            throw(ArgumentError("unknown field :$k in SciML action; expected a subset of $allowed"))
    end
    new_p = if haskey(action, :p)
        ap = action.p
        (prob.p isa NamedTuple && ap isa NamedTuple) ? merge(prob.p, ap) :
        (prob.p isa AbstractDict && ap isa NamedTuple) ? _merge_params(prob.p, ap) : ap
    else
        prob.p
    end
    new_u0 = haskey(action, :u0) ? action.u0 : prob.u0
    new_tspan = haskey(action, :tspan) ? action.tspan : prob.tspan
    extra = haskey(action, :kwargs) ? pairs(action.kwargs) : pairs((;))
    return SciMLBase.remake(prob; u0=new_u0, tspan=new_tspan, p=new_p, extra...)
end

function _is_stochastic_sciml(prob::SciMLBase.AbstractSciMLProblem)
    return prob isa SciMLBase.AbstractSDEProblem || prob isa SciMLBase.AbstractJumpProblem
end

function DecisionAnalysis.simulate(prob::SciMLBase.AbstractSciMLProblem; rng::AbstractRNG)
    if _is_stochastic_sciml(prob)
        return SciMLBase.solve(prob; seed=rand(rng, UInt64))
    end
    return SciMLBase.solve(prob)
end

function DecisionAnalysis.outcome(sol::SciMLBase.AbstractSciMLSolution)
    SciMLBase.successful_retcode(sol) ||
        throw(ErrorException("SciML solve did not succeed (retcode = $(sol.retcode))"))
    u_end = sol.u[end]
    if u_end isa Real
        return (; final=float(u_end))
    elseif u_end isa AbstractVector{<:Real}
        n = length(u_end)
        names = (ntuple(i -> Symbol(:u, i), n)..., :total)
        vals = (ntuple(i -> float(u_end[i]), n)..., float(sum(u_end)))
        return NamedTuple{names}(vals)
    else
        throw(ArgumentError("default SciML outcome expects a real scalar or real vector final state, " *
                            "got $(typeof(u_end)); define a custom `DecisionAnalysis.outcome` method"))
    end
end

end # module
