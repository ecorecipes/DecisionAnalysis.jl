# Certificates for Lean: one decision problem and the numbers the package claims for
# it, as exact rationals in JSON. The schema is a contract with the Lean checker and is
# written down in docs/src/certificates.md; change neither side alone.

"""
    CERTIFICATE_SCHEMA

The schema identifier written into every certificate, `"decision-certificate/1"`.
"""
const CERTIFICATE_SCHEMA = "decision-certificate/1"

_rational_string(x::Integer) = string(x)
function _rational_string(x::Rational)
    return isone(denominator(x)) ? string(numerator(x)) :
           string(numerator(x), "/", denominator(x))
end

function _json_string(s::AbstractString)
    io = IOBuffer()
    print(io, '"')
    for c in s
        if c == '"'
            print(io, "\\\"")
        elseif c == '\\'
            print(io, "\\\\")
        elseif c == '\n'
            print(io, "\\n")
        elseif c == '\r'
            print(io, "\\r")
        elseif c == '\t'
            print(io, "\\t")
        elseif c < ' '
            print(io, "\\u", string(UInt16(c); base=16, pad=4))
        else
            print(io, c)
        end
    end
    print(io, '"')
    return String(take!(io))
end

_json(x::Real) = _json_string(_rational_string(x))
_json(x::Symbol) = _json_string(String(x))
_json(v::AbstractVector) = "[" * join((_json(x) for x in v), ", ") * "]"
_json(M::AbstractMatrix) = "[" * join((_json(M[i, :]) for i in axes(M, 1)), ", ") * "]"
# Zero-based indices are JSON numbers, not rationals-as-strings.
_json_indices(v::AbstractVector{<:Integer}) = "[" * join(v .- 1, ", ") * "]"

function _write_object(io::IO, pairs, indent::Int)
    pad = " "^indent
    println(io, "{")
    for (i, (key, value)) in enumerate(pairs)
        print(io, pad, "  ", _json_string(key), ": ", value)
        println(io, i == length(pairs) ? "" : ",")
    end
    print(io, pad, "}")
    return nothing
end

function _require_exact(what::AbstractString, A::AbstractArray)
    _is_exact(eltype(A)) || throw(InexactCertificateError(what, eltype(A)))
    return nothing
end

# A certificate states exact numbers for a finite problem; a sample is an estimate.
function _no_sample(problem::DecisionProblem, operation::Symbol)
    problem.belief isa SampleBelief &&
        throw(BeliefFormError(operation, :SampleBelief,
                              "a certificate states exact numbers for a finite problem " *
                              "and a sample gives estimates; if the draws are the whole " *
                              "population, build the problem with FiniteBelief(sample)"))
    return nothing
end

"""
    export_decision_certificate(io::IO, problem::DecisionProblem; experiment=nothing)
    export_decision_certificate(path::AbstractString, problem; experiment=nothing)

Write a certificate for `problem`: the data of the problem and the numbers this
package claims for it, as JSON with schema `"decision-certificate/1"`
([`CERTIFICATE_SCHEMA`](@ref)), for an independent checker (the Lean development)
to recompute in exact arithmetic.

The problem must be exact: payoffs, prior and likelihood of `Rational` or integer
type, otherwise [`InexactCertificateError`](@ref) is thrown. It needs a single prior
([`FiniteBelief`](@ref) or [`PointBelief`](@ref)). Only the feasible actions are
written, and action indices in the claims refer to that list.

Every number is a string `"n/d"` in lowest terms, or `"n"` when the denominator is
one; action indices are JSON integers counted **from zero**. The object holds
`schema`, `actions`, `states`, `payoff` (rows = actions), `prior`, with an experiment
`signals` and `likelihood` (rows = signals), and `claims`: `expected_utility`,
`act_now_value`, `bayes_actions`, `regret` (rows = actions), `expected_regret`,
`max_regret`, `minimax_regret_actions`, `evpi`, and with an experiment
`value_with_information` and `evsi`. The manual page "Certificates" gives the
definition of each claim.

The path form returns the path; the `IO` form returns `nothing`.
"""
function export_decision_certificate(io::IO, problem::DecisionProblem;
                                     experiment::Union{Nothing,Experiment}=nothing)
    _no_sample(problem, :export_decision_certificate)
    prior = _prior(problem, :export_decision_certificate)
    Q = _feasible_Q(problem)
    _require_exact("payoff", Q)
    _require_exact("prior", prior)
    if experiment !== nothing
        _check_states(experiment, problem.payoff.states, "payoff table")
        _require_exact("likelihood", experiment.likelihood)
    end
    # Exact ties only: the checker compares rationals for equality.
    bayes = decide(problem, MaxExpectedUtility(); atol=0, rtol=0)
    minimax = decide(problem, MinimaxRegret(); atol=0, rtol=0)
    actions = feasible_actions(problem)
    indices(result) = findall(in(result.chosen), actions)
    claims = Pair{String,String}["expected_utility" => _json(bayes.scores),
                                 "act_now_value" => _json(act_now_value(problem)),
                                 "bayes_actions" => _json_indices(indices(bayes)),
                                 "regret" => _json(_regret_matrix(Q)),
                                 "expected_regret" => _json(_regret_matrix(Q) * prior),
                                 "max_regret" => _json(vec(maximum(_regret_matrix(Q);
                                                                   dims=2))),
                                 "minimax_regret_actions" => _json_indices(indices(minimax)),
                                 "evpi" => _json(evpi(problem))]
    if experiment !== nothing
        push!(claims,
              "value_with_information" => _json(value_with_information(problem,
                                                                       experiment)),
              "evsi" => _json(evsi(problem, experiment)))
    end
    top = Pair{String,String}["schema" => _json_string(CERTIFICATE_SCHEMA),
                              "actions" => _json(actions),
                              "states" => _json(problem.payoff.states),
                              "payoff" => _json(Q), "prior" => _json(prior)]
    if experiment !== nothing
        push!(top, "signals" => _json(experiment.signals),
              "likelihood" => _json(experiment.likelihood))
    end
    claims_io = IOBuffer()
    _write_object(claims_io, claims, 2)
    push!(top, "claims" => String(take!(claims_io)))
    _write_object(io, top, 0)
    println(io)
    return nothing
end

function export_decision_certificate(path::AbstractString, problem::DecisionProblem;
                                     experiment::Union{Nothing,Experiment}=nothing)
    open(path, "w") do io
        return export_decision_certificate(io, problem; experiment)
    end
    return path
end

# Witnesses
# ---------
#
# What the linear programmes find, written so that a checker can confirm the answer
# without trusting the solver. Both are verified here in exact arithmetic before a byte
# is written.

"""
    GARBLING_WITNESS_SCHEMA

The schema identifier of a garbling witness, `"garbling-witness/1"`.
"""
const GARBLING_WITNESS_SCHEMA = "garbling-witness/1"

"""
    EADMISSIBLE_WITNESS_SCHEMA

The schema identifier of an E-admissibility witness, `"e-admissible-witness/1"`.
"""
const EADMISSIBLE_WITNESS_SCHEMA = "e-admissible-witness/1"

function _to_path(f, path::AbstractString)
    open(f, path, "w")
    return path
end

"""
    export_garbling_witness(io::IO, e′::Experiment, e::Experiment; optimizer=nothing,
                            rounding_tol=(1e-8, 1e-10, 1e-12))
    export_garbling_witness(path::AbstractString, e′, e; optimizer=nothing,
                            rounding_tol=(1e-8, 1e-10, 1e-12))

Write the proof that `e′` is a garbling of `e`: the two likelihoods and the matrix
`G` found by [`is_garbling_of`](@ref), as JSON with schema `"garbling-witness/1"`
([`GARBLING_WITNESS_SCHEMA`](@ref)). The object holds `schema`, `experiment` (the
likelihood of `e`, rows = signals), `garbled` (the likelihood of `e′`) and `garbling`
(`G[z′][z] = P(z′ | z)`, rows = signals of `e′`, columns = signals of `e`), every
number a string `"n/d"` or `"n"` as in [`export_decision_certificate`](@ref).

Both likelihoods must be of `Rational` or integer type
([`InexactCertificateError`](@ref) otherwise). By default the programme is solved
exactly and no rounding is involved; `optimizer` and `rounding_tol` choose another
route (see [`solve_program`](@ref)). The solver is never trusted: whatever found the
witness, it is checked here in exact arithmetic before a byte is written: `G` is
nonnegative, every column of `G` sums to one, and `G * L_e` equals `L_e′` entry for
entry. [`WitnessError`](@ref) is thrown when `e′` is not a garbling of `e`, or if
that check fails; [`ExactVerificationError`](@ref) when a floating-point optimiser
reports a garbling that does not round to an exact one.

The path form returns the path; the `IO` form returns `nothing`.
"""
function export_garbling_witness(io::IO, e′::Experiment, e::Experiment; optimizer=nothing,
                                 rounding_tol=_ROUNDING_TOLS)
    _require_exact("likelihood", e.likelihood)
    _require_exact("likelihood of the garbled experiment", e′.likelihood)
    result = garbling_result(e′, e; atol=0, optimizer, rounding_tol)
    G = result.witness
    G === nothing &&
        throw(WitnessError("the first experiment is not a garbling of the second" *
                           (result.exact ? "" :
                            ", to the tolerance of the floating-point optimiser") *
                           ": there is no witness"))
    (all(>=(0), G) && all(isone, sum(G; dims=1)) && G * e.likelihood == e′.likelihood) ||
        throw(WitnessError("the garbling matrix found did not pass the exact check " *
                           "(nonnegative, columns summing to one, product equal to " *
                           "the garbled likelihood)"))
    _write_object(io,
                  Pair{String,String}["schema" => _json_string(GARBLING_WITNESS_SCHEMA),
                                      "experiment" => _json(e.likelihood),
                                      "garbled" => _json(e′.likelihood),
                                      "garbling" => _json(G)], 0)
    println(io)
    return nothing
end

function export_garbling_witness(path::AbstractString, e′::Experiment, e::Experiment;
                                 kw...)
    return _to_path(io -> export_garbling_witness(io, e′, e; kw...), path)
end

"""
    export_eadmissible_witness(io::IO, problem::DecisionProblem, action;
                               optimizer=nothing, rounding_tol=(1e-8, 1e-10, 1e-12))
    export_eadmissible_witness(path::AbstractString, problem, action; optimizer=nothing,
                               rounding_tol=(1e-8, 1e-10, 1e-12))

Write the proof that `action` (a label) is E-admissible in `problem`: a prior in the
credal set under which it maximises expected utility, with the convex weights that
place that prior in the set, as JSON with schema `"e-admissible-witness/1"`
([`EADMISSIBLE_WITNESS_SCHEMA`](@ref)). The object holds `schema`, `payoff` (rows =
the feasible actions), `priors` (the generating points of the credal set, one per
row), `weights` (one per generating point), `prior` (the mixture
`Σ_k weights[k] · priors[k]`) and `action` (a JSON integer counted from zero among
the feasible actions); every other number is a string `"n/d"` or `"n"`.

The belief is a [`CredalSet`](@ref), or a single prior read as a one-point set. The
payoffs and priors must be of `Rational` or integer type
([`InexactCertificateError`](@ref) otherwise). By default the E-admissibility
programme is solved exactly and no rounding is involved; `optimizer` and
`rounding_tol` choose another route (see [`solve_program`](@ref)). The solver is
never trusted: whatever found the weights, the witness is checked here in exact
arithmetic before a byte is written: the weights are nonnegative and sum to one, the
prior is their mixture, and no action has a higher expected utility under it.
[`WitnessError`](@ref) is thrown when the action is not E-admissible, or if that
check fails; [`ExactVerificationError`](@ref) when a floating-point optimiser reports
it E-admissible with weights that do not round to a witness; an `ArgumentError` when
the action is not a feasible action.

The path form returns the path; the `IO` form returns `nothing`.
"""
function export_eadmissible_witness(io::IO, problem::DecisionProblem,
                                    action::Union{Symbol,AbstractString};
                                    optimizer=nothing, rounding_tol=_ROUNDING_TOLS)
    _no_sample(problem, :export_eadmissible_witness)
    label = Symbol(action)
    actions = feasible_actions(problem)
    a = findfirst(==(label), actions)
    a === nothing &&
        throw(ArgumentError("$label is not among the feasible actions $actions"))
    E, v = _expectation_matrix(problem, EAdmissible())
    v.form === :PointBelief &&
        (v = _View(_feasible_Q(problem), [_prior(problem, :export_eadmissible_witness)],
                   v.form))
    Q, priors = v.Q, v.priors
    _require_exact("payoff", Q)
    foreach(q -> _require_exact("credal set", q), priors)
    admitted, weights, exact, _ = _eadmissible_witness(_expectations(Q, priors), a;
                                                       optimizer, rounding_tol)
    admitted ||
        throw(WitnessError("$label is not E-admissible" *
                           (exact ? "" :
                            ", to the tolerance of the floating-point optimiser") *
                           ": under every prior in the credal set some other action " *
                           "has a higher expected utility"))
    prior = [sum(weights[k] * priors[k][j] for k in eachindex(weights))
             for j in axes(Q, 2)]
    eu = Q * prior
    (all(>=(0), weights) && isone(sum(weights)) && eu[a] == maximum(eu)) ||
        throw(WitnessError("the weights found for $label did not pass the exact check " *
                           "(nonnegative, summing to one, the action optimal under " *
                           "the mixture)"))
    _write_object(io,
                  Pair{String,String}["schema" => _json_string(EADMISSIBLE_WITNESS_SCHEMA),
                                      "payoff" => _json(Q),
                                      "priors" => _json(permutedims(reduce(hcat, priors))),
                                      "weights" => _json(weights),
                                      "prior" => _json(prior), "action" => string(a - 1)],
                  0)
    println(io)
    return nothing
end

function export_eadmissible_witness(path::AbstractString, problem::DecisionProblem,
                                    action::Union{Symbol,AbstractString}; kw...)
    return _to_path(io -> export_eadmissible_witness(io, problem, action; kw...), path)
end

# Certified "no" answers
# ----------------------

"""
    NOT_EADMISSIBLE_WITNESS_SCHEMA

The schema identifier, `"not-e-admissible-witness/1"`, written by
[`export_not_eadmissible_witness`](@ref).
"""
const NOT_EADMISSIBLE_WITNESS_SCHEMA = "not-e-admissible-witness/1"

"""
    NOT_GARBLING_WITNESS_SCHEMA

The schema identifier, `"not-garbling-witness/1"`, written by
[`export_not_garbling_witness`](@ref).
"""
const NOT_GARBLING_WITNESS_SCHEMA = "not-garbling-witness/1"

# Whether the mixed action `μ` strictly beats action `a` at every generating point,
# exactly. `E[b, k]` is the expected utility of `b` at point `k`.
function _strictly_dominates(E::AbstractMatrix, μ::AbstractVector, a::Int)
    return all(>=(0), μ) && isone(sum(μ)) &&
           all(k -> sum(μ[b] * E[b, k] for b in axes(E, 1)) > E[a, k], axes(E, 2))
end

# A mixed action that strictly beats `a` at every generating point, verified exactly,
# or `nothing` when none was found (in particular when `a` is E-admissible).
function _mixed_dominator(E::AbstractMatrix, a::Int; optimizer=nothing,
                          rounding_tol=_ROUNDING_TOLS)
    sol = solve_program(_mixed_dominance_program(E, a); optimizer, rounding_tol)
    sol.status === :optimal || throw(LinearProgramError(:mixed_dominance, sol.status))
    μ = sol.exact ? sol.values.μ : _clean_weights(sol.values.μ)
    μ === nothing && return nothing
    return _strictly_dominates(_BigRat.(E), μ, a) ? collect(_BigRat, μ) : nothing
end

# value(e) = Σ_z max_a Σ_θ p(θ) L[z, θ] Q[a, θ], the joint form.
function _joint_value(Q::AbstractMatrix, p::AbstractVector, L::AbstractMatrix)
    return sum(maximum(Q * (p .* view(L, z, :))) for z in axes(L, 1))
end

# A payoff table (one action per signal of e′) under which, with a uniform prior, e′
# is worth strictly more than e, verified exactly; or `nothing` when none was found
# (in particular when e′ is a garbling of e).
function _separating_payoff(e′::Experiment, e::Experiment; optimizer=nothing,
                            rounding_tol=_ROUNDING_TOLS)
    sol = solve_program(separating_payoff_model(e′, e); optimizer, rounding_tol)
    sol.status === :optimal || throw(LinearProgramError(:separating_payoff, sol.status))
    V = Matrix{_BigRat}(_BigRat.(sol.values.V))
    S = size(V, 2)
    prior = fill(_BigRat(1, S), S)
    _joint_value(V, prior, _BigRat.(e′.likelihood)) >
    _joint_value(V, prior, _BigRat.(e.likelihood)) || return nothing
    return V, prior
end

"""
    export_not_eadmissible_witness(io::IO, problem::DecisionProblem, action;
                                   optimizer=nothing, rounding_tol=(1e-8, 1e-10, 1e-12))
    export_not_eadmissible_witness(path::AbstractString, problem, action; kw...)

Write the proof that `action` (a label) is **not** E-admissible in `problem`: a mixed
action that is strictly better than it at every generating point of the credal set,
and therefore under every prior of the set, as JSON with schema
`"not-e-admissible-witness/1"` ([`NOT_EADMISSIBLE_WITNESS_SCHEMA`](@ref)). The object
holds `schema`, `payoff` (rows = the feasible actions), `priors` (the generating
points, one per row), `action` (a zero-based index into the feasible actions),
`mixed_action` (a probability vector over the feasible actions) and the labels
`actions` and `states`.

The mixed action is found by the programme of [`mixed_dominance_model`](@ref), the
dual of the E-admissibility programme, by any route of [`solve_program`](@ref). The
solver is never trusted: the mixture is checked here in exact arithmetic before a
byte is written (nonnegative, summing to one, and strictly better than `action` at
every generating point), so a "no" found by a floating-point optimiser becomes an
exact statement once its rounded witness passes. Such a witness exists whenever the
action is not E-admissible.

The payoffs and the credal set must be exact ([`InexactCertificateError`](@ref)
otherwise). [`WitnessError`](@ref) is thrown when no witness passes the check, which
is the case when the action is E-admissible. The path form returns the path; the
`IO` form returns `nothing`.
"""
function export_not_eadmissible_witness(io::IO, problem::DecisionProblem,
                                        action::Union{Symbol,AbstractString};
                                        optimizer=nothing, rounding_tol=_ROUNDING_TOLS)
    _no_sample(problem, :export_not_eadmissible_witness)
    label = Symbol(action)
    actions = feasible_actions(problem)
    a = findfirst(==(label), actions)
    a === nothing &&
        throw(ArgumentError("$label is not among the feasible actions $actions"))
    _, v = _expectation_matrix(problem, EAdmissible())
    v.form === :PointBelief &&
        (v = _View(_feasible_Q(problem),
                   [_prior(problem, :export_not_eadmissible_witness)], v.form))
    Q, priors = v.Q, v.priors
    _require_exact("payoff", Q)
    foreach(q -> _require_exact("credal set", q), priors)
    E = _expectations(Q, priors)
    μ = _mixed_dominator(E, a; optimizer, rounding_tol)
    μ === nothing &&
        throw(WitnessError("no mixed action strictly better than $label at every " *
                           "generating point passed the exact check; the action is " *
                           "E-admissible, or the optimiser's answer could not be " *
                           "verified"))
    _write_object(io,
                  Pair{String,String}["schema" => _json_string(NOT_EADMISSIBLE_WITNESS_SCHEMA),
                                      "actions" => _json(actions),
                                      "states" => _json(problem.payoff.states),
                                      "payoff" => _json(Q),
                                      "priors" => "[" *
                                                  join((_json(q) for q in priors), ", ") *
                                                  "]", "action" => string(a - 1),
                                      "mixed_action" => _json(μ)], 0)
    println(io)
    return nothing
end

function export_not_eadmissible_witness(path::AbstractString, problem::DecisionProblem,
                                        action::Union{Symbol,AbstractString}; kw...)
    return _to_path(io -> export_not_eadmissible_witness(io, problem, action; kw...), path)
end

"""
    export_not_garbling_witness(io::IO, e′::Experiment, e::Experiment; optimizer=nothing,
                                rounding_tol=(1e-8, 1e-10, 1e-12))
    export_not_garbling_witness(path::AbstractString, e′, e; kw...)

Write the proof that `e′` is **not** a garbling of `e`: a decision problem in which
deciding after `e′` is worth strictly more than deciding after `e`, which a garbling
can never be. JSON with schema `"not-garbling-witness/1"`
([`NOT_GARBLING_WITNESS_SCHEMA`](@ref)): `schema`, `experiment` (the likelihood of
`e`, rows = signals), `garbled` (the likelihood of `e′`), `payoff` (the separating
payoff table, one action per signal of `e′`) and `prior` (uniform, so that every
state is visible).

The payoff table is found by the programme of [`separating_payoff_model`](@ref), by
any route of [`solve_program`](@ref). The solver is never trusted: before a byte is
written, both values are recomputed in exact arithmetic in the joint form of
[`value_with_information`](@ref) and the one of `e′` is checked to be strictly
larger. Any table that passes is a valid witness, whatever rounding it went through,
so a "no" found by a floating-point optimiser becomes an exact statement. Such a
witness exists whenever `e′` is not a garbling of `e`.

Both likelihoods must be exact ([`InexactCertificateError`](@ref) otherwise).
[`WitnessError`](@ref) is thrown when no table passes the check, which is the case
when `e′` is a garbling of `e`. The path form returns the path; the `IO` form
returns `nothing`.
"""
function export_not_garbling_witness(io::IO, e′::Experiment, e::Experiment;
                                     optimizer=nothing, rounding_tol=_ROUNDING_TOLS)
    _require_exact("likelihood", e.likelihood)
    _require_exact("likelihood of the garbled experiment", e′.likelihood)
    found = _separating_payoff(e′, e; optimizer, rounding_tol)
    found === nothing &&
        throw(WitnessError("no payoff table under which the first experiment is worth " *
                           "strictly more than the second passed the exact check; the " *
                           "first is a garbling of the second, or the optimiser's " *
                           "answer could not be verified"))
    V, prior = found
    _write_object(io,
                  Pair{String,String}["schema" => _json_string(NOT_GARBLING_WITNESS_SCHEMA),
                                      "experiment" => _json(e.likelihood),
                                      "garbled" => _json(e′.likelihood),
                                      "payoff" => _json(V), "prior" => _json(prior)], 0)
    println(io)
    return nothing
end

function export_not_garbling_witness(path::AbstractString, e′::Experiment, e::Experiment;
                                     kw...)
    return _to_path(io -> export_not_garbling_witness(io, e′, e; kw...), path)
end
