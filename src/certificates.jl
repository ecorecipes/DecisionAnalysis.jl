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
    export_garbling_witness(io::IO, e′::Experiment, e::Experiment)
    export_garbling_witness(path::AbstractString, e′, e)

Write the proof that `e′` is a garbling of `e`: the two likelihoods and the matrix
`G` found by [`is_garbling_of`](@ref), as JSON with schema `"garbling-witness/1"`
([`GARBLING_WITNESS_SCHEMA`](@ref)). The object holds `schema`, `experiment` (the
likelihood of `e`, rows = signals), `garbled` (the likelihood of `e′`) and `garbling`
(`G[z′][z] = P(z′ | z)`, rows = signals of `e′`, columns = signals of `e`), every
number a string `"n/d"` or `"n"` as in [`export_decision_certificate`](@ref).

Both likelihoods must be of `Rational` or integer type
([`InexactCertificateError`](@ref) otherwise): the programme is then solved exactly
and no rounding is involved. Before writing, the witness is checked in exact
arithmetic: `G` is nonnegative, every column of `G` sums to one, and `G * L_e`
equals `L_e′` entry for entry. [`WitnessError`](@ref) is thrown when `e′` is not a
garbling of `e`, or if that check fails.

The path form returns the path; the `IO` form returns `nothing`.
"""
function export_garbling_witness(io::IO, e′::Experiment, e::Experiment)
    _require_exact("likelihood", e.likelihood)
    _require_exact("likelihood of the garbled experiment", e′.likelihood)
    G = is_garbling_of(e′, e; atol=0)
    G === nothing &&
        throw(WitnessError("the first experiment is not a garbling of the second: " *
                           "the deficiency is positive and there is no witness"))
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

function export_garbling_witness(path::AbstractString, e′::Experiment, e::Experiment)
    return _to_path(io -> export_garbling_witness(io, e′, e), path)
end

"""
    export_eadmissible_witness(io::IO, problem::DecisionProblem, action)
    export_eadmissible_witness(path::AbstractString, problem, action)

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
([`InexactCertificateError`](@ref) otherwise): the E-admissibility programme is then
solved exactly and no rounding is involved. Before writing, the witness is checked
in exact arithmetic: the weights are nonnegative and sum to one, the prior is their
mixture, and no action has a higher expected utility under it.
[`WitnessError`](@ref) is thrown when the action is not E-admissible, or if that
check fails; an `ArgumentError` when the action is not a feasible action.

The path form returns the path; the `IO` form returns `nothing`.
"""
function export_eadmissible_witness(io::IO, problem::DecisionProblem,
                                    action::Union{Symbol,AbstractString})
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
    slack, weights = _e_admissibility_lp(_expectations(Q, priors), a)
    iszero(slack) ||
        throw(WitnessError("$label is not E-admissible: under every prior in the " *
                           "credal set some other action has a higher expected utility"))
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
                                    action::Union{Symbol,AbstractString})
    return _to_path(io -> export_eadmissible_witness(io, problem, action), path)
end
