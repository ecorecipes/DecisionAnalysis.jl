# Exceptions of DecisionAnalysis (ADR 0013). Every exception the package defines lives in
# this file, with its `showerror` method, and subtypes the root `DecisionAnalysisError`.
#
# - Library code never calls a bare `error("...")`.
# - Invalid arguments and keywords raise `ArgumentError`, not a type from this file (for
#   example a Hurwicz index outside [0, 1] or a partition of the wrong length).
# - A typed error from FiniteKernels passes through unchanged.

"""
    DecisionAnalysisError

Abstract supertype of every exception that DecisionAnalysis defines:
[`InvalidBeliefError`](@ref), [`InvalidPayoffError`](@ref),
[`InvalidExperimentError`](@ref), [`LabelMismatchError`](@ref),
[`BeliefFormError`](@ref), [`NoFeasibleActionError`](@ref),
[`InvalidBaselineError`](@ref), [`MissingBaselineError`](@ref),
[`InexactCertificateError`](@ref), [`WitnessError`](@ref) and
[`LinearProgramError`](@ref). It marks the
package that introduced an error, not a kind of failure. Invalid arguments and
keywords raise Base's `ArgumentError` instead, which is outside this root.
"""
abstract type DecisionAnalysisError <: Exception end

"""
    InvalidBeliefError(msg)

Thrown when a belief is not what its type requires: a probability vector with a
negative or non-finite entry or a sum different from one, an empty credal set, an
empty scenario set, or duplicate state labels.
"""
struct InvalidBeliefError <: DecisionAnalysisError
    msg::String
end

function Base.showerror(io::IO, e::InvalidBeliefError)
    return print(io, "InvalidBeliefError: ", e.msg)
end

"""
    InvalidPayoffError(msg)

Thrown when a [`PayoffTable`](@ref) is constructed from a matrix that is empty, has
a non-finite entry, or does not match its action and state labels.
"""
struct InvalidPayoffError <: DecisionAnalysisError
    msg::String
end

function Base.showerror(io::IO, e::InvalidPayoffError)
    return print(io, "InvalidPayoffError: ", e.msg)
end

"""
    InvalidExperimentError(msg)

Thrown when the likelihood of an [`Experiment`](@ref), or a garbling matrix passed to
[`garble`](@ref), is not column-stochastic (entries nonnegative, every column
summing to one) or does not match its labels.
"""
struct InvalidExperimentError <: DecisionAnalysisError
    msg::String
end

function Base.showerror(io::IO, e::InvalidExperimentError)
    return print(io, "InvalidExperimentError: ", e.msg)
end

"""
    LabelMismatchError(what, expected, got)

Thrown when two objects that must share labels do not: the states of a belief or an
experiment against the states of a payoff table, or the actions of two problems
compared by [`ParetoFront`](@ref). `what` names the pairing.
"""
struct LabelMismatchError <: DecisionAnalysisError
    what::String
    expected::Vector{Symbol}
    got::Vector{Symbol}
end

function Base.showerror(io::IO, e::LabelMismatchError)
    return print(io, "LabelMismatchError: ", e.what, ": expected ", e.expected, ", got ",
                 e.got)
end

"""
    BeliefFormError(operation, belief, msg)

Thrown when `operation` is not defined for the form of belief it was given, for
example an expectation over a [`ScenarioSet`](@ref) (which carries no probabilities)
or [`evpi`](@ref) on a [`CredalSet`](@ref) (use [`robust_evpi`](@ref)). `belief` is
the name of the belief type and `msg` says what to use instead.
"""
struct BeliefFormError <: DecisionAnalysisError
    operation::Symbol
    belief::Symbol
    msg::String
end

function Base.showerror(io::IO, e::BeliefFormError)
    return print(io, "BeliefFormError: ", e.operation, " is not defined for a ", e.belief,
                 ": ", e.msg)
end

"""
    NoFeasibleActionError(msg)

Thrown when the feasibility filter of a [`DecisionProblem`](@ref) leaves no action.
"""
struct NoFeasibleActionError <: DecisionAnalysisError
    msg::String
end

function Base.showerror(io::IO, e::NoFeasibleActionError)
    return print(io, "NoFeasibleActionError: ", e.msg)
end

"""
    InvalidBaselineError(baseline, actions)

Thrown when the `baseline` of a [`DecisionProblem`](@ref) is not one of the `actions`
of its payoff table.
"""
struct InvalidBaselineError <: DecisionAnalysisError
    baseline::Symbol
    actions::Vector{Symbol}
end

function Base.showerror(io::IO, e::InvalidBaselineError)
    return print(io, "InvalidBaselineError: the baseline ", e.baseline,
                 " is not among the actions ", e.actions)
end

"""
    MissingBaselineError(operation)

Thrown when `operation` ([`gain_over_baseline`](@ref) or
[`expected_gain_over_baseline`](@ref)) is asked of a [`DecisionProblem`](@ref) that
was built without a `baseline`.
"""
struct MissingBaselineError <: DecisionAnalysisError
    operation::Symbol
end

function Base.showerror(io::IO, e::MissingBaselineError)
    return print(io, "MissingBaselineError: ", e.operation, " needs a problem with a ",
                 "baseline action; pass `baseline=` to DecisionProblem")
end

"""
    InexactCertificateError(what, eltype)

Thrown by [`export_decision_certificate`](@ref) when part of the problem (`what`:
the payoff, the prior or the likelihood) has an element type that is not an exact
rational or integer. A certificate is checked in exact arithmetic, so a
floating-point problem cannot be certified.
"""
struct InexactCertificateError <: DecisionAnalysisError
    what::String
    eltype::Type
end

function Base.showerror(io::IO, e::InexactCertificateError)
    return print(io, "InexactCertificateError: the ", e.what, " has element type ",
                 e.eltype, "; a certificate needs Rational or Integer entries")
end

"""
    WitnessError(msg)

Thrown by [`export_garbling_witness`](@ref) and [`export_eadmissible_witness`](@ref)
when there is no witness to export (the experiment is not a garbling, the action is
not E-admissible) or when the witness found does not pass the exact check that is
made before anything is written.
"""
struct WitnessError <: DecisionAnalysisError
    msg::String
end

function Base.showerror(io::IO, e::WitnessError)
    return print(io, "WitnessError: ", e.msg)
end

"""
    LinearProgramError(problem, status)

Thrown when a linear programme that is feasible and bounded by construction is
reported otherwise by the built-in solver. `problem` names the programme
(`:e_admissibility`, `:deficiency`, `:evpi_envelope`) and `status` is `:infeasible`
or `:unbounded`. Seeing this error means a bug, not bad input.
"""
struct LinearProgramError <: DecisionAnalysisError
    problem::Symbol
    status::Symbol
end

function Base.showerror(io::IO, e::LinearProgramError)
    return print(io, "LinearProgramError: the ", e.problem, " programme was reported ",
                 e.status, ", which its construction rules out")
end
