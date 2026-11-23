"""
    DecisionAnalysis

Standard decision analysis on finite problems, with no category theory required: a
payoff table, a belief about its states, rules for choosing, Savage regret, and the
value of information, all exact when the inputs are rational.

A [`PayoffTable`](@ref) holds the conditional expected utility `Q[a, θ]` of each
action in each state (rows = actions, columns = states). A belief is a
[`PointBelief`](@ref), a [`FiniteBelief`](@ref), a [`CredalSet`](@ref) or a
[`ScenarioSet`](@ref), and a [`DecisionProblem`](@ref) pairs the two.
[`decide`](@ref) applies a rule: [`MaxExpectedUtility`](@ref), [`Maximin`](@ref),
[`Maximax`](@ref), [`MinimaxRegret`](@ref), [`GammaMaximin`](@ref),
[`Satisfice`](@ref), or the set-valued [`EAdmissible`](@ref), [`Maximal`](@ref),
[`IntervalDominance`](@ref) and [`ParetoFront`](@ref). A scalar rule is an aggregator,
an optional transform and a selector ([`DecisionRule`](@ref)), and the same pieces
without the selector are a [`RobustnessMetric`](@ref).

An [`Experiment`](@ref) is a likelihood matrix `L[z, θ] = P(z | θ)` (rows = signals,
columns = states); [`bayes_invert`](@ref), [`garble`](@ref),
[`is_garbling_of`](@ref) and [`deficiency`](@ref) work on experiments, and
[`evpi`](@ref), [`evppi`](@ref), [`evsi`](@ref) and [`enbs`](@ref) value information
exactly. [`DecisionReport`](@ref) puts every applicable rule side by side, and
[`export_decision_certificate`](@ref), [`export_garbling_witness`](@ref) and
[`export_eadmissible_witness`](@ref) write a problem, its claimed numbers and the
witnesses found by linear programme for the Lean checker.

Version 0.1 is finite and exact; sample-based methods come later. The package depends
on FiniteKernels for kernels and states and never on Catlab, GATlab or ACSets. It
defines none of `compose`, `otimes`, `id`, `mcopy`, `delete`, `braid`, `dom`, `codom`,
and exports no `regret` or `regrets`.
"""
module DecisionAnalysis

using FiniteKernels: FiniteKernels, FiniteKernel, FiniteSpace, axis_names, factors,
                     joint_states, kernel_matrix
using LinearAlgebra: dot
using Random: AbstractRNG
using Statistics: mean
using Tables: Tables

# Exceptions (ADR 0013)
export DecisionAnalysisError, InvalidBeliefError, InvalidPayoffError,
       InvalidExperimentError, LabelMismatchError, BeliefFormError,
       NoFeasibleActionError, InvalidBaselineError, MissingBaselineError,
       InexactCertificateError, WitnessError, LinearProgramError
# Beliefs
export AbstractBelief, PointBelief, FiniteBelief, CredalSet, ScenarioSet, state_labels,
       probabilities, extreme_points, random_belief, random_credal_set
# Payoff tables and problems
export PayoffTable, action_labels, random_payoff_table, DecisionProblem,
       feasible_actions, feasible_payoff, restrict_scenarios, gain_over_baseline,
       expected_gain_over_baseline
# Aggregators
export AbstractAggregator, Expectation, WorstCase, BestCase, Hurwicz, Laplace,
       LowerExpectation, UpperExpectation, AlphaMaxmin, CVaR, Quantile, aggregate
# Regret (never a bare `regret` or `regrets`)
export regret_table, relative_regret_table, expected_regret, max_regret
# Rules
export AbstractTransform, IdentityTransform, RegretTransform, RelativeRegretTransform,
       RobustnessMetric, robustness, maximin_metric, maximax_metric, hurwicz_metric,
       laplace_metric, minimax_regret_metric, AbstractSelector, Optimize, Threshold,
       AbstractRule, DecisionRule, MaxExpectedUtility, Maximin, Maximax, MinimaxRegret,
       GammaMaximin, Satisfice, EAdmissible, Maximal, IntervalDominance, ParetoFront,
       DecisionResult, first_choice, decide
# Experiments
export Experiment, signal_labels, random_experiment, BayesianInversion, bayes_invert,
       garble, is_garbling_of, deficiency
# Value of information
export act_now_value, value_with_information, evpi, evppi, evsi, enbs, RobustVOI,
       robust_evpi, robust_evsi
# Fidelity
export total_variation, decision_loss_bound
# Reports and certificates
export DecisionReport, CERTIFICATE_SCHEMA, export_decision_certificate,
       GARBLING_WITNESS_SCHEMA, export_garbling_witness, EADMISSIBLE_WITNESS_SCHEMA,
       export_eadmissible_witness

include("errors.jl")
include("lp.jl")
include("beliefs.jl")
include("payoff.jl")
include("problem.jl")
include("aggregators.jl")
include("regret.jl")
include("rules.jl")
include("experiments.jl")
include("voi.jl")
include("fidelity.jl")
include("report.jl")
include("certificates.jl")
include("show.jl")

end # module
