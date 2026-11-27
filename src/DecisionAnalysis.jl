"""
    DecisionAnalysis

Standard decision analysis on finite problems: a
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

A payoff table can be assembled from tables of outcome counts ([`combine_payoffs`](@ref),
[`individuals_payoff`](@ref), [`infection_payoff`](@ref), and `+`, `-`, scalar `*` on
tables with matching labels).

Every linear programme is stated once as a JuMP model by an exported builder
([`eadmissibility_model`](@ref), [`garbling_model`](@ref), [`deficiency_model`](@ref),
[`evpi_envelope_model`](@ref), [`mixed_dominance_model`](@ref),
[`separating_payoff_model`](@ref)) and solved by [`solve_program`](@ref): by default with
the built-in exact simplex, otherwise with the `optimizer` given, exact or floating
point. No solver is trusted: witnesses are verified in exact arithmetic before they
are returned.

Sample-based analysis sits on the same definitions. A [`SampleBelief`](@ref) holds
draws of named parameters with weights; an [`OutcomeTable`](@ref) holds what a model
produced by draw × action × replicate, built from any table of runs or by
[`outcomes`](@ref) through the adapter verbs ([`with_parameters`](@ref),
[`apply_action`](@ref), [`simulate`](@ref), [`outcome`](@ref)); a utility counted in
individuals ([`IndividualsUtility`](@ref), [`InfectionUtility`](@ref)) turns it into a
payoff table with its Monte Carlo standard errors kept apart ([`payoff_table`](@ref)).
[`evpi`](@ref), [`evppi`](@ref) and [`evsi`](@ref) called with a method
([`SampleAverage`](@ref), [`Grouping`](@ref), [`StrongOakley`](@ref),
[`Sadatsafavi`](@ref), [`SplineRegression`](@ref), [`SummaryStatistics`](@ref),
[`ImportanceSampling`](@ref), [`GaussianApproximation`](@ref)) return a
[`VOIEstimate`](@ref): the value, its standard error and its selection biases, never
folded together. [`policy_regions`](@ref), [`weight_sweep`](@ref) and
[`rank_reversals`](@ref) show how the choice moves with a weight.

Finite problems are exact; sample-based results are estimates of the same quantities.
The package depends
on JuMP for stating linear programmes (and on no solver), and never on Catlab, GATlab
or ACSets. FiniteKernels is optional: when it is loaded, a [`FiniteBelief`](@ref) can be
built from one of its states and an [`Experiment`](@ref) from one of its kernels. GAM.jl is an optional regression
backend through a weak-dependency extension ([`GAMRegression`](@ref)). It
defines none of `compose`, `otimes`, `id`, `mcopy`, `delete`, `braid`, `dom`, `codom`,
and exports no `regret` or `regrets`.
"""
module DecisionAnalysis

using JuMP: JuMP
using LinearAlgebra: ColumnNorm, Diagonal, I, Symmetric, cholesky, dot, eigen, issuccess,
                     logdet, qr, tr
using Random: AbstractRNG, Xoshiro, randn!
using SpecialFunctions: erfc
using Statistics: mean
using Logging: NullLogger, with_logger
using Tables: Tables

# Exceptions (ADR 0013)
export DecisionAnalysisError, InvalidBeliefError, InvalidPayoffError,
       InvalidExperimentError, LabelMismatchError, BeliefFormError,
       NoFeasibleActionError, InvalidBaselineError, MissingBaselineError,
       InexactCertificateError, WitnessError, LinearProgramError,
       ExactVerificationError, InvalidOutcomeError, MissingExtensionError
# Beliefs
export AbstractBelief, PointBelief, FiniteBelief, SampleBelief, CredalSet, ScenarioSet,
       state_labels, probabilities, extreme_points, random_belief, random_credal_set,
       effective_sample_size
# Outcome tables, adapter verbs and utilities
export OutcomeTable, outcomes, outcome_names, replicate_counts, run_counts, failed_runs,
       combine_actions,
       with_parameters, apply_action, simulate, outcome, parameter_draws, draw_weights,
       AbstractUtility, IndividualsUtility, InfectionUtility, FunctionUtility,
       payoff_table, single_replicate, crn_correlation, LinearUtility,
       ExponentialUtility, PowerUtility, AdditiveUtility, MultiplicativeUtility,
       inverse_utility, certainty_equivalent, risk_premium
# Payoff tables and problems
export PayoffTable, action_labels, random_payoff_table, combine_payoffs,
       individuals_payoff, infection_payoff, DecisionProblem,
       feasible_actions, feasible_payoff, restrict_scenarios, gain_over_baseline,
       expected_gain_over_baseline
# Aggregators
export AbstractAggregator, Expectation, WorstCase, BestCase, Hurwicz, Laplace,
       LowerExpectation, UpperExpectation, AlphaMaxmin, CVaR, Quantile, aggregate
# Regret (never a bare `regret` or `regrets`)
export RegretTable, regret_table, relative_regret_table, expected_regret, max_regret
# Rules
export AbstractTransform, IdentityTransform, RegretTransform, RelativeRegretTransform,
       RobustnessMetric, robustness, maximin_metric, maximax_metric, hurwicz_metric,
       laplace_metric, minimax_regret_metric, AbstractSelector, Optimize, Threshold,
       AbstractRule, DecisionRule, MaxExpectedUtility, Maximin, Maximax, MinimaxRegret,
       GammaMaximin, Satisfice, EAdmissible, Maximal, IntervalDominance, ParetoFront,
       DecisionResult, first_choice, decide
# Experiments
export Experiment, signal_labels, random_experiment, gaussian_experiment,
       BayesianInversion, bayes_invert, garble, is_garbling_of, garbling_result,
       GarblingResult, deficiency
# Value of information
export act_now_value, value_with_information, evpi, evppi, hypothesis_partition,
       two_stage_value, evsi, enbs,
       information_can_change_decision, RobustVOI,
       robust_evpi, robust_evsi, voi_profile, expected_variance_reduction,
       optimal_design, normal_form, survey_or_act, select_under_budget,
       value_of_waiting, adaptive_management
# Sample-based value of information
export VOIEstimate, AbstractVOIMethod, AbstractRegression, SampleAverage, Grouping,
       StrongOakley, Sadatsafavi, SplineRegression, GAMRegression,
       GaussianProcessRegression, MARSRegression,
       fitted_conditional_means, SummaryStatistics, ImportanceSampling,
       importance_loglikelihoods, GaussianApproximation, variance_components,
       evppi_nested, MomentMatching, AbstractStudyDesign, BinomialStudy,
       TwoArmBinomialStudy, NormalStudy, simulate_study, study_loglikelihood,
       study_posterior, variance_reduction_table, equivalent_binomial_sample_size,
       population_value, enbs_curve, optimal_sample_size, regression_check,
       compare_regressions, ScenarioAxis, decision_axis, uncertainty_axis, ScenarioDesign,
       design_class, scenarios, design_actions, design_draws, design_belief,
       fractional_design, factor_effects, scenario_forecast, decision_robustness,
       expected_utility_estimates, expected_gain_estimates, conditional_problem,
       probability_optimal
# Sensitivity
export policy_regions, one_way_thresholds, weight_sweep, rank_reversals
# Linear programmes as JuMP models
export DecisionProgram, ProgramSolution, solve_program, eadmissibility_model,
       garbling_model, deficiency_model, evpi_envelope_model, mixed_dominance_model,
       separating_payoff_model
# Fidelity
export total_variation, decision_loss_bound, payoff_range, largest_action_range,
       largest_regret, deficiency_value_bound
# Reports and certificates
export DecisionReport, CERTIFICATE_SCHEMA, export_decision_certificate,
       GARBLING_WITNESS_SCHEMA, export_garbling_witness, EADMISSIBLE_WITNESS_SCHEMA,
       export_eadmissible_witness, NOT_EADMISSIBLE_WITNESS_SCHEMA,
       export_not_eadmissible_witness, NOT_GARBLING_WITNESS_SCHEMA,
       export_not_garbling_witness

include("errors.jl")
include("lp.jl")
include("beliefs.jl")
include("payoff.jl")
include("adapters.jl")
include("outcomes.jl")
include("problem.jl")
include("utilities.jl")
include("aggregators.jl")
include("regret.jl")
include("rules.jl")
include("experiments.jl")
include("voi.jl")
include("voi_sample.jl")
include("gaussian_process.jl")
include("mars.jl")
include("studies.jl")
include("voi_model.jl")
include("sensitivity.jl")
include("programs.jl")
include("solve.jl")
include("fidelity.jl")
include("design.jl")
include("survey.jl")
include("adaptive.jl")
include("estimation.jl")
include("scenarios.jl")
include("report.jl")
include("certificates.jl")
include("show.jl")

end # module
