# Sample-based analysis

Outcome tables from model runs, utilities counted in individuals, the estimators of
the value of information with their standard errors, and decision sensitivity.

## Outcome tables

```@docs
OutcomeTable
outcomes
outcome_names
replicate_counts
run_counts
failed_runs
combine_actions
crn_correlation
```

## Adapter verbs

```@docs
with_parameters
apply_action
simulate
outcome
parameter_draws
draw_weights
```

## Utilities

```@docs
AbstractUtility
IndividualsUtility
InfectionUtility
FunctionUtility
payoff_table
```

Utilities with a risk attitude and over several attributes are on [their own page](utilities.md).

## Estimates and their errors

```@docs
VOIEstimate
expected_utility_estimates
probability_optimal
expected_gain_estimates
variance_components
conditional_problem
```

## Methods

```@docs
AbstractVOIMethod
AbstractRegression
SampleAverage
Grouping
StrongOakley
Sadatsafavi
SplineRegression
GAMRegression
GaussianProcessRegression
fitted_conditional_means
```

## Studies for EVSI

```@docs
SummaryStatistics
ImportanceSampling
importance_loglikelihoods
GaussianApproximation
```

## Sensitivity

```@docs
policy_regions
one_way_thresholds
two_way_policy_regions
decision_curve
weight_sweep
rank_reversals
```
