# API Reference

```@docs
DecisionAnalysis
```

## Beliefs

```@docs
AbstractBelief
PointBelief
FiniteBelief
CredalSet
ScenarioSet
state_labels
probabilities
extreme_points
random_belief
random_credal_set
```

## Payoff tables and problems

```@docs
PayoffTable
action_labels
random_payoff_table
DecisionProblem
feasible_actions
feasible_payoff
restrict_scenarios
gain_over_baseline
expected_gain_over_baseline
```

## Aggregators

```@docs
AbstractAggregator
Expectation
WorstCase
BestCase
Hurwicz
Laplace
LowerExpectation
UpperExpectation
AlphaMaxmin
CVaR
Quantile
aggregate
```

## Regret

```@docs
regret_table
relative_regret_table
expected_regret
max_regret
```

## Rules

```@docs
decide
DecisionResult
first_choice
AbstractRule
DecisionRule
MaxExpectedUtility
Maximin
Maximax
MinimaxRegret
GammaMaximin
Satisfice
EAdmissible
Maximal
IntervalDominance
ParetoFront
```

## Rule combinators and robustness metrics

```@docs
AbstractTransform
IdentityTransform
RegretTransform
RelativeRegretTransform
AbstractSelector
Optimize
Threshold
RobustnessMetric
robustness
maximin_metric
maximax_metric
hurwicz_metric
laplace_metric
minimax_regret_metric
```

## Experiments

```@docs
Experiment
signal_labels
random_experiment
bayes_invert
BayesianInversion
garble
is_garbling_of
deficiency
```

## Value of information

```@docs
act_now_value
value_with_information
evpi
evppi
evsi
enbs
robust_evpi
robust_evsi
RobustVOI
```

## Fidelity

```@docs
total_variation
decision_loss_bound
```

## Reports and certificates

```@docs
DecisionReport
export_decision_certificate
CERTIFICATE_SCHEMA
export_garbling_witness
GARBLING_WITNESS_SCHEMA
export_eadmissible_witness
EADMISSIBLE_WITNESS_SCHEMA
```

## Exceptions

```@docs
DecisionAnalysisError
InvalidBeliefError
InvalidPayoffError
InvalidExperimentError
LabelMismatchError
BeliefFormError
NoFeasibleActionError
InvalidBaselineError
MissingBaselineError
InexactCertificateError
WitnessError
LinearProgramError
```
