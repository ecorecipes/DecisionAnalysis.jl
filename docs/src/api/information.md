# Experiments and exact value of information

The definitions on finite problems. The same functions, called with a method or a study, estimate these quantities from a sample: see [Sample-based analysis](samples.md).

## Experiments

```@docs
Experiment
signal_labels
random_experiment
gaussian_experiment
bayes_invert
BayesianInversion
garble
is_garbling_of
garbling_result
GarblingResult
deficiency
```

## Value of information

```@docs
act_now_value
value_with_information
evpi
evppi
hypothesis_partition
two_stage_value
evsi
enbs
information_can_change_decision
robust_evpi
robust_evsi
RobustVOI
```

## Outcomes, estimation and design

```@docs
voi_profile
expected_variance_reduction
optimal_design
value_of_waiting
survey_or_act
select_under_budget
```

## Influence diagrams

```@docs
normal_form
```

## Fidelity

```@docs
total_variation
decision_loss_bound
payoff_range
largest_action_range
largest_regret
deficiency_value_bound
```
