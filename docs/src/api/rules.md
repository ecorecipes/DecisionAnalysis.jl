# Aggregators, regret and rules

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
SmoothAmbiguity
CVaR
Quantile
aggregate
```

## Regret

```@docs
regret_table
RegretTable
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
StochasticDominance
```

## Stochastic dominance, info-gap and scenario discovery

```@docs
fsd_dominates
ssd_dominates
info_gap_robustness
info_gap_opportuneness
prim_boxes
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
