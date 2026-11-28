# Risk attitude and multi-attribute utilities

A non-linear utility is applied to every run before anything is averaged, and its
expected utility is no longer a count of individuals; the certainty equivalent
returns to individuals.

## Utilities of one outcome

```@docs
LinearUtility
ExponentialUtility
PowerUtility
inverse_utility
certainty_equivalent
risk_premium
```

## Several attributes

```@docs
AdditiveUtility
MultiplicativeUtility
```

## Certainty-equivalent value of information

```@docs
ce_evpi
ce_evppi
ce_evsi
```
