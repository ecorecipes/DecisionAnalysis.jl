# Model-based methods, study designs and estimation

The methods that follow the R package [`voi`](https://chjackson.github.io/voi/) and the
reviews of Jackson et al. (2019, *Journal of the American Statistical Association*
114:1436; 2021, *Epidemiologic Methods* 10:20210012; 2022, *Annual Review of Statistics
and Its Application* 9:95): a further regression, the estimators that run the model
itself, standard study designs, value of information for a model that estimates a
quantity and takes no decision, and the net benefit of sampling over sample sizes. The
other methods of `voi` are on the page [Sample-based analysis](samples.md).

| `voi` | here |
|:--|:--|
| `evpi`, `evppi` (`gam`, `gp`, `so`, `sal`) | [`evpi`](@ref), [`evppi`](@ref) with [`GAMRegression`](@ref), [`GaussianProcessRegression`](@ref), [`StrongOakley`](@ref), [`Sadatsafavi`](@ref) |
| `evppi(method = "earth")` | [`MARSRegression`](@ref) |
| `evppi_mc` | [`evppi_nested`](@ref) |
| `evsi` by regression, `"is"`, `"mm"` | [`SummaryStatistics`](@ref), [`ImportanceSampling`](@ref), [`MomentMatching`](@ref) |
| `study = "binary"`, `"trial_binary"`, `"normal_known"` | [`BinomialStudy`](@ref), [`TwoArmBinomialStudy`](@ref), [`NormalStudy`](@ref) |
| `evppivar`, `evsivar` | [`expected_variance_reduction`](@ref), [`variance_reduction_table`](@ref) |
| `pop_voi`, `enbs`, `enbs_opt` | [`population_value`](@ref), [`enbs_curve`](@ref), [`optimal_sample_size`](@ref) |
| `check_regression` | [`regression_check`](@ref), [`compare_regressions`](@ref) |
| `evppi(method = "inla")`, `"bart"` | not available: see below |

Not available here: the INLA/SPDE Gaussian process of Heath, Manolopoulou and Baio
(2016), which needs the R package INLA (the built-in Gaussian process serves the same
case, five or more focal parameters), and Bayesian additive regression trees. Inputs in
cost-effectiveness form (costs, effects and a willingness to pay) and the import from
`heemod` are specific to health economics and to R.

## A further regression, and checks of a regression

```@docs
MARSRegression
regression_check
compare_regressions
```

## Estimators that run the model

```@docs
evppi_nested
MultilevelMC
MomentMatching
```

## Study designs

```@docs
AbstractStudyDesign
BinomialStudy
TwoArmBinomialStudy
NormalStudy
CountStudy
OccupancyStudy
CaptureRecaptureStudy
MvNormalStudy
simulate_study
study_loglikelihood
study_posterior
```

## Estimation

```@docs
variance_reduction_table
equivalent_binomial_sample_size
```

## Population value and the net benefit of sampling

```@docs
population_value
enbs_curve
optimal_sample_size
```
