# Scenario designs

A scenario design is a set of scenarios built from a few **axes**, each either a
decision (an intervention, under the control of whoever the design is for) or an
uncertainty. This page follows Runge et al. (2024, "Scenario design for infectious
disease projections: integrating concepts from decision analysis and experimental
design", *Epidemics*), who classify designs by what their axes are and by their
purpose, and apply the principles of experimental design to them: control, replication,
pairing of replicates across scenarios, and fractional factorials.

| purpose | axes | what to do with the runs |
|:--|:--|:--|
| decision making | decisions | compare the actions: [`decide`](@ref), gains over no intervention |
| value of information | a decision and an uncertainty | [`decision_robustness`](@ref): does the better action depend on the uncertainty? With probabilities, [`evpi`](@ref) and [`evppi`](@ref) |
| sensitivity analysis | uncertainties | [`factor_effects`](@ref); with a belief, [`variance_reduction_table`](@ref) |
| situational awareness | uncertainties, at plausible bounds | the range of the projections |
| horizon scanning | uncertainties, at extremes | the worst and best cases: [`Maximin`](@ref), [`MinimaxRegret`](@ref) over the scenario set |
| forecasting | uncertainties, with probabilities | [`scenario_forecast`](@ref) |

A design plans a table of runs: [`design_actions`](@ref) and [`design_draws`](@ref) go
to [`outcomes`](@ref), which pairs the replicates across scenarios by common random
numbers, and `DecisionProblem(design, table, utility)` is the problem that the runs
answer, over a [`ScenarioSet`](@ref) when the uncertainty axes have no probabilities.
The model of a multi-model hub is one more uncertainty axis.

```@docs
ScenarioAxis
decision_axis
uncertainty_axis
ScenarioDesign
design_class
scenarios
design_actions
design_draws
design_belief
fractional_design
factor_effects
scenario_forecast
decision_robustness
```
